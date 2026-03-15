"""Simplex APGM binder design worker (runs on Modal GPU)."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import operator
import time
from functools import reduce

import modal
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys
    from mosaic.models.protenix import Protenix2025
    from mosaic.optimizers import simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain
    import mosaic.losses.structure_prediction as sp
except ImportError:
    pass

from mosaic_tui.design_common import (
    GPU_VOLUMES,
    DesignStartMsg,
    ErrorMsg,
    GpuContext,
    GpuDoneMsg,
    RankingConfig,
    RankingMsg,
    ResultMsg,
    SimplexConfig,
    StatusMsg,
    StepMsg,
    app,
    build_result_row,
    cache_volume,
    configure_jax_cache,
    image,
    load_template_chain,
    make_ranker,
    sample_hyperparams,
    validate_design_inputs,
)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 8,
    volumes=GPU_VOLUMES,
)
def run_designs(
    num_designs: int,
    binder_length: int,
    cif_content: str | None,
    chain_id: str | None,
    hotspots: list[int] | None,
    run_name: str,
    start_idx: int,
    queue_id: str,
    gpu_id: int,
    method: SimplexConfig,
    ranking: RankingConfig,
    target_seq: str = "",
) -> list[dict]:
    """Run a chunk of binder designs on a single GPU with queue-based progress."""
    configure_jax_cache()

    queue = modal.Queue.from_id(queue_id)
    ctx = GpuContext(gpu_id, queue)

    config_dict = {
        "method_type": "simplex",
        "method": dataclasses.asdict(method),
        "ranking": dataclasses.asdict(ranking),
        "hotspots": hotspots,
    }

    try:
        ctx.send(StatusMsg(gpu=gpu_id, text="Loading models...", hw=ctx.hw_stats()))

        folder = Protenix2025()
        mpnn = load_mpnn_sol(0.05)
        ctx.send(StatusMsg(gpu=gpu_id, text="Models loaded", hw=ctx.hw_stats()))

        if cif_content is not None:
            assert chain_id is not None
            _template_st, template_chain = load_template_chain(cif_content, chain_id)
        else:
            template_chain = None

        use_msa = method.use_msa

        features, design_writer = folder.binder_features(
            binder_length=binder_length,
            chains=[
                TargetChain(
                    sequence=target_seq,
                    use_msa=use_msa,
                    template_chain=template_chain,
                )
            ],
        )

        validate_design_inputs(
            cif_content=cif_content,
            chain_id=chain_id,
            target_seq=target_seq,
            template_chain=template_chain,
            binder_length=binder_length,
            features=features,
        )

        cys_idx = TOKENS.index("C")
        bias = jnp.zeros((binder_length, 20)).at[:, cys_idx].set(-1e6)

        lw = method.loss_weights
        sp_loss = reduce(
            operator.add,
            [
                w * t
                for w, t in [
                    (
                        lw.binder_target_contact,
                        sp.BinderTargetContact(epitope_idx=hotspots),
                    ),
                    (lw.within_binder_contact, sp.WithinBinderContact()),
                    (
                        lw.inverse_folding,
                        InverseFoldingSequenceRecovery(
                            mpnn,
                            temp=jnp.array(method.mpnn_temp),
                            bias=bias,
                        ),
                    ),
                    (lw.target_binder_pae, sp.TargetBinderPAE()),
                    (lw.binder_target_pae, sp.BinderTargetPAE()),
                    (lw.iptm, sp.IPTMLoss()),
                    (lw.within_binder_pae, sp.WithinBinderPAE()),
                    (lw.ptm, sp.pTMEnergy()),
                    (lw.plddt, sp.PLDDTLoss()),
                ]
                if w
            ],
        )

        multisample_loss = folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=method.recycling_steps,
            num_samples=method.num_samples,
        )
        design_loss = NoCys(loss=multisample_loss)

        # --- JIT warmup ---
        ctx.send(StatusMsg(gpu=gpu_id, text="JIT warmup...", hw=ctx.hw_stats()))
        t0 = time.perf_counter()
        _dummy = jax.nn.softmax(
            jax.random.gumbel(jax.random.key(0), shape=(binder_length, 19))
        )
        with contextlib.redirect_stdout(io.StringIO()):
            simplex_APGM(
                loss_function=design_loss,
                x=_dummy,
                n_steps=1,
                stepsize=1.0,
                scale=1.0,
                logspace=False,
            )
        warmup_time = time.perf_counter() - t0
        ctx.send(
            StatusMsg(
                gpu=gpu_id,
                text=f"JIT done ({warmup_time:.0f}s), designing...",
                hw=ctx.hw_stats(),
            )
        )

        ranker = make_ranker(
            folder=folder,
            binder_length=binder_length,
            target_seq=target_seq,
            template_chain=template_chain,
            ranking=ranking,
        )

        # --- Design loop ---
        stepsize_base = np.sqrt(binder_length)
        fo = method.fixed_optim
        rng = np.random.default_rng()
        results = []

        def make_traj_fn(phase: str, total_steps: int):
            """Create a trajectory callback that pushes per-step loss to queue."""
            counter = [0]

            def _traj_fn(aux, x):
                counter[0] += 1
                loss_val = float(aux["loss"])
                ctx.send(
                    StepMsg(
                        gpu=gpu_id,
                        design_idx=idx,
                        phase=phase,
                        step=counter[0],
                        total_steps=total_steps,
                        loss=loss_val,
                        hw=ctx.hw_stats(),
                    )
                )
                return loss_val

            return _traj_fn

        for i in range(num_designs):
            idx = start_idx + i
            seed = int(rng.integers(0, 0xFFFFFF))
            design_rng = np.random.default_rng(seed)
            hp = sample_hyperparams(design_rng, method.hyperparam_ranges)

            ctx.send(
                DesignStartMsg(
                    gpu=gpu_id,
                    design_idx=idx,
                    design_num=i + 1,
                    n_designs=num_designs,
                    seed=seed,
                    hw=ctx.hw_stats(),
                )
            )

            dt0 = time.perf_counter()

            pssm = hp.init_scale * jax.random.gumbel(
                key=jax.random.key(seed), shape=(binder_length, 19)
            )

            # Phase 1: explore (logspace=False, use best_x)
            with contextlib.redirect_stdout(io.StringIO()):
                _, pssm, traj_p1 = simplex_APGM(
                    loss_function=design_loss,
                    x=jax.nn.softmax(pssm),
                    n_steps=hp.p1_steps,
                    stepsize=hp.p1_stepsize_factor * stepsize_base,
                    momentum=hp.p1_momentum,
                    scale=fo.p1_scale,
                    logspace=False,
                    max_gradient_norm=1.0,
                    trajectory_fn=make_traj_fn("p1", hp.p1_steps),
                )

            # Phase 2: refine (logspace=True, use x)
            with contextlib.redirect_stdout(io.StringIO()):
                pssm, _, traj_p2 = simplex_APGM(
                    loss_function=design_loss,
                    x=jnp.log(pssm + 1e-5),
                    n_steps=hp.p2_steps,
                    stepsize=hp.p2_stepsize_factor * stepsize_base,
                    momentum=fo.p2_momentum,
                    scale=fo.p2_scale,
                    logspace=True,
                    max_gradient_norm=1.0,
                    trajectory_fn=make_traj_fn("p2", hp.p2_steps),
                )

            # Phase 3: polish (logspace=True, use x)
            with contextlib.redirect_stdout(io.StringIO()):
                pssm, _, traj_p3 = simplex_APGM(
                    loss_function=design_loss,
                    x=jnp.log(pssm + 1e-5),
                    n_steps=hp.p3_steps,
                    stepsize=hp.p3_stepsize_factor * stepsize_base,
                    momentum=fo.p3_momentum,
                    scale=fo.p3_scale,
                    logspace=True,
                    max_gradient_norm=1.0,
                    trajectory_fn=make_traj_fn("p3", hp.p3_steps),
                )

            # Extract sequence
            full_pssm = NoCys.sequence(pssm)
            seq_indices = full_pssm.argmax(-1)
            seq_str = "".join(TOKENS[int(j)] for j in seq_indices)

            design_time = time.perf_counter() - dt0
            design_loss_val = float(traj_p3[-1]) if traj_p3 else float("nan")

            # --- Ranking ---
            ctx.send(RankingMsg(gpu=gpu_id, design_idx=idx, hw=ctx.hw_stats()))
            rt0 = time.perf_counter()
            rank_result = ranker(seq_str)
            rank_time = time.perf_counter() - rt0

            row = build_result_row(
                idx=idx,
                seed=seed,
                binder_length=binder_length,
                seq_str=seq_str,
                design_loss=design_loss_val,
                rank_result=rank_result,
                design_time=design_time,
                rank_time=rank_time,
                hyperparams=dataclasses.asdict(hp),
                config_dict=config_dict,
            )
            row["trajectory"] = {
                "p1": list(traj_p1),
                "p2": list(traj_p2),
                "p3": list(traj_p3),
            }
            results.append(row)

            ctx.send(
                ResultMsg(
                    gpu=gpu_id,
                    data={k: v for k, v in row.items() if k not in ("trajectory",)},
                )
            )

        cache_volume.commit()

    except Exception as e:
        ctx.send(ErrorMsg(gpu=gpu_id, text=str(e)))
        raise
    finally:
        ctx.send(GpuDoneMsg(gpu=gpu_id))

    return results
