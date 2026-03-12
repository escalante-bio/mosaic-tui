"""BoltzGen diffusion-based binder design worker (runs on Modal GPU)."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
import modal
import numpy as np
from mosaic.common import TOKENS
from mosaic.models.boltzgen import (
    CoordsToToken,
    Sampler,
    load_boltzgen,
    load_features_and_structure_writer,
)
from mosaic.models.protenix import Protenix2025

from mosaic_tui.design_common import (
    GPU_VOLUMES,
    BoltzGenConfig,
    DesignStartMsg,
    ErrorMsg,
    GpuContext,
    GpuDoneMsg,
    RankingConfig,
    RankingMsg,
    ResultMsg,
    StatusMsg,
    StepMsg,
    app,
    build_result_row,
    cache_volume,
    configure_jax_cache,
    image,
    load_template_chain,
    make_ranker,
)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 8,
    volumes=GPU_VOLUMES,
)
def run_boltzgen_designs(
    num_designs: int,
    binder_length: int,
    cif_content: str | None,
    chain_id: str | None,
    hotspots: list[int] | None,
    run_name: str,
    start_idx: int,
    queue_id: str,
    gpu_id: int,
    method: BoltzGenConfig,
    ranking: RankingConfig,
    target_seq: str = "",
) -> list[dict]:
    """Run BoltzGen binder designs on a single GPU with queue-based progress."""
    configure_jax_cache()

    queue = modal.Queue.from_id(queue_id)
    ctx = GpuContext(gpu_id, queue)

    config_dict = {
        "method_type": "boltzgen",
        "method": dataclasses.asdict(method),
        "ranking": dataclasses.asdict(ranking),
        "hotspots": hotspots,
    }

    try:
        assert cif_content is not None, "BoltzGen requires a structure file (--cif)"
        assert chain_id is not None, "BoltzGen requires a chain ID"
        ctx.send(StatusMsg(gpu=gpu_id, text="Loading models...", hw=ctx.hw_stats()))

        boltzgen = load_boltzgen()
        if method.use_rl_checkpoint:
            boltzgen = eqx.tree_deserialise_leaves(
                "/root/.boltz/boltzgen_checkpoints/diverse_rl.eqx", boltzgen
            )
        folder = Protenix2025()
        ctx.send(StatusMsg(gpu=gpu_id, text="Models loaded", hw=ctx.hw_stats()))

        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as f:
            f.write(cif_content)
            cif_path = f.name

        _template_st, template_chain = load_template_chain(cif_content, chain_id)

        hotspot_line = ""
        if hotspots:
            hotspot_line = (
                f"            binding_types:\n"
                f"              binding: [{','.join(str(h) for h in hotspots)}]\n"
            )

        yaml_str = (
            f"entities:\n"
            f"  - file:\n"
            f"      path: target.cif\n"
            f"      include:\n"
            f"        - chain:\n"
            f'            id: "{chain_id}"\n'
            f"{hotspot_line}"
            f"  - protein:\n"
            f"      id: B\n"
            f'      sequence: "{binder_length}"\n'
            f"      msa: -1\n"
        )

        cif_files = {"target.cif": Path(cif_path)}

        def build_features() -> tuple:
            return load_features_and_structure_writer(yaml_str, files=cif_files)

        @eqx.filter_jit
        def _sample(
            sampler: Sampler,
            structure_module: eqx.Module,
            num_sampling_steps: int,
            step_scale: float,
            noise_scale: float,
            key: jax.Array,
        ) -> jax.Array:
            return sampler(
                structure_module=structure_module,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                noise_scale=noise_scale,
                key=key,
            )

        # --- JIT warmup ---
        ctx.send(StatusMsg(gpu=gpu_id, text="JIT warmup...", hw=ctx.hw_stats()))
        t0 = time.perf_counter()

        warmup_features, _ = build_features()
        warmup_sampler = Sampler.from_features(
            model=boltzgen,
            features=warmup_features,
            recycling_steps=method.recycling_steps,
            key=jax.random.key(0),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            _sample(
                warmup_sampler,
                boltzgen.structure_module,
                method.num_sampling_steps,
                method.step_scale,
                method.noise_scale,
                key=jax.random.key(0),
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
        rng = np.random.default_rng()
        results = []

        for i in range(num_designs):
            idx = start_idx + i
            seed = int(rng.integers(0, 0xFFFFFF))

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
            key = jax.random.key(seed)

            features, _ = build_features()
            coords2token = CoordsToToken(features)

            sampler = Sampler.from_features(
                model=boltzgen,
                features=features,
                recycling_steps=method.recycling_steps,
                key=key,
            )
            coords = _sample(
                sampler,
                boltzgen.structure_module,
                method.num_sampling_steps,
                method.step_scale,
                method.noise_scale,
                key=jax.random.fold_in(key, 1),
            )

            token_indices = coords2token(coords[0])
            seq_str = "".join(TOKENS[int(t)] for t in token_indices)

            ctx.send(
                StepMsg(
                    gpu=gpu_id,
                    design_idx=idx,
                    phase="sample",
                    step=1,
                    total_steps=1,
                    loss=0.0,
                    hw=ctx.hw_stats(),
                )
            )

            design_time = time.perf_counter() - dt0

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
                design_loss=0.0,
                rank_result=rank_result,
                design_time=design_time,
                rank_time=rank_time,
                hyperparams={
                    "num_sampling_steps": method.num_sampling_steps,
                    "step_scale": method.step_scale,
                    "noise_scale": method.noise_scale,
                },
                config_dict=config_dict,
            )
            results.append(row)

            ctx.send(ResultMsg(gpu=gpu_id, data={k: v for k, v in row.items()}))

        cache_volume.commit()

    except Exception as e:
        ctx.send(ErrorMsg(gpu=gpu_id, text=str(e)))
        raise
    finally:
        ctx.send(GpuDoneMsg(gpu=gpu_id))

    return results
