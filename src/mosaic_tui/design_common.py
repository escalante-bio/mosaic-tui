"""Shared Modal image, app, volumes, and utilities for binder design workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import modal
import numpy as np
from biotite.structure import AtomArray

from mosaic.common import TOKENS, LinearCombination
from mosaic.losses.protenix import biotite_array_to_gemmi_struct
from mosaic.models.protenix import Protenix
from mosaic.structure_prediction import StructurePrediction, TargetChain

from mosaic_tui.ranking_loss import RankingLoss

# ---------------------------------------------------------------------------
# Target data model (discriminated union)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CifTarget:
    path: str
    chain: str


@dataclass(frozen=True)
class SeqTarget:
    sequence: str


Target = CifTarget | SeqTarget


def target_label(t: Target) -> str:
    """Human-readable label for a target."""
    match t:
        case CifTarget(path=path, chain=chain):
            return f"{PurePath(path).stem} chain {chain}"
        case SeqTarget(sequence=seq):
            return f"seq ({len(seq)}aa)"


# ---------------------------------------------------------------------------
# Config data model (frozen dataclasses)
# ---------------------------------------------------------------------------

LOSS_NAMES = {
    "binder_target_contact": "BTContact",
    "within_binder_contact": "WBContact",
    "inverse_folding": "IF",
    "target_binder_pae": "TBPAE",
    "binder_target_pae": "BTPAE",
    "iptm": "iPTM",
    "within_binder_pae": "WBPAE",
    "ptm": "pTM",
    "plddt": "pLDDT",
}


@dataclass(frozen=True)
class LossWeights:
    binder_target_contact: float = 1.0
    within_binder_contact: float = 1.0
    inverse_folding: float = 10.0
    target_binder_pae: float = 0.05
    binder_target_pae: float = 0.05
    iptm: float = 0.025
    within_binder_pae: float = 0.4
    ptm: float = 0.025
    plddt: float = 0.1

    def describe(self) -> str:
        terms = []
        for key, name in LOSS_NAMES.items():
            w = getattr(self, key)
            if w:
                terms.append(f"{w:g}*{name}" if w != 1.0 else name)
        return " + ".join(terms)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float


@dataclass(frozen=True)
class HyperparamRanges:
    p1_steps: Range = field(default_factory=lambda: Range(100, 110))
    p1_stepsize_factor: Range = field(default_factory=lambda: Range(0.08, 0.30))
    p1_momentum: Range = field(default_factory=lambda: Range(0.1, 0.5))
    p2_steps: Range = field(default_factory=lambda: Range(30, 70))
    p2_stepsize_factor: Range = field(default_factory=lambda: Range(0.3, 0.7))
    p3_steps: Range = field(default_factory=lambda: Range(10, 25))
    p3_stepsize_factor: Range = field(default_factory=lambda: Range(0.3, 0.7))
    init_scale: Range = field(default_factory=lambda: Range(0.75, 5.0))


@dataclass(frozen=True)
class FixedOptim:
    p1_scale: float = 1.0
    p2_scale: float = 1.25
    p3_scale: float = 1.4
    p2_momentum: float = 0.0
    p3_momentum: float = 0.0


@dataclass(frozen=True)
class RankingConfig:
    num_samples: int = 6
    recycling_steps: int = 10
    fast_ranking: bool = False
    use_msa: bool = True
    rmsd_cutoff: float = 5.0


@dataclass(frozen=True)
class RunParams:
    binder_length: int = 120
    num_designs: int = 32
    num_gpus: int = 1
    run: str = ""
    trim_terminals: bool = True


@dataclass(frozen=True)
class SimplexConfig:
    loss_weights: LossWeights = field(default_factory=LossWeights)
    hyperparam_ranges: HyperparamRanges = field(default_factory=HyperparamRanges)
    fixed_optim: FixedOptim = field(default_factory=FixedOptim)
    mpnn_temp: float = 0.001
    recycling_steps: int = 6
    num_samples: int = 4
    use_msa: bool = True

    def describe(self) -> str:
        return (
            f"Protenix R={self.recycling_steps}"
            f" S={self.num_samples}: {self.loss_weights.describe()}"
        )


@dataclass(frozen=True)
class BoltzGenConfig:
    use_rl_checkpoint: bool = True
    num_sampling_steps: int = 500
    step_scale: float = 2.0
    noise_scale: float = 0.88
    recycling_steps: int = 3

    def describe(self) -> str:
        weights = "RL" if self.use_rl_checkpoint else "base"
        return (
            f"BoltzGen ({weights}): steps={self.num_sampling_steps}"
            f" scale={self.step_scale} noise={self.noise_scale}"
        )


MethodConfig = SimplexConfig | BoltzGenConfig


@dataclass(frozen=True)
class DesignConfig:
    run_params: RunParams = field(default_factory=RunParams)
    method: MethodConfig = field(default_factory=SimplexConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)


def default_config() -> DesignConfig:
    """Construct DesignConfig with all default values."""
    return DesignConfig()


def default_method(method_type: str) -> MethodConfig:
    """Return default method config for the given type string."""
    match method_type:
        case "boltzgen":
            return BoltzGenConfig()
        case _:
            return SimplexConfig()


def config_from_dict(d: dict) -> DesignConfig:
    """Reconstruct DesignConfig from a saved config dict (config.json)."""
    method_type = d.get("method_type", "simplex")
    method_data = d.get("method", {})
    match method_type:
        case "boltzgen":
            bg_fields = {f.name for f in BoltzGenConfig.__dataclass_fields__.values()}
            method: MethodConfig = BoltzGenConfig(
                **{k: v for k, v in method_data.items() if k in bg_fields}
            )
        case _:
            method = SimplexConfig(
                loss_weights=LossWeights(**method_data.get("loss_weights", {})),
                hyperparam_ranges=HyperparamRanges(
                    **{
                        k: Range(lo=v["lo"], hi=v["hi"])
                        for k, v in method_data.get("hyperparam_ranges", {}).items()
                    }
                ),
                fixed_optim=FixedOptim(**method_data.get("fixed_optim", {})),
                mpnn_temp=method_data.get("mpnn_temp", 0.001),
                recycling_steps=method_data.get("recycling_steps", 6),
                num_samples=method_data.get("num_samples", 4),
                use_msa=method_data.get("use_msa", True),
            )

    rp = d.get("run_params", {})
    rk = d.get("ranking", {})
    return DesignConfig(
        run_params=RunParams(
            binder_length=rp.get("binder_length", 120),
            num_designs=rp.get("num_designs", 32),
            num_gpus=rp.get("num_gpus", 1),
            run=rp.get("run", d.get("run_name", "")),
            trim_terminals=rp.get("trim_terminals", True),
        ),
        method=method,
        ranking=RankingConfig(
            num_samples=rk.get("num_samples", 6),
            recycling_steps=rk.get("recycling_steps", 10),
            fast_ranking=rk.get("fast_ranking", False),
            use_msa=rk.get("use_msa", True),
            rmsd_cutoff=rk.get("rmsd_cutoff", 5.0),
        ),
    )


# --- Modal image ---

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget")
    .add_local_file(
        str(_PROJECT_ROOT / "pyproject.toml"), "/app/pyproject.toml", copy=True
    )
    .add_local_file(str(_PROJECT_ROOT / "uv.lock"), "/app/uv.lock", copy=True)
    .workdir("/app")
    .run_commands(
        "uv export --frozen --no-hashes --no-emit-project > /tmp/requirements.txt"
        " && uv pip install --system -r /tmp/requirements.txt"
    )
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
    .add_local_file(
        str(_PKG_DIR / "__init__.py"), "/app/mosaic_tui/__init__.py", copy=True
    )
    .add_local_file(
        str(_PKG_DIR / "design_common.py"),
        "/app/mosaic_tui/design_common.py",
        copy=True,
    )
    .add_local_file(
        str(_PKG_DIR / "ranking_loss.py"), "/app/mosaic_tui/ranking_loss.py", copy=True
    )
    .add_local_file(
        str(_PKG_DIR / "worker_simplex.py"),
        "/app/mosaic_tui/worker_simplex.py",
        copy=True,
    )
    .add_local_file(
        str(_PKG_DIR / "worker_boltzgen.py"),
        "/app/mosaic_tui/worker_boltzgen.py",
        copy=True,
    )
)

# --- Modal app + volumes ---

app = modal.App(name="binder-design-rich")

protenix_volume = modal.Volume.from_name("protenix-weights", create_if_missing=True)
boltzgen_volume = modal.Volume.from_name("boltzgen-weights", create_if_missing=True)

WEIGHTS_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/.protenix": protenix_volume,
    "/root/.boltz": boltzgen_volume,
}

cache_volume = modal.Volume.from_name("jax-cache", create_if_missing=True)

GPU_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/jax_cache": cache_volume,
    **WEIGHTS_VOLUMES,
}


@app.function(image=image, volumes=WEIGHTS_VOLUMES, timeout=60 * 30)
def download_weights() -> None:
    """Download all model weights into volumes (runs once, no GPU)."""
    from pathlib import Path

    marker = Path("/root/.protenix/.complete")
    if marker.exists():
        return

    import subprocess

    from protenix.backend import download_data, _resolve_model_path

    # Protenix weights
    download_data()
    _resolve_model_path("protenix_base_20250630_v1.0.0")
    protenix_volume.commit()

    # BoltzGen weights
    boltz_dir = Path("/root/.boltz")
    boltz_checkpoints = ["boltzgen1_adherence.ckpt", "boltzgen1_diverse.ckpt"]
    if not all((boltz_dir / ckpt).exists() for ckpt in boltz_checkpoints):
        from mosaic.models.boltzgen import load_boltzgen

        load_boltzgen()

    # RL post-training checkpoint
    dest = boltz_dir / "boltzgen_checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    rl_path = dest / "diverse_rl.eqx"
    if not rl_path.exists():
        subprocess.run(
            [
                "wget",
                "-O",
                str(rl_path),
                "https://huggingface.co/escalante-bio/boltzgen-posttraining/resolve/main/boltzgen1_diverse_rl.eqx",
            ],
            check=True,
        )
    boltzgen_volume.commit()

    marker.touch()
    protenix_volume.commit()


# --- GPU worker utilities ---


def configure_jax_cache() -> None:
    """Set JAX persistent compilation cache to /jax_cache volume."""
    import jax

    jax.config.update("jax_compilation_cache_dir", "/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )


class GpuContext:
    """GPU monitoring + queue messaging for remote workers."""

    def __init__(self, gpu_id: int, queue: modal.Queue) -> None:
        self.gpu_id = gpu_id
        self._queue = queue
        self._call_count = 0
        self._hw_cache: HWStats | None = None

        self._nvml_handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def hw_stats(self) -> HWStats | None:
        """Sample GPU utilization, power, temperature. Throttled to every 5th call."""
        self._call_count += 1
        if self._call_count % 5 != 1 and self._hw_cache is not None:
            return self._hw_cache
        if self._nvml_handle is None:
            return None
        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self._hw_cache = HWStats(
                gpu_util=util.gpu,
                power_w=round(power_mw / 1000),
                temp_c=temp,
            )
            return self._hw_cache
        except Exception:
            return None

    def send(self, msg: WorkerMessage) -> None:
        """Put a message on the queue (best-effort, never crashes worker)."""
        try:
            self._queue.put(msg)
        except Exception:
            pass


def rename_binder_residues(
    st: gemmi.Structure, seq_str: str, chain_idx: int = 0
) -> None:
    """Rename UNK binder residues to proper 3-letter codes in-place."""
    three_letter = gemmi.expand_one_letter_sequence(seq_str, gemmi.ResidueKind.AA)
    chain = st[0][chain_idx]
    for i, res in enumerate(chain):
        if i < len(three_letter) and res.name == "UNK":
            res.name = three_letter[i]


def _disulf_key(
    chain1: str, seq1: str, chain2: str, seq2: str
) -> tuple[str, str, str, str]:
    """Canonical key for a disulfide bond pair (order-independent)."""
    a, b = (chain1, seq1), (chain2, seq2)
    if a > b:
        a, b = b, a
    return (a[0], a[1], b[0], b[1])


def _detect_disulfide_bonds(s: gemmi.Structure) -> int:
    """Detect disulfide bonds from SG-SG distances and add as connections."""
    existing = {
        _disulf_key(
            conn.partner1.chain_name,
            str(conn.partner1.res_id.seqid),
            conn.partner2.chain_name,
            str(conn.partner2.res_id.seqid),
        )
        for conn in s.connections
        if conn.type == gemmi.ConnectionType.Disulf
    }

    ns = gemmi.NeighborSearch(s[0], s.cell, 3.0)
    ns.populate()

    added = 0
    seen: set[tuple[str, str, str, str]] = set()
    for chain in s[0]:
        for res in chain:
            if res.name != "CYS":
                continue
            sg = res.find_atom("SG", "\0")
            if sg is None:
                continue
            for mark in ns.find_atoms(sg.pos, "\0", radius=2.5):
                cra = mark.to_cra(s[0])
                if cra.residue.name != "CYS" or cra.atom.name != "SG":
                    continue
                if cra.residue == res:
                    continue

                pair_key = _disulf_key(
                    chain.name,
                    str(res.seqid),
                    cra.chain.name,
                    str(cra.residue.seqid),
                )
                if pair_key in seen or pair_key in existing:
                    continue
                seen.add(pair_key)

                conn = gemmi.Connection()
                conn.name = f"disulf{len(existing) + added + 1}"
                conn.type = gemmi.ConnectionType.Disulf
                conn.partner1.chain_name = chain.name
                conn.partner1.res_id.seqid = res.seqid
                conn.partner1.atom_name = "SG"
                conn.partner2.chain_name = cra.chain.name
                conn.partner2.res_id.seqid = cra.residue.seqid
                conn.partner2.atom_name = "SG"
                s.connections.append(conn)
                added += 1

    return added


def preprocess_target_cif(
    path: str, chain_id: str, *, trim_terminals: bool = True
) -> str:
    """Preprocess a target CIF: clean up, fill gaps, renumber 0-based.

    Returns the preprocessed structure as an mmCIF string with all residues
    numbered 0, 1, 2, ... so downstream consumers get consistent 0-indexed positions.

    When trim_terminals is False, unresolved terminal residues from the entity
    sequence are included (with no coordinates).
    """
    raw = gemmi.read_structure(path)

    # Extract single model + single chain (drops other chains, ligands, metals)
    sel = gemmi.Selection(f"/{raw[0].name}/{chain_id}")
    s = sel.copy_structure_selection(raw)
    s.remove_alternative_conformations()
    s.remove_hydrogens()
    s.remove_ligands_and_waters()
    s.remove_empty_chains()

    ch = s[0].find_chain(chain_id)
    polymer = ch.get_polymer()
    entity = s.get_entity_of(polymer)
    if entity is None:
        raise ValueError(f"No polymer entity found for chain {chain_id} in {path}")

    full_seq_3 = entity.full_sequence
    res_by_label = {r.label_seq: r for r in polymer}

    if trim_terminals:
        first, last = min(res_by_label), max(res_by_label)
    else:
        first, last = 1, len(full_seq_3)

    # Build chain with gaps filled
    new_ch = gemmi.Chain(chain_id)
    seq3: list[str] = []
    n_gaps = 0
    for pos in range(first, last + 1):
        if pos in res_by_label:
            new_ch.add_residue(res_by_label[pos])
            seq3.append(res_by_label[pos].name)
        else:
            res = gemmi.Residue()
            res.name = full_seq_3[pos - 1]
            res.seqid = gemmi.SeqId(str(pos))
            new_ch.add_residue(res)
            seq3.append(full_seq_3[pos - 1])
            n_gaps += 1

    if n_gaps:
        print(f"Filled {n_gaps} gap residues")

    # Build old_seqid -> new_seqid mapping, then renumber 0-based.
    # Keep label_seq 1-based to match entity_poly_seq.num (CIF convention).
    old_to_new: dict[str, str] = {}
    for i, residue in enumerate(new_ch):
        old_to_new[str(residue.seqid)] = str(i)
    for i, residue in enumerate(new_ch):
        residue.seqid = gemmi.SeqId(str(i))
        residue.label_seq = i + 1

    s[0].remove_chain(chain_id)
    s[0].add_chain(new_ch)

    # Update connection references (e.g. disulfide bonds) to new seqids
    for conn in s.connections:
        for partner in [conn.partner1, conn.partner2]:
            if partner.chain_name == chain_id:
                new_seq = old_to_new.get(str(partner.res_id.seqid))
                if new_seq is not None:
                    partner.res_id.seqid = gemmi.SeqId(new_seq)

    # Auto-detect disulfide bonds not annotated in the file
    n_disulf = _detect_disulfide_bonds(s)
    if n_disulf:
        print(f"Detected {n_disulf} disulfide bonds from coordinates")

    # Rebuild entity metadata, then restore full_sequence (setup_entities
    # resets it to only resolved residues, losing unresolved terminals).
    s.setup_entities()
    s.assign_serial_numbers()
    entity = s.get_entity_of(s[0].find_chain(chain_id).get_polymer())
    if entity is not None:
        entity.full_sequence = seq3

    return s.make_mmcif_document().as_string()


def target_seq_from_cif(cif_content: str, chain_id: str) -> str:
    """Extract one-letter target sequence from a (preprocessed) CIF string.

    Reads from the entity full_sequence so unresolved residues are included.
    """
    doc = gemmi.cif.read_string(cif_content)
    st = gemmi.make_structure_from_block(doc.sole_block())
    ch = st[0].find_chain(chain_id)
    polymer = ch.get_polymer()
    entity = st.get_entity_of(polymer)
    if entity is not None:
        return gemmi.one_letter_code(entity.full_sequence)
    return gemmi.one_letter_code([r.name for r in polymer])


def _seq_to_one_hot(seq_str: str) -> jax.Array:
    """Convert amino acid string to one-hot JAX array."""
    indices = jnp.array([TOKENS.index(c) for c in seq_str], dtype=jnp.int32)
    return jax.nn.one_hot(indices, 20)


@eqx.filter_jit
def _eval_ranking_with_pred(
    loss: RankingLoss, pssm: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """JIT-compiled fused ranking loss + structure prediction."""
    return loss(pssm, key=key)


@dataclass(frozen=True)
class RankingResult:
    """Output of ranking a single designed sequence."""

    ranking_loss: float
    iptm: float
    mean_plddt: float
    monomer_rmsd: float
    pdb_string: str
    monomer_pdb_string: str


def _refold_and_check(
    pred: StructurePrediction,
    mono_pred: StructurePrediction,
    seq_str: str,
    rank_val: float,
    rmsd_cutoff: float = 5.0,
) -> RankingResult:
    """Compare complex vs monomer structures and build ranking result."""
    binder_length = len(seq_str)
    complex_cas = [r.sole_atom("CA").pos for r in pred.st[0][0]]
    mono_cas = [r.sole_atom("CA").pos for r in mono_pred.st[0][0]]
    sup = gemmi.superpose_positions(complex_cas[:binder_length], mono_cas)

    rename_binder_residues(pred.st, seq_str, chain_idx=0)
    rename_binder_residues(mono_pred.st, seq_str, chain_idx=0)

    return RankingResult(
        ranking_loss=rank_val if sup.rmsd <= rmsd_cutoff else float("inf"),
        iptm=float(pred.iptm),
        mean_plddt=float(pred.plddt.mean()),
        monomer_rmsd=sup.rmsd,
        pdb_string=pred.st.make_pdb_string(),
        monomer_pdb_string=mono_pred.st.make_pdb_string(),
    )


def _rank_and_refold(
    folder: Protenix,
    seq_oh: jax.Array,
    rank_val: float,
    pred: StructurePrediction,
    mono_features: dict[str, np.ndarray],
    mono_writer: AtomArray,
    recycling_steps: int,
    seq_str: str,
    rmsd_cutoff: float = 5.0,
) -> RankingResult:
    """Refold as monomer and compare to complex prediction."""
    mono_pred = folder.predict(
        PSSM=seq_oh,
        features=mono_features,
        writer=mono_writer,
        recycling_steps=recycling_steps,
        key=jax.random.key(0),
    )
    return _refold_and_check(pred, mono_pred, seq_str, rank_val, rmsd_cutoff)


class FastRanker(eqx.Module):
    """Ranks using pre-built design features (skips feature rebuild per sequence)."""

    folder: Protenix
    ranking_terms: LinearCombination
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    rmsd_cutoff: float = eqx.field(static=True)
    mono_features: dict[str, np.ndarray]
    mono_writer: AtomArray
    design_features: dict[str, np.ndarray]
    design_writer: AtomArray

    def __call__(self, seq_str: str) -> RankingResult:
        seq_oh = _seq_to_one_hot(seq_str)
        loss = RankingLoss(
            model=self.folder.protenix,
            features=self.design_features,
            loss=self.ranking_terms,
            recycling_steps=self.recycling_steps,
            sampling_steps=self.folder.default_sample_steps,
            num_samples=self.num_samples,
        )
        rank_val, coords, pae, plddt, iptm = _eval_ranking_with_pred(
            loss, seq_oh, key=jax.random.key(0)
        )
        pred = StructurePrediction(
            st=biotite_array_to_gemmi_struct(self.design_writer, np.array(coords)),
            plddt=plddt,
            pae=pae,
            iptm=float(iptm),
        )
        return _rank_and_refold(
            self.folder,
            seq_oh,
            float(rank_val),
            pred,
            self.mono_features,
            self.mono_writer,
            self.recycling_steps,
            seq_str,
            self.rmsd_cutoff,
        )


class FullRanker(eqx.Module):
    """Ranks by rebuilding features from the designed sequence each call."""

    folder: Protenix
    target_seq: str = eqx.field(static=True)
    template_chain: gemmi.Chain | None
    use_msa: bool = eqx.field(static=True)
    ranking_terms: LinearCombination
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    rmsd_cutoff: float = eqx.field(static=True)
    mono_features: dict[str, np.ndarray]
    mono_writer: AtomArray

    def __call__(self, seq_str: str) -> RankingResult:
        seq_oh = _seq_to_one_hot(seq_str)
        rank_features, rank_writer = self.folder.target_only_features(
            chains=[
                TargetChain(sequence=seq_str, use_msa=False),
                TargetChain(
                    sequence=self.target_seq,
                    use_msa=self.use_msa,
                    template_chain=self.template_chain,
                ),
            ]
        )
        loss = RankingLoss(
            model=self.folder.protenix,
            features=rank_features,
            loss=self.ranking_terms,
            recycling_steps=self.recycling_steps,
            sampling_steps=self.folder.default_sample_steps,
            num_samples=self.num_samples,
        )
        rank_val, coords, pae, plddt, iptm = _eval_ranking_with_pred(
            loss, seq_oh, key=jax.random.key(0)
        )
        pred = StructurePrediction(
            st=biotite_array_to_gemmi_struct(rank_writer, np.array(coords)),
            plddt=plddt,
            pae=pae,
            iptm=float(iptm),
        )
        return _rank_and_refold(
            self.folder,
            seq_oh,
            float(rank_val),
            pred,
            self.mono_features,
            self.mono_writer,
            self.recycling_steps,
            seq_str,
            self.rmsd_cutoff,
        )


Ranker = FastRanker | FullRanker


# --- Worker message types ---


@dataclass(frozen=True)
class HWStats:
    gpu_util: int
    power_w: int
    temp_c: int


@dataclass(frozen=True)
class StatusMsg:
    gpu: int
    text: str
    hw: HWStats | None = None


@dataclass(frozen=True)
class DesignStartMsg:
    gpu: int
    design_idx: int
    design_num: int
    n_designs: int
    seed: int
    hw: HWStats | None = None


@dataclass(frozen=True)
class StepMsg:
    gpu: int
    design_idx: int
    phase: str
    step: int
    total_steps: int
    loss: float
    hw: HWStats | None = None


@dataclass(frozen=True)
class RankingMsg:
    gpu: int
    design_idx: int
    hw: HWStats | None = None


@dataclass(frozen=True)
class ResultMsg:
    gpu: int
    data: dict


@dataclass(frozen=True)
class ErrorMsg:
    gpu: int
    text: str


@dataclass(frozen=True)
class GpuDoneMsg:
    gpu: int


WorkerMessage = (
    StatusMsg
    | DesignStartMsg
    | StepMsg
    | RankingMsg
    | ResultMsg
    | ErrorMsg
    | GpuDoneMsg
)


def validate_design_inputs(
    cif_content: str | None,
    chain_id: str | None,
    target_seq: str,
    template_chain: gemmi.Chain | None,
    binder_length: int,
    features: dict[str, np.ndarray],
) -> None:
    """Validate consistency between CIF, target sequence, template, and features.

    Three independent checks:
    1. Re-derive sequence from CIF and compare to passed-in target_seq.
    2. Template chain amino acid count must match target sequence length.
    3. Template mask resolved-residue count must match template chain atoms.

    Raises ValueError with a specific message on any mismatch.
    """
    # 1. Re-derive sequence from CIF and compare to passed-in target_seq
    if cif_content is not None:
        derived_seq = target_seq_from_cif(cif_content, chain_id)  # type: ignore[arg-type]
        if derived_seq != target_seq:
            raise ValueError(
                f"target_seq ({len(target_seq)} aa) does not match"
                f" sequence derived from CIF ({len(derived_seq)} aa)"
            )

    # 2. Template chain amino acid count must match target sequence
    if template_chain is not None:
        n_aa = sum(
            1
            for r in template_chain
            if gemmi.find_tabulated_residue(r.name).is_amino_acid()
        )
        if n_aa != len(target_seq):
            raise ValueError(
                f"Template chain has {n_aa} amino acid residues"
                f" but target sequence has {len(target_seq)}"
            )

    # 3. Template mask must reflect resolved residues from the template chain
    if template_chain is not None:
        n_with_atoms = sum(1 for r in template_chain if len(r) > 0)
        tmpl_mask = features["template_pseudo_beta_mask"][0]
        target_diag = np.diag(tmpl_mask[binder_length:, binder_length:])
        n_masked = int(np.count_nonzero(target_diag))
        if n_masked != n_with_atoms:
            raise ValueError(
                f"Template mask has {n_masked} resolved positions in"
                f" target region but template chain has {n_with_atoms}"
                f" residues with coordinates"
            )


def build_result_row(
    idx: int,
    seed: int,
    binder_length: int,
    seq_str: str,
    design_loss: float,
    rank_result: RankingResult,
    design_time: float,
    rank_time: float,
    hyperparams: dict,
    config_dict: dict,
) -> dict:
    """Build the standard result row dict shared by all workers."""
    return dict(
        design_idx=idx,
        seed=seed,
        binder_length=binder_length,
        sequence=seq_str,
        design_loss=design_loss,
        ranking_loss=rank_result.ranking_loss,
        iptm=rank_result.iptm,
        mean_plddt=rank_result.mean_plddt,
        monomer_rmsd=rank_result.monomer_rmsd,
        design_time_s=design_time,
        rank_time_s=rank_time,
        hyperparams=hyperparams,
        config=config_dict,
        pdb_string=rank_result.pdb_string,
        monomer_pdb_string=rank_result.monomer_pdb_string,
    )


def load_template_chain(
    cif_content: str, chain_id: str
) -> tuple[gemmi.Structure, gemmi.Chain]:
    """Parse a CIF string and extract a single chain for use as a template.

    Returns (structure, chain) tuple so the structure stays alive -- gemmi chains
    are references into their parent structure.  Gap residues lost during the
    CIF round-trip (no atoms = no atom_site entries) are reconstructed from the
    entity full_sequence.
    """
    doc = gemmi.cif.read_string(cif_content)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    chain = structure[0].find_chain(chain_id)

    entity = structure.get_entity_of(chain.get_polymer())
    if entity is None or len(entity.full_sequence) == len(chain):
        return structure, chain

    # Gap residues were lost on CIF round-trip — rebuild from entity sequence
    full_seq = entity.full_sequence
    res_by_seqid = {str(r.seqid): r for r in chain}
    new_chain = gemmi.Chain(chain.name)
    for i, name in enumerate(full_seq):
        existing = res_by_seqid.get(str(i))
        if existing is not None:
            new_chain.add_residue(existing)
        else:
            res = gemmi.Residue()
            res.name = name
            res.seqid = gemmi.SeqId(str(i))
            res.label_seq = i + 1
            new_chain.add_residue(res)

    structure[0].remove_chain(chain.name)
    structure[0].add_chain(new_chain)
    return structure, structure[0].find_chain(chain_id)  # type: ignore[arg-type]


def make_ranker(
    folder: Protenix,
    binder_length: int,
    target_seq: str,
    template_chain: gemmi.Chain | None,
    ranking: RankingConfig,
) -> Ranker:
    """Build a FastRanker or FullRanker with standard iPTM+iPSAE ranking terms."""
    import mosaic.losses.structure_prediction as sp

    ranking_terms = (
        1.0 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE()
    )
    mono_features, mono_writer = folder.binder_features(
        binder_length=binder_length, chains=[]
    )
    if ranking.fast_ranking:
        design_features, design_writer = folder.binder_features(
            binder_length=binder_length,
            chains=[
                TargetChain(
                    sequence=target_seq,
                    use_msa=ranking.use_msa,
                    template_chain=template_chain,
                )
            ],
        )
        validate_design_inputs(
            cif_content=None,
            chain_id=None,
            target_seq=target_seq,
            template_chain=template_chain,
            binder_length=binder_length,
            features=design_features,
        )
        return FastRanker(
            folder=folder,
            ranking_terms=ranking_terms,
            recycling_steps=ranking.recycling_steps,
            num_samples=ranking.num_samples,
            rmsd_cutoff=ranking.rmsd_cutoff,
            mono_features=mono_features,
            mono_writer=mono_writer,
            design_features=design_features,
            design_writer=design_writer,
        )
    return FullRanker(
        folder=folder,
        target_seq=target_seq,
        template_chain=template_chain,
        use_msa=ranking.use_msa,
        ranking_terms=ranking_terms,
        recycling_steps=ranking.recycling_steps,
        num_samples=ranking.num_samples,
        rmsd_cutoff=ranking.rmsd_cutoff,
        mono_features=mono_features,
        mono_writer=mono_writer,
    )


@dataclass(frozen=True)
class SampledHyperparams:
    p1_steps: int
    p1_stepsize_factor: float
    p1_momentum: float
    p2_steps: int
    p2_stepsize_factor: float
    p3_steps: int
    p3_stepsize_factor: float
    init_scale: float


def sample_hyperparams(
    rng: np.random.Generator, ranges: HyperparamRanges
) -> SampledHyperparams:
    """Sample optimizer hyperparameters from given ranges."""
    return SampledHyperparams(
        p1_steps=int(
            rng.integers(int(ranges.p1_steps.lo), int(ranges.p1_steps.hi) + 1)
        ),
        p1_stepsize_factor=float(
            rng.uniform(ranges.p1_stepsize_factor.lo, ranges.p1_stepsize_factor.hi)
        ),
        p1_momentum=float(rng.uniform(ranges.p1_momentum.lo, ranges.p1_momentum.hi)),
        p2_steps=int(
            rng.integers(int(ranges.p2_steps.lo), int(ranges.p2_steps.hi) + 1)
        ),
        p2_stepsize_factor=float(
            rng.uniform(ranges.p2_stepsize_factor.lo, ranges.p2_stepsize_factor.hi)
        ),
        p3_steps=int(
            rng.integers(int(ranges.p3_steps.lo), int(ranges.p3_steps.hi) + 1)
        ),
        p3_stepsize_factor=float(
            rng.uniform(ranges.p3_stepsize_factor.lo, ranges.p3_stepsize_factor.hi)
        ),
        init_scale=float(rng.uniform(ranges.init_scale.lo, ranges.init_scale.hi)),
    )
