"""GPU-dependent shared utilities: ranking evaluation, structure comparison, ranker factory.

This module is only imported inside Modal GPU workers (never at TUI import time),
so it freely depends on JAX, equinox, and mosaic.
"""

from __future__ import annotations

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray

from mosaic.common import TOKENS, LinearCombination
from mosaic.losses.protenix import biotite_array_to_gemmi_struct
from mosaic.models.protenix import Protenix
from mosaic.structure_prediction import StructurePrediction, TargetChain

from mosaic_tui.design_common import (
    RankingConfig,
    RankingResult,
    validate_design_inputs,
)
from mosaic_tui.ranking_loss import RankingLoss


def rename_binder_residues(
    st: gemmi.Structure, seq_str: str, chain_idx: int = 0
) -> None:
    """Rename UNK binder residues to proper 3-letter codes in-place."""
    three_letter = gemmi.expand_one_letter_sequence(seq_str, gemmi.ResidueKind.AA)
    chain = st[0][chain_idx]
    for i, res in enumerate(chain):
        if i < len(three_letter) and res.name == "UNK":
            res.name = three_letter[i]


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
