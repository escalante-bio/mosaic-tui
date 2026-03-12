"""Fused ranking loss that returns structure predictions alongside the loss."""

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

import equinox as eqx

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.protenix import (
    ProtenixFromTrunkOutput,
    get_trunk_state,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.models.protenix import Protenix as Protenij


class RankingLoss(eqx.Module):
    """Compute ranking loss and extract structure predictions in one pass."""

    model: Protenij
    features: PyTree
    loss: LossTerm | LinearCombination
    recycling_steps: int = eqx.field(static=True)
    sampling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)

    def __call__(
        self, sequence: jax.Array, *, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        features = set_binder_sequence(sequence, self.features)
        initial_embedding, trunk_state = get_trunk_state(
            model=self.model,
            features=features,
            initial_recycling_state=None,
            recycling_steps=self.recycling_steps,
            key=key,
        )

        def sample_fn(
            key: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            output = ProtenixFromTrunkOutput(
                model=self.model,
                features=features,
                key=key,
                initial_embedding=initial_embedding,
                trunk_state=trunk_state,
                sampling_steps=self.sampling_steps,
            )
            v, _ = self.loss(sequence=sequence, output=output, key=key)
            iptm = -IPTMLoss()(sequence=sequence, output=output, key=key)[0]
            return v, output.structure_coordinates[0], output.pae, output.plddt, iptm

        vs, all_coords, all_paes, all_plddts, all_iptms = jax.vmap(sample_fn)(
            jax.random.split(key, self.num_samples)
        )
        best = jnp.argmin(vs)
        return (
            jnp.mean(vs),
            all_coords[best],
            all_paes[best],
            all_plddts[best],
            all_iptms[best],
        )
