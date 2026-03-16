"""CLI entrypoint and Modal startup screen for binder design.

Usage:
    uv run mosaic --cif target.cif --chain A --binder-length 100 \
        --num-designs 30 --num-gpus 5
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Static

from mosaic_tui.design_common import app, download_weights

# Register worker functions so Modal discovers them before app.run()
import mosaic_tui.worker_boltzgen as _worker_boltzgen  # noqa: F401
import mosaic_tui.worker_simplex as _worker_simplex  # noqa: F401

if TYPE_CHECKING:
    from mosaic_tui.ascimol import ProteinViewer
    from mosaic_tui.design_common import DesignConfig, Target


class StartupScreen(App[None]):
    """Fullscreen spinning protein shown while Modal connects."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #structure {
        width: 1fr;
        height: 1fr;
    }
    #status {
        text-align: center;
        color: $accent;
        text-style: bold;
        dock: bottom;
        height: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        viewer: ProteinViewer | None,
        exit_stack: ExitStack,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._exit_stack = exit_stack

    def compose(self) -> ComposeResult:
        if self._viewer is not None:
            from mosaic_tui.dashboard import StructureViewer

            yield StructureViewer(self._viewer, time.monotonic(), id="structure")
        else:
            yield Static("", id="structure")
        yield Static("Building Modal image + launching app...", id="status")

    def on_mount(self) -> None:
        if self._viewer is not None:
            self.set_interval(0.25, self._update_frame)
        self.run_worker(self._connect_modal, thread=True)

    def _update_frame(self) -> None:
        from mosaic_tui.dashboard import StructureViewer

        self.query_one("#structure", StructureViewer).render_frame()

    def _connect_modal(self) -> None:
        self._exit_stack.enter_context(app.run())
        self.call_from_thread(
            self.query_one("#status", Static).update,
            "Downloading model weights...",
        )
        download_weights.remote()
        self.call_from_thread(self.exit)


def _run_with_spinner(
    target: Target, hotspots: list[int] | None, config: DesignConfig
) -> None:
    """Start Modal app with a fullscreen spinning protein, then run design."""
    from mosaic_tui.design_common import CifTarget, SeqTarget
    from mosaic_tui.orchestrator import design

    viewer: ProteinViewer | None = None
    match target:
        case CifTarget(path=cif):
            import gemmi
            from mosaic_tui.ascimol import ProteinViewer

            target_st = gemmi.read_structure(cif)
            target_st.remove_ligands_and_waters()
            viewer = ProteinViewer.from_gemmi(target_st)
        case SeqTarget():
            pass

    exit_stack = ExitStack()
    try:
        StartupScreen(viewer, exit_stack).run()
        design(target=target, hotspots=hotspots, config=config)
    finally:
        exit_stack.close()


def main() -> None:
    import argparse
    import dataclasses
    import sys
    from pathlib import Path

    import orjson

    from mosaic_tui.config_screen import run_config_screen
    from mosaic_tui.design_common import (
        BoltzGenConfig,
        CifTarget,
        DesignConfig,
        LossWeights,
        RankingConfig,
        RunParams,
        SeqTarget,
        SimplexConfig,
        config_from_dict,
        default_method,
    )

    parser = argparse.ArgumentParser(
        description="Protein binder design with rich live dashboard"
    )
    parser.add_argument("--cif", default=None, help="Path to .cif or .pdb file")
    parser.add_argument("--chain", default="A", help="Target chain ID (default: A)")
    parser.add_argument(
        "--sequence",
        default=None,
        help="Target amino acid sequence (1-letter codes). Mutually exclusive with --cif.",
    )
    parser.add_argument(
        "--binder-length", type=int, default=120, help="Binder length (default: 120)"
    )
    parser.add_argument(
        "--num-designs", type=int, default=None, help="Total designs (default: 32)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="Number of B200 GPUs (default: 1)"
    )
    parser.add_argument(
        "--recycling-steps", type=int, default=6, help="Recycling steps (default: 6)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=4, help="Diffusion samples (default: 4)"
    )
    parser.add_argument(
        "--hotspots", default="", help="Comma-separated 0-indexed residue indices"
    )
    parser.add_argument(
        "--no-msa", action="store_true", help="Disable MSA for target chain"
    )
    parser.add_argument("--run", default="", help="Run name (default: auto timestamp)")
    parser.add_argument(
        "--full-ranking",
        action="store_true",
        help="Rebuild features per sequence for ranking (slower, more accurate)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: contact losses only, 1 recycle step for design+ranking",
    )
    parser.add_argument(
        "--method",
        choices=["simplex", "boltzgen"],
        default=None,
        help="Design method: simplex (gradient optimization) or boltzgen (diffusion sampling)",
    )
    parser.add_argument(
        "--rl-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the RL post-training BoltzGen checkpoint",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Include unresolved terminal residues from entity sequence",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip config screen, use CLI defaults directly",
    )
    args = parser.parse_args()

    # --- Resume detection ---
    resume_config_path = None
    if args.run:
        p = Path("results") / args.run / "config.json"
        if p.exists():
            resume_config_path = p

    if resume_config_path is not None:
        saved = orjson.loads(resume_config_path.read_bytes())
        config = config_from_dict(saved)

        rp = config.run_params
        overrides: dict = {}
        if args.num_designs is not None:
            overrides["run_params"] = dataclasses.replace(
                rp, num_designs=args.num_designs
            )
        if args.num_gpus is not None:
            rp = overrides.get("run_params", rp)
            overrides["run_params"] = dataclasses.replace(rp, num_gpus=args.num_gpus)
        if args.method is not None:
            overrides["method"] = default_method(args.method)
        if overrides:
            config = dataclasses.replace(config, **overrides)

        saved_target = saved["target"]
        if saved_target["type"] == "sequence":
            target: Target = SeqTarget(sequence=saved_target["sequence"])
        else:
            target = CifTarget(
                path=str(Path("results") / args.run / saved_target["cif"]),
                chain=saved_target["chain"],
            )

        saved_hotspots = saved.get("hotspots")
        initial_hotspots = set(saved_hotspots) if saved_hotspots else set()

        if not args.no_config:
            result = run_config_screen(
                config,
                target=target,
                initial_hotspots=initial_hotspots,
                locked_categories={"Ranking"},
                show_method_picker=args.method is None,
            )
            if result is None:
                sys.exit(0)
            config, hotspot_list = result
        else:
            hotspot_list = sorted(initial_hotspots) if initial_hotspots else None

        print(f"Resuming run '{args.run}'")
        _run_with_spinner(target=target, hotspots=hotspot_list, config=config)
    else:
        if args.cif and args.sequence:
            parser.error("--cif and --sequence are mutually exclusive")
        if not args.cif and not args.sequence:
            parser.error("--cif or --sequence is required for new runs")
        if args.sequence and args.method == "boltzgen":
            parser.error("BoltzGen requires a structure file (--cif)")

        if args.cif:
            target = CifTarget(path=args.cif, chain=args.chain)
        else:
            target = SeqTarget(sequence=args.sequence)

        method = default_method(args.method or "simplex")

        use_msa = not args.no_msa
        match method:
            case SimplexConfig() as m:
                method = dataclasses.replace(
                    m,
                    recycling_steps=args.recycling_steps,
                    num_samples=args.num_samples,
                    use_msa=use_msa,
                )
            case BoltzGenConfig() as m:
                if args.rl_checkpoint is not None:
                    method = dataclasses.replace(
                        m, use_rl_checkpoint=args.rl_checkpoint
                    )
                else:
                    method = m

        config = DesignConfig(
            run_params=RunParams(
                binder_length=args.binder_length,
                num_designs=args.num_designs if args.num_designs is not None else 32,
                num_gpus=args.num_gpus if args.num_gpus is not None else 1,
                run=args.run,
                trim_terminals=not args.no_trim,
            ),
            method=method,
            ranking=RankingConfig(
                fast_ranking=not args.full_ranking,
                use_msa=use_msa,
            ),
        )

        if args.fast:
            match config.method:
                case SimplexConfig() as m:
                    config = dataclasses.replace(
                        config,
                        method=dataclasses.replace(
                            m,
                            loss_weights=LossWeights(
                                binder_target_contact=1.0,
                                within_binder_contact=1.0,
                                inverse_folding=0.0,
                                target_binder_pae=0.0,
                                binder_target_pae=0.0,
                                iptm=0.0,
                                within_binder_pae=0.0,
                                ptm=0.0,
                                plddt=0.0,
                            ),
                            recycling_steps=1,
                        ),
                        ranking=dataclasses.replace(
                            config.ranking,
                            recycling_steps=1,
                        ),
                    )
                case BoltzGenConfig():
                    config = dataclasses.replace(
                        config,
                        ranking=dataclasses.replace(
                            config.ranking,
                            recycling_steps=1,
                        ),
                    )

        initial_hotspots = (
            {int(x.strip()) for x in args.hotspots.split(",") if x.strip()}
            if args.hotspots
            else set()
        )

        if not args.no_config:
            result = run_config_screen(
                config,
                target=target,
                initial_hotspots=initial_hotspots,
                show_method_picker=args.method is None,
            )
            if result is None:
                sys.exit(0)
            config, hotspot_list = result
        else:
            hotspot_list = sorted(initial_hotspots) or None

        _run_with_spinner(target=target, hotspots=hotspot_list, config=config)


if __name__ == "__main__":
    main()
