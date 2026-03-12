"""Design orchestration: dispatches GPU workers and runs the dashboard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from mosaic_tui.design_common import DesignConfig, Target


def load_existing_results(run_dir: str | Path) -> list[dict]:
    """Load all completed design results from a run directory."""
    import orjson

    results = []
    for p in sorted(Path(run_dir).glob("design_*.json")):
        if "_monomer" in p.name:
            continue
        data = orjson.loads(p.read_bytes())
        if "ranking_loss" not in data:
            continue
        if data["ranking_loss"] is None:
            data["ranking_loss"] = float("inf")
        results.append(data)
    return results


def _spawn_designs(
    config: DesignConfig,
    gpu_id: int,
    n: int,
    start_idx: int,
    binder_length: int,
    cif_content: str | None,
    chain: str | None,
    target_seq: str,
    hotspot_list: list[int] | None,
    queue_object_id: str,
) -> modal.functions.FunctionCall:
    """Spawn the right remote function based on config.method."""
    from mosaic_tui.design_common import BoltzGenConfig, SimplexConfig
    from mosaic_tui.worker_boltzgen import run_boltzgen_designs
    from mosaic_tui.worker_simplex import run_designs

    match config.method:
        case SimplexConfig():
            return run_designs.spawn(
                num_designs=n,
                binder_length=binder_length,
                cif_content=cif_content,
                chain_id=chain,
                hotspots=hotspot_list,
                run_name=config.run_params.run,
                start_idx=start_idx,
                queue_id=queue_object_id,
                gpu_id=gpu_id,
                method=config.method,
                ranking=config.ranking,
                target_seq=target_seq,
            )
        case BoltzGenConfig():
            return run_boltzgen_designs.spawn(
                num_designs=n,
                binder_length=binder_length,
                cif_content=cif_content,
                chain_id=chain,
                hotspots=hotspot_list,
                run_name=config.run_params.run,
                start_idx=start_idx,
                queue_id=queue_object_id,
                gpu_id=gpu_id,
                method=config.method,
                ranking=config.ranking,
                target_seq=target_seq,
            )


def design(
    target: Target,
    hotspots: list[int] | None = None,
    config: DesignConfig | None = None,
) -> None:
    """Design protein binders with rich live dashboard."""
    import dataclasses
    import datetime
    import shutil

    import orjson

    import polars as pl
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from mosaic_tui.design_common import (
        BoltzGenConfig,
        CifTarget,
        SeqTarget,
        SimplexConfig,
        default_config,
        target_label,
    )
    from mosaic_tui.dashboard import DesignDashboard, iptm_color

    if config is None:
        config = default_config()

    rp = config.run_params
    binder_length = rp.binder_length
    num_designs = rp.num_designs
    num_gpus = rp.num_gpus
    ranking_desc = (
        f"Protenix R={config.ranking.recycling_steps}"
        f" S={config.ranking.num_samples} iPTM+ipSAE"
        + (" [FAST]" if config.ranking.fast_ranking else "")
    )

    console = Console()

    hotspot_list = hotspots
    run_name = rp.run or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = dataclasses.replace(
        config, run_params=dataclasses.replace(rp, run=run_name)
    )

    tlabel = target_label(target)

    match target:
        case CifTarget(path=cif, chain=chain):
            import gemmi
            from mosaic_tui.ascimol import ProteinViewer
            from mosaic_tui.design_common import (
                preprocess_target_cif,
                target_seq_from_cif,
            )

            cif_path = Path(cif)
            chain_id: str | None = chain
            preprocessed = preprocess_target_cif(
                str(cif_path), chain, trim_terminals=rp.trim_terminals
            )
            cif_content: str | None = preprocessed
            target_seq = target_seq_from_cif(preprocessed, chain)

            doc = gemmi.cif.read_string(preprocessed)
            target_st = gemmi.make_structure_from_block(doc.sole_block())
            target_viewer: ProteinViewer | None = ProteinViewer.from_gemmi(
                target_st,
                title=f"{cif_path.stem} chain {chain}",
            )
        case SeqTarget(sequence=seq):
            cif_path = None  # type: ignore[assignment]
            cif_content = None
            chain_id = None
            target_seq = seq
            target_viewer = None

    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(run_dir)
    n_new = num_designs - len(existing)

    match config.method:
        case SimplexConfig():
            method_type = "simplex"
        case BoltzGenConfig():
            method_type = "boltzgen"

    match target:
        case CifTarget(path=cif, chain=chain):
            target_dict = {"type": "cif", "cif": Path(cif).name, "chain": chain}
        case SeqTarget(sequence=seq):
            target_dict = {"type": "sequence", "sequence": seq}

    saved_config: dict = {
        "run_name": run_name,
        "target": target_dict,
        "hotspots": hotspot_list,
        "run_params": dataclasses.asdict(config.run_params),
        "method_type": method_type,
        "method": dataclasses.asdict(config.method),
        "ranking": dataclasses.asdict(config.ranking),
    }

    (run_dir / "config.json").write_bytes(
        orjson.dumps(saved_config, option=orjson.OPT_INDENT_2)
    )

    if not existing and cif_path is not None:
        dest = run_dir / cif_path.name
        if cif_path.resolve() != dest.resolve():
            shutil.copy2(cif_path, dest)
    if not existing:
        lock_file = Path("uv.lock")
        if lock_file.exists():
            shutil.copy2(lock_file, run_dir / "uv.lock")

    if existing:
        console.print(
            f"[cyan]Resuming run[/cyan] '{run_name}': "
            f"{len(existing)}/{num_designs} completed"
            + (f", launching {n_new} new" if n_new > 0 else ", all done")
            + f"  [cyan]Ranking:[/cyan] {ranking_desc}"
        )
    else:
        console.print(
            f"[cyan]Run:[/cyan] {run_name}  "
            f"[cyan]Target:[/cyan] {tlabel}  "
            f"[cyan]L=[/cyan]{binder_length}  "
            f"[cyan]Designs:[/cyan] {num_designs} on {num_gpus}\u00d7B200  "
            f"[cyan]Ranking:[/cyan] {ranking_desc}"
        )

    if n_new <= 0:
        all_results = existing
        elapsed_str = "0:00:00"
        interrupted = False
    else:
        next_idx = max((r["design_idx"] for r in existing), default=-1) + 1
        effective_gpus = min(num_gpus, n_new)
        designs_per_gpu = n_new // effective_gpus
        remainder = n_new % effective_gpus

        gpu_chunks = []
        idx = next_idx
        for i in range(effective_gpus):
            n = designs_per_gpu + (1 if i < remainder else 0)
            if n > 0:
                gpu_chunks.append((i, n, idx))
                idx += n

        with modal.Queue.ephemeral() as queue:
            handles = [
                _spawn_designs(
                    config=config,
                    gpu_id=gpu_id,
                    n=n,
                    start_idx=start_idx,
                    binder_length=binder_length,
                    cif_content=cif_content,
                    chain=chain_id,
                    target_seq=target_seq,
                    hotspot_list=hotspot_list,
                    queue_object_id=queue.object_id,
                )
                for gpu_id, n, start_idx in gpu_chunks
            ]

            t_start = time.monotonic()
            dashboard = DesignDashboard(
                queue=queue,
                handles=handles,
                gpu_chunks=gpu_chunks,
                num_designs=num_designs,
                run_name=run_name,
                target_label=tlabel,
                binder_length=binder_length,
                num_gpus=effective_gpus,
                target_viewer=target_viewer,
                existing_results=existing,
                ranking_desc=ranking_desc,
                loss_desc=config.method.describe(),
                show_loss_chart=isinstance(config.method, SimplexConfig),
            )
            dashboard_result = dashboard.run()
            if dashboard_result is None:
                interrupted = True
                completed_results = existing
            else:
                interrupted = dashboard_result.interrupted
                completed_results = dashboard_result.completed_results
                for err in dashboard_result.errors:
                    console.print(f"[red]Error: {err}[/red]")

            elapsed_s = int(time.monotonic() - t_start)
            elapsed_str = f"{elapsed_s // 3600}:{(elapsed_s % 3600) // 60:02d}:{elapsed_s % 60:02d}"

        all_results = completed_results

    if not all_results:
        console.print(
            "[yellow]No designs completed.[/yellow]"
            if interrupted
            else "[red]No results collected![/red]"
        )
        return

    METRIC_KEYS = [
        "design_idx",
        "seed",
        "binder_length",
        "sequence",
        "design_loss",
        "ranking_loss",
        "iptm",
        "mean_plddt",
        "monomer_rmsd",
        "design_time_s",
        "rank_time_s",
    ]
    rows = [{k: r[k] for k in METRIC_KEYS} for r in all_results]
    for row, r in zip(rows, all_results):
        row["seed"] = f"{row['seed']:06x}"
        row["method_type"] = r.get("config", {}).get("method_type", "")
    df = pl.DataFrame(rows).sort("ranking_loss").with_row_index("rank", offset=1)
    df.write_csv(run_dir / "summary.csv")

    stats = df.select(
        n=pl.lit(len(df)),
        iptm_mean=pl.col("iptm").mean(),
        iptm_std=pl.col("iptm").std(),
        iptm_max=pl.col("iptm").max(),
        plddt_mean=pl.col("mean_plddt").mean(),
        rank_loss_mean=pl.col("ranking_loss").mean(),
        rank_loss_std=pl.col("ranking_loss").std(),
        rmsd_mean=pl.col("monomer_rmsd").mean(),
        design_s_mean=pl.col("design_time_s").mean(),
        rank_s_mean=pl.col("rank_time_s").mean(),
    ).row(0, named=True)

    console.print()
    banner = Text()
    if interrupted:
        banner.append(
            f"BINDER DESIGN INTERRUPTED ({len(all_results)}/{num_designs} collected)",
            style="bold yellow",
        )
    else:
        banner.append("BINDER DESIGN COMPLETE", style="bold cyan")
    banner.append(f"  \u2502  {tlabel}")
    banner.append(f"  \u2502  L={binder_length}")
    banner.append(f"  \u2502  {run_name}")
    banner.append(f"  \u2502  {elapsed_str}")
    console.print(Panel(banner, style="yellow" if interrupted else "cyan"))

    stats_table = Table(title="Summary", title_style="bold cyan", border_style="cyan")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Designs", str(stats["n"]))
    stats_table.add_row(
        "iPTM",
        f"{stats['iptm_mean']:.4f} \u00b1 {(stats['iptm_std'] or 0):.4f}"
        f"  (max {stats['iptm_max']:.4f})",
    )
    stats_table.add_row("pLDDT", f"{stats['plddt_mean']:.2f}")
    stats_table.add_row(
        "Rank loss",
        f"{stats['rank_loss_mean']:.4f} \u00b1 {(stats['rank_loss_std'] or 0):.4f}",
    )
    stats_table.add_row("RMSD", f"{stats['rmsd_mean']:.2f}")
    stats_table.add_row("Design time", f"{stats['design_s_mean']:.1f}s avg")
    stats_table.add_row("Rank time", f"{stats['rank_s_mean']:.1f}s avg")
    console.print(stats_table)

    top5_table = Table(
        title="Top 5 Designs", title_style="bold green", border_style="green"
    )
    top5_table.add_column("rank", justify="right", width=4)
    top5_table.add_column("idx", justify="right", width=4)
    top5_table.add_column("iptm", justify="right", width=7)
    top5_table.add_column("plddt", justify="right", width=7)
    top5_table.add_column("rmsd", justify="right", width=6)
    top5_table.add_column("rank_loss", justify="right", width=10)
    top5_table.add_column("sequence")

    for r in df.head(5).iter_rows(named=True):
        color = iptm_color(r["iptm"])
        top5_table.add_row(
            str(r["rank"]),
            str(r["design_idx"]),
            f"[{color}]{r['iptm']:.4f}[/{color}]",
            f"{r['mean_plddt']:.2f}",
            f"{r['monomer_rmsd']:.2f}",
            f"{r['ranking_loss']:.4f}",
            r["sequence"][:50] + "...",
        )

    console.print(top5_table)
    console.print(
        f"\n[dim]Local:  results/{run_name}/  (summary.csv, config.json)[/dim]"
    )
    console.print(
        f"[dim]Volume: modal volume get design-results"
        f" {run_name}/ results/{run_name}/[/dim]"
    )
