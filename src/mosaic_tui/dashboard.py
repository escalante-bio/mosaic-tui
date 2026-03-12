"""Textual-based live design dashboard.

Replaces the Rich Live loop in design_rich.py with a full Textual app.
Polls a Modal Queue for progress messages from GPU workers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import modal
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static
from textual_plotext import PlotextPlot

from mosaic_tui.design_common import (
    DesignStartMsg,
    ErrorMsg,
    GpuDoneMsg,
    HWStats,
    RankingMsg,
    ResultMsg,
    StatusMsg,
    StepMsg,
    WorkerMessage,
)

if TYPE_CHECKING:
    from mosaic_tui.ascimol import ProteinViewer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLOTEXT_COLORS = [
    "cyan",
    "red",
    "green",
    "yellow",
    "magenta",
    "blue",
    "orange",
    "white",
]

GPU_CELL_W = 38  # visible characters per GPU cell (excluding Rich markup)
B200_COST_PER_SEC = 0.001736  # $/sec per GPU (modal.com/pricing)


# ---------------------------------------------------------------------------
# GPU state tracking
# ---------------------------------------------------------------------------


@dataclass
class GPUState:
    last_msg: WorkerMessage = field(
        default_factory=lambda: StatusMsg(gpu=0, text="Waiting...")
    )
    losses: list[float] = field(default_factory=list)
    error: str | None = None


def iptm_color(val: float) -> str:
    if val >= 0.8:
        return "green"
    if val >= 0.6:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
Screen {
    background: $surface;
}

#gpu-panel {
    height: 5;
    border: solid $primary;
    padding: 0 1;
}

#gpu-panel Static {
    height: 3;
}

#lower {
    height: 1fr;
}

#left-col {
    width: 1fr;
}

#right-col {
    width: 1fr;
}

#loss-chart {
    height: 1fr;
    border: solid $warning;
}

#structure-panel {
    height: 1fr;
    border: solid $primary-background;
}

#results-table {
    height: 1fr;
    border: solid $success;
}

#progress-bar {
    height: 1;
    margin: 0 1;
}

#status-line {
    height: 1;
    margin: 0 1;
    color: $text;
}
"""


def _hw_suffix(hw: HWStats | None) -> str:
    """Format hardware stats as a compact string."""
    if hw is None:
        return ""
    return f"{hw.gpu_util}% {hw.power_w}W {hw.temp_c}\u00b0C"


def _format_gpu_cell(gid: int, gs: GPUState) -> tuple[str, str]:
    """Return (plain_text, rich_markup) for one GPU status cell."""
    match gs.last_msg:
        case StatusMsg(text=text):
            plain = f"{gid} {text[:20]}"
            cell = f"[bold cyan]{gid}[/] [dim]{text[:20]}[/]"
        case StepMsg(phase=phase, step=step, total_steps=total, loss=loss):
            plain = f"{gid} {phase} {step}/{total} {loss:.2f}"
            cell = (
                f"[bold cyan]{gid}[/]"
                f" [green]{phase}[/]"
                f" {step}/{total}"
                f" [bold yellow]{loss:.2f}[/]"
            )
        case DesignStartMsg():
            plain = f"{gid} starting design"
            cell = f"[bold cyan]{gid}[/] [dim]starting design[/]"
        case RankingMsg():
            plain = f"{gid} ranking"
            cell = f"[bold cyan]{gid}[/] [magenta]ranking[/]"
        case GpuDoneMsg():
            plain = f"{gid} done"
            cell = f"[bold cyan]{gid}[/] [bold green]done[/]"
        case ErrorMsg(text=text):
            plain = f"{gid} err"
            cell = f"[bold cyan]{gid}[/] [bold red]err[/]"
        case _:
            plain = f"{gid} ?"
            cell = f"[bold cyan]{gid}[/] ?"

    # Append hardware stats
    hw = getattr(gs.last_msg, "hw", None)
    hw_str = _hw_suffix(hw)
    if hw_str:
        budget = GPU_CELL_W - len(plain) - 1
        if budget >= 4:
            hw_str = hw_str[:budget]
            plain += f" {hw_str}"
            cell += f" [dim]{hw_str}[/]"

    # Pad to fixed width
    pad = GPU_CELL_W - len(plain)
    if pad > 0:
        cell += " " * pad
    return plain, cell


class GPUPanel(Widget):
    """Displays per-GPU worker status in a horizontal grid."""

    DEFAULT_CSS = ""

    def compose(self) -> ComposeResult:
        yield Static("Workers", id="gpu-title")

    def update_gpus(self, gpu_state: dict[int, GPUState]) -> None:
        cells: list[str] = []
        for gid in sorted(gpu_state.keys()):
            _plain, cell = _format_gpu_cell(gid, gpu_state[gid])
            cells.append(cell)

        title = self.query_one("#gpu-title", Static)
        if not cells:
            title.update("[dim]No workers yet[/]")
            return

        width = self.size.width or 120
        col_w = GPU_CELL_W + 5  # cell + "  │  " separator
        cells_per_row = max(1, width // col_w)
        rows = []
        for i in range(0, len(cells), cells_per_row):
            rows.append("  \u2502  ".join(cells[i : i + cells_per_row]))
        title.update("\n".join(rows))


class StructureViewer(Static):
    """Renders a spinning protein structure via ascimol."""

    DEFAULT_CSS = ""

    def __init__(self, viewer: ProteinViewer | None, t_start: float, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._viewer = viewer
        self._t_start = t_start
        self._label = "target"
        self._color_by = "ss"
        self._charset = "dots"
        self._selected_data: dict | None = None
        self._show_complex = False

    def set_viewer(
        self, viewer: ProteinViewer, label: str, color_by: str = "ss"
    ) -> None:
        self._viewer = viewer
        self._label = label
        self._color_by = color_by

    def set_design(self, data: dict) -> None:
        """Store design data and rebuild viewer in current view mode."""
        self._selected_data = data
        self._rebuild_design_viewer()

    def toggle_complex(self) -> None:
        """Toggle between binder-only and full complex views."""
        if self._selected_data is None:
            return
        self._show_complex = not self._show_complex
        self._rebuild_design_viewer()

    def _rebuild_design_viewer(self) -> None:
        """Build viewer from selected design data for current view mode."""
        data = self._selected_data
        if data is None:
            return
        key = "pdb_string" if self._show_complex else "monomer_pdb_string"
        pdb = data.get(key)
        if pdb is None:
            return
        import gemmi
        from mosaic_tui.ascimol import ProteinViewer

        st = gemmi.read_pdb_string(pdb)
        suffix = " (complex)" if self._show_complex else ""
        viewer = ProteinViewer.from_gemmi(
            st,
            title=f"#{data['design_idx']} iptm={data['iptm']:.3f}{suffix}",
        )
        color = "chain" if self._show_complex else "ss"
        self.set_viewer(
            viewer,
            label=f"design_{data['seed']:06x}",
            color_by=color,
        )

    def render_frame(self) -> None:
        if self._viewer is None:
            return
        elapsed = time.monotonic() - self._t_start
        angle_y = (elapsed * 12.0) % 360
        angle_x = 15.0 * math.sin(elapsed * 0.6)
        w = max(10, self.size.width - 2)
        h = max(5, self.size.height - 2)
        frame = self._viewer._render_frame(
            color_by=self._color_by,
            charset=self._charset,
            w=w,
            h=h,
            rotation=(angle_x, angle_y, 0.0),
        )
        self.update(frame)


# ---------------------------------------------------------------------------
# Dashboard App
# ---------------------------------------------------------------------------


@dataclass
class DashboardResult:
    """Returned by the dashboard app."""

    completed_results: list[dict]
    interrupted: bool
    errors: list[str] = field(default_factory=list)


class DesignDashboard(App[DashboardResult]):
    """Live design monitoring dashboard."""

    CSS = DASHBOARD_CSS

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Stop", priority=True),
        Binding("d", "toggle_charset", "Dots/Unicode"),
        Binding("b", "toggle_complex", "Binder/Complex"),
    ]

    def __init__(
        self,
        queue: modal.Queue,
        handles: list[modal.functions.FunctionCall],
        gpu_chunks: list[tuple[int, int, int]],
        num_designs: int,
        run_name: str,
        target_label: str,
        binder_length: int,
        num_gpus: int,
        target_viewer: ProteinViewer | None = None,
        existing_results: list[dict] | None = None,
        ranking_desc: str = "",
        loss_desc: str = "Loss",
        show_loss_chart: bool = True,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._handles = handles
        self._gpu_chunks = gpu_chunks
        self._num_designs = num_designs
        self._run_name = run_name
        self._target_label = target_label
        self._binder_length = binder_length
        self._num_gpus = num_gpus
        self._target_viewer = target_viewer
        self._ranking_desc = ranking_desc
        self._loss_desc = loss_desc
        self._show_loss_chart = show_loss_chart
        self._run_dir = Path("results") / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._gpu_state: dict[int, GPUState] = {}
        for gpu_id, _n, _ in gpu_chunks:
            self._gpu_state[gpu_id] = GPUState()

        self._completed_results: list[dict] = list(existing_results or [])
        self._sorted_results: list[dict] = []
        self._sort_key: str = "ranking_loss"
        self._sort_reverse: bool = False
        self._gpus_done = 0
        self._num_active_gpus = len(gpu_chunks)
        self._t_start = time.monotonic()
        self._interrupted = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield GPUPanel(id="gpu-panel")
        with Horizontal(id="lower"):
            with Vertical(id="left-col"):
                yield DataTable(id="results-table")
                if self._show_loss_chart:
                    yield PlotextPlot(id="loss-chart")
            with Vertical(id="right-col"):
                yield StructureViewer(
                    self._target_viewer,
                    self._t_start,
                    id="structure-panel",
                )
        yield ProgressBar(total=self._num_designs, id="progress-bar")
        yield Static("", id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "MOSAIC v0.01"
        self.sub_title = (
            f"{self._target_label}"
            f" | L={self._binder_length}"
            f" | {self._num_gpus}\u00d7B200"
            f" | {self._num_designs} designs"
            f" | {self._ranking_desc}"
            f" | {self._run_name}"
        )

        gpu_panel = self.query_one("#gpu-panel", GPUPanel)
        gpu_panel.border_title = self._loss_desc

        table = self.query_one("#results-table", DataTable)
        table.add_columns(
            "#",
            "idx",
            "seed",
            "iptm",
            "plddt",
            "rmsd",
            "rank_loss",
            "design_s",
            "rank_s",
            "cost",
            "sequence",
        )
        table.cursor_type = "row"

        if self._show_loss_chart:
            chart = self.query_one("#loss-chart", PlotextPlot)
            chart.plt.theme("dark")

        # Pre-populate from existing results (resume)
        if self._completed_results:
            pbar = self.query_one("#progress-bar", ProgressBar)
            pbar.advance(len(self._completed_results))
            self._refresh_results_table()

        # Start polling
        self.set_interval(0.15, self._poll_queue)
        self.set_interval(0.25, self._update_structure)
        self.set_interval(1.0, self._update_status_line)

        if self._num_active_gpus == 0:
            self.set_timer(
                0.5,
                lambda: self.exit(
                    DashboardResult(
                        completed_results=self._completed_results,
                        interrupted=False,
                    )
                ),
            )

    # -- Queue processing ---------------------------------------------------

    def _poll_queue(self) -> None:
        batch = 0
        while batch < 50:
            msg = self._queue.get(block=False)
            if msg is None:
                break
            batch += 1
            self._handle_message(msg)

        if batch > 0:
            self._refresh_gpu_panel()
            self._refresh_loss_chart()

        if self._gpus_done >= self._num_active_gpus:
            errors = [
                f"GPU {gid}: {gs.error}"
                for gid, gs in self._gpu_state.items()
                if gs.error is not None
            ]
            self.exit(
                DashboardResult(
                    completed_results=self._completed_results,
                    interrupted=False,
                    errors=errors,
                )
            )

    def _handle_message(self, msg: WorkerMessage) -> None:
        if isinstance(msg, ResultMsg):
            self._completed_results.append(msg.data)
            self._save_design(msg.data)
            self._refresh_results_table()
            self.query_one("#progress-bar", ProgressBar).advance(1)
            return

        gs = self._gpu_state.get(msg.gpu)
        if gs is None:
            return

        gs.last_msg = msg
        match msg:
            case DesignStartMsg():
                gs.losses = []
            case StepMsg(loss=loss):
                gs.losses.append(loss)
            case ErrorMsg(text=text):
                gs.error = text
            case GpuDoneMsg():
                self._gpus_done += 1

    # -- Panel updates ------------------------------------------------------

    def _refresh_gpu_panel(self) -> None:
        self.query_one(GPUPanel).update_gpus(self._gpu_state)

    def _refresh_loss_chart(self) -> None:
        if not self._show_loss_chart:
            return
        chart = self.query_one("#loss-chart", PlotextPlot)
        plt = chart.plt
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")

        for gid in sorted(self._gpu_state.keys()):
            gs = self._gpu_state[gid]
            if gs.losses:
                plt.plot(
                    gs.losses,
                    label=f"GPU {gid}",
                    color=PLOTEXT_COLORS[gid % len(PLOTEXT_COLORS)],
                )

        chart.refresh()

    def _refresh_results_table(self) -> None:
        table = self.query_one("#results-table", DataTable)

        # Remember which design the cursor was on
        old_cursor_idx = None
        if self._sorted_results and table.row_count > 0:
            row = table.cursor_row
            if 0 <= row < len(self._sorted_results):
                old_cursor_idx = self._sorted_results[row]["design_idx"]

        table.clear()
        sk = self._sort_key
        self._sorted_results = sorted(
            self._completed_results,
            key=lambda r: r.get(sk, 0),
            reverse=self._sort_reverse,
        )
        for rank, r in enumerate(self._sorted_results, 1):
            table.add_row(
                str(rank),
                str(r["design_idx"]),
                f"{r.get('seed', 0):06x}",
                Text(f"{r['iptm']:.3f}", style=iptm_color(r["iptm"])),
                f"{r['mean_plddt']:.2f}",
                Text(
                    f"{r['monomer_rmsd']:.1f}",
                    style="bold red" if r["monomer_rmsd"] > 3.0 else "",
                ),
                f"{r['ranking_loss']:.3f}",
                f"{r.get('design_time_s', 0):.0f}s",
                f"{r.get('rank_time_s', 0):.0f}s",
                f"${(r.get('design_time_s', 0) + r.get('rank_time_s', 0)) * B200_COST_PER_SEC:.2f}",
                r["sequence"],
            )

        # Restore cursor position
        cursor_row = 0
        if old_cursor_idx is not None:
            for i, r in enumerate(self._sorted_results):
                if r["design_idx"] == old_cursor_idx:
                    table.move_cursor(row=i)
                    cursor_row = i
                    break

        # Update sequence + viewer for the cursor row
        if self._sorted_results:
            self._select_design(self._sorted_results[cursor_row])

    # -- Design selection (sequence panel + structure viewer) ----------------

    def _select_design(self, data: dict) -> None:
        """Update structure viewer for the selected design."""
        if data.get("monomer_pdb_string") is None:
            return
        self.query_one(StructureViewer).set_design(data)

    _COLUMN_SORT_KEYS: dict[str, str] = {
        "idx": "design_idx",
        "seed": "seed",
        "iptm": "iptm",
        "plddt": "mean_plddt",
        "rmsd": "monomer_rmsd",
        "rank_loss": "ranking_loss",
        "design_s": "design_time_s",
        "rank_s": "rank_time_s",
    }

    _SORT_DESCENDING: set[str] = {"iptm", "mean_plddt"}

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row = event.cursor_row
        if 0 <= row < len(self._sorted_results):
            self._select_design(self._sorted_results[row])

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        label = str(event.label)
        sort_key = self._COLUMN_SORT_KEYS.get(label)
        if sort_key is None:
            return
        if self._sort_key == sort_key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key = sort_key
            self._sort_reverse = sort_key in self._SORT_DESCENDING
        self._refresh_results_table()

    # -- Periodic updates ---------------------------------------------------

    def _update_structure(self) -> None:
        self.query_one(StructureViewer).render_frame()

    def _update_status_line(self) -> None:
        elapsed = int(time.monotonic() - self._t_start)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        n_done = len(self._completed_results)
        pct = n_done / self._num_designs * 100 if self._num_designs else 0
        cost = elapsed * self._num_gpus * B200_COST_PER_SEC
        self.query_one("#status-line", Static).update(
            f" {h}:{m:02d}:{s:02d}"
            f"  \u2502  {n_done}/{self._num_designs} designs ({pct:.0f}%)"
            f"  \u2502  ${cost:.2f}"
            f"  \u2502  {self._run_name}"
        )

    # -- Actions ------------------------------------------------------------

    def action_toggle_charset(self) -> None:
        sv = self.query_one(StructureViewer)
        sv._charset = "unicode" if sv._charset == "dots" else "dots"

    def action_toggle_complex(self) -> None:
        self.query_one(StructureViewer).toggle_complex()

    def action_interrupt(self) -> None:
        self._interrupted = True
        for h in self._handles:
            try:
                h.cancel()
            except Exception:
                pass
        self.exit(
            DashboardResult(
                completed_results=self._completed_results,
                interrupted=True,
            )
        )

    # -- File I/O -----------------------------------------------------------

    def _save_design(self, data: dict) -> None:
        import orjson

        seed_hex = f"{data['seed']:06x}"
        json_path = self._run_dir / f"design_{seed_hex}.json"
        json_path.write_bytes(orjson.dumps(data))
        if "pdb_string" in data:
            (self._run_dir / f"design_{seed_hex}.pdb").write_text(data["pdb_string"])
        if "monomer_pdb_string" in data:
            (self._run_dir / f"design_{seed_hex}_monomer.pdb").write_text(
                data["monomer_pdb_string"]
            )
