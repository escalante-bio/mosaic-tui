"""Interactive fullscreen configuration screen for design_rich.py.

Uses Textual for the TUI. Arrow keys navigate, Enter edits,
F5 launches, Esc aborts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from mosaic_tui.design_common import (
    BoltzGenConfig,
    CifTarget,
    DesignConfig,
    FixedOptim,
    HyperparamRanges,
    LossWeights,
    MethodConfig,
    Range,
    RankingConfig,
    SeqTarget,
    SimplexConfig,
    Target,
    default_config,
    default_method,
    target_label,
)

if TYPE_CHECKING:
    from mosaic_tui.ascimol import ProteinViewer


# ---------------------------------------------------------------------------
# Field descriptors (bridge between DesignConfig and form widgets)
# ---------------------------------------------------------------------------


@dataclass
class ScalarField:
    category: str
    key: str
    field_type: str  # "float", "int", "bool", "str"
    default: float | int | bool | str


@dataclass
class RangeField:
    category: str
    key: str
    field_type: str  # "int" or "float"
    default_lo: float | int
    default_hi: float | int


def _ft(v: object) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    return "str"


def _method_categories(
    method: MethodConfig,
) -> list[tuple[str, list[ScalarField | RangeField]]]:
    """Build field descriptors for the method-specific tabs."""
    categories: list[tuple[str, list[ScalarField | RangeField]]] = []
    match method:
        case SimplexConfig():
            defaults = SimplexConfig()

            loss_fields: list[ScalarField | RangeField] = [
                ScalarField("Loss", "recycling_steps", "int", defaults.recycling_steps),
                ScalarField("Loss", "num_samples", "int", defaults.num_samples),
                ScalarField("Loss", "use_msa", "bool", defaults.use_msa),
            ]
            for key in LossWeights.__dataclass_fields__:
                loss_fields.append(
                    ScalarField(
                        "Loss", key, "float", getattr(defaults.loss_weights, key)
                    )
                )
            loss_fields.append(
                ScalarField("Loss", "mpnn_temp", "float", defaults.mpnn_temp)
            )
            categories.append(("Loss", loss_fields))

            opt_fields: list[ScalarField | RangeField] = []
            for key in HyperparamRanges.__dataclass_fields__:
                r = getattr(defaults.hyperparam_ranges, key)
                ft = "int" if key.endswith("_steps") else "float"
                opt_fields.append(
                    RangeField(
                        "Optimizer",
                        key,
                        ft,
                        r.lo if ft == "float" else int(r.lo),
                        r.hi if ft == "float" else int(r.hi),
                    )
                )
            for key in FixedOptim.__dataclass_fields__:
                opt_fields.append(
                    ScalarField(
                        "Optimizer",
                        key,
                        "float",
                        getattr(defaults.fixed_optim, key),
                    )
                )
            categories.append(("Optimizer", opt_fields))

        case BoltzGenConfig():
            bg_defaults = BoltzGenConfig()
            bg_fields: list[ScalarField | RangeField] = []
            for key in BoltzGenConfig.__dataclass_fields__:
                bg_fields.append(
                    ScalarField(
                        "BoltzGen",
                        key,
                        _ft(getattr(bg_defaults, key)),
                        getattr(bg_defaults, key),
                    )
                )
            categories.append(("BoltzGen", bg_fields))

    return categories


def _field_descriptors(
    config: DesignConfig,
) -> list[tuple[str, list[ScalarField | RangeField]]]:
    """Build ordered (category_name, fields) list from a DesignConfig."""
    defaults = default_config()
    categories: list[tuple[str, list[ScalarField | RangeField]]] = []

    # Run Parameters (shared)
    drp = defaults.run_params
    rp_fields: list[ScalarField | RangeField] = [
        ScalarField("Run Parameters", key, _ft(getattr(drp, key)), getattr(drp, key))
        for key in ("binder_length", "num_designs", "num_gpus", "run")
    ]
    categories.append(("Run Parameters", rp_fields))

    # Method-specific tabs
    categories.extend(_method_categories(config.method))

    # Ranking (shared)
    drk = defaults.ranking
    rk_fields: list[ScalarField | RangeField] = [
        ScalarField("Ranking", key, _ft(getattr(drk, key)), getattr(drk, key))
        for key in RankingConfig.__dataclass_fields__
    ]
    categories.append(("Ranking", rk_fields))

    return categories


def _get_value(config: DesignConfig, category: str, key: str) -> object:
    """Read a value from DesignConfig by category+key."""
    match category:
        case "Run Parameters":
            return getattr(config.run_params, key)
        case "Loss":
            assert isinstance(config.method, SimplexConfig)
            m = config.method
            if hasattr(m.loss_weights, key):
                return getattr(m.loss_weights, key)
            return getattr(m, key)
        case "Optimizer":
            assert isinstance(config.method, SimplexConfig)
            m = config.method
            if hasattr(m.hyperparam_ranges, key):
                return getattr(m.hyperparam_ranges, key)
            if hasattr(m.fixed_optim, key):
                return getattr(m.fixed_optim, key)
            return getattr(m, key)
        case "BoltzGen":
            assert isinstance(config.method, BoltzGenConfig)
            return getattr(config.method, key)
        case "Ranking":
            return getattr(config.ranking, key)
        case _:
            raise KeyError(category)


# ---------------------------------------------------------------------------
# Textual TUI
# ---------------------------------------------------------------------------


CSS = """
Screen {
    background: $surface;
}

.field-row {
    layout: horizontal;
    height: 3;
    padding: 0 1;
}
.field-row:focus-within {
    background: $accent 15%;
}

.field-label {
    width: 32;
    height: 3;
    content-align-vertical: middle;
    padding: 1 1 0 1;
    color: $text;
}

.field-input {
    width: 20;
    height: 3;
}

.range-dash {
    width: 3;
    height: 3;
    content-align: center middle;
    padding: 1 0 0 0;
    color: $text-muted;
}

.field-default {
    width: 20;
    height: 3;
    content-align-vertical: middle;
    padding: 1 1 0 0;
    color: $text-muted;
}

.field-switch {
    width: 12;
    height: 3;
    padding: 0 1;
}

.changed .field-input {
    color: $warning;
}

HotspotTab {
    height: 1fr;
}
"""

PICKER_CSS = """
Screen {
    background: $surface;
    align: center middle;
}

#picker-box {
    width: 60;
    height: auto;
    padding: 2 4;
    border: solid $primary;
}

.method-btn {
    width: 100%;
    margin: 1 0;
}
"""


class MethodPicker(App[str | None]):
    """Intro screen to choose the design method."""

    CSS = PICKER_CSS

    BINDINGS = [
        Binding("escape", "abort", "Abort", priority=True),
    ]

    def __init__(self, target: Target) -> None:
        super().__init__()
        self._target = target

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        yield Header()
        with Vertical(id="picker-box"):
            yield Static("Choose a design method:\n")
            yield Button(
                "Hallucination (Simplex)",
                id="simplex",
                classes="method-btn",
                variant="primary",
            )
            yield Static(
                "[dim]Gradient optimization through Protenix structure prediction. Slow, flexible, high-quality. [/dim]\n",
            )
            if isinstance(self._target, CifTarget):
                yield Button(
                    "BoltzGen",
                    id="boltzgen",
                    classes="method-btn",
                    variant="primary",
                )
                yield Static(
                    "[dim]Generative diffusion model for binder sequences + structures. Very fast, high-quality with RL model if you want ~100 AA binders. [/dim]",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "MOSAIC v0.01"
        self.sub_title = target_label(self._target)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.exit(event.button.id)

    def action_abort(self) -> None:
        self.exit(None)


class HotspotTab(Static):
    """Interactive hotspot selection with optional 3D structure viewer.

    Extends Static directly (like the dashboard StructureViewer) so there
    are no parent-child height-resolution issues inside TabPane.
    When viewer is None (sequence-only target), shows only the sequence bar.
    """

    can_focus = True

    def __init__(
        self,
        viewer: "ProteinViewer | None",
        residues: list[tuple[int, str]],
        initial_hotspots: set[int],
        **kwargs,
    ) -> None:
        super().__init__("", **kwargs)
        self._viewer = viewer
        self._residues = residues
        self.hotspots: set[int] = set(initial_hotspots)
        self._cursor_idx: int = 0
        self._rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 4, self._refresh_frame)

    def _build_sequence_bar(self, width: int) -> Text:
        """Build a colored sequence bar showing cursor (yellow) and hotspots (red)."""
        import gemmi

        seq_1 = gemmi.one_letter_code([rn for _, rn in self._residues])
        n = len(seq_1)
        cursor_resseq = self._residues[self._cursor_idx][0] if self._residues else -1

        # Visible window centered on cursor
        half = width // 2
        start = max(0, self._cursor_idx - half)
        end = min(n, start + width)
        if end - start < width:
            start = max(0, end - width)

        bar = Text()
        for i in range(start, end):
            resseq = self._residues[i][0]
            aa = seq_1[i]
            if resseq == cursor_resseq:
                bar.append(aa, style="bold black on rgb(240,220,60)")
            elif resseq in self.hotspots:
                bar.append(aa, style="bold white on rgb(230,60,60)")
            else:
                bar.append(aa, style="rgb(120,140,170)")
        return bar

    def _refresh_frame(self) -> None:
        if self._viewer is not None:
            self._viewer._highlights = {}
            for rs in self.hotspots:
                self._viewer._highlights[rs] = "hotspot"
            if self._residues:
                cursor_resseq = self._residues[self._cursor_idx][0]
                self._viewer._highlights[cursor_resseq] = "cursor"

        w = max(20, self.size.width)

        # Sequence bar at top
        seq_bar = self._build_sequence_bar(w)

        output = seq_bar
        output.append("\n")

        if self._viewer is not None:
            # Render 3D frame (reserve 2 lines: sequence bar + status bar)
            h = max(5, self.size.height - 2)
            frame = self._viewer._render_frame(
                color_by="highlight",
                charset="dots",
                w=w,
                h=h,
                rotation=self._rotation,
            )
            output.append_text(frame)

        if self._residues:
            resseq, resname = self._residues[self._cursor_idx]
            hs_list = sorted(self.hotspots)
            hs_str = ",".join(str(h) for h in hs_list) if hs_list else "none"
            controls = "\u2190\u2192 move  \u2191\u2193 jump  Enter toggle"
            if self._viewer is not None:
                controls += "  QE/WS rotate"
            output.append(
                f"\n Res {resseq} {resname} [{self._cursor_idx + 1}/{len(self._residues)}]"
                f"  \u2502  Hotspots: {hs_str} ({len(hs_list)})"
                f"  \u2502  {controls}",
                style="reverse",
            )

        self.update(output)

    def on_key(self, event) -> None:
        if not self._residues:
            return
        n = len(self._residues)
        match event.key:
            case "left":
                self._cursor_idx = max(0, self._cursor_idx - 1)
            case "right":
                self._cursor_idx = min(n - 1, self._cursor_idx + 1)
            case "up":
                self._cursor_idx = max(0, self._cursor_idx - 10)
            case "down":
                self._cursor_idx = min(n - 1, self._cursor_idx + 10)
            case "enter" | "space":
                resseq = self._residues[self._cursor_idx][0]
                self.hotspots.symmetric_difference_update({resseq})
            case "q" if self._viewer is not None:
                rx, ry, rz = self._rotation
                self._rotation = (rx, ry - 15, rz)
            case "e" if self._viewer is not None:
                rx, ry, rz = self._rotation
                self._rotation = (rx, ry + 15, rz)
            case "w" if self._viewer is not None:
                rx, ry, rz = self._rotation
                self._rotation = (rx - 15, ry, rz)
            case "s" if self._viewer is not None:
                rx, ry, rz = self._rotation
                self._rotation = (rx + 15, ry, rz)
            case _:
                return
        event.prevent_default()


class FieldRow(Widget):
    """A single scalar config field: label + input."""

    DEFAULT_CSS = ""

    def __init__(
        self,
        desc: ScalarField,
        value: object,
        disabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(classes="field-row", **kwargs)
        self.desc = desc
        self._init_value = value
        self._disabled = disabled

    def compose(self) -> ComposeResult:
        yield Static(self.desc.key, classes="field-label")
        if self.desc.field_type == "bool":
            sw = Switch(
                value=bool(self._init_value),
                classes="field-switch",
                disabled=self._disabled,
            )
            sw.field_desc = self.desc
            yield sw
            yield Static(
                f"(default: {'yes' if self.desc.default else 'no'})",
                classes="field-default",
            )
        else:
            val_str = _fmt(self._init_value, self.desc.field_type)
            inp = Input(
                value=val_str,
                placeholder=_fmt(self.desc.default, self.desc.field_type),
                type="number" if self.desc.field_type in ("int", "float") else "text",
                classes="field-input",
                disabled=self._disabled,
            )
            inp.field_desc = self.desc
            yield inp
            default_str = _fmt(self.desc.default, self.desc.field_type)
            yield Static(f"(default: {default_str})", classes="field-default")

    def on_input_changed(self, event: Input.Changed) -> None:
        desc = getattr(event.input, "field_desc", None)
        if desc and event.value != _fmt(desc.default, desc.field_type):
            self.add_class("changed")
        else:
            self.remove_class("changed")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        desc = getattr(event.switch, "field_desc", None)
        if desc and event.value != desc.default:
            self.add_class("changed")
        else:
            self.remove_class("changed")


class RangeRow(Widget):
    """A range config field: label + lo input + dash + hi input."""

    DEFAULT_CSS = ""

    def __init__(
        self,
        desc: RangeField,
        value: Range,
        disabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(classes="field-row", **kwargs)
        self.desc = desc
        self._range = value
        self._disabled = disabled

    def compose(self) -> ComposeResult:
        yield Static(self.desc.key, classes="field-label")
        lo_str = _fmt(self._range.lo, self.desc.field_type)
        lo_inp = Input(
            value=lo_str,
            placeholder=_fmt(self.desc.default_lo, self.desc.field_type),
            type="number",
            classes="field-input",
            id=f"{self.desc.key}__lo",
            disabled=self._disabled,
        )
        lo_inp.field_desc = self.desc
        lo_inp.range_part = "lo"
        yield lo_inp
        yield Static(" \u2014 ", classes="range-dash")
        hi_str = _fmt(self._range.hi, self.desc.field_type)
        hi_inp = Input(
            value=hi_str,
            placeholder=_fmt(self.desc.default_hi, self.desc.field_type),
            type="number",
            classes="field-input",
            id=f"{self.desc.key}__hi",
            disabled=self._disabled,
        )
        hi_inp.field_desc = self.desc
        hi_inp.range_part = "hi"
        yield hi_inp
        yield Static(
            f"(default: {_fmt(self.desc.default_lo, self.desc.field_type)}"
            f"\u2014{_fmt(self.desc.default_hi, self.desc.field_type)})",
            classes="field-default",
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        desc = getattr(event.input, "field_desc", None)
        part = getattr(event.input, "range_part", None)
        if desc and part:
            default = self.desc.default_lo if part == "lo" else self.desc.default_hi
            if event.value != _fmt(default, desc.field_type):
                self.add_class("changed")
                return
        # Check if either sub-field is changed
        lo_inp = self.query_one(f"#{self.desc.key}__lo", Input)
        hi_inp = self.query_one(f"#{self.desc.key}__hi", Input)
        lo_changed = lo_inp.value != _fmt(self.desc.default_lo, self.desc.field_type)
        hi_changed = hi_inp.value != _fmt(self.desc.default_hi, self.desc.field_type)
        if lo_changed or hi_changed:
            self.add_class("changed")
        else:
            self.remove_class("changed")


def _fmt(val: object, ft: str) -> str:
    if ft == "int":
        return str(int(val))
    if ft == "float":
        return f"{val:g}"
    return str(val)


_DIVIDER_BEFORE: dict[str, set[str]] = {
    "Loss": {"mpnn_temp"},
}

_DIVIDER_BETWEEN_RANGE_AND_SCALAR = {"Optimizer"}


class ConfigScreen(App[tuple[DesignConfig, list[int] | None] | None]):
    """Fullscreen config editor. Returns (DesignConfig, hotspots) on launch, None on abort."""

    CSS = CSS

    BINDINGS = [
        Binding("f5", "launch", "Launch", priority=True),
        Binding("ctrl+s", "launch", "Launch", priority=True),
        Binding("escape", "abort", "Abort", priority=True),
    ]

    def __init__(
        self,
        config: DesignConfig,
        tlabel: str,
        viewer: "ProteinViewer | None" = None,
        residues: list[tuple[int, str]] | None = None,
        initial_hotspots: set[int] | None = None,
        locked_categories: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._tlabel = tlabel
        self._field_descs = _field_descriptors(config)
        self._viewer = viewer
        self._residues = residues or []
        self._initial_hotspots = initial_hotspots or set()
        self._locked_categories = locked_categories or set()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            for cat_name, fields in self._field_descs:
                locked = cat_name in self._locked_categories
                tab_title = f"{cat_name} (locked)" if locked else cat_name
                with TabPane(tab_title):
                    divider_keys = _DIVIDER_BEFORE.get(cat_name, set())
                    insert_range_divider = cat_name in _DIVIDER_BETWEEN_RANGE_AND_SCALAR
                    prev_was_range = False
                    for desc in fields:
                        is_range = isinstance(desc, RangeField)
                        if desc.key in divider_keys:
                            yield Static("\u2500" * 60, classes="field-row")
                        if insert_range_divider and prev_was_range and not is_range:
                            yield Static("\u2500" * 60, classes="field-row")
                        val = _get_value(self._config, desc.category, desc.key)
                        if is_range:
                            yield RangeRow(desc, val, disabled=locked)
                        else:
                            yield FieldRow(desc, val, disabled=locked)
                        prev_was_range = is_range
            if self._viewer is not None or self._residues:
                with TabPane("Hotspots"):
                    yield HotspotTab(
                        self._viewer,
                        self._residues,
                        self._initial_hotspots,
                    )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "MOSAIC v0.01"
        self.sub_title = f"{self._tlabel} | Configuration"

    def action_launch(self) -> None:
        config = self._collect_config()
        hotspot_list = self._collect_hotspots()
        self.exit((config, hotspot_list))

    def action_abort(self) -> None:
        self.exit(None)

    def _collect_hotspots(self) -> list[int] | None:
        """Read hotspots from HotspotTab, if present."""
        try:
            tab = self.query_one(HotspotTab)
            hs = sorted(tab.hotspots)
            return hs if hs else None
        except Exception:
            return None

    def _collect_config(self) -> DesignConfig:
        """Read all widget values back into a frozen DesignConfig."""
        vals: dict[str, dict[str, object]] = {}
        for cat_name, fields in self._field_descs:
            cat_vals: dict[str, object] = {}
            for desc in fields:
                if isinstance(desc, RangeField):
                    lo_inp = self.query_one(f"#{desc.key}__lo", Input)
                    hi_inp = self.query_one(f"#{desc.key}__hi", Input)
                    lo = _parse(lo_inp.value, desc.field_type, desc.default_lo)
                    hi = _parse(hi_inp.value, desc.field_type, desc.default_hi)
                    cat_vals[desc.key] = Range(lo=lo, hi=hi)
                elif desc.field_type == "bool":
                    for sw in self.query(Switch):
                        if getattr(sw, "field_desc", None) is desc:
                            cat_vals[desc.key] = sw.value
                            break
                else:
                    for inp in self.query(Input):
                        if getattr(inp, "field_desc", None) is desc:
                            cat_vals[desc.key] = _parse(
                                inp.value, desc.field_type, desc.default
                            )
                            break
            vals[cat_name] = cat_vals

        rp = vals["Run Parameters"]
        rk = vals["Ranking"]

        method: MethodConfig
        match self._config.method:
            case SimplexConfig():
                loss = vals["Loss"]
                opt = vals["Optimizer"]
                non_weight_keys = (
                    "mpnn_temp",
                    "recycling_steps",
                    "num_samples",
                    "use_msa",
                )
                loss_kv = {
                    k: float(v) for k, v in loss.items() if k not in non_weight_keys
                }
                hr_kv = {k: v for k, v in opt.items() if isinstance(v, Range)}
                fo_kv = {
                    k: float(v) for k, v in opt.items() if not isinstance(v, Range)
                }
                method = SimplexConfig(
                    loss_weights=LossWeights(**loss_kv),
                    hyperparam_ranges=HyperparamRanges(**hr_kv),
                    fixed_optim=FixedOptim(**fo_kv),
                    mpnn_temp=float(loss["mpnn_temp"]),
                    recycling_steps=int(loss["recycling_steps"]),
                    num_samples=int(loss["num_samples"]),
                    use_msa=bool(loss["use_msa"]),
                )
            case BoltzGenConfig():
                bg = vals["BoltzGen"]
                method = BoltzGenConfig(
                    num_sampling_steps=int(bg["num_sampling_steps"]),
                    step_scale=float(bg["step_scale"]),
                    noise_scale=float(bg["noise_scale"]),
                    recycling_steps=int(bg["recycling_steps"]),
                    use_rl_checkpoint=bool(bg["use_rl_checkpoint"]),
                )

        return DesignConfig(
            run_params=dataclasses.replace(
                self._config.run_params,
                binder_length=int(rp["binder_length"]),
                num_designs=int(rp["num_designs"]),
                num_gpus=int(rp["num_gpus"]),
                run=str(rp.get("run", "")),
            ),
            method=method,
            ranking=dataclasses.replace(
                self._config.ranking,
                num_samples=int(rk["num_samples"]),
                recycling_steps=int(rk["recycling_steps"]),
                fast_ranking=bool(rk["fast_ranking"]),
                use_msa=bool(rk["use_msa"]),
                rmsd_cutoff=float(rk["rmsd_cutoff"]),
            ),
        )


def _parse(text: str, ft: str, fallback: object) -> object:
    """Parse a string into the appropriate type, falling back on error."""
    text = text.strip()
    if not text:
        return fallback
    try:
        if ft == "int":
            return int(text)
        if ft == "float":
            return float(text)
        return text
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_config_screen(
    config: DesignConfig,
    target: Target,
    initial_hotspots: set[int] | None = None,
    locked_categories: set[str] | None = None,
    show_method_picker: bool = False,
) -> tuple[DesignConfig, list[int] | None] | None:
    """Show fullscreen config form.

    Returns (DesignConfig, hotspot_list) on launch, or None if aborted.
    """
    import dataclasses

    import gemmi

    if show_method_picker:
        method_choice = MethodPicker(target).run()
        if method_choice is None:
            return None
        config = dataclasses.replace(config, method=default_method(method_choice))

    tlabel = target_label(target)
    viewer: ProteinViewer | None = None
    residues: list[tuple[int, str]] = []

    match target:
        case CifTarget(path=cif, chain=chain):
            from mosaic_tui.ascimol import ProteinViewer
            from mosaic_tui.design_common import preprocess_target_cif

            cif_content = preprocess_target_cif(
                cif, chain, trim_terminals=config.run_params.trim_terminals
            )
            doc = gemmi.cif.read_string(cif_content)
            st = gemmi.make_structure_from_block(doc.sole_block())
            viewer = ProteinViewer.from_gemmi(st, title=tlabel)

            ch_obj = st[0].find_chain(chain)
            entity = st.get_entity_of(ch_obj.get_polymer())
            if entity is not None:
                residues = [(i, name) for i, name in enumerate(entity.full_sequence)]
            else:
                for res in ch_obj:
                    if res.find_atom("CA", "\0"):
                        residues.append((res.seqid.num, res.name))
        case SeqTarget(sequence=seq):
            three_letter = gemmi.expand_one_letter_sequence(seq, gemmi.ResidueKind.AA)
            residues = [(i, name) for i, name in enumerate(three_letter)]

    config_app = ConfigScreen(
        config,
        tlabel,
        viewer=viewer,
        residues=residues,
        initial_hotspots=initial_hotspots or set(),
        locked_categories=locked_categories,
    )
    return config_app.run()
