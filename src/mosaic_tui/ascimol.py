"""
ascimol — ASCII Protein Structure Viewer

A text-based cartoon ribbon renderer for terminal use.
Renders protein structures as colored Unicode art with depth shading,
designed for use with `rich` in Python.

Usage:
    from ascimol import ProteinViewer
    viewer = ProteinViewer.from_gemmi(structure)
    text = viewer._render_frame("cartoon", "ss", "dots", 80, 40, (0, 30, 0))
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from rich.text import Text


# ── Color Palettes ─────────────────────────────────────────────────

CHAIN_COLORS = [
    (27, 158, 119),  # teal (Dark2)
    (217, 95, 2),  # orange (Dark2)
    (117, 112, 179),  # purple (Dark2)
    (231, 41, 138),  # pink (Dark2)
    (102, 166, 30),  # green (Dark2)
    (230, 171, 2),  # yellow (Dark2)
    (166, 118, 29),  # brown (Dark2)
    (102, 102, 102),  # gray (Dark2)
]

SS_COLORS = {
    "H": (220, 80, 80),  # helix → red
    "E": (80, 160, 220),  # sheet → blue
    "C": (160, 160, 160),  # coil  → gray
}

# Depth shading characters (from dense/close to sparse/far)
SHADE_CHARS_UNICODE = "█▓▒░·"
SHADE_CHARS_DOTS = "●◉◎○◌"

# Ribbon geometry
RIBBON_WIDTH = {"H": 1.6, "E": 2.4, "C": 0.6}
RIBBON_THICKNESS = {"H": 1.6, "E": 0.5, "C": 0.6}
RIBBON_PPR = 8  # spline points per residue


# ── Data Classes ───────────────────────────────────────────────────


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str
    bfactor: float = 0.0
    ss: str = "C"  # secondary structure: H=helix, E=sheet, C=coil


@dataclass
class Residue:
    resseq: int
    chain: str
    resname: str
    ca: np.ndarray
    o: np.ndarray | None
    ss: str = "C"
    bfactor: float = 0.0


@dataclass
class ProteinViewer:
    atoms: list[Atom]
    title: str = "Structure"
    _highlights: dict[int, str] = field(
        default_factory=dict
    )  # resseq → "hotspot" | "cursor"

    # ── Constructors ────────────────────────────────────────────

    @classmethod
    def from_string(cls, pdb_string: str, **kwargs) -> "ProteinViewer":
        """Parse PDB-format string directly."""
        atoms, title = _parse_pdb(pdb_string)
        viewer = cls(atoms=atoms, title=title, **kwargs)
        if not any(a.ss != "C" for a in atoms):
            viewer._assign_ss_from_ca()
        return viewer

    @classmethod
    def from_gemmi(cls, structure, model_idx: int = 0, **kwargs) -> "ProteinViewer":
        """Create viewer from a gemmi.Structure (or Model/Chain).

        Args:
            structure: gemmi.Structure, gemmi.Model, or gemmi.Chain
            model_idx: model index (only used if structure is gemmi.Structure)
        """
        import gemmi

        # Normalize input to a model + optional structure for SS
        st = None
        if isinstance(structure, gemmi.Structure):
            st = structure
            model = structure[model_idx]
        elif isinstance(structure, gemmi.Model):
            model = structure
        elif isinstance(structure, gemmi.Chain):
            # Wrap single chain in a temporary model
            model = gemmi.Model("1")
            model.add_chain(structure)
        else:
            raise TypeError(
                f"Expected gemmi.Structure/Model/Chain, got {type(structure)}"
            )

        # Parse secondary structure annotations from structure
        helix_ranges = []
        sheet_ranges = []
        if st is not None:
            for helix in st.helices:
                try:
                    helix_ranges.append(
                        (
                            helix.start.chain_name,
                            helix.start.res_id.seqid.num,
                            helix.end.res_id.seqid.num,
                        )
                    )
                except Exception:
                    pass
            for sheet in st.sheets:
                for strand in sheet.strands:
                    try:
                        sheet_ranges.append(
                            (
                                strand.start.chain_name,
                                strand.start.res_id.seqid.num,
                                strand.end.res_id.seqid.num,
                            )
                        )
                    except Exception:
                        pass

        atoms = []
        serial = 1
        for chain in model:
            for residue in chain:
                for atom in residue:
                    ss = "C"
                    rnum = residue.seqid.num
                    cname = chain.name
                    for ch, start, end in helix_ranges:
                        if cname == ch and start <= rnum <= end:
                            ss = "H"
                            break
                    if ss == "C":
                        for ch, start, end in sheet_ranges:
                            if cname == ch and start <= rnum <= end:
                                ss = "E"
                                break

                    atoms.append(
                        Atom(
                            serial=serial,
                            name=atom.name,
                            resname=residue.name,
                            chain=cname,
                            resseq=rnum,
                            x=atom.pos.x,
                            y=atom.pos.y,
                            z=atom.pos.z,
                            element=atom.element.name,
                            bfactor=atom.b_iso,
                            ss=ss,
                        )
                    )
                    serial += 1

        title = kwargs.pop("title", None)
        if title is None:
            title = st.name if st and st.name else "Structure"
        viewer = cls(atoms=atoms, title=title, **kwargs)
        if not helix_ranges and not sheet_ranges:
            viewer._assign_ss_from_ca()
        return viewer

    # ── Secondary Structure Assignment ─────────────────────────

    def _assign_ss_from_ca(self) -> None:
        """Assign secondary structure from CA geometry when annotations are missing.

        Uses CA-CA distance patterns:
          - Helix: d(i, i+3) in [4.5, 6.0] Å (α-helix ~5.3 Å)
          - Sheet: d(i, i+2) in [6.2, 7.5] Å (extended ~6.7 Å)
        """
        chains = self._extract_residues()
        ss_map: dict[tuple[str, int], str] = {}

        for chain_id, residues in chains.items():
            n = len(residues)
            if n < 4:
                continue
            cas = [r.ca for r in residues]

            is_helix = [False] * n
            for i in range(n - 3):
                d = np.linalg.norm(cas[i] - cas[i + 3])
                if 4.5 < d < 6.0:
                    for j in range(i, i + 4):
                        is_helix[j] = True

            is_sheet = [False] * n
            for i in range(n - 2):
                if is_helix[i]:
                    continue
                d = np.linalg.norm(cas[i] - cas[i + 2])
                if 6.2 < d < 7.5:
                    for j in range(i, i + 3):
                        if not is_helix[j]:
                            is_sheet[j] = True

            for i, res in enumerate(residues):
                if is_helix[i]:
                    ss_map[(chain_id, res.resseq)] = "H"
                elif is_sheet[i]:
                    ss_map[(chain_id, res.resseq)] = "E"

        for atom in self.atoms:
            ss = ss_map.get((atom.chain, atom.resseq))
            if ss:
                atom.ss = ss

    # ── Ribbon Geometry ─────────────────────────────────────────

    def _extract_residues(self) -> dict[str, list[Residue]]:
        """Group atoms by residue, extract CA and O positions."""
        residue_atoms: dict[tuple[str, int], list[Atom]] = {}
        for a in self.atoms:
            residue_atoms.setdefault((a.chain, a.resseq), []).append(a)

        chains: dict[str, list[Residue]] = {}
        for (chain, resseq), atoms in sorted(residue_atoms.items()):
            ca = next((a for a in atoms if a.name == "CA"), None)
            if not ca:
                continue
            o = next((a for a in atoms if a.name == "O"), None)
            ca_pos = np.array([ca.x, ca.y, ca.z])
            o_pos = np.array([o.x, o.y, o.z]) if o else None
            chains.setdefault(chain, []).append(
                Residue(resseq, chain, ca.resname, ca_pos, o_pos, ca.ss, ca.bfactor)
            )
        return chains

    def _build_ribbon_mesh(
        self,
        residues: list[Residue],
    ) -> list[tuple[np.ndarray, str, float, str, int, float]]:
        """Build ribbon surface points for one chain.

        Returns list of (position, ss, bfactor, chain, resseq, face_dot)
        where face_dot encodes surface orientation for lighting.
        """
        n = len(residues)
        if n < 3:
            return []

        ca = np.array([r.ca for r in residues])
        t = np.arange(n, dtype=float)
        t_fine = np.linspace(0, n - 1, n * RIBBON_PPR)

        try:
            cs = [CubicSpline(t, ca[:, ax], bc_type="natural") for ax in range(3)]
        except Exception:
            return []

        spine = np.column_stack([c(t_fine) for c in cs])
        tangents = np.column_stack([c(t_fine, 1) for c in cs])
        tn = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
        tangents /= tn

        # Peptide plane orientation from Cα→O
        o_dirs = []
        for r in residues:
            if r.o is not None:
                d = r.o - r.ca
                nm = np.linalg.norm(d)
                o_dirs.append(d / nm if nm > 0.1 else np.array([0.0, 1.0, 0.0]))
            else:
                o_dirs.append(np.array([0.0, 1.0, 0.0]))
        o_dirs_arr = np.array(o_dirs)

        # Fix flipping
        for i in range(1, len(o_dirs_arr)):
            if np.dot(o_dirs_arr[i], o_dirs_arr[i - 1]) < 0:
                o_dirs_arr[i] = -o_dirs_arr[i]

        # Interpolate O-directions along spline
        o_cs = [CubicSpline(t, o_dirs_arr[:, ax], bc_type="natural") for ax in range(3)]
        o_fine = np.column_stack([c(t_fine) for c in o_cs])

        # Local coordinate frame: binormal (across ribbon) and normal (face)
        binormals = np.cross(tangents, o_fine)
        binormals /= np.linalg.norm(binormals, axis=1, keepdims=True).clip(1e-8)
        normals = np.cross(binormals, tangents)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)

        # Generate cross-section surface points
        points = []
        for i, tf in enumerate(t_fine):
            res_idx = min(int(tf), n - 1)
            res = residues[res_idx]
            ss = res.ss
            w_half = RIBBON_WIDTH[ss] * 0.5
            th_half = RIBBON_THICKNESS[ss] * 0.5
            P, B, N = spine[i], binormals[i], normals[i]

            if ss == "H":
                # Helix: elliptical tube cross-section
                for j in range(10):
                    a = 2 * math.pi * j / 10
                    offset = B * w_half * math.cos(a) + N * th_half * math.sin(a)
                    points.append(
                        (
                            P + offset,
                            ss,
                            res.bfactor,
                            res.chain,
                            res.resseq,
                            math.sin(a),
                        )
                    )
            elif ss == "E":
                # Sheet: flat wide ribbon with arrow taper at C-terminus
                next_idx = min(res_idx + 1, n - 1)
                is_end = (
                    (residues[next_idx].ss != "E") if next_idx != res_idx else False
                )
                local_t = tf - int(tf)

                if is_end:
                    if local_t < 0.3:
                        w_scale = 1.0 + local_t * 3.0  # widen to 1.9x
                    else:
                        w_scale = max(0.0, (1.0 - local_t) * 1.9 / 0.7)  # taper to 0
                    w_half *= w_scale

                n_across = 8
                for j in range(n_across):
                    frac = j / (n_across - 1) - 0.5
                    for side, fd in [(1.0, 0.8), (-1.0, -0.5)]:
                        pt = P + B * w_half * 2 * frac + N * th_half * side
                        points.append((pt, ss, res.bfactor, res.chain, res.resseq, fd))
                    pt = P + B * w_half * 2 * frac
                    points.append((pt, ss, res.bfactor, res.chain, res.resseq, 0.3))

                for edge in [-1.0, 1.0]:
                    for k in range(3):
                        s = (k / 2) - 0.5
                        pt = P + B * w_half * edge + N * th_half * s
                        points.append((pt, ss, res.bfactor, res.chain, res.resseq, 0.2))
            else:
                # Coil: thin round tube
                for j in range(8):
                    a = 2 * math.pi * j / 8
                    offset = B * w_half * math.cos(a) + N * th_half * math.sin(a)
                    points.append(
                        (
                            P + offset,
                            ss,
                            res.bfactor,
                            res.chain,
                            res.resseq,
                            math.sin(a),
                        )
                    )

        return points

    # ── Cartoon Cache & Rendering ──────────────────────────────

    def _ensure_cartoon_cache(self) -> dict[str, np.ndarray] | None:
        """Build and cache ribbon mesh geometry (static across frames)."""
        if hasattr(self, "_cartoon_cache"):
            return self._cartoon_cache  # type: ignore[has-type]

        chains = self._extract_residues()
        if not chains:
            self._cartoon_cache = None
            return None

        all_pts = []
        for chain_id in sorted(chains.keys()):
            all_pts.extend(self._build_ribbon_mesh(chains[chain_id]))
        if not all_pts:
            self._cartoon_cache = None
            return None

        positions = np.array([p[0] for p in all_pts])
        center = positions.mean(axis=0)
        positions -= center

        # Pre-extract metadata as NumPy arrays
        n = len(all_pts)
        ss_arr = np.empty(n, dtype="U1")
        bf_arr = np.empty(n, dtype=np.float64)
        ch_arr = np.empty(n, dtype="U1")
        rseq_arr = np.empty(n, dtype=np.int32)
        fdot_arr = np.empty(n, dtype=np.float64)
        for i, (_, ss, bf, ch, rseq, fdot) in enumerate(all_pts):
            ss_arr[i] = ss
            bf_arr[i] = bf
            ch_arr[i] = ch
            rseq_arr[i] = rseq
            fdot_arr[i] = fdot

        self._cartoon_cache = {
            "positions": positions,
            "ss": ss_arr,
            "bf": bf_arr,
            "ch": ch_arr,
            "rseq": rseq_arr,
            "fdot": fdot_arr,
        }
        return self._cartoon_cache

    def _ribbon_colors_vectorized(
        self,
        ss_arr: np.ndarray,
        ch_arr: np.ndarray,
        rseq_arr: np.ndarray,
        bf_arr: np.ndarray,
        color_by: str,
    ) -> np.ndarray:
        """Vectorized color lookup for all ribbon points."""
        n = len(ss_arr)
        colors = np.empty((n, 3), dtype=np.float64)

        if color_by == "highlight":
            colors[:] = (120, 140, 170)
            for rseq, hl_type in self._highlights.items():
                mask = rseq_arr == rseq
                if hl_type == "cursor":
                    colors[mask] = (240, 220, 60)
                elif hl_type == "hotspot":
                    colors[mask] = (230, 60, 60)
        elif color_by == "ss":
            for ss_type, color in SS_COLORS.items():
                colors[ss_arr == ss_type] = color
            known = np.zeros(n, dtype=bool)
            for ss_type in SS_COLORS:
                known |= ss_arr == ss_type
            if not known.all():
                colors[~known] = SS_COLORS["C"]
        elif color_by == "chain":
            for ch_id in np.unique(ch_arr):
                idx = ord(ch_id) - ord("A") if ch_id.isalpha() else 0
                colors[ch_arr == ch_id] = CHAIN_COLORS[idx % len(CHAIN_COLORS)]
        else:
            # Fallback: SS coloring
            for ss_type, color in SS_COLORS.items():
                colors[ss_arr == ss_type] = color
            known = np.zeros(n, dtype=bool)
            for ss_type in SS_COLORS:
                known |= ss_arr == ss_type
            if not known.all():
                colors[~known] = (180, 180, 180)

        return colors

    def _render_cartoon_buffers(
        self,
        color_by: str,
        charset: str,
        w: int,
        h: int,
        rotation: tuple[float, float, float],
        stable_scale: bool = False,
    ) -> tuple[str, np.ndarray, np.ndarray] | None:
        """Render cartoon ribbon to character-grid buffers.

        Args:
            stable_scale: use bounding sphere so scale doesn't change with
                rotation (good for animations, slightly smaller).

        Returns (shade_chars, shade_buf, color_buf) or None if no backbone:
            shade_chars: str of depth-shade characters for the charset
            shade_buf: (h, w) intp array, -1 = empty, else index into shade_chars
            color_buf: (h, w, 3) uint8 RGB array
        """
        cache = self._ensure_cartoon_cache()
        if cache is None:
            return None

        positions = cache["positions"]
        ss_arr = cache["ss"]
        bf_arr = cache["bf"]
        ch_arr = cache["ch"]
        rseq_arr = cache["rseq"]
        fdot_arr = cache["fdot"]

        # Rotate (copy to avoid mutating cache)
        rx, ry, rz = [math.radians(a) for a in rotation]
        rotated = _rotate(positions, rx, ry, rz)

        xs, ys, zs = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        zmin, zmax = zs.min(), zs.max()
        depth_norm = (zs - zmin) / (zmax - zmin or 1.0)

        if stable_scale:
            # Bounding sphere: rotation-invariant scale (no drift in animations)
            bounding_r = np.linalg.norm(positions, axis=1).max() or 1.0
            xrange = 2 * bounding_r
            yrange = 2 * bounding_r
        else:
            # Tight data-driven bounds: fills the frame for the current rotation
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            xrange = xmax - xmin or 1.0
            yrange = ymax - ymin or 1.0

        # Vectorized color + brightness
        base_colors = self._ribbon_colors_vectorized(
            ss_arr, ch_arr, rseq_arr, bf_arr, color_by
        )
        bri = np.clip(
            0.55 + 0.45 * np.maximum(0, fdot_arr) - depth_norm * 0.2, 0.25, 1.0
        )
        final_colors = np.clip(base_colors * bri[:, None], 0, 255).astype(np.uint8)

        # Vectorized shade index
        lighting = np.clip(depth_norm * 0.7 + (1.0 - fdot_arr) * 0.15, 0, 1)

        shade_chars = SHADE_CHARS_DOTS if charset == "dots" else SHADE_CHARS_UNICODE
        ns = len(shade_chars)

        shade_idx = np.clip((lighting * (ns - 1)).astype(np.intp), 0, ns - 1)

        pad = 2
        ew, eh = w - 2 * pad, h - 2 * pad
        scale = min(ew / xrange, eh / yrange * 2.0)

        if stable_scale:
            pxi = (xs * scale + w / 2).astype(np.intp)
            pyi = (-ys * scale / 2.0 + h / 2).astype(np.intp)
        else:
            x_off = (ew - xrange * scale) / 2
            y_off = (eh - yrange * scale / 2.0) / 2
            pxi = ((xs - xmin) * scale + pad + x_off).astype(np.intp)
            pyi = (eh - (ys - ymin) * scale / 2.0 + pad - y_off).astype(np.intp)

        # Painter's algorithm: sort back-to-front, fancy-index
        valid = (pxi >= 0) & (pxi < w) & (pyi >= 0) & (pyi < h)
        order = np.argsort(-zs)
        order = order[valid[order]]

        z_buf = np.full((h, w), np.inf)
        color_buf = np.zeros((h, w, 3), dtype=np.uint8)
        shade_buf = np.full((h, w), -1, dtype=np.intp)

        pyi_s, pxi_s = pyi[order], pxi[order]
        z_buf[pyi_s, pxi_s] = zs[order]
        color_buf[pyi_s, pxi_s] = final_colors[order]
        shade_buf[pyi_s, pxi_s] = shade_idx[order]

        # Vectorized gap filling
        filled = shade_buf >= 0
        _fill_gaps_np(color_buf, shade_buf, z_buf, filled, w, h)

        return shade_chars, shade_buf, color_buf

    def _render_frame(
        self,
        color_by: str,
        charset: str,
        w: int,
        h: int,
        rotation: tuple[float, float, float],
    ) -> Text:
        """Render cartoon ribbon to a Rich Text object."""
        result = self._render_cartoon_buffers(color_by, charset, w, h, rotation)
        if result is None:
            return Text("No backbone for cartoon")

        shade_chars, shade_buf, color_buf = result
        filled = shade_buf >= 0

        # Build Text with run-length batching
        text = Text()
        for row in range(h):
            row_filled = filled[row]
            row_shade = shade_buf[row]
            row_color = color_buf[row]
            run_chars = []
            run_style = ""
            for col in range(w):
                if row_filled[col]:
                    ch = shade_chars[row_shade[col]]
                    r, g, b = row_color[col]
                    style = f"rgb({r},{g},{b})"
                else:
                    ch = " "
                    style = ""
                if style == run_style:
                    run_chars.append(ch)
                else:
                    if run_chars:
                        text.append("".join(run_chars), style=run_style)
                    run_chars = [ch]
                    run_style = style
            if run_chars:
                text.append("".join(run_chars), style=run_style)
            if row < h - 1:
                text.append("\n")
        return text


# ── Gap Filling ─────────────────────────────────────────────────────


def _fill_gaps_np(
    color_buf: np.ndarray,
    shade_buf: np.ndarray,
    z_buf: np.ndarray,
    filled: np.ndarray,
    w: int,
    h: int,
) -> None:
    """Vectorized gap filling using NumPy arrays."""
    # Horizontal: empty cells with filled neighbors on both sides
    h_gaps = ~filled[:, 1:-1] & filled[:, :-2] & filled[:, 2:]
    with np.errstate(invalid="ignore"):
        h_z_ok = np.abs(z_buf[:, :-2] - z_buf[:, 2:]) < 3.0
    h_fill = h_gaps & h_z_ok
    if h_fill.any():
        rows, cols = np.where(h_fill)
        cols += 1  # offset for the 1:-1 slice
        z_buf[rows, cols] = (z_buf[rows, cols - 1] + z_buf[rows, cols + 1]) * 0.5
        color_buf[rows, cols] = (
            (
                color_buf[rows, cols - 1].astype(np.int16)
                + color_buf[rows, cols + 1].astype(np.int16)
            )
            // 2
        ).astype(np.uint8)
        shade_buf[rows, cols] = (
            shade_buf[rows, cols - 1] + shade_buf[rows, cols + 1]
        ) // 2
        filled[rows, cols] = True

    # Vertical: empty cells with filled neighbors above and below
    v_gaps = ~filled[1:-1, :] & filled[:-2, :] & filled[2:, :]
    with np.errstate(invalid="ignore"):
        v_z_ok = np.abs(z_buf[:-2, :] - z_buf[2:, :]) < 3.0
    v_avg_z = (z_buf[:-2, :] + z_buf[2:, :]) * 0.5
    v_fill = v_gaps & v_z_ok & (v_avg_z < z_buf[1:-1, :])
    if v_fill.any():
        rows, cols = np.where(v_fill)
        rows += 1
        z_buf[rows, cols] = v_avg_z[rows - 1, cols]
        color_buf[rows, cols] = (
            (
                color_buf[rows - 1, cols].astype(np.int16)
                + color_buf[rows + 1, cols].astype(np.int16)
            )
            // 2
        ).astype(np.uint8)
        shade_buf[rows, cols] = (
            shade_buf[rows - 1, cols] + shade_buf[rows + 1, cols]
        ) // 2


# ── PDB Parsing ─────────────────────────────────────────────────────


def _parse_pdb(content: str) -> tuple[list[Atom], str]:
    atoms = []
    title = "Unknown"
    helix_ranges = []  # (chain, start, end)
    sheet_ranges = []

    for line in content.split("\n"):
        rec = line[:6].strip()

        if rec == "TITLE" and title == "Unknown":
            title = line[10:].strip()

        elif rec == "HELIX":
            try:
                chain = line[19]
                start = int(line[21:25].strip())
                end = int(line[33:37].strip())
                helix_ranges.append((chain, start, end))
            except (ValueError, IndexError):
                pass

        elif rec == "SHEET":
            try:
                chain = line[21]
                start = int(line[22:26].strip())
                end = int(line[33:37].strip())
                sheet_ranges.append((chain, start, end))
            except (ValueError, IndexError):
                pass

        elif rec in ("ATOM", "HETATM"):
            try:
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21]
                resseq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                bfactor = float(line[60:66].strip()) if len(line) >= 66 else 0.0
                element = line[76:78].strip() if len(line) >= 78 else name[0]

                atoms.append(
                    Atom(
                        serial=serial,
                        name=name,
                        resname=resname,
                        chain=chain,
                        resseq=resseq,
                        x=x,
                        y=y,
                        z=z,
                        element=element,
                        bfactor=bfactor,
                    )
                )
            except (ValueError, IndexError):
                pass

    # Assign secondary structure
    for atom in atoms:
        for chain, start, end in helix_ranges:
            if atom.chain == chain and start <= atom.resseq <= end:
                atom.ss = "H"
                break
        for chain, start, end in sheet_ranges:
            if atom.chain == chain and start <= atom.resseq <= end:
                atom.ss = "E"
                break

    return atoms, title


# ── 3D Math ─────────────────────────────────────────────────────────


def _rotate(coords: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """Apply Euler rotation (XYZ order)."""
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return (R @ coords.T).T
