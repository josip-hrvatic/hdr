#!/usr/bin/env python3
# Run:
#   pip install streamlit pandas openpyxl numpy matplotlib scipy
#   streamlit run app.py

import io
import os
import re
import hashlib
import colorsys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib.lines import Line2D
from matplotlib import colormaps
from scipy.stats import gaussian_kde
import streamlit as st


# ---------------------------
# Utilities
# ---------------------------

def nice_label(colname: str) -> str:
    s = str(colname)
    if "baseline" in s.lower():
        return "baseline"
    m = re.search(r"pruning\s*=\s*([-+]?\d*\.?\d+)", s, flags=re.I)
    n = re.search(r"parsimony\s*=\s*([-+]?\d*\.?\d+)", s, flags=re.I)
    if m and n:
        return f"pr={m.group(1)}, pa={n.group(1)}"
    if m:
        return f"pr={m.group(1)}"
    if n:
        return f"pa={n.group(1)}"
    return s


def sort_key(lbl: str) -> Tuple[float, float]:
    if lbl.lower() == "baseline":
        return (float("-inf"), float("-inf"))
    m = re.search(r"pr=([-+]?\d*\.?\d+)", lbl)
    n = re.search(r"pa=([-+]?\d*\.?\d+)", lbl)
    pr = float(m.group(1)) if m else float("inf")
    pa = float(n.group(1)) if n else float("inf")
    return (pr, pa)


def hdr_prob_to_level(Z: np.ndarray, dx: float, dy: float, probs: Sequence[float]) -> Dict[float, float]:
    """
    Highest Density Region (HDR) thresholds.
    Returns {prob -> density_level}, where region {Z >= level} contains ~prob of total mass.
    """
    z = Z.ravel()
    z = z[np.isfinite(z)]
    if z.size < 10:
        return {}

    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    mass = np.cumsum(z_sorted) * (dx * dy)
    total = mass[-1]
    if not np.isfinite(total) or total <= 0:
        return {}

    out: Dict[float, float] = {}
    for p in probs:
        p = float(p)
        if not (0.0 < p < 1.0):
            continue
        target = p * total
        idx = int(np.searchsorted(mass, target, side="left"))
        idx = min(idx, len(z_sorted) - 1)
        out[p] = float(z_sorted[idx])
    return out


def parse_probs(text: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in re.split(r"[,\s]+", text.strip()) if p.strip()]
    probs = tuple(float(p) for p in parts)
    for p in probs:
        if not (0.0 < p < 1.0):
            raise ValueError(f"HDR probability must be in (0,1): got {p}")
    if len(probs) < 2:
        raise ValueError("Provide at least two HDR probabilities (e.g., 0.8 0.95).")
    return probs


def parse_range(text: str) -> Optional[Tuple[float, float]]:
    t = text.strip()
    if not t:
        return None
    parts = [p.strip() for p in re.split(r"[,\s]+", t) if p.strip()]
    if len(parts) != 2:
        raise ValueError("Range must have exactly two numbers, e.g. 0,10")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Range must be two finite numbers with min < max.")
    return (lo, hi)


def parse_alphas(text: str, nbands: int) -> Optional[Tuple[float, ...]]:
    """
    Accept empty => None (use defaults)
    Else accept comma/space separated list of length = nbands
    """
    t = text.strip()
    if not t:
        return None
    parts = [p.strip() for p in re.split(r"[,\s]+", t) if p.strip()]
    vals = tuple(float(p) for p in parts)
    if len(vals) != nbands:
        raise ValueError(f"Fill alphas must have exactly {nbands} values (got {len(vals)}).")
    for a in vals:
        if not (0.0 <= a <= 1.0):
            raise ValueError("Fill alphas must be in [0,1].")
    return vals


def distinct_colors(n: int) -> List[str]:
    """
    Return n *unique* color hex strings.
    Uses the current style cycle first, then several qualitative palettes,
    then falls back to a golden-ratio HSV generator (still unique).
    """
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    base_colors = prop_cycle.by_key().get("color", []) if prop_cycle is not None else []

    out: List[str] = []
    seen = set()

    def add_color(c) -> None:
        hx = to_hex(c, keep_alpha=False) if not isinstance(c, str) else c
        hx = hx.lower()
        if hx not in seen:
            out.append(hx)
            seen.add(hx)

    for c in base_colors:
        add_color(c)
        if len(out) >= n:
            return out[:n]

    palettes = ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2"]
    for cmap_name in palettes:
        cmap = colormaps.get_cmap(cmap_name)
        N = getattr(cmap, "N", 256)
        for t in np.linspace(0.0, 1.0, N, endpoint=True):
            add_color(cmap(t))
            if len(out) >= n:
                return out[:n]

    golden = 0.618033988749895
    h = 0.0
    while len(out) < n:
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        add_color(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    return out[:n]


@dataclass(frozen=True)
class Variant:
    variant_id: str          # internal unique key
    display: str             # shown to user in multiselect
    file_label: str          # user-facing label (no hash)
    colname: str             # original column name
    nice: str                # nice_label(colname)
    x: np.ndarray
    y: np.ndarray


@st.cache_data(show_spinner=False)
def load_variants_from_bytes(
    file_bytes: bytes,
    file_tag: str,           # INTERNAL stable id (may include hash)
    file_label: str,         # USER-FACING label (no hash)
    original_filename: str,
    sheet_data: str,
    sheet_sizes: str,
    min_pairs: int = 5
) -> Tuple[List[Variant], Optional[str]]:
    """
    Returns (variants, error_message). Never raises, so the UI won't crash on one bad file.
    """
    try:
        buf = io.BytesIO(file_bytes)
        df_data = pd.read_excel(buf, sheet_name=sheet_data, engine="openpyxl")
        buf.seek(0)
        df_sizes = pd.read_excel(buf, sheet_name=sheet_sizes, engine="openpyxl")
    except Exception as e:
        return [], f"{original_filename}: failed to read sheets '{sheet_data}'/'{sheet_sizes}' ({e})"

    common_cols = [c for c in df_data.columns if c in df_sizes.columns]
    if not common_cols:
        return [], f"{original_filename}: no common columns between '{sheet_data}' and '{sheet_sizes}'"

    n = min(len(df_data), len(df_sizes))
    out: List[Variant] = []

    for col in common_cols:
        x = pd.to_numeric(df_sizes[col].iloc[:n], errors="coerce").to_numpy()
        y = pd.to_numeric(df_data[col].iloc[:n], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < min_pairs:
            continue

        nice = nice_label(col)

        variant_id = f"{file_tag}::{col}"
        display = f"{file_label} — {nice}   (col: {col})"
        out.append(Variant(variant_id, display, file_label, str(col), nice, x, y))

    if not out:
        return [], f"{original_filename}: no usable variants (need >= {min_pairs} finite (x,y) pairs per column)"

    return out, None


def make_plot(
    variants: List[Variant],
    probs: Sequence[float],
    fill_alphas: Optional[Sequence[float]],
    no_points: bool,
    point_alpha: float,
    point_size: float,
    bw_adjust: float,
    gridsize: int,
    view_xlim: Optional[Tuple[float, float]],
    view_ylim: Optional[Tuple[float, float]],
    show_legends: bool = True,
    figsize: Tuple[float, float] = (8.4, 6.6),
):
    variants_sorted = sorted(variants, key=lambda v: (sort_key(v.nice), v.file_label, v.nice))

    allx = np.concatenate([v.x for v in variants_sorted])
    ally = np.concatenate([v.y for v in variants_sorted])

    xmin, xmax = float(allx.min()), float(allx.max())
    ymin, ymax = float(ally.min()), float(ally.max())

    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    pad = 0.08
    base_xlim = (xmin - pad * dx, xmax + pad * dx)
    base_ylim = (ymin - pad * dy, ymax + pad * dy)

    view_xlim = view_xlim if view_xlim is not None else base_xlim
    view_ylim = view_ylim if view_ylim is not None else base_ylim

    base_xlim = (min(base_xlim[0], view_xlim[0]), max(base_xlim[1], view_xlim[1]))
    base_ylim = (min(base_ylim[0], view_ylim[0]), max(base_ylim[1], view_ylim[1]))

    gridsize = int(gridsize)
    xi = np.linspace(*base_xlim, gridsize)
    yi = np.linspace(*base_ylim, gridsize)
    Xg, Yg = np.meshgrid(xi, yi)
    grid_dx = float(xi[1] - xi[0])
    grid_dy = float(yi[1] - yi[0])

    probs_outer_to_inner = tuple(sorted(set(float(p) for p in probs), reverse=True))
    probs_inner_to_outer = tuple(sorted(probs_outer_to_inner))

    style_cycle = ["solid", "dashed", "dotted", "dashdot"]
    linestyle_for = {p: style_cycle[i % len(style_cycle)] for i, p in enumerate(probs_inner_to_outer)}

    nbands = len(probs_outer_to_inner)
    default_alphas = (0.15, 0.35, 0.55, 0.70, 0.85, 0.95)
    if fill_alphas is None:
        fill_alphas = default_alphas[:nbands]
    else:
        fill_alphas = tuple(float(a) for a in fill_alphas)
        if len(fill_alphas) != nbands:
            raise ValueError(f"Need exactly {nbands} fill alphas (got {len(fill_alphas)}).")

    alpha_for_prob = {p: fill_alphas[i] for i, p in enumerate(probs_outer_to_inner)}

    colors = distinct_colors(len(variants_sorted))
    group_colors = {v.variant_id: colors[i] for i, v in enumerate(variants_sorted)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*view_xlim)
    ax.set_ylim(*view_ylim)
    ax.grid(True, alpha=0.20)
    ax.set_xlabel("Size")
    ax.set_ylabel("Twt")

    if not no_points:
        for v in variants_sorted:
            ax.scatter(
                v.x, v.y,
                s=float(point_size),
                alpha=float(point_alpha),
                edgecolors="none",
                color=group_colors[v.variant_id],
                zorder=1,
            )

    for v in variants_sorted:
        color = group_colors[v.variant_id]
        try:
            kde = gaussian_kde(np.vstack([v.x, v.y]))
            if bw_adjust != 1.0:
                kde.set_bandwidth(bw_method=kde.factor * float(bw_adjust))

            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

            prob_level = hdr_prob_to_level(Z, grid_dx, grid_dy, probs=probs_outer_to_inner)
            if len(prob_level) < 2:
                continue

            levels = [prob_level[p] for p in probs_outer_to_inner]
            if not all(np.isfinite(levels)):
                continue

            zmax = float(np.nanmax(Z))
            if not np.isfinite(zmax):
                continue

            levels_sorted = np.array(levels, dtype=float)
            do_fill = not np.any(np.diff(levels_sorted) <= 0)

            if do_fill:
                fill_levels = list(levels_sorted) + [zmax * 1.0001]
                rgba_colors = [to_rgba(color, alpha=a) for a in fill_alphas]
                ax.contourf(
                    Xg, Yg, Z,
                    levels=fill_levels,
                    colors=rgba_colors,
                    antialiased=True,
                    zorder=2,
                )

            for p in probs_outer_to_inner:
                lvl = prob_level[p]
                if not np.isfinite(lvl):
                    continue
                ax.contour(
                    Xg, Yg, Z,
                    levels=[lvl],
                    colors=[color],
                    linewidths=0.01,
                    linestyles=linestyle_for.get(p, "solid"),
                    zorder=3,
                )
        except Exception:
            continue

    if show_legends:
        group_handles = [
            Line2D([0], [0], color=group_colors[v.variant_id], lw=3, label=f"{v.file_label} — {v.nice}")
            for v in variants_sorted
        ]
        leg_groups = ax.legend(handles=group_handles, title="Variants", loc="upper right", frameon=True)
        ax.add_artist(leg_groups)

        style_handles = [
            Line2D([0], [0], color="black", lw=6, alpha=alpha_for_prob[p], label=f"HDR {int(round(p*100))}%")
            for p in probs_inner_to_outer
        ]
        ax.legend(handles=style_handles, title="Region", loc="lower right", frameon=True)

    fig.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str, dpi: int = 900) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format=fmt, dpi=dpi, bbox_inches="tight")
    return bio.getvalue()


# ---------------------------
# Built-in example files
# ---------------------------

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"

SAMPLE_FILES = {
    "Pruning": "pruning.xlsx",
    "Parsimony": "parsimony.xlsx",
    "Pruning + Parsimony": "pruning+parsimony.xlsx",
}

def resolve_sample_path(filename: str) -> Optional[Path]:
    for d in (EXAMPLES_DIR, APP_DIR):
        p = d / filename
        if p.exists() and p.is_file():
            return p
    return None

@st.cache_data(show_spinner=False)
def read_local_bytes(path_str: str) -> bytes:
    with open(path_str, "rb") as f:
        return f.read()


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="KDE HDR Variant Selector", layout="centered")
st.title("KDE + HDR plot: select variants across multiple .xlsx files")

with st.sidebar:
    st.header("Inputs")
    sheet_data = st.text_input("Sheet for y-values", value="Data")
    sheet_sizes = st.text_input("Sheet for x-values", value="Sizes")
    min_pairs = st.number_input("Min (x,y) pairs per variant", min_value=3, max_value=1000, value=5, step=1)

    st.header("KDE / HDR")
    probs_text = st.text_input("HDR probabilities (comma/space separated)", value="0.50 0.80 0.95")
    gridsize = st.slider("Grid size", min_value=60, max_value=400, value=180, step=10)
    bw_adjust = st.slider("Bandwidth adjust", min_value=0.25, max_value=3.0, value=1.0, step=0.05)

    st.header("Fill transparency (optional)")
    st.caption("Provide one alpha per HDR band (outer→inner), e.g. for 3 probs: `0.15 0.35 0.6`")
    alphas_text = st.text_input("Fill alphas", value="")  # empty => defaults

    st.header("Points")
    no_points = st.checkbox("Hide raw points", value=False)
    point_alpha = st.slider("Point alpha", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    point_size = st.slider("Point size", min_value=1.0, max_value=80.0, value=10.0, step=1.0)

    st.header("Legend")
    show_legends = st.checkbox("Show legends", value=True)

    st.header("View (optional)")
    xrange_text = st.text_input("x-range (e.g. 0,10)", value="")
    yrange_text = st.text_input("y-range (e.g. 0,1)", value="")

    st.header("Export")
    export_dpi = st.select_slider("Export DPI", options=[150, 300, 600, 900], value=300)

# --- Data source chooser ---
data_source = st.radio(
    "Data source",
    ["Use built-in files (recommended)", "Upload your own files"],
    index=0,
)

input_files: List[Dict[str, object]] = []

if data_source == "Upload your own files":
    uploaded = st.file_uploader(
        "Upload one or more .xlsx files (each must have Sheets: Sizes + Data, with matching columns)",
        type=["xlsx"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("Upload files, or switch to 'Use built-in files'.")
        st.stop()

    # create nice labels without hashes, numbering duplicates
    bases = [os.path.basename(uf.name) for uf in uploaded]
    base_total: Dict[str, int] = {}
    for b in bases:
        base_total[b] = base_total.get(b, 0) + 1
    base_seen: Dict[str, int] = {}

    for uf in uploaded:
        file_bytes = uf.getvalue()
        base = os.path.basename(uf.name)
        h = hashlib.md5(file_bytes).hexdigest()[:8]

        base_seen[base] = base_seen.get(base, 0) + 1
        k = base_seen[base]
        file_label = base if base_total[base] == 1 else f"{base} ({k})"

        input_files.append({
            "base": base,
            "hash": h,
            "bytes": file_bytes,
            "original": uf.name,
            "label": file_label,
        })

else:
    picked = st.multiselect(
        "Choose built-in files",
        options=list(SAMPLE_FILES.keys()),
        default=list(SAMPLE_FILES.keys()),
    )
    if not picked:
        st.warning("Pick at least one built-in file (or switch to uploads).")
        st.stop()

    missing: List[str] = []
    resolved: List[Tuple[str, Path]] = []
    for label in picked:
        fn = SAMPLE_FILES[label]
        p = resolve_sample_path(fn)
        if p is None:
            missing.append(fn)
        else:
            resolved.append((label, p))

    if missing:
        st.error(
            "Missing built-in files on disk:\n"
            f"- " + "\n- ".join(missing) + "\n\n"
            f"Expected them in:\n- {str(EXAMPLES_DIR)}\n- {str(APP_DIR)}"
        )
        st.stop()

    for label, path in resolved:
        file_bytes = read_local_bytes(str(path))
        base = os.path.basename(path.name)
        h = hashlib.md5(file_bytes).hexdigest()[:8]
        input_files.append({
            "base": base,
            "hash": h,
            "bytes": file_bytes,
            "original": str(path),
            "label": label,  # friendly label for UI/legend
        })

# Stable ordering for IDs across reruns
input_files.sort(key=lambda d: (str(d["base"]), str(d["hash"]), str(d["label"])))

# Load all variants across all input files (robust to bad files)
all_variants: List[Variant] = []
errors: List[str] = []

dup_counts_hash: Dict[Tuple[str, str], int] = {}
for item in input_files:
    base = str(item["base"])
    h = str(item["hash"])
    file_bytes = item["bytes"]  # type: ignore[assignment]
    original_filename = str(item["original"])
    file_label = str(item["label"])

    # INTERNAL tag: stable & unique (hash included)
    key = (base, h)
    dup_counts_hash[key] = dup_counts_hash.get(key, 0) + 1
    hsuffix = dup_counts_hash[key]
    file_tag = f"{base}#{h}" if hsuffix == 1 else f"{base}#{h}-{hsuffix}"

    vs, err = load_variants_from_bytes(
        file_bytes=file_bytes,     # type: ignore[arg-type]
        file_tag=file_tag,
        file_label=file_label,
        original_filename=original_filename,
        sheet_data=sheet_data,
        sheet_sizes=sheet_sizes,
        min_pairs=int(min_pairs),
    )
    all_variants.extend(vs)
    if err:
        errors.append(err)

if errors:
    with st.expander("Some files had issues (click to view)"):
        for e in errors:
            st.warning(e)

if not all_variants:
    st.error("No usable variants found in the selected files.")
    st.stop()

# Variant selection (fix "select twice" by using widget key and not fighting Streamlit state)
id_to_display = {v.variant_id: v.display for v in all_variants}
all_ids = list(id_to_display.keys())

SEL_KEY = "selected_ids"
if SEL_KEY not in st.session_state:
    st.session_state[SEL_KEY] = []

# Reset selection to NONE when available SET changes
available_sig = tuple(sorted(all_ids))
if st.session_state.get("available_sig") != available_sig:
    st.session_state["available_sig"] = available_sig
    st.session_state[SEL_KEY] = []

# Drop stale selections
st.session_state[SEL_KEY] = [i for i in st.session_state[SEL_KEY] if i in all_ids]

c1, c2, _ = st.columns([1, 1, 3])
with c1:
    if st.button("Select all"):
        st.session_state[SEL_KEY] = all_ids.copy()
        st.rerun()
with c2:
    if st.button("Select none"):
        st.session_state[SEL_KEY] = []
        st.rerun()

st.multiselect(
    "Select variants to include in the plot",
    options=all_ids,
    key=SEL_KEY,
    format_func=lambda k: id_to_display.get(k, k),
)

selected_ids = st.session_state[SEL_KEY]
selected = [v for v in all_variants if v.variant_id in set(selected_ids)]

# Parse settings
try:
    probs = parse_probs(probs_text)
    view_xlim = parse_range(xrange_text)
    view_ylim = parse_range(yrange_text)

    probs_outer_to_inner = tuple(sorted(set(float(p) for p in probs), reverse=True))
    fill_alphas = parse_alphas(alphas_text, nbands=len(probs_outer_to_inner))
except Exception as e:
    st.error(str(e))
    st.stop()

st.write(f"**Found:** {len(all_variants)} variants across {len(input_files)} input files.")
st.write(f"**Selected:** {len(selected)} variants.")

if len(selected) == 0:
    st.warning("Select at least one variant.")
    st.stop()

# Plot
fig = make_plot(
    variants=selected,
    probs=probs,
    fill_alphas=fill_alphas,
    no_points=no_points,
    point_alpha=float(point_alpha),
    point_size=float(point_size),
    bw_adjust=float(bw_adjust),
    gridsize=int(gridsize),
    view_xlim=view_xlim,
    view_ylim=view_ylim,
    show_legends=show_legends,
)

st.pyplot(fig)

# Download buttons
png_bytes = fig_to_bytes(fig, fmt="png", dpi=int(export_dpi))
pdf_bytes = fig_to_bytes(fig, fmt="pdf", dpi=int(export_dpi))

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download PNG", data=png_bytes, file_name="kde_hdr.png", mime="image/png")
with col2:
    st.download_button("Download PDF", data=pdf_bytes, file_name="kde_hdr.pdf", mime="application/pdf")

plt.close(fig)
