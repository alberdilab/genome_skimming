#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustered heatmap (with row/column dendrograms) from a square distance matrix.

- Color scale defaults to [0, max off-diagonal distance].
- Sample names are shown on BOTH axes by default (use --no-ticklabels to hide).

Example:
python heatmap_hclust_figure.py \
  --dist data/distances.tsv \
  --out results/hclust/ehi.heatmap.png \
  --method average --optimal-ordering \
  --title "EHI distance – hclust"
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# -------- I/O --------

def read_distance_matrix(path: str) -> pd.DataFrame:
    """Load a square distance matrix with sample IDs both as columns and index."""
    df = None
    for sep, kw in [("\t", {}), (",", {}), (r"\s+", {"engine": "python"})]:
        try:
            tmp = pd.read_csv(path, sep=sep, index_col=0, **kw)
            # Heuristic: index/columns should largely overlap
            if len(set(tmp.index) & set(tmp.columns)) > max(1, len(tmp)//2):
                df = tmp
                break
        except Exception:
            pass
    if df is None or df.empty:
        raise ValueError(f"Could not read distance matrix from: {path}")

    # Align rows/cols to intersection (handles stray token like 'le' in top-left)
    if not df.columns.equals(df.index):
        common = [s for s in df.index if s in df.columns]
        if len(common) < 2:
            raise ValueError("Row/column labels do not match sufficiently to form a square matrix.")
        df = df.loc[common, common]

    if df.shape[0] != df.shape[1]:
        raise ValueError("Distance matrix is not square.")

    # Numeric + sanity checks
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Non-numeric values detected in distance matrix.")
    if not np.allclose(df.values, df.values.T, atol=1e-10):
        max_asym = np.abs(df.values - df.values.T).max()
        raise ValueError(f"Distance matrix is not symmetric (max asymmetry={max_asym:.3g}).")
    if (df.values < -1e-12).any():
        raise ValueError("Negative distances found; expected non-negative distances.")
    if not np.allclose(np.diag(df.values), 0.0, atol=1e-12):
        sys.stderr.write("Warning: diagonal not all zeros; setting diagonal to 0.\n")
        v = df.values.copy()
        np.fill_diagonal(v, 0.0)
        df = pd.DataFrame(v, index=df.index, columns=df.columns)

    return df

# -------- Plotting --------

def plot_clustered_heatmap(dist: pd.DataFrame,
                           Z_rows,
                           Z_cols,
                           out_path: Path,
                           cmap: str = "viridis",
                           vmin: float | None = None,
                           vmax: float | None = None,
                           title: str = "Hierarchical clustering heatmap",
                           show_ticks: bool = True):
    """Create dendrograms + heatmap (matplotlib only) and save to out_path."""
    # Compute leaf orders (without drawing)
    row_leaves = dendrogram(Z_rows, no_plot=True)["leaves"]
    col_leaves = dendrogram(Z_cols, no_plot=True)["leaves"]

    # Reorder matrix
    dist_re = dist.iloc[row_leaves, :].iloc[:, col_leaves]
    n = dist_re.shape[0]

    # Figure geometry scales with n but caps to avoid monster figures
    heat_size = min(max(4, 0.25 * n), 12)     # inches
    dend_size = min(max(1.5, 0.08 * n), 3.5)  # inches

    fig = plt.figure(figsize=(dend_size + heat_size + 0.8,
                              dend_size + heat_size + 0.6))
    gs = gridspec.GridSpec(nrows=2, ncols=2,
                           width_ratios=[dend_size, heat_size],
                           height_ratios=[dend_size, heat_size],
                           wspace=0.02, hspace=0.02)

    # Top dendrogram (columns)
    ax_col = fig.add_subplot(gs[0, 1])
    dendrogram(Z_cols, ax=ax_col, color_threshold=None, no_labels=True)
    ax_col.set_xticks([])
    ax_col.set_yticks([])

    # Left dendrogram (rows)
    ax_row = fig.add_subplot(gs[1, 0])
    dendrogram(Z_rows, ax=ax_row, orientation="right", color_threshold=None, no_labels=True)
    ax_row.set_xticks([])
    ax_row.set_yticks([])

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(dist_re.values, aspect="auto", origin="lower",
                        interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    if show_ticks:
        ax_heat.set_xticks(np.arange(n))
        ax_heat.set_xticklabels(dist_re.columns, rotation=90, fontsize=7)
        ax_heat.set_yticks(np.arange(n))
        ax_heat.set_yticklabels(dist_re.index, fontsize=7)
    else:
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

    ax_heat.set_title(title, fontsize=10, pad=6)

    # Robust colorbar (avoids axes_grid1 sizing issues)
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label("Distance", rotation=90)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

# -------- Main --------

def main():
    ap = argparse.ArgumentParser(description="Clustered heatmap (figure only) from a distance matrix.")
    ap.add_argument("--dist", required=True, help="Path to square distance matrix (TSV/CSV/space).")
    ap.add_argument("--out", required=True, help="Output figure path (.png/.pdf/.svg).")
    ap.add_argument("--method", default="average",
                    choices=["single", "complete", "average", "weighted",
                             "centroid", "median", "ward"],
                    help="Linkage method. Default: average (UPGMA).")
    ap.add_argument("--optimal-ordering", action="store_true",
                    help="Use optimal leaf ordering (prettier, a bit slower).")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    ap.add_argument("--vmin", type=float, help="Color scale minimum. Default: 0.")
    ap.add_argument("--vmax", type=float, help="Color scale maximum. Default: max off-diagonal.")
    ap.add_argument("--title", default="Hierarchical clustering heatmap")
    ap.add_argument("--no-ticklabels", action="store_true",
                    help="Hide tick labels (useful for very large matrices).")
    args = ap.parse_args()

    # Load distances
    dist_df = read_distance_matrix(args.dist)
    A = dist_df.values.astype(float)
    n = A.shape[0]

    # Default color scale: [0, max off-diagonal]
    vmin = 0.0 if args.vmin is None else float(args.vmin)
    if args.vmax is None:
        # exclude diagonal from max computation
        A_off = A.copy()
        np.fill_diagonal(A_off, np.nan)
        vmax = float(np.nanmax(A_off)) if np.isfinite(np.nanmax(A_off)) else 0.0
    else:
        vmax = float(args.vmax)
    if not np.isfinite(vmax):
        vmax = 0.0
    if vmax <= vmin:
        vmax = vmin + 1e-9  # avoid invalid range

    # Linkage for rows/columns (same because D is symmetric)
    condensed = squareform(A, checks=False)  # symmetry already validated
    Z = linkage(condensed, method=args.method, optimal_ordering=args.optimal_ordering)

    # Plot
    out_path = Path(args.out)
    plot_clustered_heatmap(
        dist=dist_df, Z_rows=Z, Z_cols=Z, out_path=out_path,
        cmap=args.cmap, vmin=vmin, vmax=vmax,
        title=args.title, show_ticks=not args.no_ticklabels
    )
    print(f"Wrote clustered heatmap: {out_path}")

if __name__ == "__main__":
    main()
