#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCoA (classical MDS) from a precomputed distance matrix.

Input
-----
A square table (TSV/CSV/whitespace-delimited) with samples as both
rows and columns. The first row is a header with sample IDs; the first
column contains the row IDs (same order as columns). Diagonal should
be zero; matrix should be symmetric.

Output
------
- <out_prefix>.coordinates.tsv        # sample scores for components
- <out_prefix>.eigenvalues.tsv        # eigenvalues and explained variance
- <out_prefix>.plot.png               # 2D scatter (PCoA1 vs PCoA2 by default)

Optional
--------
- --metadata: 2+ column table; must contain a column matching sample IDs
  (default column name 'sample'). A --color-by column can be used to color points.

Usage
-----
python pcoa_from_distance.py \
  --dist distances.tsv \
  --out-prefix results/pcoa/ehi \
  --n-components 3 \
  --metadata meta.tsv --sample-col sample --color-by lineage
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def read_distance_matrix(path: str) -> pd.DataFrame:
    """Load a square distance matrix with sample IDs both as columns and index."""
    # Try tab, then CSV, then whitespace
    for sep, kw in [("\t", {}), (",", {}), (r"\s+", {"engine": "python"})]:
        try:
            df = pd.read_csv(path, sep=sep, index_col=0, **kw)
            # Heuristic: if the first column name accidentally carried over (e.g., 'le'),
            # ensure columns and index overlap > 0; otherwise retry with next sep.
            if len(set(df.index) & set(df.columns)) > max(1, len(df) // 2):
                break
        except Exception:
            df = None
    if df is None or df.empty:
        raise ValueError(f"Could not read distance matrix from: {path}")

    # Ensure ordering of columns matches index; if not, try to reindex
    if not df.columns.equals(df.index):
        # Keep only the intersecting labels and align
        common = [s for s in df.index if s in df.columns]
        if len(common) < 2:
            raise ValueError("Row/column labels do not match sufficiently to form a square matrix.")
        df = df.loc[common, common]

    # Basic sanity checks
    if (df.shape[0] != df.shape[1]):
        raise ValueError("Distance matrix is not square.")
    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Non-numeric values detected in distance matrix.")
    # Symmetry check (allow small numerical noise)
    if not np.allclose(df.values, df.values.T, atol=1e-10):
        max_asym = np.abs(df.values - df.values.T).max()
        raise ValueError(f"Distance matrix is not symmetric (max asymmetry={max_asym:.3g}).")
    # Zero diagonal
    if not np.allclose(np.diag(df.values), 0.0, atol=1e-12):
        # Zero it out but warn
        sys.stderr.write("Warning: diagonal not all zeros; setting diagonal to 0.\n")
        v = df.values.copy()
        np.fill_diagonal(v, 0.0)
        df = pd.DataFrame(v, index=df.index, columns=df.columns)

    return df

def classical_pcoa(D: np.ndarray, n_components: int = 2):
    """
    Classical MDS / PCoA from a distance matrix D.

    Returns
    -------
    coords : (n_samples, n_components) array
    eigvals : (n_samples,) array of eigenvalues (descending)
    explained : (n_samples,) array of explained variance ratios over positive eigvals
    """
    # Double-centering: B = -0.5 * J * D^2 * J
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * J.dot(D2).dot(J)

    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigenvalues for variance accounting
    pos = eigvals > 0
    pos_eigvals = eigvals[pos]
    total_pos = pos_eigvals.sum() if pos_eigvals.size else 1.0
    explained = np.zeros_like(eigvals)
    if pos_eigvals.size:
        explained[pos] = pos_eigvals / total_pos

    # Coordinates: scale eigenvectors by sqrt of eigenvalues (for positive ones)
    Lhalf = np.sqrt(np.maximum(eigvals, 0))
    coords_full = eigvecs * Lhalf

    # Select components
    k = min(n_components, coords_full.shape[1])
    coords = coords_full[:, :k]
    return coords, eigvals, explained

def load_metadata(path, sample_col="sample"):
    meta = pd.read_csv(path, sep=None, engine="python")
    if sample_col not in meta.columns:
        raise ValueError(f"Metadata missing required sample column '{sample_col}'.")
    return meta

def plot_scatter(scores: pd.DataFrame,
                 explained: np.ndarray,
                 out_path: Path,
                 color_by=None,
                 meta: pd.DataFrame | None = None,
                 title="PCoA"):
    xlab = f"PCoA1 ({explained[0]*100:.1f}% var)"
    ylab = f"PCoA2 ({explained[1]*100:.1f}% var)" if len(explained) > 1 else "PCoA2"
    fig, ax = plt.subplots(figsize=(6, 5))

    if meta is not None and color_by and color_by in meta.columns:
        m = meta.set_index(meta.columns[0])  # typically sample ID column
        # align to scores index
        m = m.reindex(scores.index)
        groups = m[color_by].astype(str).fillna("NA")
        for g in sorted(groups.unique()):
            mask = (groups == g).values
            ax.scatter(scores.iloc[mask, 0], scores.iloc[mask, 1], label=g, s=40, alpha=0.9)
        ax.legend(title=color_by, fontsize=8, frameon=True)
    else:
        ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], s=40, alpha=0.9)

    for name, (x, y) in scores.iloc[:, :2].iterrows():
        ax.text(x, y, f" {name}", fontsize=7, va="center")

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.axhline(0, lw=0.5, alpha=0.5)
    ax.axvline(0, lw=0.5, alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="PCoA (classical MDS) from a distance matrix.")
    ap.add_argument("--dist", required=True, help="Path to square distance matrix (TSV/CSV/space).")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (TSVs + plot).")
    ap.add_argument("--n-components", type=int, default=2, help="Number of components to compute.")
    ap.add_argument("--metadata", help="Optional metadata file (TSV/CSV).")
    ap.add_argument("--sample-col", default="sample", help="Column in metadata with sample IDs.")
    ap.add_argument("--color-by", help="Metadata column to color points by.")
    ap.add_argument("--title", default="PCoA from distance matrix", help="Plot title.")
    ap.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Plot format.")
    args = ap.parse_args()

    dist_df = read_distance_matrix(args.dist)
    D = dist_df.values.astype(float)

    coords, eigvals, explained = classical_pcoa(D, n_components=args.n_components)

    # Write coordinates
    comp_cols = [f"PCoA{i+1}" for i in range(coords.shape[1])]
    scores_df = pd.DataFrame(coords, index=dist_df.index, columns=comp_cols)
    coord_path = Path(f"{args.out_prefix}_coordinates.tsv")
    scores_df.to_csv(coord_path, sep="\t", index=True)

    # Write eigenvalues + explained variance
    ev_df = pd.DataFrame({
        "eigenvalue": eigvals,
        "explained_variance_ratio": explained
    })
    ev_path = Path(f"{args.out_prefix}_eigenvalues.tsv")
    ev_df.to_csv(ev_path, sep="\t", index=False)

    # Plot (first two components if available)
    plot_path = Path(f"{args.out_prefix}.{args.plot_format}")
    meta = None
    if args.metadata:
        meta = load_metadata(args.metadata, sample_col=args.sample_col)
        # Ensure the first column is the sample ID for simple alignment
        if meta.columns[0] != args.sample_col:
            # Move sample_col to be first
            cols = [args.sample_col] + [c for c in meta.columns if c != args.sample_col]
            meta = meta[cols]
    # Use explained over positive eigenvalues for labels
    plot_scatter(scores_df, explained, plot_path, args.color_by, meta, title=args.title)

    # Console summary (useful in logs)
    tot = (eigvals[eigvals > 0].sum() if np.any(eigvals > 0) else 0.0)
    exp1 = explained[0]*100 if explained.size else 0.0
    exp2 = explained[1]*100 if explained.size > 1 else 0.0
    print(f"PCoA complete. Positive-eigenvalue variance sum: {tot:.6g}. "
          f"PCoA1={exp1:.1f}%, PCoA2={exp2:.1f}%")
    print(f"Wrote: {coord_path}, {ev_path}, {plot_path}")

if __name__ == "__main__":
    main()

