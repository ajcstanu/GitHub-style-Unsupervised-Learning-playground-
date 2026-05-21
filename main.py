"""
=============================================================
  FULL UNSUPERVISED LEARNING SUITE
  Algorithms : K-Means · DBSCAN · PCA · Hierarchical
  Datasets   : Synthetic · CSV · sklearn built-ins
=============================================================
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – swap to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import (load_iris, load_digits, load_wine,
                               load_breast_cancer, make_blobs,
                               make_moons, make_circles)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR / STYLE
# ──────────────────────────────────────────────────────────────────────────────
PALETTE   = ["#00d4aa","#ff6b6b","#ffd93d","#a78bfa",
             "#38bdf8","#fb923c","#4ade80","#f472b6","#e879f9","#94a3b8"]
NOISE_CLR = "#555555"
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_CLR  = "#e6edf3"
GRID_CLR  = "#21262d"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_CLR,
        "axes.labelcolor":   TEXT_CLR,
        "axes.titlecolor":   TEXT_CLR,
        "xtick.color":       TEXT_CLR,
        "ytick.color":       TEXT_CLR,
        "text.color":        TEXT_CLR,
        "grid.color":        GRID_CLR,
        "grid.linewidth":    0.5,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_CLR,
        "font.family":       "monospace",
    })

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_builtin(name: str):
    """Return (X_scaled, feature_names, dataset_title)."""
    loaders = {
        "iris":           load_iris,
        "digits":         load_digits,
        "wine":           load_wine,
        "breast_cancer":  load_breast_cancer,
    }
    if name not in loaders:
        raise ValueError(f"Unknown built-in: {name}. Choose from {list(loaders)}")
    bunch  = loaders[name]()
    X      = StandardScaler().fit_transform(bunch.data)
    feats  = list(bunch.feature_names) if hasattr(bunch, "feature_names") else \
             [f"f{i}" for i in range(X.shape[1])]
    return X, feats, name.capitalize()


def load_synthetic(kind: str = "blobs", n: int = 400):
    """Return (X_scaled, feature_names, title)."""
    gen = {
        "blobs":   lambda: make_blobs(n_samples=n, centers=5,
                                      cluster_std=0.9, random_state=42),
        "moons":   lambda: make_moons(n_samples=n, noise=0.08, random_state=42),
        "circles": lambda: make_circles(n_samples=n, noise=0.06,
                                        factor=0.5, random_state=42),
    }
    if kind not in gen:
        raise ValueError(f"Unknown synthetic: {kind}. Choose blobs/moons/circles")
    X, _ = gen[kind]()
    X    = StandardScaler().fit_transform(X)
    return X, ["x", "y"], f"Synthetic – {kind}"


def load_csv(path: str, target_col: str = None):
    """Return (X_scaled, feature_names, title). Drops target column if given."""
    df = pd.read_csv(path)
    print(f"[CSV] Loaded {df.shape[0]} rows × {df.shape[1]} cols from '{path}'")
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    num = df.select_dtypes(include=[np.number])
    num = num.dropna()
    X   = StandardScaler().fit_transform(num.values)
    return X, list(num.columns), os.path.basename(path)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ALGORITHMS
# ──────────────────────────────────────────────────────────────────────────────

# ── 2a. K-Means ───────────────────────────────────────────────────────────────
def run_kmeans(X, k=4):
    model  = KMeans(n_clusters=k, init="k-means++",
                    n_init=20, max_iter=500, random_state=42)
    labels = model.fit_predict(X)
    return model, labels


def elbow_silhouette(X, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(m.inertia_)
        silhouettes.append(silhouette_score(X, m.labels_))
    return list(k_range), inertias, silhouettes


# ── 2b. DBSCAN ────────────────────────────────────────────────────────────────
def run_dbscan(X, eps=0.5, min_samples=5):
    model  = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(X)
    return model, labels


# ── 2c. PCA ───────────────────────────────────────────────────────────────────
def run_pca(X, n_components=None):
    n_max  = min(X.shape)
    nc     = n_components or n_max
    model  = PCA(n_components=nc, random_state=42)
    X_proj = model.fit_transform(X)
    return model, X_proj


# ── 2d. Hierarchical ──────────────────────────────────────────────────────────
def run_hierarchical(X, k=4, linkage_method="ward"):
    model  = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X)
    # scipy linkage matrix for dendrogram (subsample for big datasets)
    n      = min(len(X), 200)
    Z      = linkage(X[:n], method=linkage_method)
    return model, labels, Z


# ──────────────────────────────────────────────────────────────────────────────
# 3.  METRICS HELPER
# ──────────────────────────────────────────────────────────────────────────────
def cluster_metrics(X, labels):
    valid = labels[labels >= 0]
    n_cls = len(set(valid)) if len(set(valid)) > 1 else 0
    if n_cls < 2:
        return {"clusters": n_cls, "silhouette": None,
                "davies_bouldin": None, "calinski_harabasz": None}
    mask  = labels >= 0
    Xv, lv = X[mask], labels[mask]
    return {
        "clusters":           n_cls,
        "noise_points":       int((labels == -1).sum()),
        "silhouette":         round(silhouette_score(Xv, lv), 4),
        "davies_bouldin":     round(davies_bouldin_score(Xv, lv), 4),
        "calinski_harabasz":  round(calinski_harabasz_score(Xv, lv), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PCA 2-D PROJECTION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _pca2(X):
    """Always return 2-D PCA for scatter plots."""
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(X)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PLOTTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _scatter_clusters(ax, X2, labels, title, centroids=None):
    unique = sorted(set(labels))
    for lbl in unique:
        mask   = labels == lbl
        color  = NOISE_CLR if lbl == -1 else PALETTE[lbl % len(PALETTE)]
        marker = "x" if lbl == -1 else "o"
        alpha  = 0.4 if lbl == -1 else 0.80
        lab    = "Noise" if lbl == -1 else f"C{lbl}"
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=color, marker=marker, s=22, alpha=alpha, label=lab)
    if centroids is not None:
        c2 = PCA(n_components=2, random_state=42).fit(X2)  # already 2-D
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   s=220, marker="*", c="white", edgecolors="#ff6b6b",
                   linewidths=1.2, zorder=6, label="Centroids")
    ax.set_title(title, pad=8, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)


# ── Figure 1 : K-Means ────────────────────────────────────────────────────────
def plot_kmeans(X, title, k=4, out_dir="."):
    _style()
    X2             = _pca2(X)
    model, labels  = run_kmeans(X, k)
    ks, iner, sils = elbow_silhouette(X)

    fig = plt.figure(figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f"K-Means  ·  {title}", fontsize=14, fontweight="bold",
                 color=TEXT_CLR, y=1.01)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Scatter
    ax0 = fig.add_subplot(gs[0])
    _scatter_clusters(ax0, X2, labels, f"k={k}  (PCA 2-D projection)",
                      centroids=model.cluster_centers_ if X.shape[1]==2 else None)

    # Elbow
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(ks, iner, "o-", color="#00d4aa", lw=2, ms=7)
    ax1.axvline(k, color="#ffd93d", linestyle="--", lw=1.5, label=f"k={k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(ks, sils, "o-", color="#a78bfa", lw=2, ms=7)
    ax2.axvline(k, color="#ffd93d", linestyle="--", lw=1.5, label=f"k={k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "01_kmeans.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [saved] {path}")
    return labels


# ── Figure 2 : DBSCAN ─────────────────────────────────────────────────────────
def plot_dbscan(X, title, eps=0.5, min_samples=5, out_dir="."):
    _style()
    X2            = _pca2(X)
    _, labels     = run_dbscan(X, eps, min_samples)
    eps_vals      = np.round(np.arange(0.2, 1.6, 0.1), 2)
    n_clusters_   = []
    noise_counts  = []
    for e in eps_vals:
        lbl = DBSCAN(eps=e, min_samples=min_samples).fit_predict(X)
        n_clusters_.append(len(set(lbl)) - (1 if -1 in lbl else 0))
        noise_counts.append((lbl == -1).sum())

    fig = plt.figure(figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f"DBSCAN  ·  {title}  ·  ε={eps}  minPts={min_samples}",
                 fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.01)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    _scatter_clusters(ax0, X2, labels, "Cluster Assignment (PCA 2-D)")

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(eps_vals, n_clusters_, "o-", color="#ff6b6b", lw=2, ms=7)
    ax1.axvline(eps, color="#ffd93d", linestyle="--", lw=1.5, label=f"ε={eps}")
    ax1.set_xlabel("ε"); ax1.set_ylabel("# Clusters")
    ax1.set_title("ε Sensitivity", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(eps_vals, noise_counts, "o-", color="#ffd93d", lw=2, ms=7)
    ax2.axvline(eps, color="#ff6b6b", linestyle="--", lw=1.5, label=f"ε={eps}")
    ax2.set_xlabel("ε"); ax2.set_ylabel("Noise Points")
    ax2.set_title("Noise vs ε", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "02_dbscan.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [saved] {path}")
    return labels


# ── Figure 3 : PCA ────────────────────────────────────────────────────────────
def plot_pca(X, feature_names, title, out_dir="."):
    _style()
    model, X_proj = run_pca(X)
    ev_ratio      = model.explained_variance_ratio_
    cumvar        = np.cumsum(ev_ratio)
    n_show        = min(20, len(ev_ratio))

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(f"PCA  ·  {title}", fontsize=14, fontweight="bold",
                 color=TEXT_CLR, y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)

    # 2-D scatter
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(X_proj[:, 0], X_proj[:, 1], c="#00d4aa", s=15, alpha=0.6)
    ax0.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}%)")
    ax0.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}%)")
    ax0.set_title("PC1 vs PC2 Projection", fontweight="bold"); ax0.grid(True, alpha=0.3)

    # Scree
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(range(1, n_show+1), ev_ratio[:n_show]*100,
            color=["#a78bfa" if i < 5 else "#30363d" for i in range(n_show)])
    ax1.set_xlabel("Component"); ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot", fontweight="bold"); ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative variance
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(range(1, len(cumvar)+1), cumvar*100, "o-", color="#38bdf8", lw=2, ms=5)
    ax2.axhline(95, color="#ffd93d", linestyle="--", lw=1.5, label="95%")
    ax2.axhline(99, color="#ff6b6b", linestyle="--", lw=1.5, label="99%")
    ax2.set_xlabel("Components"); ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Explained Variance", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Biplot (top 2 components, top 8 features)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.scatter(X_proj[:, 0], X_proj[:, 1], c="#00d4aa", s=10, alpha=0.4)
    loadings = model.components_[:2].T
    scale    = np.abs(X_proj).max() / np.abs(loadings).max()
    top_idx  = np.argsort(np.sqrt(loadings[:,0]**2 + loadings[:,1]**2))[-8:]
    for i in top_idx:
        lx, ly = loadings[i, 0]*scale*0.6, loadings[i, 1]*scale*0.6
        fname  = feature_names[i] if i < len(feature_names) else f"f{i}"
        ax3.annotate("", xy=(lx,ly), xytext=(0,0),
                     arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5))
        ax3.text(lx*1.08, ly*1.08, fname, color="#ffd93d", fontsize=8)
    ax3.set_title("Biplot  (top 8 features)", fontweight="bold"); ax3.grid(True, alpha=0.3)

    # Correlation heatmap of loadings
    ax4 = fig.add_subplot(gs[1, 2])
    n_feat   = min(10, len(feature_names))
    heat_data = np.abs(model.components_[:5, :n_feat])
    im = ax4.imshow(heat_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax4.set_xticks(range(n_feat))
    ax4.set_xticklabels(feature_names[:n_feat], rotation=45, ha="right", fontsize=7)
    ax4.set_yticks(range(5)); ax4.set_yticklabels([f"PC{i+1}" for i in range(5)], fontsize=8)
    ax4.set_title("|Loadings| Heatmap (PC1-5)", fontweight="bold")
    plt.colorbar(im, ax=ax4, fraction=0.04)

    path = os.path.join(out_dir, "03_pca.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Figure 4 : Hierarchical ───────────────────────────────────────────────────
def plot_hierarchical(X, title, k=4, out_dir="."):
    _style()
    X2 = _pca2(X)
    linkage_methods = ["ward", "complete", "average", "single"]

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.suptitle(f"Hierarchical Clustering  ·  {title}", fontsize=14,
                 fontweight="bold", color=TEXT_CLR, y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.4, hspace=0.45)

    # Ward scatter
    _, labels_ward, Z_ward = run_hierarchical(X, k=k, linkage_method="ward")
    ax0 = fig.add_subplot(gs[0, :2])
    _scatter_clusters(ax0, X2, labels_ward, f"Ward Linkage  k={k}  (PCA 2-D)")

    # Dendrogram
    ax1 = fig.add_subplot(gs[0, 2:])
    dendrogram(Z_ward, ax=ax1, truncate_mode="level", p=5,
               leaf_rotation=90, leaf_font_size=7,
               color_threshold=0.65*Z_ward[:,2].max(),
               above_threshold_color="#30363d")
    ax1.set_title("Dendrogram  (Ward, first 200 pts)", fontweight="bold")
    ax1.set_xlabel("Sample index"); ax1.set_ylabel("Distance")
    ax1.grid(True, alpha=0.3, axis="y")

    # 4 linkage comparisons
    for col, method in enumerate(linkage_methods):
        _, lbl, _ = run_hierarchical(X, k=k, linkage_method=method)
        ax = fig.add_subplot(gs[1, col])
        _scatter_clusters(ax, X2, lbl, method.capitalize())

    path = os.path.join(out_dir, "04_hierarchical.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [saved] {path}")
    return labels_ward


# ── Figure 5 : Metrics summary ────────────────────────────────────────────────
def plot_metrics(metrics_all: dict, out_dir="."):
    _style()
    names  = list(metrics_all.keys())
    scores = {
        "Silhouette ↑":         [m.get("silhouette")        or 0 for m in metrics_all.values()],
        "Calinski-Harabasz ↑":  [m.get("calinski_harabasz") or 0 for m in metrics_all.values()],
        "Davies-Bouldin ↓":     [m.get("davies_bouldin")    or 0 for m in metrics_all.values()],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK_BG)
    fig.suptitle("Clustering Metrics Comparison", fontsize=13,
                 fontweight="bold", color=TEXT_CLR)
    for ax, (metric, vals) in zip(axes, scores.items()):
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars   = ax.bar(names, vals, color=colors, edgecolor=GRID_CLR, linewidth=0.8)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("Score"); ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*max(vals or [1]),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT_CLR)
    plt.tight_layout()
    path = os.path.join(out_dir, "05_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_all(X, feature_names, title, k=4, eps=0.5, min_samples=5, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Dataset : {title}")
    print(f"  Shape   : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Output  : {out_dir}/")
    print(f"{'='*60}\n")

    metrics = {}

    print("► K-Means")
    km_labels = plot_kmeans(X, title, k=k, out_dir=out_dir)
    metrics["K-Means"] = cluster_metrics(X, km_labels)
    print(f"  Metrics: {metrics['K-Means']}")

    print("\n► DBSCAN")
    db_labels = plot_dbscan(X, title, eps=eps, min_samples=min_samples, out_dir=out_dir)
    metrics["DBSCAN"] = cluster_metrics(X, db_labels)
    print(f"  Metrics: {metrics['DBSCAN']}")

    print("\n► PCA")
    plot_pca(X, feature_names, title, out_dir=out_dir)
    _, X2d = run_pca(X, n_components=2)
    ev_ratio = PCA(n_components=2, random_state=42).fit(X).explained_variance_ratio_
    print(f"  PC1+PC2 explains {sum(ev_ratio)*100:.1f}% of variance")

    print("\n► Hierarchical")
    hc_labels = plot_hierarchical(X, title, k=k, out_dir=out_dir)
    metrics["Hierarchical"] = cluster_metrics(X, hc_labels)
    print(f"  Metrics: {metrics['Hierarchical']}")

    print("\n► Metrics Summary")
    plot_metrics(metrics, out_dir=out_dir)

    # ── print comparison table ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Algorithm':<18} {'Clusters':>9} {'Silhouette':>12} {'Davies-B':>10} {'Calinski':>12}")
    print(f"{'─'*65}")
    for algo, m in metrics.items():
        print(f"  {algo:<18} {str(m.get('clusters','–')):>9} "
              f"{str(m.get('silhouette','–')):>12} "
              f"{str(m.get('davies_bouldin','–')):>10} "
              f"{str(m.get('calinski_harabasz','–')):>12}")
    print(f"{'─'*65}\n")

    print(f"✓ All plots saved to '{out_dir}/'")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="Unsupervised Learning Suite  –  K-Means · DBSCAN · PCA · Hierarchical",
        formatter_class=argparse.RawTextHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--builtin",   choices=["iris","digits","wine","breast_cancer"],
                     help="Use a sklearn built-in dataset")
    src.add_argument("--synthetic", choices=["blobs","moons","circles"], default="blobs",
                     help="Use a synthetic dataset  (default: blobs)")
    src.add_argument("--csv",       metavar="PATH",
                     help="Path to a CSV file")

    p.add_argument("--target",      metavar="COL", default=None,
                   help="Column name to drop from CSV (target / label)")
    p.add_argument("--k",           type=int, default=4,   help="Clusters for K-Means / Hierarchical")
    p.add_argument("--eps",         type=float, default=0.5, help="DBSCAN eps")
    p.add_argument("--min_samples", type=int, default=5,   help="DBSCAN min_samples")
    p.add_argument("--out",         default="output",      help="Output directory for plots")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT – also exposes demo() for direct import
# ──────────────────────────────────────────────────────────────────────────────
def demo():
    """Run all algorithms on all dataset modes automatically."""
    datasets = [
        ("synthetic", "blobs"),
        ("synthetic", "moons"),
        ("synthetic", "circles"),
        ("builtin",   "iris"),
        ("builtin",   "wine"),
    ]
    for mode, name in datasets:
        out = f"output/{mode}_{name}"
        if mode == "synthetic":
            X, feats, title = load_synthetic(name)
            eps = 0.3
        else:
            X, feats, title = load_builtin(name)
            eps = 0.8
        run_all(X, feats, title, k=4, eps=eps, min_samples=5, out_dir=out)


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.csv:
        X, feats, title = load_csv(args.csv, args.target)
    elif args.builtin:
        X, feats, title = load_builtin(args.builtin)
    else:
        X, feats, title = load_synthetic(args.synthetic)

    run_all(X, feats, title,
            k=args.k, eps=args.eps, min_samples=args.min_samples,
            out_dir=args.out)
