"""
Visualization functions for batch correction evaluation on MALDI MSI data.
"""
import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _subsample_for_plot(
    data: np.ndarray,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    sample_size: int = 10000,
    random_state: int = 42,
):
    """Subsample data for plotting."""
    n = len(data)
    if n <= sample_size:
        return data, batch_labels, tissue_labels
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    return data[idx], batch_labels[idx], tissue_labels[idx]


def plot_pca_before_after(
    raw_data: np.ndarray,
    corrected_data: np.ndarray,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    method_name: str,
    output_dir: str,
    sample_size: int = 10000,
) -> str:
    """
    Plot 2x2 PCA: colored by batch (before/after) and tissue (before/after).

    Parameters
    ----------
    raw_data : np.ndarray
        Raw feature matrix.
    corrected_data : np.ndarray
        Batch-corrected feature matrix.
    batch_labels : np.ndarray
        Batch labels.
    tissue_labels : np.ndarray
        Tissue type labels.
    method_name : str
        Name of correction method for plot title.
    output_dir : str
        Directory to save figure.
    sample_size : int, optional
        Number of samples to use for PCA plot (default 10000).

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_s, bl_s, tl_s = _subsample_for_plot(raw_data, batch_labels, tissue_labels, sample_size)
    cor_s, bl_sc, tl_sc = _subsample_for_plot(corrected_data, batch_labels, tissue_labels, sample_size)

    pca = PCA(n_components=2, random_state=42)
    raw_pca = pca.fit_transform(raw_s)
    pca2 = PCA(n_components=2, random_state=42)
    cor_pca = pca2.fit_transform(cor_s)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"PCA Before/After {method_name}", fontsize=14)

    # Row 0: batch coloring
    for ax, pca_data, labels, title in [
        (axes[0, 0], raw_pca, bl_s, "Before — Batch"),
        (axes[0, 1], cor_pca, bl_sc, f"After ({method_name}) — Batch"),
    ]:
        sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="tab10", s=3, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="Batch")

    # Row 1: tissue coloring
    for ax, pca_data, labels, title in [
        (axes[1, 0], raw_pca, tl_s, "Before — Tissue"),
        (axes[1, 1], cor_pca, tl_sc, f"After ({method_name}) — Tissue"),
    ]:
        sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="Set1", s=3, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="Tissue")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"pca_before_after_{method_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PCA plot saved to {save_path}")
    return save_path


def plot_umap_before_after(
    raw_data: np.ndarray,
    corrected_data: np.ndarray,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    method_name: str,
    output_dir: str,
    sample_size: int = 10000,
) -> str:
    """
    Plot 2x2 UMAP: colored by batch (before/after) and tissue (before/after).

    Parameters
    ----------
    raw_data : np.ndarray
        Raw feature matrix.
    corrected_data : np.ndarray
        Batch-corrected feature matrix.
    batch_labels : np.ndarray
        Batch labels.
    tissue_labels : np.ndarray
        Tissue type labels.
    method_name : str
        Name of correction method for plot title.
    output_dir : str
        Directory to save figure.
    sample_size : int, optional
        Number of samples to use for UMAP (default 10000).

    Returns
    -------
    str
        Path to saved figure.
    """
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        raise

    os.makedirs(output_dir, exist_ok=True)

    raw_s, bl_s, tl_s = _subsample_for_plot(raw_data, batch_labels, tissue_labels, sample_size)
    cor_s, bl_sc, tl_sc = _subsample_for_plot(corrected_data, batch_labels, tissue_labels, sample_size)

    reducer = umap.UMAP(n_components=2, random_state=42)
    raw_emb = reducer.fit_transform(raw_s)
    reducer2 = umap.UMAP(n_components=2, random_state=42)
    cor_emb = reducer2.fit_transform(cor_s)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"UMAP Before/After {method_name}", fontsize=14)

    for ax, emb, labels, is_batch, title in [
        (axes[0, 0], raw_emb, bl_s, True, "Before — Batch"),
        (axes[0, 1], cor_emb, bl_sc, True, f"After ({method_name}) — Batch"),
        (axes[1, 0], raw_emb, tl_s, False, "Before — Tissue"),
        (axes[1, 1], cor_emb, tl_sc, False, f"After ({method_name}) — Tissue"),
    ]:
        cmap = "tab10" if is_batch else "Set1"
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap, s=3, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"umap_before_after_{method_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"UMAP plot saved to {save_path}")
    return save_path


def plot_metrics_heatmap(results_df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot heatmap of all metrics × all method+normalization combinations.

    Direction-aware coloring: green = good, red = bad.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation.
    output_dir : str
        Directory to save figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Batch metrics: lower = better (invert for heatmap: green = low)
    batch_metrics = ["ami_batch", "ari_batch", "silhouette_batch", "pcr", "kbet"]
    # Bio metrics: higher = better
    bio_metrics = ["batch_entropy", "ami_tissue", "ari_tissue", "silhouette_tissue",
                   "knn_accuracy", "mcc", "macro_f1", "bio_batch_f1"]

    metric_cols = [c for c in results_df.columns if c not in ("method", "normalization")]
    if not metric_cols:
        logger.warning("No metric columns found in results_df.")
        return ""

    results_df = results_df.copy()
    results_df["combo"] = results_df["method"] + " / " + results_df["normalization"]
    pivot = results_df.set_index("combo")[metric_cols]

    # Normalize each column to [0,1] for display; handle constant columns (zero range)
    col_range = pivot.max() - pivot.min()
    col_range[col_range < 1e-10] = 1.0  # avoid division by near-zero
    pivot_norm = (pivot - pivot.min()) / col_range

    # For batch metrics: invert so that green = low (good removal)
    for col in batch_metrics:
        if col in pivot_norm.columns:
            pivot_norm[col] = 1.0 - pivot_norm[col]

    fig, ax = plt.subplots(figsize=(max(10, len(metric_cols) * 1.2), max(6, len(pivot_norm) * 0.5)))
    sns.heatmap(
        pivot_norm,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Score (green=good)"},
    )
    ax.set_title("Batch Correction Metrics Heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Method / Normalization")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "metrics_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {save_path}")
    return save_path


def plot_radar_chart(results_df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot radar/spider chart comparing all methods across key metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation.
    output_dir : str
        Directory to save figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    key_metrics = [c for c in ["ami_batch", "silhouette_batch", "kbet", "knn_accuracy",
                                "silhouette_tissue", "bio_batch_f1"] if c in results_df.columns]
    if len(key_metrics) < 3:
        logger.warning("Not enough metrics for radar chart.")
        return ""

    # Average per method
    method_avg = results_df.groupby("method")[key_metrics].mean()

    # Normalize to [0,1]
    norm = (method_avg - method_avg.min()) / (method_avg.max() - method_avg.min() + 1e-8)
    # Invert batch metrics
    for col in ["ami_batch", "silhouette_batch", "kbet"]:
        if col in norm.columns:
            norm[col] = 1.0 - norm[col]

    N = len(key_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(norm)))
    for (method, row), color in zip(norm.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=method, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(key_metrics, size=10)
    ax.set_title("Radar Chart: Method Comparison", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    save_path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Radar chart saved to {save_path}")
    return save_path


def plot_batch_distribution(
    data: np.ndarray,
    batch_labels: np.ndarray,
    title: str,
    output_dir: str,
) -> str:
    """
    Boxplot of per-batch mean intensity distributions.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray
        Batch labels.
    title : str
        Plot title.
    output_dir : str
        Directory to save figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_batches = np.unique(batch_labels)
    batch_means = [data[batch_labels == b].mean(axis=1) for b in unique_batches]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(batch_means, labels=[f"Batch {b}" for b in unique_batches])
    ax.set_title(title)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mean Intensity")
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"batch_distribution_{safe_title}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Batch distribution plot saved to {save_path}")
    return save_path


def plot_metric_comparison_boxplot(
    results_df: pd.DataFrame,
    metric: str,
    output_dir: str,
) -> str:
    """
    Boxplot comparing a single metric across all methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation.
    metric : str
        Name of the metric column to plot.
    output_dir : str
        Directory to save figure.

    Returns
    -------
    str
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = results_df["method"].unique()
    data_by_method = [results_df[results_df["method"] == m][metric].dropna().values for m in methods]
    ax.boxplot(data_by_method, labels=methods)
    ax.set_title(f"{metric} Comparison Across Methods")
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"boxplot_{metric}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Metric boxplot saved to {save_path}")
    return save_path
