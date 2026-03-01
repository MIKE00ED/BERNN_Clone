"""
Main evaluation pipeline for batch correction methods.

Evaluates pyCombat and NormAE on MALDI MSI data across all 6 normalizations.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.evaluation.batch_metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def evaluate_single_dataset(
    data: np.ndarray,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    method_name: str,
    normalization_name: str,
    sample_size: int = 50000,
) -> dict:
    """
    Evaluate a single batch-corrected dataset.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch labels.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.
    method_name : str
        Name of the correction method (e.g., 'raw', 'pycombat', 'normae').
    normalization_name : str
        Name of the normalization (e.g., 'TIC_max').
    sample_size : int, optional
        Subsampling size for expensive metrics (default 50000).

    Returns
    -------
    dict
        Dictionary with keys: method, normalization, and all metric values.
    """
    logger.info(f"Evaluating {method_name} / {normalization_name}")
    metrics = compute_all_metrics(data, batch_labels, tissue_labels, sample_size=sample_size)
    result = {"method": method_name, "normalization": normalization_name}
    result.update(metrics)
    return result


def run_full_evaluation(
    data_dict: Dict[str, Dict[str, pd.DataFrame]],
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    sample_size: int = 50000,
) -> pd.DataFrame:
    """
    Run full evaluation across all normalizations and correction methods.

    Parameters
    ----------
    data_dict : dict
        Nested dict: {normalization_name: {'raw': df, 'pycombat': df, 'normae': df}}.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch labels.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.
    sample_size : int, optional
        Subsampling size for expensive metrics (default 50000).

    Returns
    -------
    pd.DataFrame
        Results table with rows = method+normalization combos, cols = metrics.
    """
    from tqdm import tqdm

    rows = []
    for norm_name, method_dfs in tqdm(data_dict.items(), desc="Evaluating normalizations"):
        for method_name, df in method_dfs.items():
            try:
                row = evaluate_single_dataset(
                    df.values,
                    batch_labels,
                    tissue_labels,
                    method_name,
                    norm_name,
                    sample_size=sample_size,
                )
                rows.append(row)
            except Exception as e:
                logger.error(f"Evaluation failed for {norm_name}/{method_name}: {e}")

    return pd.DataFrame(rows)


def get_best_method(
    results_df: pd.DataFrame, primary_metric: str = "bio_batch_f1"
) -> pd.Series:
    """
    Return the row with the best combined score.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from run_full_evaluation.
    primary_metric : str, optional
        Column name of the primary metric (default 'bio_batch_f1').

    Returns
    -------
    pd.Series
        Row with the highest primary metric value.
    """
    if primary_metric not in results_df.columns:
        raise ValueError(f"Metric '{primary_metric}' not found in results DataFrame.")
    best_idx = results_df[primary_metric].idxmax()
    return results_df.loc[best_idx]


def print_results_table(results_df: pd.DataFrame) -> None:
    """
    Pretty-print the results table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from run_full_evaluation.
    """
    print(tabulate(results_df, headers="keys", tablefmt="grid", floatfmt=".4f"))


def save_results(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Save results DataFrame to CSV.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from run_full_evaluation.
    output_path : str
        Path to output CSV file.
    """
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
