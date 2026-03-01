"""
Data loading and utility functions for MALDI MSI prostate cancer dataset.

Dataset summary:
- 114 prostate cancer patients
- 125 slides, 4 samples per slide
- 159 m/z features
- ~3,000,000 spectra (pixels)
- 4 tissue types
- 7 batches (unequal sizes: 30, 4, 12, 24, 32, 18, 5 slides)
- 6 normalization versions
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZATION_NAMES = ["TIC_max", "TIC_area", "RMS_area", "RMS_max", "Median_max", "Median_area"]

BATCH_INFO = {
    "batch_1": 30,
    "batch_2": 4,
    "batch_3": 12,
    "batch_4": 24,
    "batch_5": 32,
    "batch_6": 18,
    "batch_7": 5,
}


def load_maldi_data(filepath: str, file_format: str = "csv") -> pd.DataFrame:
    """
    Load MALDI MSI data from CSV or HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to data file.
    file_format : str, optional
        File format: 'csv' or 'hdf5' (default 'csv').

    Returns
    -------
    pd.DataFrame
        Feature matrix with samples as rows and m/z features as columns.
    """
    if file_format == "csv":
        logger.info(f"Loading CSV data from {filepath}")
        return pd.read_csv(filepath, index_col=0)
    elif file_format in ("hdf5", "h5"):
        logger.info(f"Loading HDF5 data from {filepath}")
        import h5py
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            logger.info(f"HDF5 keys: {keys}")
            data_key = keys[0]
            data = f[data_key][:]
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'csv' or 'hdf5'.")


def subsample_pixels(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    n_samples: int = 50000,
    stratify_by: str = "tissue",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Stratified subsampling from ~3M pixels for downstream analysis.

    Parameters
    ----------
    data : pd.DataFrame, shape (n_pixels, n_features)
        Full pixel feature matrix.
    batch_labels : np.ndarray, shape (n_pixels,)
        Batch labels.
    tissue_labels : np.ndarray, shape (n_pixels,)
        Tissue type labels.
    n_samples : int, optional
        Number of pixels to sample (default 50000).
    stratify_by : str, optional
        Stratification variable: 'tissue' or 'batch' (default 'tissue').
    random_state : int, optional
        Random seed (default 42).

    Returns
    -------
    tuple
        (sampled_data, sampled_batch_labels, sampled_tissue_labels)
    """
    n = len(data)
    if n <= n_samples:
        return data, batch_labels, tissue_labels

    if stratify_by == "tissue":
        strat_labels = tissue_labels
    else:
        strat_labels = batch_labels

    rng = np.random.RandomState(random_state)
    unique, counts = np.unique(strat_labels, return_counts=True)
    indices = []
    for cls, cnt in zip(unique, counts):
        cls_idx = np.where(strat_labels == cls)[0]
        n_take = max(2, int(n_samples * cnt / n))
        n_take = min(n_take, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        indices.extend(chosen.tolist())

    indices = np.array(indices[:n_samples])
    logger.info(f"Subsampled {len(indices)} pixels from {n} total.")
    return (
        data.iloc[indices].reset_index(drop=True),
        batch_labels[indices],
        tissue_labels[indices],
    )


def get_batch_info() -> Dict[str, int]:
    """
    Return slide counts per batch.

    Returns
    -------
    dict
        Mapping from batch name to number of slides.
        batch_1: 30, batch_2: 4, batch_3: 12, batch_4: 24,
        batch_5: 32, batch_6: 18, batch_7: 5
    """
    return BATCH_INFO.copy()


def get_normalization_names() -> List[str]:
    """
    Return list of normalization version names.

    Returns
    -------
    list of str
        ['TIC_max', 'TIC_area', 'RMS_area', 'RMS_max', 'Median_max', 'Median_area']
    """
    return NORMALIZATION_NAMES.copy()


def validate_data(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
) -> bool:
    """
    Validate shapes, labels, and data quality.

    Parameters
    ----------
    data : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch labels.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.

    Returns
    -------
    bool
        True if validation passes, raises ValueError otherwise.

    Raises
    ------
    ValueError
        If shapes don't match or data has critical quality issues.
    """
    n = len(data)
    if len(batch_labels) != n:
        raise ValueError(
            f"batch_labels length {len(batch_labels)} != data rows {n}"
        )
    if len(tissue_labels) != n:
        raise ValueError(
            f"tissue_labels length {len(tissue_labels)} != data rows {n}"
        )

    # Check for all-NaN features
    all_nan_cols = data.columns[data.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"{len(all_nan_cols)} features are all-NaN: {all_nan_cols[:5]}")

    # Check minimum samples per batch
    for b in np.unique(batch_labels):
        n_b = np.sum(batch_labels == b)
        if n_b < 2:
            logger.warning(f"Batch {b} has fewer than 2 samples ({n_b}).")

    logger.info(
        f"Validation passed: {n} samples, {data.shape[1]} features, "
        f"{len(np.unique(batch_labels))} batches, {len(np.unique(tissue_labels))} tissue types."
    )
    return True


def prepare_data_dict(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all 6 normalization versions from a directory.

    Expects files named: <normalization_name>.csv or <normalization_name>.h5

    Parameters
    ----------
    data_dir : str
        Path to directory containing normalization files.

    Returns
    -------
    dict
        Mapping from normalization name to DataFrame.
    """
    data_dict = {}
    for norm_name in NORMALIZATION_NAMES:
        for ext, fmt in [(".csv", "csv"), (".h5", "hdf5"), (".hdf5", "hdf5")]:
            fpath = os.path.join(data_dir, norm_name + ext)
            if os.path.exists(fpath):
                logger.info(f"Loading {norm_name} from {fpath}")
                data_dict[norm_name] = load_maldi_data(fpath, file_format=fmt)
                break
        else:
            logger.warning(f"No file found for normalization {norm_name} in {data_dir}")
    return data_dict
