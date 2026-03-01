"""
ComBat batch correction using pycombat package.

Implements parametric empirical Bayes batch correction for MALDI MSI data.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def apply_pycombat(
    data_df: pd.DataFrame,
    batch_labels: np.ndarray,
    covariate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply pyCombat batch correction to a feature matrix.

    Uses the parametric empirical Bayes framework (Johnson et al., 2007) to
    remove batch effects while preserving biological variation. Features with
    zero variance are dropped before correction and re-added afterward.

    Parameters
    ----------
    data_df : pd.DataFrame, shape (n_samples, n_features)
        Input feature matrix with samples as rows and features as columns.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment integer or string label for each sample.
    covariate_df : pd.DataFrame, optional
        Additional covariates to preserve during correction, shape (n_samples, n_covariates).

    Returns
    -------
    pd.DataFrame
        Batch-corrected DataFrame with the same shape as input.

    Notes
    -----
    pyCombat expects data as (n_features, n_samples), so transposition is handled
    internally. Features with zero variance across samples are excluded from
    correction and restored unchanged afterward.
    """
    try:
        from combat.pycombat import pycombat
    except ImportError:
        logger.error(
            "pycombat is not installed. Install with: pip install pycombat"
        )
        raise

    logger.info(f"Applying pyCombat to {data_df.shape[0]} samples, {data_df.shape[1]} features")

    # Handle NaN
    data_filled = data_df.fillna(0.0)

    # Identify zero-variance features (using small threshold to handle float precision)
    _VAR_THRESHOLD = 1e-10
    variances = data_filled.var(axis=0)
    zero_var_cols = variances[variances < _VAR_THRESHOLD].index.tolist()
    nonzero_var_cols = variances[variances >= _VAR_THRESHOLD].index.tolist()

    if len(nonzero_var_cols) == 0:
        logger.warning("All features have zero variance; returning original data.")
        return data_df.copy()

    if zero_var_cols:
        logger.warning(f"Dropping {len(zero_var_cols)} zero-variance features before ComBat.")

    data_nonzero = data_filled[nonzero_var_cols]

    # pycombat expects (n_features, n_samples)
    data_T = data_nonzero.T  # (n_features, n_samples)

    batch_series = pd.Series(batch_labels, index=data_T.columns)

    try:
        if covariate_df is not None:
            corrected_T = pycombat(data_T, batch_series, covariate_df)
        else:
            corrected_T = pycombat(data_T, batch_series)
    except Exception as e:
        logger.error(f"pyCombat failed: {e}")
        raise

    corrected = corrected_T.T  # back to (n_samples, n_features)

    # Re-add zero-variance columns unchanged
    result = data_df.copy()
    result[nonzero_var_cols] = corrected.values

    logger.info("pyCombat correction completed.")
    return result


def apply_pycombat_per_normalization(
    data_dict: Dict[str, pd.DataFrame],
    batch_labels: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """
    Apply pyCombat to each of the 6 normalization versions.

    Parameters
    ----------
    data_dict : dict
        Mapping from normalization name to DataFrame (n_samples, n_features).
        Expected keys: TIC_max, TIC_area, RMS_area, RMS_max, Median_max, Median_area.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.

    Returns
    -------
    dict
        Mapping from normalization name to corrected DataFrame.
    """
    corrected = {}
    for norm_name, df in tqdm(data_dict.items(), desc="pyCombat per normalization"):
        logger.info(f"Applying pyCombat to normalization: {norm_name}")
        try:
            corrected[norm_name] = apply_pycombat(df, batch_labels)
        except Exception as e:
            logger.error(f"pyCombat failed for {norm_name}: {e}")
            corrected[norm_name] = df.copy()
    return corrected
