"""
Evaluation metrics for batch correction methods on MALDI MSI data.

Batch effect removal metrics (lower = better):
- AMI/ARI between KMeans clusters and batch labels
- Silhouette score using batch labels
- PCR: PC1 variance explained by batch labels
- Batch entropy in kNN neighborhoods
- kBET approximation (chi-squared test)

Biological signal preservation metrics (higher = better):
- AMI/ARI between KMeans clusters and tissue labels
- Silhouette score using tissue labels
- MCC
- kNN accuracy
- Macro F1

Combined metric:
- bio_batch_f1
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2

logger = logging.getLogger(__name__)

_EPSILON = 1e-10
_MIN_TISSUE_CLUSTERS = 4


def _clean_data(data: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    data = np.array(data, dtype=float)
    data = np.where(np.isfinite(data), data, 0.0)
    return data


def _stratified_subsample(
    data: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int = 42,
) -> tuple:
    """Stratified subsample of data and labels."""
    n = len(data)
    if n <= sample_size:
        return data, labels
    rng = np.random.RandomState(random_state)
    unique, counts = np.unique(labels, return_counts=True)
    indices = []
    for cls, cnt in zip(unique, counts):
        cls_idx = np.where(labels == cls)[0]
        n_take = max(2, int(sample_size * cnt / n))
        n_take = min(n_take, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        indices.extend(chosen.tolist())
    indices = np.array(indices[:sample_size])
    return data[indices], labels[indices]


def compute_ami_batch(data: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute Adjusted Mutual Information between KMeans clusters and batch labels.

    Lower values indicate better batch effect removal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.

    Returns
    -------
    float
        AMI score between KMeans clusters and batch labels.
    """
    try:
        data = _clean_data(data)
        n_batches = len(np.unique(batch_labels))
        if n_batches < 2:
            logger.warning("Only one batch found; AMI batch is 0.")
            return 0.0
        kmeans = KMeans(n_clusters=n_batches, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        return float(adjusted_mutual_info_score(batch_labels, cluster_labels))
    except Exception as e:
        logger.error(f"compute_ami_batch failed: {e}")
        return float("nan")


def compute_ari_batch(data: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between KMeans clusters and batch labels.

    Lower values indicate better batch effect removal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.

    Returns
    -------
    float
        ARI score between KMeans clusters and batch labels.
    """
    try:
        data = _clean_data(data)
        n_batches = len(np.unique(batch_labels))
        if n_batches < 2:
            logger.warning("Only one batch found; ARI batch is 0.")
            return 0.0
        kmeans = KMeans(n_clusters=n_batches, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        return float(adjusted_rand_score(batch_labels, cluster_labels))
    except Exception as e:
        logger.error(f"compute_ari_batch failed: {e}")
        return float("nan")


def compute_silhouette_batch(
    data: np.ndarray, batch_labels: np.ndarray, sample_size: int = 50000
) -> float:
    """
    Compute Silhouette score using batch labels.

    Subsamples to sample_size if data is larger. Lower values indicate
    better batch effect removal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.
    sample_size : int, optional
        Maximum number of samples to use (default 50000).

    Returns
    -------
    float
        Silhouette score using batch labels.
    """
    try:
        data = _clean_data(data)
        if len(np.unique(batch_labels)) < 2:
            logger.warning("Only one batch; silhouette_batch is 0.")
            return 0.0
        data_s, labels_s = _stratified_subsample(data, batch_labels, sample_size)
        if len(np.unique(labels_s)) < 2:
            return 0.0
        return float(silhouette_score(data_s, labels_s, random_state=42))
    except Exception as e:
        logger.error(f"compute_silhouette_batch failed: {e}")
        return float("nan")


def compute_pcr(data: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Principal Component Regression: variance of PC1 explained by batch labels.

    Computes PCA on data, then regresses PC1 scores against one-hot encoded
    batch labels using linear regression R².

    Lower values indicate better batch effect removal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.

    Returns
    -------
    float
        R² of PC1 regressed on batch labels.
    """
    try:
        data = _clean_data(data)
        pca = PCA(n_components=1, random_state=42)
        pc1 = pca.fit_transform(data)  # (n, 1)
        le = LabelEncoder()
        batch_encoded = le.fit_transform(batch_labels).reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(batch_encoded, pc1)
        ss_res = np.sum((pc1 - reg.predict(batch_encoded)) ** 2)
        ss_tot = np.sum((pc1 - np.mean(pc1)) ** 2)
        if ss_tot == 0:
            return 0.0
        r2 = float(1 - ss_res / ss_tot)
        return max(0.0, r2)
    except Exception as e:
        logger.error(f"compute_pcr failed: {e}")
        return float("nan")


def compute_batch_entropy(
    data: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 100,
    sample_size: int = 50000,
) -> float:
    """
    Entropy of batch label distribution in kNN neighborhoods.

    Higher values indicate better batch mixing (less batch effect).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.
    n_neighbors : int, optional
        Number of nearest neighbors (default 100).
    sample_size : int, optional
        Maximum number of samples to use (default 50000).

    Returns
    -------
    float
        Mean entropy of batch distribution in kNN neighborhoods.
    """
    try:
        data = _clean_data(data)
        data_s, labels_s = _stratified_subsample(data, batch_labels, sample_size)
        unique_batches = np.unique(labels_s)
        n_batches = len(unique_batches)
        if n_batches < 2:
            return 0.0
        k = min(n_neighbors, len(data_s) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(data_s)
        _, indices = nbrs.kneighbors(data_s)
        entropies = []
        for i in range(len(data_s)):
            neighbor_idx = indices[i][1:]  # exclude self
            neighbor_batches = labels_s[neighbor_idx]
            counts = np.array(
                [np.sum(neighbor_batches == b) for b in unique_batches], dtype=float
            )
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
        return float(np.mean(entropies))
    except Exception as e:
        logger.error(f"compute_batch_entropy failed: {e}")
        return float("nan")


def compute_kbet_python(
    data: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 30,
    sample_size: int = 10000,
) -> float:
    """
    Pure Python kBET approximation using chi-squared test on batch proportions
    in kNN neighborhoods.

    Lower rejection rate indicates better batch mixing (less batch effect).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.
    k : int, optional
        Number of nearest neighbors (default 30).
    sample_size : int, optional
        Maximum number of samples to use (default 10000).

    Returns
    -------
    float
        Rejection rate (fraction of neighborhoods that fail chi-squared test).
    """
    try:
        data = _clean_data(data)
        data_s, labels_s = _stratified_subsample(data, batch_labels, sample_size)
        n = len(data_s)
        unique_batches, global_counts = np.unique(labels_s, return_counts=True)
        n_batches = len(unique_batches)
        if n_batches < 2:
            return 0.0
        global_props = global_counts / n
        k_actual = min(k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(data_s)
        _, indices = nbrs.kneighbors(data_s)
        rejections = 0
        for i in range(n):
            neighbor_idx = indices[i][1:]
            neighbor_labels = labels_s[neighbor_idx]
            observed = np.array(
                [np.sum(neighbor_labels == b) for b in unique_batches], dtype=float
            )
            expected = global_props * k_actual
            # chi-squared statistic
            chi2_stat = np.sum(
                (observed - expected) ** 2 / (expected + _EPSILON)
            )
            df = n_batches - 1
            p_val = 1 - chi2.cdf(chi2_stat, df)
            if p_val < 0.05:
                rejections += 1
        return float(rejections / n)
    except Exception as e:
        logger.error(f"compute_kbet_python failed: {e}")
        return float("nan")


def compute_ami_tissue(data: np.ndarray, tissue_labels: np.ndarray) -> float:
    """
    Compute AMI between KMeans(n_clusters=4) clusters and tissue type labels.

    Higher values indicate better biological signal preservation.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.

    Returns
    -------
    float
        AMI score between KMeans clusters and tissue labels.
    """
    try:
        data = _clean_data(data)
        # Use at least _MIN_TISSUE_CLUSTERS (4) clusters to match expected tissue types
        # in MALDI MSI dataset; actual unique labels may be fewer during subsampling.
        n_tissues = max(_MIN_TISSUE_CLUSTERS, len(np.unique(tissue_labels)))
        kmeans = KMeans(n_clusters=n_tissues, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        return float(adjusted_mutual_info_score(tissue_labels, cluster_labels))
    except Exception as e:
        logger.error(f"compute_ami_tissue failed: {e}")
        return float("nan")


def compute_ari_tissue(data: np.ndarray, tissue_labels: np.ndarray) -> float:
    """
    Compute ARI between KMeans(n_clusters=4) clusters and tissue type labels.

    Higher values indicate better biological signal preservation.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.

    Returns
    -------
    float
        ARI score between KMeans clusters and tissue labels.
    """
    try:
        data = _clean_data(data)
        # Use at least _MIN_TISSUE_CLUSTERS (4) clusters to match expected tissue types
        # in MALDI MSI dataset; actual unique labels may be fewer during subsampling.
        n_tissues = max(_MIN_TISSUE_CLUSTERS, len(np.unique(tissue_labels)))
        kmeans = KMeans(n_clusters=n_tissues, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        return float(adjusted_rand_score(tissue_labels, cluster_labels))
    except Exception as e:
        logger.error(f"compute_ari_tissue failed: {e}")
        return float("nan")


def compute_silhouette_tissue(
    data: np.ndarray, tissue_labels: np.ndarray, sample_size: int = 50000
) -> float:
    """
    Compute Silhouette score using tissue labels.

    Subsamples to sample_size if data is larger. Higher values indicate
    better biological signal preservation.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.
    sample_size : int, optional
        Maximum number of samples to use (default 50000).

    Returns
    -------
    float
        Silhouette score using tissue labels.
    """
    try:
        data = _clean_data(data)
        if len(np.unique(tissue_labels)) < 2:
            logger.warning("Only one tissue type; silhouette_tissue is 0.")
            return 0.0
        data_s, labels_s = _stratified_subsample(data, tissue_labels, sample_size)
        if len(np.unique(labels_s)) < 2:
            return 0.0
        return float(silhouette_score(data_s, labels_s, random_state=42))
    except Exception as e:
        logger.error(f"compute_silhouette_tissue failed: {e}")
        return float("nan")


def compute_mcc(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> float:
    """
    Compute Matthews Correlation Coefficient.

    Higher values indicate better biological signal preservation.

    Parameters
    ----------
    true_labels : np.ndarray, shape (n_samples,)
        True class labels.
    predicted_labels : np.ndarray, shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Matthews Correlation Coefficient.
    """
    try:
        return float(matthews_corrcoef(true_labels, predicted_labels))
    except Exception as e:
        logger.error(f"compute_mcc failed: {e}")
        return float("nan")


def compute_knn_accuracy(
    data: np.ndarray,
    tissue_labels: np.ndarray,
    n_neighbors: int = 15,
    sample_size: int = 50000,
) -> float:
    """
    Compute kNN classifier accuracy with train/test split.

    Higher values indicate better biological signal preservation.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    tissue_labels : np.ndarray, shape (n_samples,)
        Tissue type labels.
    n_neighbors : int, optional
        Number of nearest neighbors for kNN classifier (default 15).
    sample_size : int, optional
        Maximum number of samples to use (default 50000).

    Returns
    -------
    float
        kNN classification accuracy.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        data = _clean_data(data)
        data_s, labels_s = _stratified_subsample(data, tissue_labels, sample_size)
        unique, counts = np.unique(labels_s, return_counts=True)
        if len(unique) < 2 or np.min(counts) < 2:
            logger.warning("Too few samples per class for kNN accuracy.")
            return float("nan")
        X_train, X_test, y_train, y_test = train_test_split(
            data_s, labels_s, test_size=0.2, random_state=42, stratify=labels_s
        )
        k = min(n_neighbors, len(X_train) - 1)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        return float(knn.score(X_test, y_test))
    except Exception as e:
        logger.error(f"compute_knn_accuracy failed: {e}")
        return float("nan")


def compute_macro_f1(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> float:
    """
    Compute macro-averaged F1 score.

    Higher values indicate better biological signal preservation.

    Parameters
    ----------
    true_labels : np.ndarray, shape (n_samples,)
        True class labels.
    predicted_labels : np.ndarray, shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Macro-averaged F1 score.
    """
    try:
        return float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0))
    except Exception as e:
        logger.error(f"compute_macro_f1 failed: {e}")
        return float("nan")


def compute_bio_batch_f1(bio_score: float, batch_score: float) -> float:
    """
    Combined bio-batch F1 score.

    Harmonic mean of biological signal preservation and batch effect removal.

    Parameters
    ----------
    bio_score : float
        Biological signal preservation score (higher = better).
    batch_score : float
        Batch effect score (lower = better; will be inverted internally).

    Returns
    -------
    float
        Combined F1 score.
    """
    batch_removed = 1.0 - batch_score
    denom = batch_removed + bio_score
    if denom == 0:
        return 0.0
    return float(2.0 * batch_removed * bio_score / (denom + 1e-8))


def compute_all_metrics(
    data: np.ndarray,
    batch_labels: np.ndarray,
    tissue_labels: np.ndarray,
    tissue_preds: Optional[np.ndarray] = None,
    sample_size: int = 50000,
) -> dict:
    """
    Compute all evaluation metrics for batch correction.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix (batch-corrected or raw).
    batch_labels : np.ndarray, shape (n_samples,)
        Batch assignment for each sample.
    tissue_labels : np.ndarray, shape (n_samples,)
        True tissue type labels.
    tissue_preds : np.ndarray, optional
        Predicted tissue labels (for MCC and macro_f1).
    sample_size : int, optional
        Subsampling size for expensive metrics (default 50000).

    Returns
    -------
    dict
        Dictionary with keys: ami_batch, ari_batch, silhouette_batch, pcr,
        batch_entropy, kbet, ami_tissue, ari_tissue, silhouette_tissue,
        mcc (if tissue_preds), knn_accuracy, macro_f1 (if tissue_preds),
        bio_batch_f1.
    """
    data = np.array(data, dtype=float)
    batch_labels = np.array(batch_labels)
    tissue_labels = np.array(tissue_labels)

    results = {}

    logger.info("Computing batch effect metrics...")
    results["ami_batch"] = compute_ami_batch(data, batch_labels)
    results["ari_batch"] = compute_ari_batch(data, batch_labels)
    results["silhouette_batch"] = compute_silhouette_batch(data, batch_labels, sample_size)
    results["pcr"] = compute_pcr(data, batch_labels)
    results["batch_entropy"] = compute_batch_entropy(data, batch_labels, sample_size=sample_size)
    results["kbet"] = compute_kbet_python(data, batch_labels, sample_size=min(10000, sample_size))

    logger.info("Computing biological signal metrics...")
    results["ami_tissue"] = compute_ami_tissue(data, tissue_labels)
    results["ari_tissue"] = compute_ari_tissue(data, tissue_labels)
    results["silhouette_tissue"] = compute_silhouette_tissue(data, tissue_labels, sample_size)
    results["knn_accuracy"] = compute_knn_accuracy(data, tissue_labels, sample_size=sample_size)

    if tissue_preds is not None:
        results["mcc"] = compute_mcc(tissue_labels, tissue_preds)
        results["macro_f1"] = compute_macro_f1(tissue_labels, tissue_preds)

    # Combined metric: use silhouette_batch as batch score, knn_accuracy as bio score
    batch_score = results.get("silhouette_batch", 0.0)
    bio_score = results.get("knn_accuracy", 0.0)
    if np.isnan(batch_score):
        batch_score = 0.0
    if np.isnan(bio_score):
        bio_score = 0.0
    results["bio_batch_f1"] = compute_bio_batch_f1(bio_score, batch_score)

    return results
