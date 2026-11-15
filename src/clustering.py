"""
Clustering Module for LD-MGF Research
Implements K-Means, Agglomerative Clustering, DBSCAN, OPTICS, and Gaussian Mixture Models (GMM).
Provides timing, execution pipelines, and evaluation integration.
"""

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from src.evaluation import evaluate_all_metrics


def run_kmeans(X, k=2, random_state=42):
    """Executes K-Means clustering."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return labels, elapsed, model


def run_agglomerative(X, k=2, linkage='ward'):
    """Executes Agglomerative Hierarchical Clustering."""
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return labels, elapsed, model


def run_dbscan(X, eps=2.5, min_samples=5):
    """Executes Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return labels, elapsed, model


def run_optics(X, min_samples=10, cluster_method='xi', xi=0.01, eps=None, min_cluster_size=0.03):
    """Executes Ordering Points To Identify the Clustering Structure (OPTICS)."""
    model = OPTICS(min_samples=min_samples, cluster_method=cluster_method, xi=xi, eps=eps, min_cluster_size=min_cluster_size)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return labels, elapsed, model


def run_gmm(X, k=2, random_state=42):
    """Executes Gaussian Mixture Model clustering."""
    model = GaussianMixture(n_components=k, random_state=random_state)
    t0 = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - t0
    return labels, elapsed, model


def run_clustering_suite(X, dataset_name, params=None):
    """
    Executes all 5 algorithms on feature matrix X, times them,
    and evaluates against all 6 CVIs including LD-MGF.
    
    Returns:
        results_df (pd.DataFrame): Comparison table of metrics and execution times.
        labels_dict (dict): Dictionary mapping model_name -> predicted labels array.
    """
    if params is None:
        params = {}
        
    k = params.get('k', 2)
    dbscan_eps = params.get('dbscan_eps', 2.2)
    dbscan_ms = params.get('dbscan_ms', 5)
    optics_ms = params.get('optics_ms', 10)
    optics_xi = params.get('optics_xi', 0.01)
    optics_method = params.get('optics_method', 'xi')
    optics_eps = params.get('optics_eps', None)
    
    models = {
        'K-Means': lambda: run_kmeans(X, k=k),
        'Agglomerative': lambda: run_agglomerative(X, k=k),
        'DBSCAN': lambda: run_dbscan(X, eps=dbscan_eps, min_samples=dbscan_ms),
        'OPTICS': lambda: run_optics(X, min_samples=optics_ms, cluster_method=optics_method, xi=optics_xi, eps=optics_eps),
        'GMM': lambda: run_gmm(X, k=k)
    }
    
    metrics_records = []
    labels_dict = {}
    
    print(f"\n--- Running Clustering Suite on {dataset_name} (N={X.shape[0]}, D={X.shape[1]}) ---")
    for model_name, runner in models.items():
        print(f"  Fitting {model_name}...", end=" ", flush=True)
        labels, elapsed, _ = runner()
        labels_dict[model_name] = labels
        unique_c = [l for l in np.unique(labels) if l != -1]
        noise_cnt = int(np.sum(labels == -1))
        print(f"done in {elapsed:.3f}s. (Clusters: {len(unique_c)}, Noise: {noise_cnt})")
        
        # Evaluate metrics
        res = evaluate_all_metrics(X, labels, model_name=model_name, dataset_name=dataset_name, execution_time=elapsed)
        metrics_records.append(res)
        
    results_df = pd.DataFrame(metrics_records)
    return results_df, labels_dict
