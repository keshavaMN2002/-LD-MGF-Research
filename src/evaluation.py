"""
Evaluation Module for LD-MGF Research
Implements standard Cluster Validity Indices (Silhouette, Davies-Bouldin, Calinski-Harabasz),
advanced indices (Dunn Index, DBCV), and the proposed LD-MGF Index.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors


def compute_dunn_index(X, labels, max_samples_per_cluster=500, random_state=42):
    """
    Computes Dunn Index: ratio of minimum inter-cluster distance to maximum intra-cluster diameter.
    Higher is better.
    """
    unique_labels = [l for l in np.unique(labels) if l != -1 and np.sum(labels == l) > 1]
    if len(unique_labels) < 2:
        return np.nan

    rng = np.random.RandomState(random_state)
    cluster_points = {}
    max_diameter = 0.0

    # Intra-cluster maximum diameter
    for l in unique_labels:
        pts = X[labels == l]
        if len(pts) <= 1:
            cluster_points[l] = pts
            continue
        if len(pts) > max_samples_per_cluster:
            sub_idx = rng.choice(len(pts), size=max_samples_per_cluster, replace=False)
            pts = pts[sub_idx]
        cluster_points[l] = pts
        
        diam = np.max(pdist(pts))
        if diam > max_diameter:
            max_diameter = diam

    if max_diameter == 0:
        return np.nan

    # Inter-cluster minimum distance
    min_inter_dist = np.inf
    for i in range(len(unique_labels)):
        pts_i = cluster_points[unique_labels[i]]
        for j in range(i + 1, len(unique_labels)):
            pts_j = cluster_points[unique_labels[j]]
            dist_ij = np.min(cdist(pts_i, pts_j))
            if dist_ij < min_inter_dist:
                min_inter_dist = dist_ij

    return float(min_inter_dist / max_diameter)


def compute_dbcv(X, labels, max_samples_per_cluster=400, random_state=42):
    """
    Density-Based Clustering Validation (DBCV) index (Moulavi et al., 2014).
    Evaluates density-compactness and density-separation via mutual reachability distance.
    Higher is better (bounded in [-1, 1]).
    """
    unique_labels = [l for l in np.unique(labels) if l != -1 and np.sum(labels == l) > 1]
    if len(unique_labels) < 2:
        return np.nan

    N, d = X.shape
    rng = np.random.RandomState(random_state)
    cluster_pts = {}
    core_dists = {}
    mst_compactness = {}

    for l in unique_labels:
        pts = X[labels == l]
        if len(pts) > max_samples_per_cluster:
            sub_idx = rng.choice(len(pts), size=max_samples_per_cluster, replace=False)
            pts = pts[sub_idx]
        cluster_pts[l] = pts
        n_l = len(pts)
        if n_l <= 1:
            mst_compactness[l] = 0.0
            core_dists[l] = np.zeros(n_l)
            continue

        D = squareform(pdist(pts))
        effective_dim = min(d, 10)
        D_safe = np.where(D == 0, np.inf, D)
        inv_D_power = np.sum(1.0 / (D_safe ** effective_dim), axis=1)
        cdist_l = (inv_D_power / (n_l - 1)) ** (-1.0 / effective_dim)
        core_dists[l] = cdist_l

        cdist_matrix = np.maximum(cdist_l[:, None], cdist_l[None, :])
        mrd = np.maximum(cdist_matrix, D)

        mst = minimum_spanning_tree(mrd)
        mst_compactness[l] = float(np.max(mst.data)) if len(mst.data) > 0 else 0.0

    sep_matrix = {}
    for i in range(len(unique_labels)):
        l_i = unique_labels[i]
        pts_i = cluster_pts[l_i]
        cdist_i = core_dists[l_i]
        for j in range(i + 1, len(unique_labels)):
            l_j = unique_labels[j]
            pts_j = cluster_pts[l_j]
            cdist_j = core_dists[l_j]

            D_ij = cdist(pts_i, pts_j)
            cdist_outer = np.maximum(cdist_i[:, None], cdist_j[None, :])
            mrd_ij = np.maximum(cdist_outer, D_ij)
            min_mrd = float(np.min(mrd_ij))
            sep_matrix[(l_i, l_j)] = min_mrd
            sep_matrix[(l_j, l_i)] = min_mrd

    vc_list = []
    weights = []
    for l in unique_labels:
        n_l = len(cluster_pts[l])
        min_sep = min([sep_matrix[(l, other)] for other in unique_labels if other != l])
        comp = mst_compactness[l]
        denom = max(min_sep, comp)
        vc = (min_sep - comp) / denom if denom > 0 else 0.0
        vc_list.append(vc)
        weights.append(n_l)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    dbcv_score = np.sum(np.array(vc_list) * np.array(weights)) / total_weight
    noise_ratio = np.sum(labels == -1) / N
    dbcv_penalized = float(dbcv_score * (1.0 - noise_ratio))
    return dbcv_penalized


def compute_ld_mgf(X, labels, k_neighbors=5, alpha=0.5, beta=0.3, gamma=0.2, random_state=42):
    """
    Computes the proposed LD-MGF (Local Density - Multi-Granularity Fusion) Index.
    
    Mathematical Formulation:
      - Local density compactness evaluates intra-cluster cohesion using k-NN adaptive kernel:
          rho(x_i) = mean(exp(- d(x_i, x_j)^2 / (2 * sigma_k^2)))
          Disp(C_k) = mean(||x_i - mu_k|| / (1 + rho_norm(x_i)))
      - Multi-granularity separation fuses:
          1. Fine-scale boundary margin (5th percentile inter-cluster distance)
          2. Core-scale density peak distance (||core_a - core_b||)
          3. Macro-scale centroid distance (||mu_a - mu_b||)
          Sep_MGF = alpha * delta_fine + beta * delta_core + gamma * delta_macro
      - Composite Index:
          LD-MGF = (min_{a != b} Sep_MGF(C_a, C_b) / avg_Disp) * (1 - noise_ratio)
          LD-MGF_norm = LD-MGF / (1 + LD-MGF) in [0, 1)
    """
    unique_labels = [l for l in np.unique(labels) if l != -1 and np.sum(labels == l) > 1]
    if len(unique_labels) < 2:
        return {
            'ld_mgf': np.nan,
            'ld_mgf_norm': np.nan,
            'sep_mgf': np.nan,
            'comp_ld': np.nan,
            'noise_ratio': float(np.sum(labels == -1) / len(labels))
        }

    N, D = X.shape
    rng = np.random.RandomState(random_state)
    min_cluster_size = min([np.sum(labels == l) for l in unique_labels])
    k = max(2, min(k_neighbors, min_cluster_size - 1))

    cluster_stats = {}
    dispersions = []

    # 1. Local Density Compactness
    for l in unique_labels:
        pts = X[labels == l]
        n_l = len(pts)
        if n_l <= 1:
            continue

        k_l = min(k, n_l - 1)
        nn = NearestNeighbors(n_neighbors=k_l + 1).fit(pts)
        distances, _ = nn.kneighbors(pts)
        knn_dist = distances[:, 1:]
        sigma = np.median(knn_dist)
        if sigma <= 1e-6:
            sigma = 1.0

        rho = np.mean(np.exp(- (knn_dist ** 2) / (2.0 * (sigma ** 2))), axis=1)
        mode_idx = np.argmax(rho)
        core_point = pts[mode_idx]
        centroid = np.mean(pts, axis=0)

        norm_rho = rho / (np.max(rho) + 1e-8)
        dist_to_mean = np.linalg.norm(pts - centroid, axis=1)
        weighted_dispersion = np.mean(dist_to_mean / (1.0 + norm_rho))
        dispersions.append(weighted_dispersion)

        cluster_stats[l] = {
            'pts': pts,
            'centroid': centroid,
            'core': core_point,
            'rho': rho
        }

    avg_dispersion = float(np.mean(dispersions)) if len(dispersions) > 0 else 1.0

    # 2. Multi-Granularity Separation
    min_fused_sep = np.inf
    for i in range(len(unique_labels)):
        l_i = unique_labels[i]
        stats_i = cluster_stats[l_i]
        pts_i = stats_i['pts']
        if len(pts_i) > 400:
            pts_i_sub = pts_i[rng.choice(len(pts_i), 400, replace=False)]
        else:
            pts_i_sub = pts_i

        for j in range(i + 1, len(unique_labels)):
            l_j = unique_labels[j]
            stats_j = cluster_stats[l_j]
            pts_j = stats_j['pts']
            if len(pts_j) > 400:
                pts_j_sub = pts_j[rng.choice(len(pts_j), 400, replace=False)]
            else:
                pts_j_sub = pts_j

            # Fine granularity: 5th percentile boundary distance
            pair_dists = cdist(pts_i_sub, pts_j_sub)
            delta_fine = np.percentile(pair_dists, 5)

            # Core granularity: distance between density modes
            delta_core = np.linalg.norm(stats_i['core'] - stats_j['core'])

            # Macro granularity: distance between centroids
            delta_macro = np.linalg.norm(stats_i['centroid'] - stats_j['centroid'])

            # Fused separation
            fused_sep = (alpha * delta_fine) + (beta * delta_core) + (gamma * delta_macro)
            if fused_sep < min_fused_sep:
                min_fused_sep = fused_sep

    noise_ratio = float(np.sum(labels == -1) / N)
    ld_mgf_raw = float((min_fused_sep / (avg_dispersion + 1e-6)) * (1.0 - noise_ratio))
    ld_mgf_norm = float(ld_mgf_raw / (1.0 + ld_mgf_raw))

    return {
        'ld_mgf': ld_mgf_raw,
        'ld_mgf_norm': ld_mgf_norm,
        'sep_mgf': float(min_fused_sep),
        'comp_ld': avg_dispersion,
        'noise_ratio': noise_ratio
    }


def evaluate_all_metrics(X, labels, model_name="", dataset_name="", execution_time=0.0):
    """
    Evaluates all clustering validity indices:
    - Silhouette Score (higher is better)
    - Davies-Bouldin Index (lower is better)
    - Calinski-Harabasz Index (higher is better)
    - Dunn Index (higher is better)
    - DBCV (higher is better)
    - Proposed LD-MGF Index (raw and normalized)
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    n_samples = len(labels)
    noise_count = int(np.sum(labels == -1))
    noise_pct = (noise_count / n_samples) * 100.0

    metrics = {
        'Dataset': dataset_name,
        'Algorithm': model_name,
        'Clusters (K)': n_clusters,
        'Noise Points': noise_count,
        'Noise (%)': round(noise_pct, 2),
        'Execution Time (s)': round(execution_time, 4),
        'Silhouette': np.nan,
        'Davies-Bouldin': np.nan,
        'Calinski-Harabasz': np.nan,
        'Dunn Index': np.nan,
        'DBCV': np.nan,
        'LD-MGF': np.nan,
        'LD-MGF (Norm)': np.nan
    }

    if n_clusters < 2:
        return metrics

    # Mask noise points for standard sklearn CVIs
    valid_mask = (labels != -1)
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    try:
        metrics['Silhouette'] = round(float(silhouette_score(X_valid, labels_valid)), 4)
    except Exception:
        pass

    try:
        metrics['Davies-Bouldin'] = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
    except Exception:
        pass

    try:
        metrics['Calinski-Harabasz'] = round(float(calinski_harabasz_score(X_valid, labels_valid)), 2)
    except Exception:
        pass

    try:
        metrics['Dunn Index'] = round(float(compute_dunn_index(X, labels)), 4)
    except Exception:
        pass

    try:
        metrics['DBCV'] = round(float(compute_dbcv(X, labels)), 4)
    except Exception:
        pass

    try:
        ld_res = compute_ld_mgf(X, labels)
        metrics['LD-MGF'] = round(float(ld_res['ld_mgf']), 4)
        metrics['LD-MGF (Norm)'] = round(float(ld_res['ld_mgf_norm']), 4)
    except Exception:
        pass

    return metrics
