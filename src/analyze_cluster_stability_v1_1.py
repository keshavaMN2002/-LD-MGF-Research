#!/usr/bin/env python
"""
Cluster stability analysis V1.1.

Adds to V1:
  1. Corrected jointly-assigned ARI (defined for any non-trivial overlap;
     degenerate one-cluster partitions are explicitly flagged).
  2. One-to-one Hungarian cluster matching, with unmatched clusters retained.
  3. Rare-cluster precision/recall/F1 and split/merge diagnostics.
  4. Optional patient-level subsampling stability by refitting selected models.

The script is deliberately exploratory: stability is not clinical validity.
Patient identifiers are used only to align records and are never model features.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except Exception:
    hdbscan = None


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def first_col(df: pd.DataFrame, candidates: list[str], required=True):
    lookup = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    if required:
        raise ValueError(
            f"Could not find any of columns {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def parse_mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    s = str(value).strip()
    if not s:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def boolish(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_labels(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    label_col = first_col(df, ["cluster_label", "label", "labels", "cluster"])
    return df[label_col].to_numpy()


def label_path_for(row: pd.Series, results_dir: Path, labels_dir: Path) -> Path:
    candidates = [
        "labels_file", "label_file", "labels_path", "label_path",
        "label_file_path", "labels_csv", "label_csv"
    ]
    col = first_col(pd.DataFrame([row]), candidates, required=False)
    if col:
        raw = str(row[col])
        p = Path(raw)
        choices = [p, results_dir / p, labels_dir / p, labels_dir / p.name]
        for q in choices:
            if q.exists():
                return q
    # Common convention: label filename is in a `labels/` folder, or a
    # filename field is stored under `label_filename`.
    col = first_col(pd.DataFrame([row]), ["label_filename", "labels_filename"], required=False)
    if col:
        p = labels_dir / str(row[col])
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not resolve label CSV for result row. Expected a labels_file/label_path "
        "column pointing to the saved per-patient label CSV."
    )


def config_key(row: pd.Series, dataset_col: str, algorithm_col: str, config_col: str | None):
    if config_col:
        return str(row[config_col])
    ignored = {dataset_col, algorithm_col}
    parts = []
    for k, v in row.items():
        if k in ignored or "label" in norm(k) or norm(k) in {"runtime", "runtime_seconds"}:
            continue
        if pd.notna(v):
            parts.append(f"{k}={v}")
    return "|".join(parts)


def get_params(row: pd.Series) -> dict:
    pcol = first_col(pd.DataFrame([row]), ["params_json", "parameters_json", "params", "parameters"], required=False)
    params = parse_mapping(row[pcol]) if pcol else {}
    # Also accept flat parameter columns.
    for k, v in row.items():
        nk = norm(k)
        if nk in {
            "eps", "min_samples", "min_cluster_size", "xi", "min_samples_optics",
            "max_eps", "metric", "cluster_selection_method", "algorithm",
            "extract_method", "min_cluster_fraction", "predecessor_correction"
        } and pd.notna(v):
            params.setdefault(nk, v)
    return params


def algorithm_name(x: Any) -> str:
    s = str(x).strip().lower()
    if "hdbscan" in s:
        return "HDBSCAN"
    if "optics" in s:
        return "OPTICS"
    if "dbscan" in s:
        return "DBSCAN"
    return str(x)


def pairwise_jaccard(a: np.ndarray, b: np.ndarray, label_a, label_b) -> float:
    aa, bb = a == label_a, b == label_b
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 0.0


def cluster_sets(labels: np.ndarray) -> list:
    return sorted([x for x in np.unique(labels) if x != -1], key=lambda x: str(x))


def match_clusters_one_to_one(a: np.ndarray, b: np.ndarray, rare_threshold: float, match_threshold: float = 0.50):
    """Hungarian assignment over all non-noise clusters; report rare-cluster outcomes."""
    ca, cb = cluster_sets(a), cluster_sets(b)
    if not ca or not cb:
        return [], {
            "n_clusters_a": len(ca), "n_clusters_b": len(cb),
            "n_matched": 0, "n_unmatched_a": len(ca), "n_unmatched_b": len(cb),
            "matched_jaccard_mean": np.nan, "matched_jaccard_median": np.nan,
            "rare_precision": np.nan, "rare_recall": np.nan, "rare_f1": np.nan,
            "rare_a_count": sum(np.sum(a == c) <= rare_threshold * len(a) for c in ca),
            "rare_b_count": sum(np.sum(b == c) <= rare_threshold * len(b) for c in cb),
            "rare_split_count": 0, "rare_merge_count": 0,
        }

    matrix = np.zeros((len(ca), len(cb)), dtype=float)
    for i, x in enumerate(ca):
        for j, y in enumerate(cb):
            matrix[i, j] = pairwise_jaccard(a, b, x, y)
    ri, ci = linear_sum_assignment(1.0 - matrix)
    matched = []
    for i, j in zip(ri, ci):
        matched.append({
            "cluster_a": ca[i], "cluster_b": cb[j],
            "jaccard": float(matrix[i, j]),
            "size_a": int(np.sum(a == ca[i])),
            "size_b": int(np.sum(b == cb[j])),
            "rare_a": bool(np.sum(a == ca[i]) <= rare_threshold * len(a)),
            "rare_b": bool(np.sum(b == cb[j]) <= rare_threshold * len(b)),
        })
    matched_a = {m["cluster_a"] for m in matched}
    matched_b = {m["cluster_b"] for m in matched}
    rare_a = {c for c in ca if np.sum(a == c) <= rare_threshold * len(a)}
    rare_b = {c for c in cb if np.sum(b == c) <= rare_threshold * len(b)}
    # A rare cluster is considered reproduced if its one-to-one match is also rare
    # and reaches the configured Jaccard threshold.
    threshold = match_threshold
    reproduced_a = {
        m["cluster_a"] for m in matched
        if m["rare_a"] and m["rare_b"] and m["jaccard"] >= threshold
    }
    reproduced_b = {
        m["cluster_b"] for m in matched
        if m["rare_a"] and m["rare_b"] and m["jaccard"] >= threshold
    }
    precision = len(reproduced_b) / len(rare_b) if rare_b else np.nan
    recall = len(reproduced_a) / len(rare_a) if rare_a else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and precision + recall else 0.0
          if np.isfinite(precision) and np.isfinite(recall) else np.nan)

    # Split: a rare source cluster overlaps >=.5 with multiple target clusters.
    # Merge: multiple source clusters overlap >=.5 with one target cluster.
    rare_split = 0
    for x in rare_a:
        if sum(matrix[ca.index(x), j] >= threshold for j in range(len(cb))) > 1:
            rare_split += 1
    rare_merge = 0
    for y in rare_b:
        if sum(matrix[i, cb.index(y)] >= threshold for i in range(len(ca))) > 1:
            rare_merge += 1

    return matched, {
        "n_clusters_a": len(ca), "n_clusters_b": len(cb),
        "n_matched": len(matched),
        "n_unmatched_a": len(ca) - len(matched),
        "n_unmatched_b": len(cb) - len(matched),
        "matched_jaccard_mean": float(np.mean([m["jaccard"] for m in matched])),
        "matched_jaccard_median": float(np.median([m["jaccard"] for m in matched])),
        "rare_precision": precision, "rare_recall": recall, "rare_f1": f1,
        "rare_a_count": len(rare_a), "rare_b_count": len(rare_b),
        "rare_split_count": rare_split, "rare_merge_count": rare_merge,
    }


def pair_metrics(a: np.ndarray, b: np.ndarray, rare_threshold: float, match_threshold: float):
    if len(a) != len(b):
        raise ValueError("Label arrays must refer to the same ordered patient cohort.")
    joint = (a != -1) & (b != -1)
    n_joint = int(joint.sum())
    ari_joint = np.nan
    joint_degenerate = False
    if n_joint >= 2:
        ua, ub = np.unique(a[joint]), np.unique(b[joint])
        # sklearn ARI is mathematically defined for one-cluster partitions too;
        # flag the degenerate case instead of silently returning NaN.
        joint_degenerate = len(ua) < 2 or len(ub) < 2
        ari_joint = float(adjusted_rand_score(a[joint], b[joint]))
    assigned_a, assigned_b = a != -1, b != -1
    union = np.logical_or(assigned_a, assigned_b).sum()
    assigned_jaccard = float(np.logical_and(assigned_a, assigned_b).sum() / union) if union else np.nan
    matched, summary = match_clusters_one_to_one(a, b, rare_threshold, match_threshold)
    return {
        "ari_all_patients": float(adjusted_rand_score(a, b)),
        "ari_jointly_assigned": ari_joint,
        "joint_assigned_n": n_joint,
        "joint_partition_degenerate": joint_degenerate,
        "assigned_patient_jaccard": assigned_jaccard,
        "coverage_a": float(assigned_a.mean()),
        "coverage_b": float(assigned_b.mean()),
        "coverage_abs_difference": float(abs(assigned_a.mean() - assigned_b.mean())),
        **summary,
    }, matched


def build_estimator(algorithm: str, params: dict):
    algorithm = algorithm_name(algorithm)
    if algorithm == "DBSCAN":
        return DBSCAN(
            eps=float(params.get("eps", 0.5)),
            min_samples=int(float(params.get("min_samples", 5))),
            metric=str(params.get("metric", "euclidean")),
        )
    if algorithm == "HDBSCAN":
        if hdbscan is None:
            raise ImportError("Install hdbscan to run HDBSCAN bootstrap refits.")
        return hdbscan.HDBSCAN(
            min_cluster_size=int(float(params.get("min_cluster_size", 5))),
            min_samples=None if params.get("min_samples") is None else int(float(params["min_samples"])),
            metric=str(params.get("metric", "euclidean")),
            cluster_selection_method=str(params.get("cluster_selection_method", "eom")),
        )
    if algorithm == "OPTICS":
        method = str(params.get("extract_method", params.get("cluster_method", "xi"))).lower()
        common = dict(
            min_samples=int(float(params.get("min_samples", 5))),
            metric=str(params.get("metric", "minkowski")),
            p=float(params.get("p", 2)),
            cluster_method=method,
        )
        if method == "dbscan":
            common["eps"] = float(params.get("eps", 0.8))
        else:
            common["xi"] = float(params.get("xi", 0.05))
            if params.get("min_cluster_size") is not None:
                v = params["min_cluster_size"]
                common["min_cluster_size"] = float(v) if isinstance(v, float) and v < 1 else int(float(v))
        return OPTICS(**common)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def load_feature_matrix(dataset: str, dataset_dir: Path, row: pd.Series):
    # Prefer an explicit matrix path from the results CSV.
    col = first_col(pd.DataFrame([row]), [
        "dataset_file", "feature_file", "input_file", "matrix_file", "data_path"
    ], required=False)
    candidates = []
    if col and pd.notna(row[col]):
        p = Path(str(row[col]))
        candidates += [p, dataset_dir / p, dataset_dir / p.name]
    candidates += [
        dataset_dir / f"{dataset}.csv",
        dataset_dir / f"{dataset}_scaled.csv",
        dataset_dir / f"{dataset}_features.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Keep only numeric columns. Exclude obvious IDs/labels/index columns.
            drop = [c for c in df.columns if norm(c) in {
                "patient_id", "id", "row_index", "index", "cluster_label", "label", "unnamed_0"
            }]
            xdf = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
            if xdf.shape[1] == 0:
                raise ValueError(f"No numeric feature columns found in {p}")
            return xdf.to_numpy(dtype=float), p
    raise FileNotFoundError(
        f"Could not find feature matrix for dataset '{dataset}'. Supply --dataset-dir "
        "containing <dataset>.csv, or include dataset_file in the results CSV."
    )


def bootstrap_stability(results: pd.DataFrame, dataset_col: str, algorithm_col: str,
                        labels_dir: Path, results_dir: Path, dataset_dir: Path,
                        reps: int, fraction: float, seed: int, max_configs: int,
                        output_dir: Path):
    rng = np.random.default_rng(seed)
    out = []
    # Choose representative configurations by assigned coverage, then retain a
    # deterministic spread rather than selecting on a validity score.
    for (dataset, algo), group in results.groupby([dataset_col, algorithm_col], dropna=False):
        group = group.copy()
        if max_configs > 0 and len(group) > max_configs:
            cov_col = first_col(group, ["coverage", "coverage_ratio", "assigned_fraction"], required=False)
            if cov_col:
                group = group.sort_values(cov_col).iloc[
                    np.linspace(0, len(group) - 1, max_configs).round().astype(int)
                ]
            else:
                group = group.iloc[np.linspace(0, len(group) - 1, max_configs).round().astype(int)]
        x, matrix_path = load_feature_matrix(str(dataset), dataset_dir, group.iloc[0])
        n = len(x)
        if n < 10:
            continue
        for _, row in group.iterrows():
            algo_name = algorithm_name(row[algorithm_col])
            params = get_params(row)
            original_path = label_path_for(row, results_dir, labels_dir)
            original_labels = load_labels(original_path)
            if len(original_labels) != n:
                raise ValueError(
                    f"Feature matrix rows ({n}) and labels ({len(original_labels)}) differ "
                    f"for {dataset}/{algo_name}. Keep identical row ordering."
                )
            for rep in range(reps):
                take_n = max(2, int(round(n * fraction)))
                idx = np.sort(rng.choice(n, size=take_n, replace=False))
                started = time.perf_counter()
                try:
                    est = build_estimator(algo_name, params)
                    sampled_labels = np.asarray(est.fit_predict(x[idx]))
                    # Compare refit labels against original labels restricted to the
                    # exact same patient rows. This avoids comparing unrelated samples.
                    reference = original_labels[idx]
                    ari = float(adjusted_rand_score(reference, sampled_labels))
                    # One-to-one matching and rare subgroup reproducibility on the
                    # common patient subset.
                    _, match_summary = match_clusters_one_to_one(
                        reference, sampled_labels, rare_threshold=0.05
                    )
                    out.append({
                        "dataset": dataset, "algorithm": algo_name,
                        "config_key": config_key(row, dataset_col, algorithm_col, None),
                        "replicate": rep + 1, "sample_fraction": fraction,
                        "sample_n": take_n, "ari_vs_full_fit_on_sample": ari,
                        "sample_coverage": float(np.mean(sampled_labels != -1)),
                        "reference_coverage_on_sample": float(np.mean(reference != -1)),
                        "runtime_seconds": time.perf_counter() - started,
                        **{f"rare_{k}": v for k, v in match_summary.items()
                           if k in {"precision", "recall", "f1", "a_count", "b_count",
                                    "split_count", "merge_count"}},
                        "feature_matrix": str(matrix_path),
                        "original_labels": str(original_path),
                        "status": "ok",
                    })
                except Exception as e:
                    out.append({
                        "dataset": dataset, "algorithm": algo_name,
                        "config_key": config_key(row, dataset_col, algorithm_col, None),
                        "replicate": rep + 1, "sample_fraction": fraction,
                        "sample_n": take_n, "status": "error", "error": repr(e),
                    })
    df = pd.DataFrame(out)
    df.to_csv(output_dir / "bootstrap_replicate_stability.csv", index=False)
    if not df.empty:
        good = df[df["status"] == "ok"]
        if not good.empty:
            summary = good.groupby(["dataset", "algorithm", "config_key"], dropna=False).agg(
                n_replicates=("ari_vs_full_fit_on_sample", "count"),
                ari_median=("ari_vs_full_fit_on_sample", "median"),
                ari_mean=("ari_vs_full_fit_on_sample", "mean"),
                ari_iqr=("ari_vs_full_fit_on_sample", lambda s: s.quantile(.75) - s.quantile(.25)),
                sample_coverage_median=("sample_coverage", "median"),
                runtime_median_seconds=("runtime_seconds", "median"),
            ).reset_index()
            summary.to_csv(output_dir / "bootstrap_stability_summary.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="parameter_sensitivity_results.csv")
    ap.add_argument("--labels-dir", type=Path, required=True,
                    help="Directory containing saved per-patient label CSVs")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rare-threshold", type=float, default=0.05,
                    help="Rare cluster size threshold as fraction of full cohort (default 0.05)")
    ap.add_argument("--rare-match-threshold", type=float, default=0.50,
                    help="Jaccard threshold used for rare-cluster reproduction (default 0.50)")
    ap.add_argument("--bootstrap-replicates", type=int, default=0,
                    help="Set >0 to refit selected configs on patient subsamples")
    ap.add_argument("--bootstrap-fraction", type=float, default=0.80)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--bootstrap-max-configs", type=int, default=3,
                    help="Configs per dataset/algorithm for bootstrap; 0 means all")
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="Directory with scaled feature CSVs; required for bootstrap")
    args = ap.parse_args()

    if not 0 < args.rare_threshold <= 1:
        ap.error("--rare-threshold must be in (0, 1].")
    if not 0 < args.bootstrap_fraction < 1:
        ap.error("--bootstrap-fraction must be in (0, 1).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    dataset_col = first_col(results, ["dataset", "dataset_name", "cohort"])
    algorithm_col = first_col(results, ["algorithm", "method", "clustering_algorithm"])
    config_col = first_col(results, ["config_id", "configuration_id", "run_id"], required=False)

    # Load each configuration's labels once.
    cache = {}
    records = []
    for i, row in results.iterrows():
        try:
            p = label_path_for(row, args.results.parent, args.labels_dir)
            labels = load_labels(p)
            key = (i, str(row[dataset_col]), algorithm_name(row[algorithm_col]))
            cache[key] = (labels, p)
            records.append((i, key))
        except Exception as e:
            warnings.warn(f"Skipping result row {i}: {e}")

    # Reconstruct local adjacent-neighbor comparisons from the V2 result grid.
    # We compare configurations differing in exactly one parameter, with the
    # same dataset/algorithm/extraction mode. Parameter distance is not assumed
    # to be adjacent unless the values are immediate neighbors in observed grid.
    neighbor_rows, match_rows = [], []
    valid_indices = [i for i, _ in records]
    params_by_i = {i: get_params(results.iloc[i]) for i in valid_indices}
    for dataset, dgroup in results.loc[valid_indices].groupby(dataset_col):
        for algo, agroup in dgroup.groupby(algorithm_col):
            indices = list(agroup.index)
            algo_name = algorithm_name(algo)
            for pos, ia in enumerate(indices):
                pa = params_by_i[ia]
                for ib in indices[pos + 1:]:
                    pb = params_by_i[ib]
                    keys = sorted(set(pa) | set(pb))
                    changed = [k for k in keys if str(pa.get(k)) != str(pb.get(k))]
                    if len(changed) != 1:
                        continue
                    changed_key = changed[0]
                    # OPTICS extraction families must not be mixed.
                    if algo_name == "OPTICS":
                        ma = str(pa.get("extract_method", pa.get("cluster_method", "xi"))).lower()
                        mb = str(pb.get("extract_method", pb.get("cluster_method", "xi"))).lower()
                        if ma != mb:
                            continue
                    # Require neighboring numeric values when a numeric parameter
                    # is changed; categorical method variants are allowed as a pair.
                    va, vb = pa.get(changed_key), pb.get(changed_key)
                    try:
                        numeric = sorted({
                            float(get_params(results.iloc[k]).get(changed_key))
                            for k in indices if get_params(results.iloc[k]).get(changed_key) is not None
                        })
                        if len(numeric) > 1 and float(va) != float(vb):
                            unique = sorted(set(numeric))
                            if abs(unique.index(float(va)) - unique.index(float(vb))) != 1:
                                continue
                    except Exception:
                        pass
                    la = cache[(ia, str(dataset), algo_name)][0]
                    lb = cache[(ib, str(dataset), algo_name)][0]
                    if len(la) != len(lb):
                        warnings.warn(f"Label length mismatch for rows {ia}/{ib}; skipped.")
                        continue
                    metrics, matches = pair_metrics(la, lb, args.rare_threshold, args.rare_match_threshold)
                    rec = {
                        "dataset": dataset, "algorithm": algo_name,
                        "result_row_a": ia, "result_row_b": ib,
                        "config_a": config_key(results.iloc[ia], dataset_col, algorithm_col, config_col),
                        "config_b": config_key(results.iloc[ib], dataset_col, algorithm_col, config_col),
                        "changed_parameter": changed_key,
                        "value_a": va, "value_b": vb,
                        **metrics,
                    }
                    neighbor_rows.append(rec)
                    for m in matches:
                        match_rows.append({
                            "dataset": dataset, "algorithm": algo_name,
                            "result_row_a": ia, "result_row_b": ib,
                            "changed_parameter": changed_key, **m,
                        })

    neighbors = pd.DataFrame(neighbor_rows)
    matches = pd.DataFrame(match_rows)
    neighbors.to_csv(args.output_dir / "neighbor_pair_stability_v1_1.csv", index=False)
    matches.to_csv(args.output_dir / "one_to_one_cluster_matches.csv", index=False)

    if not neighbors.empty:
        summary = neighbors.groupby(["dataset", "algorithm"], dropna=False).agg(
            n_neighbor_pairs=("ari_all_patients", "count"),
            ari_all_median=("ari_all_patients", "median"),
            ari_joint_median=("ari_jointly_assigned", "median"),
            joint_ari_defined_pairs=("ari_jointly_assigned", "count"),
            joint_degenerate_pairs=("joint_partition_degenerate", "sum"),
            assigned_jaccard_median=("assigned_patient_jaccard", "median"),
            coverage_abs_diff_median=("coverage_abs_difference", "median"),
            rare_precision_median=("rare_precision", "median"),
            rare_recall_median=("rare_recall", "median"),
            rare_f1_median=("rare_f1", "median"),
            rare_split_count=("rare_split_count", "sum"),
            rare_merge_count=("rare_merge_count", "sum"),
        ).reset_index()
        summary.to_csv(args.output_dir / "stability_summary_v1_1.csv", index=False)

    metadata = {
        "version": "1.1",
        "input_results": str(args.results.resolve()),
        "labels_dir": str(args.labels_dir.resolve()),
        "rare_cluster_threshold_fraction": args.rare_threshold,
        "rare_match_jaccard_threshold": args.rare_match_threshold,
        "neighbor_pair_count": len(neighbors),
        "matched_cluster_record_count": len(matches),
        "bootstrap_requested": args.bootstrap_replicates > 0,
        "limitations": [
            "Parameter-neighbor stability is local sensitivity, not bootstrap stability.",
            "All-patient ARI treats noise as a label and can be influenced by shared noise.",
            "Jointly-assigned ARI is reported with sample size and a degenerate-partition flag.",
            "Hungarian matching is one-to-one; unmatched clusters are retained in pair summaries.",
            "Rare precision/recall/F1 use a fixed 0.50 Jaccard match threshold and are exploratory.",
            "Synthetic Synthea data cannot establish clinical validity or real-world subgroup utility.",
        ],
    }
    if args.bootstrap_replicates > 0:
        if args.dataset_dir is None:
            ap.error("--dataset-dir is required when --bootstrap-replicates > 0")
        boot = bootstrap_stability(
            results, dataset_col, algorithm_col, args.labels_dir, args.results.parent,
            args.dataset_dir, args.bootstrap_replicates, args.bootstrap_fraction,
            args.bootstrap_seed, args.bootstrap_max_configs, args.output_dir
        )
        metadata["bootstrap_replicate_count"] = len(boot)
    (args.output_dir / "stability_metadata_v1_1.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("=" * 72)
    print("CLUSTER STABILITY ANALYSIS V1.1")
    print("=" * 72)
    print(f"Neighbor pairs: {len(neighbors)}")
    print(f"One-to-one cluster matches: {len(matches)}")
    print(f"Output: {args.output_dir.resolve()}")
    if args.bootstrap_replicates:
        print(f"Bootstrap refit records: {metadata.get('bootstrap_replicate_count', 0)}")


if __name__ == "__main__":
    main()
