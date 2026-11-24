"""
Expanded parameter-sensitivity benchmark for DBSCAN, OPTICS and HDBSCAN.

Run from the project root:
    python src/run_parameter_sensitivity_v2.py

Optional:
    python src/run_parameter_sensitivity_v2.py --algorithms DBSCAN OPTICS HDBSCAN
    python src/run_parameter_sensitivity_v2.py --max-workers 1

Inputs:
    output/benchmark_v1/benchmark_manifest.json
    output/benchmark_v1/features_*.csv

Each parameter configuration runs in a fresh child Python process. The child
measures its own RSS baseline and sampled peak RSS during model fitting.
`incremental_peak_rss_mb` is peak RSS minus the post-import/post-data-load
baseline. It is an approximation of additional resident memory, not an exact
allocation profiler. Child-process startup and input loading are excluded from
`runtime_seconds`; `worker_wall_seconds` includes the whole child process.

OPTICS is tested with BOTH:
  * cluster_method='xi' and an expanded xi/min_cluster_size grid
  * cluster_method='dbscan' with explicit eps extraction thresholds

All internal validity scores, including DBCV, are calculated on non-noise
points only. Always interpret them with noise_ratio and clustered_coverage.
No score alone selects a winning configuration.

DSCI, CDbw and VIASCKDE are not implemented here; no surrogate scores are used.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings

import numpy as np
import pandas as pd

# Expanded but tractable starter sensitivity grids.
DBSCAN_EPS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
DBSCAN_MIN_SAMPLES = [5, 10, 15]

OPTICS_XI = [0.01, 0.05, 0.10]
OPTICS_MIN_SAMPLES = [5, 10, 15]
OPTICS_MIN_CLUSTER_SIZE = [0.02, 0.05]
OPTICS_DBSCAN_EPS = [0.60, 0.80, 1.00]

HDBSCAN_MIN_CLUSTER_SIZE = [5, 10, 15, 20, 30]
HDBSCAN_MIN_SAMPLES = [3, 5, 10]
HDBSCAN_SELECTION_METHOD = ["eom", "leaf"]

RARE_CLUSTER_MAX_FRACTION = 0.05  # exploratory reporting threshold, not a clinical definition
RSS_SAMPLE_INTERVAL_SECONDS = 0.01


def build_grid(algorithms):
    configs = []
    if "DBSCAN" in algorithms:
        for eps, min_samples in itertools.product(DBSCAN_EPS, DBSCAN_MIN_SAMPLES):
            configs.append(("DBSCAN", {
                "eps": eps, "min_samples": min_samples, "metric": "euclidean"
            }))
    if "OPTICS" in algorithms:
        # Xi extraction: tests several steepness thresholds and minimum cluster sizes.
        for min_samples, xi, min_cluster_size in itertools.product(
            OPTICS_MIN_SAMPLES, OPTICS_XI, OPTICS_MIN_CLUSTER_SIZE
        ):
            configs.append(("OPTICS", {
                "cluster_method": "xi",
                "min_samples": min_samples,
                "xi": xi,
                "min_cluster_size": min_cluster_size,
                "metric": "minkowski",
                "p": 2,
            }))
        # DBSCAN extraction from the OPTICS reachability ordering.
        for min_samples, eps in itertools.product(
            OPTICS_MIN_SAMPLES, OPTICS_DBSCAN_EPS
        ):
            configs.append(("OPTICS", {
                "cluster_method": "dbscan",
                "min_samples": min_samples,
                "eps": eps,
                "metric": "minkowski",
                "p": 2,
            }))
    if "HDBSCAN" in algorithms:
        for min_cluster_size, min_samples, method in itertools.product(
            HDBSCAN_MIN_CLUSTER_SIZE,
            HDBSCAN_MIN_SAMPLES,
            HDBSCAN_SELECTION_METHOD,
        ):
            configs.append(("HDBSCAN", {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": method,
                "metric": "euclidean",
            }))
    return configs


def child_worker(args):
    """Execute exactly one configuration in an isolated process."""
    import psutil
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score, calinski_harabasz_score
    )

    try:
        import hdbscan
        from hdbscan.validity import validity_index as dbcv_function
    except Exception:
        hdbscan = None
        dbcv_function = None

    input_path = Path(args.input_file)
    data = pd.read_csv(input_path)
    features = json.loads(args.features_json)
    X = data[features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    params = json.loads(args.params_json)
    algorithm = args.algorithm

    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss / (1024 ** 2)
    peak_holder = {"value": baseline_rss}
    stop_event = threading.Event()

    def sample_rss():
        while not stop_event.is_set():
            try:
                current = process.memory_info().rss / (1024 ** 2)
                peak_holder["value"] = max(peak_holder["value"], current)
            except (psutil.Error, OSError):
                pass
            stop_event.wait(RSS_SAMPLE_INTERVAL_SECONDS)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        if algorithm == "DBSCAN":
            labels = DBSCAN(**params).fit_predict(X)
        elif algorithm == "OPTICS":
            labels = OPTICS(**params).fit_predict(X)
        elif algorithm == "HDBSCAN":
            if hdbscan is None:
                raise ImportError("The hdbscan package is not installed.")
            model_params = dict(params)
            metric = model_params.pop("metric", "euclidean")
            labels = hdbscan.HDBSCAN(metric=metric, **model_params).fit_predict(X)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        sampler.join()

    labels = np.asarray(labels, dtype=int)
    non_noise = labels != -1
    Xn, yn = X[non_noise], labels[non_noise]
    unique_clusters = np.unique(yn)
    n_clusters = int(len(unique_clusters))
    n_noise = int((~non_noise).sum())
    n_assigned = int(non_noise.sum())
    sizes = pd.Series(yn).value_counts().sort_values() if n_assigned else pd.Series(dtype=int)

    result = {
        "status": "COMPLETED",
        "error": "",
        "runtime_seconds": float(elapsed),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_clusters": n_clusters,
        "noise_count": n_noise,
        "noise_ratio": float(n_noise / len(X)),
        "clustered_count": n_assigned,
        "clustered_coverage": float(n_assigned / len(X)),
        "smallest_cluster_size": int(sizes.min()) if len(sizes) else 0,
        "median_cluster_size": float(sizes.median()) if len(sizes) else np.nan,
        "largest_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "rare_cluster_count_le_5pct": int((sizes <= max(1, int(np.floor(len(X) * RARE_CLUSTER_MAX_FRACTION)))).sum()),
        "patients_in_rare_clusters_le_5pct": int(sizes[sizes <= max(1, int(np.floor(len(X) * RARE_CLUSTER_MAX_FRACTION)))].sum()),
        "silhouette_non_noise": np.nan,
        "davies_bouldin_non_noise": np.nan,
        "calinski_harabasz_non_noise": np.nan,
        "dbcv_non_noise": np.nan,
        "dbcv_status": "NOT_COMPUTED",
        "validity_scope": "NON_NOISE_SUBSET_ONLY",
        "baseline_rss_mb": float(baseline_rss),
        "peak_rss_mb": float(peak_holder["value"]),
        "incremental_peak_rss_mb": float(max(0.0, peak_holder["value"] - baseline_rss)),
        "memory_method": "isolated_child_sampled_process_RSS_10ms",
    }

    if n_clusters >= 2 and n_assigned > n_clusters:
        try:
            result["silhouette_non_noise"] = float(silhouette_score(Xn, yn))
        except Exception as exc:
            result["silhouette_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        try:
            result["davies_bouldin_non_noise"] = float(davies_bouldin_score(Xn, yn))
        except Exception as exc:
            result["davies_bouldin_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        try:
            result["calinski_harabasz_non_noise"] = float(calinski_harabasz_score(Xn, yn))
        except Exception as exc:
            result["calinski_harabasz_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        if dbcv_function is not None:
            try:
                result["dbcv_non_noise"] = float(
                    dbcv_function(Xn, yn, metric="euclidean")
                )
                result["dbcv_status"] = "COMPUTED_ON_NON_NOISE_SUBSET"
            except Exception as exc:
                result["dbcv_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        else:
            result["dbcv_status"] = "HDBSCAN_PACKAGE_UNAVAILABLE"
    else:
        result["dbcv_status"] = "NOT_DEFINED_FEWER_THAN_TWO_NON_NOISE_CLUSTERS"

    result["cluster_sizes_json"] = json.dumps(
        {str(int(k)): int(v) for k, v in pd.Series(yn).value_counts().sort_index().items()}
    )
    result["labels"] = labels.tolist()

    Path(args.result_file).write_text(json.dumps(result), encoding="utf-8")


def run_one(root, dataset_key, info, algorithm, params, config_number, out_dir):
    input_file = root / "output" / "benchmark_v1" / info["file"]
    features_json = json.dumps(info["features"])
    safe_name = f"{dataset_key}_{algorithm.lower()}_{config_number:03d}"
    result_file = out_dir / "worker_results" / f"{safe_name}.json"
    label_file = out_dir / "labels" / f"{safe_name}_labels.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--input-file", str(input_file),
        "--features-json", features_json,
        "--algorithm", algorithm,
        "--params-json", json.dumps(params),
        "--result-file", str(result_file),
    ]
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=str(root)
    )
    worker_wall = time.perf_counter() - wall_start

    base_row = {
        "dataset": dataset_key,
        "algorithm": algorithm,
        "config_number": config_number,
        "parameters": json.dumps(params, sort_keys=True),
        "status": "COMPLETED" if completed.returncode == 0 and result_file.exists() else "ERROR",
        "worker_wall_seconds": float(worker_wall),
        "error": "",
    }
    if base_row["status"] == "COMPLETED":
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            labels = result.pop("labels")
            base_row.update(result)
            pd.DataFrame({
                "row_index": np.arange(len(labels)),
                "cluster_label": labels,
            }).to_csv(label_file, index=False)
            base_row["labels_file"] = str(label_file.relative_to(root))
        except Exception as exc:
            base_row["status"] = "ERROR"
            base_row["error"] = f"Could not read worker result: {type(exc).__name__}: {exc}"
    else:
        stderr = (completed.stderr or completed.stdout or "").strip()
        base_row["error"] = stderr[-4000:] if stderr else f"Worker exited with code {completed.returncode}"
    return base_row


def main():
    parser = argparse.ArgumentParser(description="Expanded isolated parameter sensitivity benchmark.")
    parser.add_argument("--algorithms", nargs="+", choices=["DBSCAN", "OPTICS", "HDBSCAN"],
                        default=["DBSCAN", "OPTICS", "HDBSCAN"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-file")
    parser.add_argument("--features-json")
    parser.add_argument("--algorithm")
    parser.add_argument("--params-json")
    parser.add_argument("--result-file")
    args = parser.parse_args()

    if args.worker:
        child_worker(args)
        return

    root = Path(__file__).resolve().parents[1]
    bench_dir = root / "output" / "benchmark_v1"
    manifest_path = bench_dir / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run src/prepare_benchmark_dataset_v1.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = root / "output" / "parameter_sensitivity_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    (output_dir / "worker_results").mkdir(exist_ok=True)

    configs = build_grid(set(args.algorithms))
    total = len(configs) * len(manifest["scaled_feature_files"])
    print("=" * 72)
    print("PARAMETER SENSITIVITY V2 — ISOLATED CHILD-PROCESS RUNS")
    print("=" * 72)
    print(f"Datasets: {len(manifest['scaled_feature_files'])}")
    print(f"Configurations per dataset: {len(configs)}")
    print(f"Total child-process runs: {total}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Output: {output_dir}")
    print("Note: runtime_seconds excludes process startup; worker_wall_seconds includes it.")
    print("")

    rows = []
    run_index = 0
    for dataset_key, info in manifest["scaled_feature_files"].items():
        print(f"\nDataset: {dataset_key} ({info['rows']} rows)")
        per_dataset_counts = {name: 0 for name in args.algorithms}
        for algorithm, params in configs:
            run_index += 1
            per_dataset_counts[algorithm] += 1
            print(
                f"[{run_index}/{total}] {algorithm} | "
                f"{json.dumps(params, sort_keys=True)}",
                flush=True,
            )
            row = run_one(
                root, dataset_key, info, algorithm, params,
                per_dataset_counts[algorithm], output_dir
            )
            rows.append(row)
            if row["status"] == "ERROR":
                print(f"    ERROR: {row['error'][:250]}")
            else:
                print(
                    f"    clusters={row.get('n_clusters')} "
                    f"noise={row.get('noise_ratio', np.nan):.1%} "
                    f"runtime={row.get('runtime_seconds', np.nan):.3f}s "
                    f"incremental_RSS={row.get('incremental_peak_rss_mb', np.nan):.1f}MB"
                )

    results = pd.DataFrame(rows)
    results_path = output_dir / "parameter_sensitivity_results.csv"
    results.to_csv(results_path, index=False)

    # Compact summaries retain coverage/noise alongside scores to discourage
    # selecting a configuration from validity scores alone.
    completed = results[results["status"] == "COMPLETED"].copy()
    summary_cols = [
        "dataset", "algorithm", "status", "n_clusters", "noise_ratio",
        "clustered_coverage", "smallest_cluster_size", "median_cluster_size",
        "largest_cluster_size", "rare_cluster_count_le_5pct",
        "patients_in_rare_clusters_le_5pct", "runtime_seconds",
        "incremental_peak_rss_mb", "silhouette_non_noise",
        "davies_bouldin_non_noise", "calinski_harabasz_non_noise",
        "dbcv_non_noise", "dbcv_status", "validity_scope",
    ]
    summary = (
        completed.groupby(["dataset", "algorithm"], dropna=False)
        .agg(
            configurations=("config_number", "count"),
            configurations_with_2plus_clusters=("n_clusters", lambda s: int((s >= 2).sum())),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_incremental_peak_rss_mb=("incremental_peak_rss_mb", "median"),
            median_noise_ratio=("noise_ratio", "median"),
            median_clustered_coverage=("clustered_coverage", "median"),
            median_dbcv_non_noise=("dbcv_non_noise", "median"),
        )
        .reset_index()
    )
    summary_path = output_dir / "parameter_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    run_metadata = {
        "benchmark": "Synthea Patient Phenotyping Parameter Sensitivity V2",
        "algorithms_requested": args.algorithms,
        "datasets": list(manifest["scaled_feature_files"].keys()),
        "configurations_per_dataset": len(configs),
        "total_runs_expected": total,
        "completed_runs": int((results["status"] == "COMPLETED").sum()),
        "error_runs": int((results["status"] != "COMPLETED").sum()),
        "rare_cluster_reporting_threshold": "cluster size <= floor(5% of dataset), minimum 1",
        "validity_metric_scope": "non-noise subset only",
        "memory_method": "one child process per configuration; sampled RSS every 10ms",
        "memory_caveat": "incremental_peak_rss_mb = sampled peak RSS - post-import/post-data-load baseline; sampling may miss brief peaks",
        "runtime_caveat": "runtime_seconds measures estimator fit/predict only; worker_wall_seconds includes child startup and input loading",
        "not_implemented": ["DSCI", "CDbw", "VIASCKDE"],
        "grid": {
            "DBSCAN_eps": DBSCAN_EPS,
            "DBSCAN_min_samples": DBSCAN_MIN_SAMPLES,
            "OPTICS_xi": OPTICS_XI,
            "OPTICS_min_samples": OPTICS_MIN_SAMPLES,
            "OPTICS_min_cluster_size": OPTICS_MIN_CLUSTER_SIZE,
            "OPTICS_dbscan_extraction_eps": OPTICS_DBSCAN_EPS,
            "HDBSCAN_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "HDBSCAN_min_samples": HDBSCAN_MIN_SAMPLES,
            "HDBSCAN_selection_method": HDBSCAN_SELECTION_METHOD,
        },
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)
    print(f"Completed: {run_metadata['completed_runs']}/{total}")
    print(f"Errors: {run_metadata['error_runs']}")
    print(f"Unified results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")
    if not summary.empty:
        print("\nMedian performance by dataset and algorithm:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
