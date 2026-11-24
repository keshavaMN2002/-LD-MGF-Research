"""
Baseline clustering benchmark: DBSCAN, OPTICS, HDBSCAN.

Run from project root:
    python src/run_baseline_clustering_v1.py

Required:
    pip install -r requirements_benchmark_v1.txt

Optional:
    pip install hdbscan
    DBCV is attempted through hdbscan.validity.validity_index.

Important:
- Noise label -1 is excluded from conventional internal metrics.
- DBCV is attempted on the non-noise subset only, and is labelled accordingly.
- DSCI, CDbw and VIASCKDE are NOT fabricated or approximated here. This runner
  records them as NOT_CONFIGURED until their exact published implementations
  are integrated and verified.
- Parameter grids are starting baselines, not claimed optimal settings.
"""
from pathlib import Path
import json
import time
import threading
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

try:
    import psutil
except ImportError:
    psutil = None

try:
    import hdbscan
    from hdbscan.validity import validity_index as hdbscan_dbcv
except Exception:
    hdbscan = None
    hdbscan_dbcv = None


# Modest, explicit baseline grids. Expand only after inspecting results.
DBSCAN_GRID = [
    {"eps": 0.5, "min_samples": 5},
    {"eps": 0.7, "min_samples": 5},
    {"eps": 1.0, "min_samples": 5},
    {"eps": 1.0, "min_samples": 10},
    {"eps": 1.2, "min_samples": 10},
]
OPTICS_GRID = [
    {"min_samples": 5, "xi": 0.05, "min_cluster_size": 0.05},
    {"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.05},
    {"min_samples": 10, "xi": 0.10, "min_cluster_size": 0.05},
]
HDBSCAN_GRID = [
    {"min_cluster_size": 5, "min_samples": 5},
    {"min_cluster_size": 10, "min_samples": 5},
    {"min_cluster_size": 15, "min_samples": 5},
    {"min_cluster_size": 10, "min_samples": 10},
]

class PeakRSS:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = None
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        if psutil is None:
            return
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                rss = process.memory_info().rss / (1024 ** 2)
                self.peak = rss if self.peak is None else max(self.peak, rss)
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self):
        if psutil is not None:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

def score_labels(X, labels):
    labels = np.asarray(labels)
    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    noise_count = int((~non_noise).sum())
    result = {
        "n_clusters": int(n_clusters),
        "noise_count": noise_count,
        "noise_ratio": float(noise_count / len(labels)),
        "silhouette_non_noise": np.nan,
        "davies_bouldin_non_noise": np.nan,
        "calinski_harabasz_non_noise": np.nan,
        "dbcv_non_noise": np.nan,
        "dbcv_status": "NOT_AVAILABLE",
    }

    if n_clusters >= 2 and int(non_noise.sum()) > n_clusters:
        Xn = X[non_noise]
        yn = labels[non_noise]
        try:
            result["silhouette_non_noise"] = float(silhouette_score(Xn, yn))
        except Exception:
            pass
        try:
            result["davies_bouldin_non_noise"] = float(
                davies_bouldin_score(Xn, yn)
            )
        except Exception:
            pass
        try:
            result["calinski_harabasz_non_noise"] = float(
                calinski_harabasz_score(Xn, yn)
            )
        except Exception:
            pass

        if hdbscan_dbcv is not None:
            try:
                result["dbcv_non_noise"] = float(
                    hdbscan_dbcv(Xn, yn, metric="euclidean")
                )
                result["dbcv_status"] = "COMPUTED_ON_NON_NOISE_SUBSET"
            except Exception as exc:
                result["dbcv_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        else:
            result["dbcv_status"] = "INSTALL_HDBSCAN_TO_ENABLE_DBCV"
    elif n_clusters < 2:
        result["dbcv_status"] = "NOT_DEFINED_FEWER_THAN_TWO_CLUSTERS"

    # Cluster-size distribution excludes noise.
    sizes = pd.Series(labels[non_noise]).value_counts()
    result["smallest_cluster_size"] = int(sizes.min()) if len(sizes) else 0
    result["largest_cluster_size"] = int(sizes.max()) if len(sizes) else 0
    result["median_cluster_size"] = float(sizes.median()) if len(sizes) else np.nan
    result["cluster_sizes"] = json.dumps(
        {str(k): int(v) for k, v in sizes.sort_index().items()}
    )
    return result

def get_algorithms():
    algorithms = {
        "DBSCAN": [
            (p, lambda X, p=p: DBSCAN(**p).fit_predict(X))
            for p in DBSCAN_GRID
        ],
        "OPTICS": [
            (p, lambda X, p=p: OPTICS(**p, metric="minkowski", p=2).fit_predict(X))
            for p in OPTICS_GRID
        ],
    }
    if hdbscan is not None:
        algorithms["HDBSCAN"] = [
            (p, lambda X, p=p: hdbscan.HDBSCAN(**p, metric="euclidean").fit_predict(X))
            for p in HDBSCAN_GRID
        ]
    else:
        algorithms["HDBSCAN"] = []
    return algorithms

def main():
    root = Path(__file__).resolve().parents[1]
    bench = root / "output" / "benchmark_v1"
    out = root / "output" / "baseline_results_v1"
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = bench / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Benchmark inputs not found. Run prepare_benchmark_dataset_v1.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    algorithms = get_algorithms()
    result_rows = []
    label_dir = out / "labels"
    label_dir.mkdir(exist_ok=True)

    for dataset_key, info in manifest["scaled_feature_files"].items():
        data = pd.read_csv(bench / info["file"])
        id_col = data.columns[0]
        ids = data[id_col].astype(str).to_numpy()
        features = info["features"]
        X = data[features].to_numpy(dtype=float)

        print(f"\nDataset: {dataset_key} | rows={len(X)} | dimensions={X.shape[1]}")

        for algorithm, configs in algorithms.items():
            if not configs:
                result_rows.append({
                    "dataset": dataset_key,
                    "algorithm": algorithm,
                    "parameters": "NOT_RUN",
                    "status": "PACKAGE_NOT_INSTALLED",
                    "error": "Install hdbscan to run HDBSCAN and enable DBCV.",
                })
                continue

            for config_index, (params, fit_predict) in enumerate(configs, start=1):
                row = {
                    "dataset": dataset_key,
                    "algorithm": algorithm,
                    "parameters": json.dumps(params, sort_keys=True),
                    "status": "COMPLETED",
                    "error": "",
                }
                try:
                    with PeakRSS() as monitor:
                        start = time.perf_counter()
                        labels = fit_predict(X)
                        elapsed = time.perf_counter() - start

                    row.update(score_labels(X, labels))
                    row["runtime_seconds"] = float(elapsed)
                    row["peak_process_rss_mb"] = (
                        float(monitor.peak) if monitor.peak is not None else np.nan
                    )
                    row["memory_measurement"] = (
                        "sampled_process_RSS" if psutil is not None else "psutil_not_installed"
                    )

                    label_file = (
                        f"{dataset_key}_{algorithm.lower()}_{config_index}_labels.csv"
                    )
                    pd.DataFrame({
                        id_col: ids,
                        "cluster_label": labels,
                    }).to_csv(label_dir / label_file, index=False)
                    row["labels_file"] = str(Path("labels") / label_file)

                except Exception as exc:
                    row["status"] = "ERROR"
                    row["error"] = f"{type(exc).__name__}: {exc}"

                result_rows.append(row)
                print(
                    f"  {algorithm} {params} -> {row['status']}"
                    + (
                        f", clusters={row.get('n_clusters')}, "
                        f"noise={row.get('noise_ratio', np.nan):.3f}, "
                        f"time={row.get('runtime_seconds', np.nan):.3f}s"
                        if row["status"] == "COMPLETED" else ""
                    )
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(out / "baseline_metrics.csv", index=False)

    index_status = pd.DataFrame([
        {"index": "DBCV", "status": (
            "ENABLED" if hdbscan_dbcv is not None else "NOT_AVAILABLE_INSTALL_HDBSCAN"
        ), "note": "Computed on non-noise subset; report this scope explicitly."},
        {"index": "DSCI", "status": "NOT_CONFIGURED",
         "note": "Integrate exact published implementation; do not substitute another metric."},
        {"index": "CDbw", "status": "NOT_CONFIGURED",
         "note": "Integrate exact published implementation; verify preprocessing and noise handling."},
        {"index": "VIASCKDE", "status": "NOT_CONFIGURED",
         "note": "Integrate exact published implementation; verify bandwidth and computational settings."},
        {"index": "Silhouette", "status": "COMPUTED_WHEN_DEFINED",
         "note": "Supplementary metric on non-noise points only; not density-based."},
        {"index": "Davies-Bouldin", "status": "COMPUTED_WHEN_DEFINED",
         "note": "Supplementary metric on non-noise points only; not density-based."},
        {"index": "Calinski-Harabasz", "status": "COMPUTED_WHEN_DEFINED",
         "note": "Supplementary metric on non-noise points only; not density-based."},
    ])
    index_status.to_csv(out / "validity_index_status.csv", index=False)

    print("\n" + "=" * 68)
    print("BASELINE RUN COMPLETE")
    print("=" * 68)
    print(f"Results: {out / 'baseline_metrics.csv'}")
    print(f"Index status: {out / 'validity_index_status.csv'}")
    if hdbscan is None:
        print("NOTE: HDBSCAN and DBCV were not run. Install hdbscan and rerun.")
    print("DSCI, CDbw and VIASCKDE remain unconfigured pending verified implementations.")

if __name__ == "__main__":
    main()
