"""
Cluster stability analysis V1 for parameter_sensitivity_v2 outputs.

Run from project root:
    python src/analyze_cluster_stability_v1.py

Optional:
    python src/analyze_cluster_stability_v1.py --results-csv output/parameter_sensitivity_v2/parameter_sensitivity_results.csv
    python src/analyze_cluster_stability_v1.py --rare-max-fraction 0.05

Design:
- Compares only neighboring parameter configurations within the same dataset,
  algorithm, and (for OPTICS) extraction method.
- Neighbor means exactly one parameter changes by one adjacent grid step.
- Reports full-cohort ARI (noise is treated as a label), jointly-assigned ARI,
  assigned-patient overlap, and rare-cluster matching.
- Rare-cluster matching uses maximum Jaccard overlap between a rare cluster
  in one run and any cluster in its neighboring run. It also records whether
  the matched cluster is itself rare.
- Does not interpret one isolated high score as reproducibility.
- Does not claim clinical validity or use patient identifiers as features.

Important:
ARI on all patients can be strongly affected by shared noise assignments.
Always inspect it alongside assigned-patient overlap and jointly-assigned ARI.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


DEFAULT_RESULTS = Path("output/parameter_sensitivity_v2/parameter_sensitivity_results.csv")
LABEL_COLUMNS = ("cluster_label", "label", "labels")


def parse_params(value):
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_path(path_text: str) -> Path:
    # The sensitivity runner writes Windows-style relative paths in its CSV.
    return Path(str(path_text).replace("\\", "/"))


def load_labels(project_root: Path, relative_path: str, expected_n: int):
    path = normalize_path(relative_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    frame = pd.read_csv(path)
    label_col = next((c for c in LABEL_COLUMNS if c in frame.columns), None)
    if label_col is None:
        raise ValueError(f"No cluster-label column in {path}; columns={list(frame.columns)}")
    labels = frame[label_col].to_numpy(dtype=int)
    if len(labels) != expected_n:
        raise ValueError(
            f"Label length mismatch for {path}: got {len(labels)}, expected {expected_n}"
        )
    # row_index is emitted by the V2 runner; verify original row ordering.
    if "row_index" in frame.columns:
        expected = np.arange(expected_n)
        if not np.array_equal(frame["row_index"].to_numpy(), expected):
            raise ValueError(f"Non-sequential row_index in label file: {path}")
    return labels


def is_noise(label):
    return int(label) == -1


def cluster_memberships(labels):
    return {
        int(c): set(np.flatnonzero(labels == c).tolist())
        for c in np.unique(labels) if int(c) != -1
    }


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def pair_metrics(labels_a, labels_b):
    assigned_a = labels_a != -1
    assigned_b = labels_b != -1
    jointly_assigned = assigned_a & assigned_b
    union_assigned = assigned_a | assigned_b
    n = len(labels_a)

    ari_all = adjusted_rand_score(labels_a, labels_b)
    ari_joint = np.nan
    if jointly_assigned.sum() >= 2 and len(np.unique(labels_a[jointly_assigned])) >= 2 \
            and len(np.unique(labels_b[jointly_assigned])) >= 2:
        ari_joint = adjusted_rand_score(
            labels_a[jointly_assigned], labels_b[jointly_assigned]
        )

    assigned_overlap = (
        float(jointly_assigned.sum() / union_assigned.sum())
        if union_assigned.sum() else 1.0
    )
    coverage_a = float(assigned_a.mean())
    coverage_b = float(assigned_b.mean())
    return {
        "ari_all_patients": float(ari_all),
        "ari_jointly_assigned": float(ari_joint) if pd.notna(ari_joint) else np.nan,
        "jointly_assigned_count": int(jointly_assigned.sum()),
        "assigned_patient_jaccard": assigned_overlap,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "coverage_absolute_difference": abs(coverage_a - coverage_b),
    }


def rare_cluster_matches(labels_a, labels_b, rare_max_fraction):
    n = len(labels_a)
    threshold = max(1, int(np.floor(n * rare_max_fraction)))
    clusters_a = cluster_memberships(labels_a)
    clusters_b = cluster_memberships(labels_b)
    sizes_b = {k: len(v) for k, v in clusters_b.items()}
    rows = []

    for cluster_a, members_a in clusters_a.items():
        size_a = len(members_a)
        if size_a > threshold:
            continue

        best_cluster_b = None
        best_jaccard = 0.0
        best_intersection = 0
        for cluster_b, members_b in clusters_b.items():
            overlap = len(members_a & members_b)
            score = jaccard(members_a, members_b)
            if score > best_jaccard:
                best_jaccard = score
                best_cluster_b = cluster_b
                best_intersection = overlap

        matched_size = sizes_b.get(best_cluster_b, 0) if best_cluster_b is not None else 0
        rows.append({
            "source_cluster_label": int(cluster_a),
            "source_cluster_size": size_a,
            "source_cluster_fraction": size_a / n,
            "best_matched_cluster_label": (
                int(best_cluster_b) if best_cluster_b is not None else np.nan
            ),
            "best_match_size": int(matched_size),
            "intersection_patients": int(best_intersection),
            "best_match_jaccard": float(best_jaccard),
            "matched_cluster_is_rare": bool(
                best_cluster_b is not None and matched_size <= threshold
            ),
            "source_cluster_reproduced_jaccard_0_5": bool(best_jaccard >= 0.5),
            "source_cluster_reproduced_jaccard_0_75": bool(best_jaccard >= 0.75),
        })
    return rows, threshold


def parameter_neighbors(records, algorithm):
    """
    Return pairs differing by exactly one adjacent numeric/categorical grid step.
    The adjacency is inferred from the values present in the results CSV.
    """
    pairs = []
    param_keys = sorted(set().union(*(r["_params"].keys() for r in records)))
    if algorithm in ("OPTICS", "HDBSCAN"):
        # Extraction/selection modes are categorical; hold them fixed while
        # comparing numeric parameter neighbors.
        categorical_mode = "cluster_method" if algorithm == "OPTICS" else "cluster_selection_method"
        param_keys = [k for k in param_keys if k != categorical_mode]

    values_by_key = {}
    for key in param_keys:
        values = []
        for rec in records:
            value = rec["_params"].get(key)
            if value is not None and value not in values:
                values.append(value)
        try:
            values_by_key[key] = sorted(values, key=float)
        except (TypeError, ValueError):
            values_by_key[key] = sorted(values, key=str)

    for i, j in combinations(range(len(records)), 2):
        a, b = records[i], records[j]
        pa, pb = a["_params"], b["_params"]

        if algorithm == "OPTICS" and pa.get("cluster_method") != pb.get("cluster_method"):
            continue
        if algorithm == "HDBSCAN" and pa.get("cluster_selection_method") != pb.get("cluster_selection_method"):
            continue

        changed = []
        adjacent = True
        for key in param_keys:
            va, vb = pa.get(key), pb.get(key)
            if va == vb:
                continue
            changed.append(key)
            vals = values_by_key.get(key, [])
            try:
                ia, ib = vals.index(va), vals.index(vb)
                if abs(ia - ib) != 1:
                    adjacent = False
            except ValueError:
                adjacent = False
        if adjacent and len(changed) == 1:
            pairs.append((a, b, changed[0]))
    return pairs


def summarize_configuration(group):
    return pd.Series({
        "runs": len(group),
        "median_neighbor_ari_all": group["ari_all_patients"].median(),
        "median_neighbor_ari_joint": group["ari_jointly_assigned"].median(),
        "median_assigned_patient_jaccard": group["assigned_patient_jaccard"].median(),
        "median_coverage_difference": group["coverage_absolute_difference"].median(),
        "neighbor_pairs_ari_ge_0_75": int((group["ari_all_patients"] >= 0.75).sum()),
        "neighbor_pairs_ari_ge_0_90": int((group["ari_all_patients"] >= 0.90).sum()),
        "rare_clusters_observed": int(group["rare_cluster_count_a"].sum()),
        "rare_cluster_median_best_jaccard": group["rare_cluster_mean_best_jaccard"].median(),
    })


def main():
    parser = argparse.ArgumentParser(
        description="Compare nearby density-clustering configurations for stability."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--rare-max-fraction", type=float, default=0.05)
    parser.add_argument("--min-coverage-for-stability-summary", type=float, default=0.10)
    args = parser.parse_args()

    if not (0 < args.rare_max_fraction <= 0.5):
        parser.error("--rare-max-fraction must be in (0, 0.5].")

    project_root = Path(__file__).resolve().parents[1]
    results_path = args.results_csv
    if not results_path.is_absolute():
        results_path = project_root / results_path
    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find {results_path}. Run run_parameter_sensitivity_v2.py first."
        )

    source = pd.read_csv(results_path)
    required = {
        "dataset", "algorithm", "config_number", "parameters", "status",
        "n_samples", "labels_file"
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {sorted(missing)}")

    source = source[source["status"] == "COMPLETED"].copy()
    if source.empty:
        raise ValueError("No completed configurations found.")

    source["_params"] = source["parameters"].map(parse_params)
    output_dir = results_path.parent / "cluster_stability_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels once. A failure is reported for that configuration and skipped.
    labels_by_key = {}
    load_errors = []
    for _, row in source.iterrows():
        key = (row["dataset"], row["algorithm"], int(row["config_number"]))
        try:
            labels_by_key[key] = load_labels(
                project_root, row["labels_file"], int(row["n_samples"])
            )
        except Exception as exc:
            load_errors.append({
                "dataset": row["dataset"], "algorithm": row["algorithm"],
                "config_number": int(row["config_number"]),
                "labels_file": row["labels_file"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    pair_rows = []
    rare_rows = []
    groups = ["dataset", "algorithm"]
    for (dataset, algorithm), group in source.groupby(groups, sort=True):
        records = []
        for _, row in group.iterrows():
            key = (dataset, algorithm, int(row["config_number"]))
            if key not in labels_by_key:
                continue
            rec = row.to_dict()
            rec["_key"] = key
            records.append(rec)

        for a, b, changed_param in parameter_neighbors(records, algorithm):
            la, lb = labels_by_key[a["_key"]], labels_by_key[b["_key"]]
            metrics = pair_metrics(la, lb)
            rare_ab, rare_threshold = rare_cluster_matches(
                la, lb, args.rare_max_fraction
            )
            rare_ba, _ = rare_cluster_matches(
                lb, la, args.rare_max_fraction
            )

            def param_string(rec):
                return json.dumps(rec["_params"], sort_keys=True)

            mean_ab = (
                float(np.mean([x["best_match_jaccard"] for x in rare_ab]))
                if rare_ab else np.nan
            )
            mean_ba = (
                float(np.mean([x["best_match_jaccard"] for x in rare_ba]))
                if rare_ba else np.nan
            )
            pair_id = (
                f"{dataset}__{algorithm}__"
                f"{int(a['config_number']):03d}_{int(b['config_number']):03d}"
            )
            pair_rows.append({
                "pair_id": pair_id,
                "dataset": dataset,
                "algorithm": algorithm,
                "config_a": int(a["config_number"]),
                "config_b": int(b["config_number"]),
                "parameters_a": param_string(a),
                "parameters_b": param_string(b),
                "changed_parameter": changed_param,
                **metrics,
                "rare_cluster_threshold_patients": rare_threshold,
                "rare_cluster_count_a": len(rare_ab),
                "rare_cluster_count_b": len(rare_ba),
                "rare_cluster_mean_best_jaccard": mean_ab,
                "rare_cluster_mean_best_jaccard_reverse": mean_ba,
                "rare_clusters_matched_jaccard_ge_0_5": int(
                    sum(x["best_match_jaccard"] >= 0.5 for x in rare_ab)
                ),
                "rare_clusters_matched_to_rare_cluster": int(
                    sum(x["matched_cluster_is_rare"] for x in rare_ab)
                ),
                "rare_cluster_reproducibility_scope": "directional_best_Jaccard",
            })

            for direction, from_run, to_run, details in (
                ("A_TO_B", a, b, rare_ab), ("B_TO_A", b, a, rare_ba)
            ):
                for detail in details:
                    rare_rows.append({
                        "pair_id": pair_id,
                        "dataset": dataset,
                        "algorithm": algorithm,
                        "direction": direction,
                        "from_config": int(from_run["config_number"]),
                        "to_config": int(to_run["config_number"]),
                        "from_parameters": param_string(from_run),
                        "to_parameters": param_string(to_run),
                        **detail,
                        "rare_threshold_patients": rare_threshold,
                    })

    pair_df = pd.DataFrame(pair_rows)
    rare_df = pd.DataFrame(rare_rows)

    pair_path = output_dir / "neighbor_pair_stability.csv"
    rare_path = output_dir / "rare_cluster_reproducibility.csv"
    pair_df.to_csv(pair_path, index=False)
    rare_df.to_csv(rare_path, index=False)

    if not pair_df.empty:
        summary_rows = []
        for (dataset_name, algorithm_name), group_frame in pair_df.groupby(
            ["dataset", "algorithm"], sort=True
        ):
            row_summary = summarize_configuration(group_frame)
            row_summary["dataset"] = dataset_name
            row_summary["algorithm"] = algorithm_name
            summary_rows.append(row_summary)
        summary = pd.DataFrame(summary_rows)
        # This secondary table avoids hiding instability caused by very low coverage.
        eligible = pair_df[
            (pair_df["coverage_a"] >= args.min_coverage_for_stability_summary)
            & (pair_df["coverage_b"] >= args.min_coverage_for_stability_summary)
        ]
        if not eligible.empty:
            eligible_summary = (
                eligible.groupby(["dataset", "algorithm"], sort=True)
                .agg(
                    eligible_neighbor_pairs=("pair_id", "count"),
                    eligible_median_ari_all=("ari_all_patients", "median"),
                    eligible_median_ari_joint=("ari_jointly_assigned", "median"),
                    eligible_median_assigned_jaccard=("assigned_patient_jaccard", "median"),
                    eligible_median_rare_match_jaccard=("rare_cluster_mean_best_jaccard", "median"),
                ).reset_index()
            )
            summary = summary.merge(eligible_summary, on=["dataset", "algorithm"], how="left")
        summary_path = output_dir / "stability_summary.csv"
        summary.to_csv(summary_path, index=False)
    else:
        summary_path = output_dir / "stability_summary.csv"
        pd.DataFrame().to_csv(summary_path, index=False)

    load_error_path = output_dir / "label_loading_errors.csv"
    pd.DataFrame(load_errors).to_csv(load_error_path, index=False)

    metadata = {
        "analysis": "Cluster Stability V1",
        "input_results_csv": str(results_path),
        "completed_configurations": int(len(source)),
        "labels_loaded": int(len(labels_by_key)),
        "label_load_errors": int(len(load_errors)),
        "neighbor_definition": (
            "Same dataset and algorithm; OPTICS extraction method held fixed; "
            "exactly one parameter changes by one adjacent observed grid step."
        ),
        "rare_cluster_definition": (
            f"Cluster size <= floor(dataset_size * {args.rare_max_fraction}), minimum 1."
        ),
        "rare_cluster_matching": (
            "For each rare cluster in source run, choose the target cluster with "
            "maximum patient-set Jaccard; report if matched target is also rare."
        ),
        "stability_metrics": {
            "ari_all_patients": "ARI on all patients; noise label -1 is treated as a cluster label.",
            "ari_jointly_assigned": "ARI only among patients assigned (non-noise) in both runs.",
            "assigned_patient_jaccard": "Intersection/union of patients assigned in either run.",
            "rare_cluster_best_match_jaccard": "Directional best Jaccard match for each rare cluster.",
        },
        "minimum_coverage_for_secondary_summary": args.min_coverage_for_stability_summary,
        "caveats": [
            "ARI can be inflated by shared noise assignments; interpret alongside coverage and assigned overlap.",
            "Jaccard best matching does not enforce one-to-one cluster matching.",
            "Nearby parameter agreement measures local parameter stability, not bootstrap or external reproducibility.",
            "Rare clusters are defined by an exploratory size threshold, not clinical rarity.",
            "Patient identifiers are not read or used; labels are compared by row order.",
        ],
    }
    metadata_path = output_dir / "stability_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 72)
    print("CLUSTER STABILITY V1 COMPLETE")
    print("=" * 72)
    print(f"Completed configurations: {len(source)}")
    print(f"Labels loaded: {len(labels_by_key)}")
    print(f"Label loading errors: {len(load_errors)}")
    print(f"Neighbor pairs analyzed: {len(pair_df)}")
    print(f"Rare-cluster directional records: {len(rare_df)}")
    print(f"\nNeighbor-pair results: {pair_path}")
    print(f"Rare-cluster results: {rare_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Label errors: {load_error_path}")
    if not pair_df.empty:
        print("\nMedian neighbor stability by dataset and algorithm:")
        display_cols = [
            "dataset", "algorithm", "runs", "median_neighbor_ari_all",
            "median_neighbor_ari_joint", "median_assigned_patient_jaccard",
            "median_coverage_difference", "rare_cluster_median_best_jaccard"
        ]
        print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
