# LD-MGF Research: Local Density – Multi-Granularity Fusion Index for Clustering Validation

This repository provides an empirical validation and benchmarking framework for unsupervised clustering in medical and epidemiological datasets, featuring the proposed **Local Density – Multi-Granularity Fusion (LD-MGF)** Cluster Validity Index (CVI).

---

## 🔬 Research Summary

### 1. Benchmark Datasets (Preserved Raw in `data/raw/`)
1. **UCI Heart Disease (Cleveland Clinic):** $N = 303$, $D = 13$ clinical features.
2. **NHANES 2017–2018 (CDC):** $N = 3,000$ adult records merged across demographics, examination, and laboratory panels ($D = 12$ cardiometabolic biomarkers).
3. **Diabetes 130-US Hospitals:** $N = 3,000$ patient encounters with 10-year clinical and medication features ($D = 43$).

*Note: In compliance with strict unsupervised evaluation standards, all disease outcome and diagnostic labels (`target`, `readmitted`, high-risk biomarker thresholds) are preserved for external verification and strictly excluded from the clustering feature space.*

### 2. Clustering Algorithms Evaluated
- **K-Means** (Partitioning / Centroid-based)
- **Agglomerative Clustering** (Hierarchical / Ward Linkage)
- **DBSCAN** (Density-based with noise detection)
- **OPTICS** (Ordering Points To Identify the Clustering Structure)
- **Gaussian Mixture Models (GMM)** (Probabilistic / Expectation-Maximization)

### 3. Cluster Validity Indices (CVIs) Evaluated
- **Silhouette Score (SC)** (Higher is better, $[-1, 1]$)
- **Davies–Bouldin Index (DBI)** (Lower is better, $[0, \infty)$)
- **Calinski–Harabasz Index (CHI)** (Higher is better, $[0, \infty)$)
- **Dunn Index (DI)** (Higher is better, $[0, \infty)$)
- **Density-Based Clustering Validation (DBCV)** (Higher is better, $[-1, 1]$)
- **Proposed LD-MGF Index** (Raw $[0, \infty)$ and Normalized $[0, 1)$)

---

## 📐 Mathematical Formulation of Proposed LD-MGF Index

### Assumptions
1. **Density Cohesion:** Cluster compactness is evaluated by local density uniformity and mode cohesion via adaptive $k$-NN kernels, rather than spatial proximity to an artificial global centroid.
2. **Multi-Granularity Margin:** Inter-cluster separation fuses three distinct scales:
   - *Fine Granularity:* 5th-percentile boundary distance ($\delta_{\text{fine}}$)
   - *Intermediate Granularity:* Distance between high-density core modes ($\delta_{\text{core}}$)
   - *Coarse Granularity:* Euclidean distance between global centroids ($\delta_{\text{macro}}$)
3. **Noise Robustness:** Clustering algorithms that isolate noise points (DBSCAN, OPTICS) are penalized proportionally by $(1 - N_{\text{noise}}/N)$.

### Formula
$$\text{LD-MGF} = \left( \frac{\min_{a \neq b} \left[ \alpha \delta_{\text{fine}}(C_a, C_b) + \beta \delta_{\text{core}}(C_a, C_b) + \gamma \delta_{\text{macro}}(C_a, C_b) \right]}{\frac{1}{K}\sum_{k=1}^K \text{Disp}_{LD}(C_k) + \epsilon} \right) \times \left(1 - \frac{N_{\text{noise}}}{N}\right)$$

$$\text{LD-MGF}_{\text{norm}} = \frac{\text{LD-MGF}}{1 + \text{LD-MGF}} \in [0, 1)$$

---

## 📁 Repository Structure

```
LD-MGF-Research/
├── data/
│   └── raw/                       # Preserved raw data
│       ├── diabetes_130/
│       ├── heart_disease/
│       └── nhanes_2017_2018/
├── notebooks/
│   └── LD_MGF_Experiments.ipynb  # Primary executed Jupyter Notebook with inline tables & plots
├── results/
│   ├── figures/                   # Exported high-resolution PCA and CVI comparison plots
│   └── metrics/
│       └── clustering_comparison_results.csv # Consolidated CVI benchmark results
├── src/
│   ├── clustering.py              # 5 clustering algorithms and suite runner
│   ├── data_inspection.py         # Non-destructive data schema inspection
│   ├── evaluation.py              # CVIs (Silhouette, DB, CH, Dunn, DBCV, LD-MGF)
│   └── preprocessing.py           # Imputation, encoding, and scaling pipelines
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🚀 How to Run

1. **Activate virtual environment:**
   ```bash
   .\.venv\Scripts\activate
   ```
2. **Open and run the notebook:**
   ```bash
   jupyter notebook notebooks/LD_MGF_Experiments.ipynb
   ```
   Or open directly in Google Colab / VS Code. All cells are pre-executed with complete inline tables, 2D PCA plots, and metric comparisons.
