# Model Notes: K-Nearest Neighbors (KNN)

**Author**: Atharva (Team 12)  
**Family**: Instance-based  

---

## Results Summary Table

| Run Name | Configuration | Train Macro F1 | CV Macro F1 | Student-Test Macro F1 | Student-Test Top-1 Acc | Student-Test Log Loss | Fit Time (s) | Inference Time (ms/obs) |
|---|---|---|---|---|---|---|---|---|
| `knn` | k=15, weights='distance', cap=200k | 1.0000 | 0.8998 ± 0.0135 | 0.9047 | 0.8448 | 0.1104 | 0.2s | 7.2061 ms |
| `knn_k5` | k=5, weights='distance', cap=200k | 1.0000 | 0.8957 ± 0.0138 | 0.9012 | 0.8338 | 0.2270 | 0.1s | 2.8298 ms |

---

## A1 why suitable
TODO (my words)

---

## A2 strengths and limitations
TODO (my words)

---

## B5 hyperparameters and initial values

| Parameter | `knn` (baseline) | `knn_k5` (variant) | Notes |
|---|---|---|---|
| `n_neighbors` ($k$) | 15 | 5 | Neighborhood size |
| `weights` | "distance" | "distance" | Inverse distance weighting |
| `max_train_rows` | 200,000 | 200,000 | Seeded subsample to bound indexing / query time |
| `n_jobs` | -1 | -1 | Parallel neighbor search across all CPU cores |
| Preprocessing | `StandardScaler` | `StandardScaler` | Mandatory for Euclidean distance metric |

TODO (my words)

---

## B6 loss/objective
TODO (my words)

---

## C7-8 student-test results

| Metric | `knn` (k=15) | `knn_k5` (k=5) |
|---|---|---|
| Top-1 Observation Accuracy | 0.8448 (84.48%) | 0.8338 (83.38%) |
| Pair Macro F1 | 0.9047 | 0.9012 |
| Pair Log Loss | 0.1104 | 0.2270 |
| Route Continuity | 0.9141 | 0.9050 |
| Inference Latency per Obs | 7.2061 ms | 2.8298 ms |
| Fraction of Pred Probs == 0.0 | 0.8880 (88.80%) | 0.9182 (91.82%) |
| Fraction of Pred Probs == 1.0 | 0.0172 (1.72%) | 0.0295 (2.95%) |
| Test Confusion Matrix `[[TN, FP], [FN, TP]]` | `[[344497, 2671], [4013, 15130]]` | `[[343990, 3178], [3868, 15275]]` |

TODO (my words)

---

## C9 overfitting/underfitting evidence

| Split / Metric | `knn` (k=15) | `knn_k5` (k=5) |
|---|---|---|
| Train Macro F1 | 1.0000 | 1.0000 |
| 5-Fold CV Macro F1 | 0.8998 ± 0.0135 | 0.8957 ± 0.0138 |
| Student-Test Macro F1 | 0.9047 | 0.9012 |
| Train vs Test F1 Gap | +0.0953 | +0.0988 |

TODO (my words)
