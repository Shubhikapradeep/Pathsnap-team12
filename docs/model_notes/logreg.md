# Model Notes: Logistic Regression

**Author**: Atharva (Team 12)  
**Family**: Linear / Probabilistic  

---

## Results Summary Table

| Run Name | Configuration | Train Macro F1 | CV Macro F1 | Student-Test Macro F1 | Student-Test Top-1 Acc | Student-Test Log Loss | Fit Time (s) | Inference Time (ms/obs) |
|---|---|---|---|---|---|---|---|---|
| `logreg` | C=1.0, max_iter=200, class_weight=None | 0.9180 | 0.9150 ± 0.0138 | 0.9141 | 0.8641 | 0.0442 | 7.1s | 0.0035 ms |
| `logreg_c0p1` | C=0.1, max_iter=200, class_weight=None | 0.9179 | 0.9150 ± 0.0137 | 0.9140 | 0.8640 | 0.0442 | 7.5s | 0.0034 ms |
| `logreg_balanced` | C=1.0, max_iter=200, class_weight='balanced' | 0.8164 | 0.8147 ± 0.0221 | 0.8207 | 0.8616 | 0.1102 | 9.0s | 0.0034 ms |

---

## A1 why suitable
TODO (my words)

---

## A2 strengths and limitations
TODO (my words)

---

## B5 hyperparameters and initial values

| Parameter | `logreg` (baseline) | `logreg_c0p1` | `logreg_balanced` | Notes |
|---|---|---|---|---|
| `C` (inverse regularization strength) | 1.0 | 0.1 | 1.0 | L2 penalty |
| `max_iter` | 200 | 200 | 200 | Maximum solver iterations |
| `class_weight` | None | None | "balanced" | Inversely proportional to class frequencies |
| `random_state` | 42 | 42 | 42 | Seed for reproducible solver initialization |
| Preprocessing | `StandardScaler` | `StandardScaler` | `StandardScaler` | Fit on train folds only |

TODO (my words)

---

## B6 loss/objective
TODO (my words)

---

## C7-8 student-test results

| Metric | `logreg` | `logreg_c0p1` | `logreg_balanced` |
|---|---|---|---|
| Top-1 Observation Accuracy | 0.8641 (86.41%) | 0.8640 (86.40%) | 0.8616 (86.16%) |
| Pair Macro F1 | 0.9141 | 0.9140 | 0.8207 |
| Pair Log Loss | 0.0442 | 0.0442 | 0.1102 |
| Route Continuity | 0.9217 | 0.9216 | 0.9250 |
| Inference Latency per Obs | 0.0035 ms | 0.0034 ms | 0.0034 ms |
| Test Confusion Matrix `[[TN, FP], [FN, TP]]` | `[[344159, 3009], [3197, 15946]]` | `[[344164, 3004], [3209, 15934]]` | `[[329000, 18168], [395, 18748]]` |

TODO (my words)

---

## C9 overfitting/underfitting evidence

| Split / Metric | `logreg` | `logreg_c0p1` | `logreg_balanced` |
|---|---|---|---|
| Train Macro F1 | 0.9180 | 0.9179 | 0.8164 |
| 5-Fold CV Macro F1 | 0.9150 ± 0.0138 | 0.9150 ± 0.0137 | 0.8147 ± 0.0221 |
| Student-Test Macro F1 | 0.9141 | 0.9140 | 0.8207 |
| Train vs Test F1 Gap | +0.0039 | +0.0039 | -0.0043 |

TODO (my words)
