
# Bayesian Hyperparameter Optimisation – PathSnap

## 1. Objective

The objective was to improve the PathSnap LightGBM classifier using Bayesian hyperparameter optimisation with Optuna.

The optimisation searched for hyperparameters that minimise mean five-fold cross-validation log loss while maintaining strong macro F1 and top-1 accuracy.

## 2. Optimisation methodology

- Model: LightGBM
- Optimisation framework: Optuna
- Sampler: Tree-structured Parzen Estimator (TPE)
- Pruner: MedianPruner
- Random seed: 42
- Cross-validation: 5-fold GroupKFold
- Grouping variable: trip_id
- Objective: Minimise mean validation log loss
- Maximum planned trials: 30
- Trials explored before stopping: 15
- Fully completed trials: 9
- Pruned trials: 6
- One additional trial was interrupted manually.

The student-test partition was not used during hyperparameter optimisation. It was used later for final model evaluation.

## 3. Hyperparameter search space

| Parameter | Search range |
|---|---|
| num_leaves | 15–255 (log scale) |
| learning_rate | 0.01–0.2 (log scale) |
| n_estimators | 100–600 |
| min_child_samples | 20–500 (log scale) |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| reg_lambda | 0.001–10 (log scale) |
| reg_alpha | 0.001–10 (log scale) |

Fixed parameters included random_state=42, n_jobs=-1, verbose=-1 and subsample_freq=1.

## 4. Best hyperparameters

The best trial was Trial 1.

| Parameter | Selected value |
|---|---:|
| num_leaves | 81 |
| learning_rate | 0.083411 |
| n_estimators | 110 |
| min_child_samples | 454 |
| subsample | 0.916221 |
| colsample_bytree | 0.606170 |
| reg_lambda | 0.005337 |
| reg_alpha | 0.005416 |

Best mean CV log loss: 0.0383003.

## 5. Cross-validation results

The selected configuration achieved:

- Mean CV macro F1: 0.9331
- CV macro F1 standard deviation: 0.0138
- Mean CV top-1 accuracy: 0.8761
- Mean CV route continuity: 0.9358
- Mean CV log loss: 0.0383

Five-fold GroupKFold was used to ensure that observations belonging to the same trip were not split between training and validation folds.

## 6. Baseline versus Bayesian-optimised LightGBM

| Metric | Baseline | Bayesian tuned |
|---|---:|---:|
| CV macro F1 | 0.9292 | 0.9331 |
| Student-test top-1 | 0.8698 | 0.8869 |
| Student-test macro F1 | 0.9322 | 0.9386 |
| Student-test log loss | 0.0570 | 0.0304 |
| Student-test route continuity | 0.8986 | 0.9321 |
| Fit time (seconds) | 14.1 | 17.0 |
| Inference time (ms/observation) | 0.0554 | 0.0137 |

The tuned model improved the measured student-test metrics compared with the baseline model.

## 7. HMM evaluation

The Bayesian-optimised LightGBM probabilities were evaluated with the PathSnap HMM decoder.

| Decoder | Top-1 accuracy | Route continuity |
|---|---:|---:|
| Classifier only | 88.69% | 93.21% |
| Classifier + default HMM | 91.96% | 97.59% |
| Classifier + tuned HMM | 92.69% | 98.09% |

The tuned HMM improved top-1 accuracy by approximately 3.00 percentage points compared with the tuned classifier alone.

The earlier baseline LightGBM with tuned HMM achieved approximately 92.75% top-1 accuracy. The Bayesian-tuned pipeline achieved approximately 92.69%, so it did not outperform that earlier HMM result in top-1 accuracy.

## 8. Deliverables

- scripts/tune_bayesian.py
- scripts/models/train_lightgbm_tuned.py
- results/bayesian/trials.csv
- results/bayesian/best_params.json
- results/bayesian/optuna_study.db
- results/experiments/lightgbm_tuned.json
- results/hmm_lightgbm_tuned/
- results/figures/bayesian_optimisation_history.png
- results/figures/bayesian_parameter_importance.png

## 9. Conclusion

Bayesian optimisation identified a LightGBM configuration that improved the baseline classifier's measured cross-validation and student-test metrics.

The best configuration achieved a mean CV log loss of 0.0383 and student-test top-1 accuracy of 88.69%.

HMM decoding further increased top-1 accuracy to approximately 92.69% and route continuity to approximately 98.09%.

The optimisation was stopped after 15 trials were explored due to time constraints. Therefore, the selected configuration is the best among the explored trials and is not guaranteed to be the global optimum.
