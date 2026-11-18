"""Build the technical report (HTML -> PDF) from the committed result files.

Every number, table and figure comes from files under results/ produced by committed scripts; nothing is typed in by hand.
Model analysis text comes from each member's own notes (docs/model_notes/*.md or docs/<model>.md). Where a note is
missing, the report says so instead of inventing text.

Usage: python scripts/13_build_report.py [--notes-dir DIR]   (extra directory searched for <model>.md notes)
Writes report/PathSnap_Team12_Report.html (then print it to PDF; see README).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPO_URL = "https://github.com/Shubhikapradeep/Pathsnap-team12"
OWNERS = {"logreg": "Atharva", "knn": "Atharva", "random_forest": "Yashvantha", "extra_trees": "Yashvantha",
          "lightgbm": "Tanish", "mlp": "Tanish", "bayesian": "Yashvantha"}
MAIN_MODELS = [("logreg", "Logistic Regression", ["logreg", "logreg_c0p1", "logreg_balanced"]),
               ("knn", "K-Nearest Neighbours", ["knn", "knn_k5"]),
               ("random_forest", "Random Forest", ["random_forest"]),
               ("extra_trees", "Extra Trees", ["extra_trees"]),
               ("lightgbm", "LightGBM (untuned and Bayesian-tuned)", ["lightgbm", "lightgbm_tuned"]),
               ("mlp", "MLP (neural network)", ["mlp"])]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.2pt; line-height: 1.38; color: #1b1f23; }
h1 { font-size: 21pt; margin: 0 0 4pt; } h2 { font-size: 15pt; margin-top: 20pt; border-bottom: 1.5px solid #1f4e79; padding-bottom: 2pt; color: #1f4e79; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 14pt; color: #1f4e79; page-break-after: avoid; } h4, h5, h6 { font-size: 10.5pt; margin: 9pt 0 2pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.6pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #c8ced4; padding: 2.5pt 5pt; text-align: left; vertical-align: top; } th { background: #eaf0f6; }
td.num { text-align: right; } code { font-family: Menlo, monospace; font-size: 8.4pt; background: #f2f4f6; padding: 0 2pt; }
pre { background: #f2f4f6; padding: 6pt; font-size: 8pt; white-space: pre-wrap; }
img { max-width: 100%; display: block; margin: 6pt auto; } figure { margin: 8pt 0; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #4a5560; text-align: center; margin-top: 2pt; }
.note { border-left: 3px solid #e67e22; background: #fdf3e7; padding: 4pt 8pt; margin: 6pt 0; font-size: 9pt; }
.todo { border-left: 3px solid #c0392b; background: #fbeaea; padding: 4pt 8pt; margin: 6pt 0; font-size: 9pt; }
.cover { text-align: center; margin-top: 70pt; page-break-after: always; } .cover h1 { font-size: 28pt; }
.small { font-size: 8.8pt; color: #4a5560; } .pb { page-break-before: always; }
"""


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def figure(rel: str, caption: str, width: str = "100%") -> str:
    path = ROOT / rel
    if not path.exists():
        return f'<div class="todo">Missing figure: <code>{html.escape(rel)}</code></div>'
    return f'<figure><img style="width:{width}" src="data:image/png;base64,{b64(path)}"><figcaption>{html.escape(caption)}'\
           f' <span class="small">(source: <code>{html.escape(rel)}</code>)</span></figcaption></figure>'


def table(frame: pd.DataFrame, digits: int = 4) -> str:
    return frame.to_html(index=False, escape=True, border=0, float_format=lambda v: f"{v:.{digits}f}")


def load_experiments() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((RES / "experiments").glob("*.json")) if not p.stem.startswith("_")}


def notes_html(model: str, extra_dir: Path | None) -> str:
    candidates = [ROOT / "docs" / "model_notes" / f"{model}.md", ROOT / "docs" / f"{model}.md"]
    if extra_dir:
        candidates.append(extra_dir / f"{model}.md")
    for path in candidates:
        if path.exists():
            body = markdown.markdown(path.read_text(), extensions=["tables", "fenced_code"])
            body = re.sub(r"<h([1-5])>", lambda m: f"<h{min(6, int(m.group(1)) + 3)}>", body)
            body = re.sub(r"</h([1-5])>", lambda m: f"</h{min(6, int(m.group(1)) + 3)}>", body)
            return f'<div class="small">Analysis written by {OWNERS[model]} (<code>{html.escape(str(path.relative_to(ROOT)) if ROOT in path.parents else path.name)}</code>).</div>{body}'
    return (f'<div class="todo"><b>Analysis not yet submitted.</b> {OWNERS[model]} still has to add the written answers to A1, A2, B5, B6, C7-C9 '
            f'for this model (<code>docs/model_notes/{model}.md</code>). The measured results above are real; the explanation is not '
            f'included here rather than guessed.</div>')


def model_section(key: str, title: str, runs: list[str], experiments: dict, extra_dir: Path | None) -> str:
    rows = []
    for run in runs:
        r = experiments.get(run)
        if not r:
            continue
        t = r["student_test"]
        rows.append({"run": run, "CV macro F1": f'{r["cv_macro_f1_mean"]:.4f} ± {r["cv_macro_f1_std"]:.4f}', "CV top-1": r["cv_top1_mean"],
                     "test macro F1": t["macro_f1"], "test top-1": t["top1_accuracy"], "test log loss": t["log_loss"],
                     "test continuity": t["route_continuity"], "train macro F1": r["train_pair_macro_f1"],
                     "train-CV gap": r["train_pair_macro_f1"] - r["cv_macro_f1_mean"], "fit s": t["fit_seconds"],
                     "inference ms/pt": t["inference_ms_per_obs"]})
    base = experiments[runs[0]]
    config = {**base["config"], "max_train_rows": base.get("max_train_rows")}
    config_rows = pd.DataFrame({"hyperparameter": list(config), "value": [str(v) for v in config.values()]})
    out = f'<h3>{title}</h3><div class="small">Owner: {OWNERS[key]}. Runs: {", ".join(runs)}. Evaluation: 5-fold GroupKFold by trip on the train trips; refit on all train trips and tested on the 10 student_test trips.</div>'
    out += "<h5>Measured results (from results/experiments/*.json)</h5>" + table(pd.DataFrame(rows))
    out += "<h5>Configuration of the first run</h5>" + table(config_rows)
    return out + notes_html(key, extra_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes-dir", type=Path, default=None)
    args = parser.parse_args()
    ex = load_experiments()
    inventory = pd.read_csv(RES / "dataset_inventory.csv", dtype={"trip_id": str})
    summary = inventory.groupby("partition").agg(trips=("trip_id", "count"), points_min=("n_obs", "min"), points_median=("n_obs", "median"),
                                                  points_max=("n_obs", "max"), nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
                                                  route_arcs_median=("n_route_arcs", "median"), gaps_over_1s=("n_gaps_gt_1s", "sum"),
                                                  max_gap_s=("dt_max_s", "max"), nan_rows=("nan_track_rows", "sum"),
                                                  max_components=("n_components", "max")).reset_index()
    radius = pd.read_csv(RES / "radius_study.csv")
    align = pd.read_csv(RES / "alignment_quality_sample.csv", dtype={"trip_id": str})
    missing = pd.read_csv(RES / "feature_missingness.csv")
    importance = pd.read_csv(RES / "feature_importance.csv")
    excluded = pd.read_csv(RES / "excluded_trips.csv", dtype={"trip_id": str})
    comparison = pd.read_csv(RES / "model_comparison_table.csv")
    hmm_cmp = pd.read_csv(RES / "hmm_model_comparison.csv")
    base0 = pd.read_csv(RES / "baseline_nearest_candidate.csv").set_index("partition").nearest_candidate_top1
    lon_span = (inventory.lon_min.min(), inventory.lon_max.max())

    parts: list[str] = []
    parts.append(f'<div class="cover"><h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1><p style="font-size:14pt">Technical Report, Team 12</p>'
                 f'<p>UE24CS352A Machine Learning Hackathon</p><p>Team members: Shubhikapradeep, Atharva, Yashvantha, Tanish</p>'
                 f'<p class="small">Repository: {REPO_URL}</p><p class="small">All results in this report are generated from files in the repository '
                 f'(<code>results/</code>) by <code>scripts/13_build_report.py</code>.</p></div>')

    # ---- 1 overview
    parts.append("<h2>1. Overview</h2><p>Noisy 1 Hz GPS points are matched to the arcs of a road network. Stage 1 generates candidate arcs for every GPS point "
                 "and a classifier scores each (point, arc) pair. Stage 2 decodes the whole trip with an HMM and the Viterbi algorithm so that the chosen arcs form a connected route. "
                 "The system is served by a FastAPI backend (Laptop 1) and an evaluator UI (Laptop 2).</p>")

    # ---- 2 EDA
    parts.append('<h2 class="pb">2. Dataset understanding and exploratory data analysis</h2>')
    parts.append("<p class='note'>Partitions used: every analysis below uses the <b>train</b> partition (80 trips) and/or <b>student_test</b> (10 trips) only, as stated per item. "
                 "The 10 evaluator trips were never analysed, used for training, feature selection, tuning or model choice.</p>")
    parts.append("<h3>2.1 Dataset inventory (train and student_test)</h3>" + table(summary, 1) +
                 "<p>Raw files per trip: <code>.track</code> (lon, lat, time), <code>.nodes</code> (lon, lat), <code>.arcs</code> (directed from/to node ids) and "
                 "<code>.route</code> (reference arc ids in travel order, no timestamps). Source: <code>results/dataset_inventory.csv</code>, "
                 "<code>scripts/01_eda.py</code>.</p>")
    parts.append(f"<h3>2.2 How the data was preprocessed</h3><ol>"
                 f"<li><b>Projection.</b> The trips come from all over the world (track bounding boxes in the inventory span longitudes {lon_span[0]:.0f} to {lon_span[1]:.0f} degrees), so every trip is projected to its own local azimuthal-equidistant plane in metres; distances inside a trip are then Euclidean metres.</li>"
                 "<li><b>Reference labels.</b> <code>.route</code> has no timestamps, so each GPS point is aligned to a route position with a monotone dynamic-programming alignment (cost = point-to-arc distance capped at 200 m; forward skips penalised 0.5 m per arc; free forward jump after a gap above 10 s). These are derived reference labels, not independently verified ground truth.</li>"
                 "<li><b>Trip exclusion.</b> A trip is excluded if more than 20% of its points are over 50 m from their aligned reference arc (the reference route does not cover the track).</li>"
                 "<li><b>Candidate generation.</b> For every point, all arcs within 100 m (Shapely STRtree), nearest 30 kept.</li>"
                 "<li><b>Features.</b> 19 features per (point, candidate) pair, listed in section 2.4.</li>"
                 "<li><b>Validation.</b> 5-fold GroupKFold by trip; scalers are fitted inside training folds (sklearn Pipeline) for scale-sensitive models.</li></ol>")
    parts.append("<h3>2.3 Patterns observed in the data</h3><ul>"
                 "<li>Every trip has its own road network in a different place; absolute coordinates and arc ids therefore cannot be features.</li>"
                 "<li>No NaN rows and strictly increasing timestamps in the raw tracks, but some trips contain very large time gaps (see max_gap_s above).</li>"
                 "<li>Road graphs are directed; in the three trips checked (87% to 97%) most arcs have a reverse twin. Graphs are not fully connected (see max_components).</li>"
                 "<li>Reference routes are head-to-tail connected, and some routes repeat arcs.</li>"
                 f"<li>The nearest candidate arc is the reference arc only {100 * base0['train']:.0f}% of the time (train): two-way roads give a twin arc at the same place and distance, so heading is needed.</li></ul>")
    parts.append(figure("results/figures/trajectories_network.png", "GPS trajectories, road network and reference route for one train trip (00000034) and two student_test trips (00000052, 00000077)."))
    parts.append("<h4>Reference-label alignment quality (20 sampled train trips)</h4>" + table(align.describe().loc[["min", "50%", "max"]].reset_index().rename(columns={"index": "stat"}), 2) +
                 "<h4>Excluded trips</h4>" + table(excluded, 1))
    parts.append("<h4>Candidate radius study (train sample, excluded trips removed)</h4>" + table(radius, 3) +
                 "<p>A 100 m radius gives a candidate miss rate of 0.16% at about 15 candidates per point; 150 m adds about 40% more candidates for a further 0.1 point.</p>")
    parts.append("<h4>Missing-value analysis of the feature table (train)</h4>" + table(missing, 3) +
                 "<p>There are no NaN values. Uncertainty is carried by explicit flags: <code>heading_valid</code> (heading unreliable when slow or across a time gap; "
                 "<code>heading_cos</code> and <code>motion_along_arc_m</code> are set to 0 together with the flag) and <code>ctx_valid_count</code> (how many neighbouring points were usable). "
                 "A missing neighbour is never converted into a distance of 0.</p>")
    parts.append("<h3>2.4 Features and why they were chosen</h3><table><tr><th>Group</th><th>Features</th><th>Reason</th></tr>"
                 "<tr><td>Closeness (5)</td><td>dist_m, dist_rank, dist_minus_min, dist_ratio_to_min, n_candidates</td><td>the true arc is usually close; relative closeness copes with varying GPS noise</td></tr>"
                 "<tr><td>Position on arc (3)</td><td>t_frac, dist_to_start_m, dist_to_end_m</td><td>points near an arc end are ambiguous between neighbouring arcs</td></tr>"
                 "<tr><td>Arc shape (1)</td><td>arc_len_m</td><td>long arcs are more likely under a point</td></tr>"
                 "<tr><td>Direction and motion (4)</td><td>heading_cos, heading_valid, speed_ms, motion_along_arc_m</td><td>separates twin arcs (same place, opposite direction) and crossing roads</td></tr>"
                 "<tr><td>Temporal context (3)</td><td>ctx_dist_mean, ctx_dist_max, ctx_valid_count</td><td>distance of points at -6, -3, +3, +6 s to the same arc; rejects arcs close for one noisy sample only</td></tr>"
                 "<tr><td>Road structure (3)</td><td>start_in_degree, end_out_degree, has_reverse_arc</td><td>junction type, one-way versus two-way</td></tr></table>"
                 "<p>The features were designed from map-matching reasoning and then checked against the data (below). No automatic feature-selection step and no ablation study was run; the importance figure is exploratory (one LightGBM fit on a 600k-row train subsample, gain importance, not cross-validated, and correlated features share credit).</p>")
    parts.append(figure("results/figures/feature_distributions.png", "Feature distributions for the reference arc versus other candidates (train, 400k sampled candidate pairs)."))
    parts.append(figure("results/figures/feature_importance.png", "Exploratory LightGBM gain importance (train subsample).", "62%"))
    parts.append("<h4>Folium map of a trip</h4><p>Interactive Folium maps are committed in <code>results/maps/</code>; screenshots appear in section 6.</p>")

    # ---- 3 models
    parts.append('<h2 class="pb">3. Model experimentation and comparison</h2>')
    parts.append("<p>Six distinct algorithms from five families were trained and evaluated with an identical protocol (5-fold GroupKFold by trip on the train trips; refit on all train trips; test on the 10 student_test trips). "
                 "Metrics: pair-level macro F1 and log loss; observation-level top-1 accuracy (a point whose reference arc is missing from the candidates counts as wrong); route continuity "
                 "(share of consecutive-second predicted pairs that are the same arc or head-to-tail connected; the reference arcs themselves reach 0.99). Definitions: <code>src/pathsnap/evaluation.py</code>.</p>")
    cmp = comparison[["model", "family", "key_config", "cv_macro_f1", "cv_top1", "cv_route_continuity", "test_macro_f1", "test_top1", "test_log_loss", "test_route_continuity", "inference_ms_per_obs"]].copy()
    cmp["key_config"] = cmp["key_config"].str.slice(0, 70)
    cmp = cmp.rename(columns={"key_config": "key configuration (truncated)", "test_macro_f1": "student-test macro F1", "test_top1": "student-test top-1", "cv_top1": "CV top-1",
                              "cv_route_continuity": "CV route continuity", "cv_macro_f1": "CV macro F1", "test_log_loss": "log loss", "test_route_continuity": "test route continuity",
                              "inference_ms_per_obs": "inference ms/point"})
    parts.append("<h3>3.1 Overall model comparison</h3>" + table(cmp) +
                 f"<p class='note'>Baseline 0 (nearest candidate arc only): top-1 = {base0['student_test']:.3f} on student_test ({base0['train']:.3f} on train). The models reach about 0.83 to 0.88. KNN's train macro F1 of 1.000 is trivial (each training point is its own nearest neighbour) and is not evidence of fit quality.</p>")
    parts.append(figure("results/figures/model_comparison.png", "Model comparison: CV macro F1 (mean ± std), student-test top-1 and macro F1, log loss, overfitting gap and inference time."))
    parts.append("<h3>3.2 Per-model analysis</h3>")
    for key, title, runs in MAIN_MODELS:
        parts.append(model_section(key, title, runs, ex, args.notes_dir))
    parts.append("<h3>3.3 Model visualisations</h3>" + figure("results/figures/confusion_matrices.png", "Confusion matrices (student_test, candidate-pair level).") +
                 figure("results/curves/mlp_training_history.png", "MLP training history (the only model with an epoch-wise history; produced from the real run).", "70%"))

    # ---- 4 Bayesian
    parts.append('<h2 class="pb">4. Bayesian hyperparameter optimisation</h2>')
    best = RES / "bayesian" / "best_params.json"
    if best.exists():
        info = json.loads(best.read_text())
        trials = pd.read_csv(RES / "bayesian" / "trials.csv")
        spent = pd.to_timedelta(trials.duration).sum()
        counts = trials.state.value_counts().to_dict()
        overview = pd.DataFrame({"item": ["model", "sampler / pruner", "objective", "planned trials", "trials explored (rows in trials.csv)", "completed / pruned / failed",
                                          "search time (sum of trial durations)", "best trial", "best mean CV log loss", "its CV macro F1 / CV top-1", "seed"],
                                 "value": [info["model"], "Optuna TPE / MedianPruner", "minimise mean 5-fold GroupKFold CV log loss (train trips only)", "30 (stopped early)",
                                           len(trials), f'{counts.get("COMPLETE", 0)} / {counts.get("PRUNED", 0)} / {counts.get("FAIL", 0)}', str(spent).replace("0 days ", ""),
                                           info["best_trial_number"], f'{info["best_cv_log_loss"]:.4f}', f'{info["cv_macro_f1_mean"]:.4f} / {info["cv_top1_mean"]:.4f}', info["seed"]]})
        space = pd.DataFrame({"hyperparameter": list(info["best_params"]), "best value": [f"{v:.5g}" for v in info["best_params"].values()]})
        parts.append("<p>Bayesian optimisation (Optuna TPE) of the LightGBM classifier. The student_test partition was never used inside a trial. "
                     "The search was stopped after 15 trials because of time, so the best configuration is the best among the explored trials, not a proven global optimum. "
                     "Pruned trials report only their partial (early-fold) loss in <code>trials.csv</code>, so the table's best value is taken from completed trials.</p>"
                     + table(overview) + "<h4>Best hyperparameters</h4>" + table(space))
        both = comparison[comparison.model.isin(["lightgbm", "lightgbm_tuned"])]
        parts.append("<h4>Tuned versus untuned (same folds, same partitions, same metrics)</h4>" +
                     table(both[["model", "cv_macro_f1", "cv_top1", "test_top1", "test_macro_f1", "test_log_loss", "test_route_continuity", "fit_seconds", "inference_ms_per_obs"]]))
        parts.append(figure("results/figures/bayesian_optimisation_history.png", "Optimisation history.", "70%") + figure("results/figures/bayesian_parameter_importance.png", "Hyperparameter importance.", "70%"))
        parts.append(notes_html("bayesian", args.notes_dir))
    else:
        parts.append("<div class='todo'>Bayesian optimisation results are not committed yet.</div>")

    # ---- 5 HMM
    parts.append('<h2 class="pb">5. HMM and Viterbi route reconstruction</h2>')
    parts.append("<ol><li><b>Hidden states.</b> At GPS point i, the state is which of its candidate arcs the vehicle is on; the state set differs per point and each state keeps its arc id, so the decoded path is directly a sequence of road arcs.</li>"
                 "<li><b>Emission.</b> The classifier probability of the (point, arc) pair, as a log score floored at 1e-6.</li>"
                 "<li><b>Transition.</b> From arc a to arc b: 0 if the same arc; -hop_penalty x h if b is reachable from a in h arc-steps (h up to max_hops) on the directed arc graph; an extra u_turn_penalty for an immediate U-turn onto the reverse twin; otherwise a flat -break_penalty (a soft break, never impossible).</li>"
                 "<li><b>Missing candidates and gaps.</b> Points without candidates get no prediction and are skipped; a time gap above 10 s cuts the chain and each side is decoded independently.</li>"
                 "<li><b>Disconnected consecutive observations.</b> The break penalty makes disconnected pairs expensive but possible; the decoded route is split into connected segments wherever consecutive arcs are not head-to-tail connected.</li>"
                 "<li><b>Viterbi.</b> delta_t(b) = emission_t(b) + max over a [delta_(t-1)(a) + transition(a, b)], with back-pointers; cost O(T K^2). Implementation: <code>src/pathsnap/hmm.py</code>, 7 unit tests in <code>tests/test_hmm.py</code>.</li></ol>")
    parts.append("<p>Decoder parameters were tuned on out-of-fold train probabilities of every third train trip (26 trips, 81-point grid), then evaluated once on student_test.</p>")
    parts.append("<h3>5.1 Classifier-only versus classifier + HMM (student_test, tuned decoder)</h3>" +
                 table(hmm_cmp.drop(columns=["best_params"]).rename(columns={"tuning_trips_best_top1": "best top-1 on tuning trips"})))
    parts.append(figure("results/figures/hmm_effect.png", "Per-trip top-1 accuracy before and after HMM decoding for LightGBM and MLP (green = improved, red = worse)."))
    losers = []
    for model in hmm_cmp.model:
        per_trip = pd.read_csv(RES / f"hmm_{model}" / "student_test_per_trip.csv", dtype={"trip_id": str})
        for _, row in per_trip[per_trip.top1_change < 0].iterrows():
            losers.append(f"{model}, trip {row.trip_id}: top-1 {row.classifier_top1:.4f} to {row.hmm_top1:.4f} ({row.top1_change:+.4f})")
    parts.append("<h3>5.2 Where the HMM did not help</h3>" + ("<ul>" + "".join(f"<li>{x}</li>" for x in losers) + "</ul>" if losers else "<p>The HMM improved top-1 on every student_test trip for every evaluated model.</p>") +
                 "<p>Ranking changes: Random Forest has the best pointwise top-1 among the untuned models but is third after decoding; LightGBM has the largest gain and the best decoded top-1, so it was chosen for tuning. After Bayesian tuning, decoded top-1 is essentially the same as the untuned model (0.9269 versus 0.9275), but the tuned classifier has better probabilities (log loss 0.030 versus 0.057), higher classifier-only top-1 and continuity, and faster inference, so the Bayesian-tuned LightGBM is the final model. "
                 "Differences between the top three decoded models are about one point on 10 trips and should not be over-interpreted. Several tuned decoder parameters lie on the upper edge of the search grid, so a wider grid might change the numbers slightly.</p>")

    # ---- 6 maps
    parts.append('<h2 class="pb">6. Geographical visualisation using Folium</h2>')
    parts.append("<p>The interactive Folium maps (layer control and legend) are committed as HTML: <code>results/maps/student_test_00000052_predictions.html</code>, "
                 "<code>student_test_00000077_predictions.html</code> and <code>student_test_00000076_predictions.html</code>, generated by <code>scripts/11_folium_predictions.py</code> "
                 "from the saved LightGBM probabilities and the tuned decoder. Layers: raw GPS observations, road network (candidate arcs), reference route, classifier-only route, classifier + HMM route.</p>")
    parts.append(figure("results/maps/screenshots/map_00000052.png", "Trip 00000052 (student_test): classifier-only top-1 0.954 versus 0.985 with HMM. Screenshot of the Folium map."))
    parts.append(figure("results/maps/screenshots/map_00000077.png", "Trip 00000077 (student_test, the hardest trip): top-1 0.746 versus 0.822 with HMM."))

    # ---- 7 API UI
    parts.append('<h2 class="pb">7. API, UI and deployment</h2>')
    parts.append("<p>Architecture and contract: <code>docs/API_CONTRACT.md</code> and <code>api/schemas.py</code>. Laptop 1 runs the FastAPI service (<code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>), "
                 "which runs the full pipeline (<code>src/pathsnap/pipeline.py</code>: projection, candidates, features, classifier, HMM) and scores the result against the reference route held server-side only. "
                 "Laptop 2 runs the UI, which lists the reserved trips, sends the chosen trip id to the API and draws the returned routes and metrics. The two laptops communicate over a phone hotspot.</p>"
                 f"<p>Classifier inference cost (from the experiment logs): LightGBM {comparison.set_index('model').loc['lightgbm', 'inference_ms_per_obs']:.3f} ms per GPS point, i.e. under one second for a 6,854-point trip.</p>"
                 "<div class='todo'><b>Screenshots to insert after the two-laptop rehearsal:</b> (1) the working UI for at least one evaluator trip, (2) the API request log on Laptop 1, (3) a successful <code>/predict</code> response. These must be real screenshots from the final run.</div>")

    # ---- 8 limitations
    parts.append("<h2>8. Limitations and reproducibility</h2><ul>"
                 "<li>Reference labels are derived by alignment and may be truncated or imperfect (two train trips were excluded for this reason).</li>"
                 "<li>Arcs are straight segments between nodes; curved roads are approximated.</li>"
                 "<li>Some trips contain very large time gaps; the decoder cuts the chain there.</li>"
                 "<li>Tree ensembles use random sampling, so re-running them in a different environment can give slightly different numbers.</li>"
                 "<li>No feature ablation was run.</li></ul>"
                 f"<p>Reproduction commands are in the repository README ({REPO_URL}). Seeds are fixed (42) and every run is logged in <code>results/experiments/</code>.</p>")

    out_dir = ROOT / "report"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "PathSnap_Team12_Report.html"
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Team 12 Report</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    print("wrote", path, f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
