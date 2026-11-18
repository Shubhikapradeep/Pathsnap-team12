# HMM development results (stand-in classifier)

These files were produced with `scripts/dev/make_standin_probabilities.py` (a HistGradientBoosting
stand-in, NOT one of the compared models) to build and debug the decoder. Do not quote them as the final HMM
results. Re-run with the final selected model:

    python scripts/05_tune_hmm.py --probs <selected_model_name>

Known caveat: the best parameters (hop_penalty 1.0, break_penalty 10, max_hops 8, u_turn_penalty 5) sit on the
upper edge of the search grid in `scripts/05_tune_hmm.py`, so a wider grid may improve slightly. Tuned vs default
parameters differed by only about 0.4 points of top-1 on student_test, so the gain from widening is expected to be small.
