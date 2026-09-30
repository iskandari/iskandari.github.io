# CONUS k=3 capacity and weighting search — September 23, 2026

Run: `conus-k3-search-20260923-01`. Parent: `conus-edges-k1to5-paired-20260923-04`.

Goal: improve k=3 raw-density R² toward 0.8 without selecting hyperparameters on reporting data. This is a follow-up experiment, not a paper replication. Status: code, metric tests, diagnosis join check, and CPU end-to-end smoke test passed. AWS launch is waiting for renewal of an expired SSO session; no new instance has launched yet. A two-trial CUDA smoke test will run before the full search.

## Frozen data and scope

Reuse the parent's exact prepared Parquet rows: 35,580,856 training rows, 2,016,620 stopping rows, and 2,020,935 reporting rows. No resampling or new splits. All k=1–5 remain pooled in each model. The 2025 test files are excluded from download and never opened. The parent archive remains unchanged.

Inherited methodology: hourly nighttime thinning, migration seasons only (March 1–June 15 and August 1–November 15), 141 contributing CONUS radars, three eligible edges per profile/k at most, upper VID ≥10 birds/km², 3 km ASL ceiling, ground proxy with accepted 1 m tolerance, 35 features, and upper-VID normalization of both predictors and labels. See [parent run](conus-gpu-run-2026-09-23.md) for full details.

## Diagnosis before fitting

The parent cube-root model's k=3 raw-density R² is 0.943 at 100 m, 0.790 at 200 m, and 0.574 at 300 m below the observed edge. The deepest target is the principal weakness. The cloud job additionally joins parent reporting predictions to frozen radar metadata, verifying example IDs, and saves metrics and squared-error shares by depth, radar, and upper-VID decile. It measures the fraction of total squared error contributed by the worst 1% of rows. These are descriptive development diagnostics, not final test results.

## Eight prespecified trials

All use cube-root inputs/targets, `hist`, seed 20260903, eta 0.05, subsample 0.8, colsample_bytree 0.8, max_bin 256, alpha/gamma 0, and a **10,000-round ceiling**. Every trial starts from scratch so capacity/weight comparisons use the same training setup.

| Trial family | Depth | Training weights | min_child_weight | lambda |
|---|---|---|---|---|
| Baseline capacity | 6, 8, 10 | Upper VID | 5 | 1 |
| Weight comparison | 6, 8, 10 | None | 5 | 1 |
| Stronger regularization | 8, 10 | Upper VID | 20 | 5 |

Changing weights also changes their interaction with regularization; this is an explicit model comparison, not a claim that the loss weighting is isolated from every numerical consequence.

## Overfitting controls and recommendation

Early stopping uses **unweighted k=3 raw-density RMSE** on `valid_stop`: decode predictions, multiply both predictions and targets by each row's upper VID, then calculate RMSE across all k=3 target rows. This aligns selection with k=3 density R² on a fixed cohort. Patience is 300 rounds with no minimum improvement threshold. Training still includes all k=1–5. The prior run instead stopped on macro normalized RMSE, so stopping criterion and maximum rounds both change here.

After all eight trials, select the smallest stopping RMSE and write `selection.json` **before opening reporting rows for new-model evaluation**. Evaluate the chosen model and the depth-6 VID control on the unchanged reporting rows. Save the same normalized/density metrics overall, by k, k/depth, and radar, plus predictions. Inspect other k values for deterioration rather than assuming a k=3 gain improves every horizon. The generated `recommendation.md` records the selected parameters and best round.

This guards against direct reporting-based selection, but repeated development use of these splits still introduces selection uncertainty. This is not cross-validation, an unseen-station test, or a guarantee against overfitting. The untouched 2025 data remain the final generalization check. A 0.8 reporting score is a target, not a promised result.

## Execution, saving, and shutdown

One `g5.8xlarge` A10G GPU worker in us-east-1; trials run sequentially. Existing IAM profile, encrypted 500 GiB disk, IMDSv2, no new inbound access. The parent's watchdog and CloudWatch protection terminate after three hours idle; the instance also terminates on success/failure and has a 24-hour maximum-runtime guard. Saving runs independently of the user's laptop. Checkpoints and progress upload every minute, with a final artifact sync and `_SUCCESS`/`_FAILED` marker. Frozen data are referenced in the parent S3 prefix rather than duplicated into this run.

```text
s3://vpts-extrapolation-863683271215/training-runs/conus-k3-search-20260923-01/
  source/bundle.tgz
  artifacts/search/diagnosis.csv
  artifacts/search/diagnosis.json
  artifacts/search/search.csv
  artifacts/search/selection.json
  artifacts/search/recommendation.md
  artifacts/search/<trial>/model.ubj, fit.json, history.json
  artifacts/search/<selected-or-control>/metrics.csv, predictions/
  logs/workload-live.log
  _SUCCESS or _FAILED
```

Source: [configuration](../config/conus-k3-search.json), [trainer](../training/search-conus-k3.py), [worker](../scripts/run-conus-k3-search.sh), [launcher](../scripts/launch-conus-k3-search.py), [metric tests](../tests/test_conus_k3_search.py).

Retrieve metric tables and models locally:

```bash
RUN_ID=conus-k3-search-20260923-01 bash scripts/collect-conus-results.sh
```

The machine-readable launch record is `metadata/runs/conus-k3-search-20260923-01.json`.
