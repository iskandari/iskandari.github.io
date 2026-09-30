# VPTS extrapolation: model development and generalization

::: {.callout}
**What was done:** prepared a CONUS radar-profile dataset; compared cube-root and untransformed models; tuned model capacity and weighting; then retrained the selected cube-root configuration with 15% of radar sites held out to test spatial and temporal transfer separately.

**Main result:** for k=3, raw-density R² is **0.7528** at familiar sites in 2025, **0.7201** at unseen sites in 2013–2024, and **0.7460** at unseen sites in 2025. Spatial transfer shows a modest reduction in historical R²; the combined site/year test shows no large additional deterioration in this split. The GPU worker completed successfully and was terminated.
:::

## What the model predicts

The model estimates **missing low-altitude bird densities from a contemporaneously observed upper radar profile**. It does not forecast migration without those upper-profile observations. Holding out 2025 tests whether the learned profile-extrapolation relationship transfers to a later year.

One XGBoost model covers **k=1–5**, where k is the number of consecutive 100 m target bins immediately below a sampled observed edge. For example, k=3 withholds three bins and predicts targets at 100, 200, and 300 m below the edge. Each reported k score pools those target depths. Eligibility requires measured targets, which are hidden from the predictors and used as ground truth. This is an artificial-missingness evaluation, not direct validation of every truly unobserved near-ground layer, and it is not a replication of the coastal paper.

## Work completed

| Stage | Question | What changed | Finding |
|---|---|---|---|
| Data preparation | Can measured profiles provide comparable extrapolation examples? | Fixed seasons, heights, sampling, normalization, and radar-night splits | 141 radars contribute eligible examples |
| Transform comparison | Does a cube-root transform help? | Cube-root versus untransformed slots and targets on identical rows | Cube root improved raw-density R²; untransformed modeling improved normalized R² |
| Capacity and weighting search | Can tuning improve k=3? | Eight prespecified configurations; selection on stopping validation | Selected depth 10 with stronger regularization; reporting k=3 remained about 0.742 |
| Crossed site/year holdout | Does the relationship transfer to new radars and a new year? | Retrain the selected cube-root configuration after excluding 21 sites entirely | Separate temporal, spatial, and combined results reported below |

The first two model experiments shared the original splits. The crossed experiment reuses their exact examples and features but **changes site membership in the splits and retrains from scratch**. No additional hyperparameter search was performed for that experiment.

## Data preparation

1. **Source:** 143 CONUS radars, 2013–2025. The existing archive already selects nighttime profiles with full-profile VID ≥6.6 birds/km² and one scan per radar/UTC hour, nearest that hour's median VID. No second thinning pass.
2. **Season:** March 1–June 15 and August 1–November 15, inclusive, by local night date.
3. **Height:** keep 100 m bins from 0–3000 m ASL; remove higher bins before normalization. Target bin bottoms must be above the station's minimum-terrain proxy, allowing the agreed 1 m discrepancy. Missing height datums retain the archive's ASL assumption.
4. **Examples:** for every k, sample up to three distinct eligible edges per profile, without replacement; seed **20260903**. Each edge requires measured targets, a measured edge bin, and **observed-upper VID ≥10 birds/km²**. Missing interior upper bins remain NA. No additional precipitation/clutter filtering was introduced.
5. **Normalize:** divide both upper densities and target density by that edge's upper VID. Targets never enter the denominator. With 100 m layers, `upper_VID = 0.1 × sum(upper densities)`, so normalized slots sum to **10**. Cube the cube-root model's predictions, clip at zero, then multiply by VID to recover birds/km³.

After seasonal selection: **2,638,711 profiles**. **141 radars** contribute eligible examples. The original transform comparison and tuning search used the following split. The crossed experiment later repartitioned these same examples by radar site:

| Split | Rows | Purpose |
|---|---:|---|
| Training | 35,580,856 | Fit models |
| Stopping validation | 2,016,620 | Select rounds and hyperparameters |
| Reporting validation | 2,020,935 | Evaluate selected models |
| 2025 test | 3,474,557 | Reserved during original comparison/tuning; evaluated in the later crossed experiment |

These are expanded target rows, not independent profiles. Original training/validation night assignments were retained; validation nights were divided by a deterministic radar-night hash. Whole radar-nights stay together. Those original experiments used grouped holdout validation, **not cross-validation or an unseen-station test**. The later crossed experiment adds the site holdout.

## Input features

Each target-bin prediction uses **35 fixed input columns**. The number of observed density values varies between examples:

| Input | Number | What it tells the model | Representation |
|---|---:|---|---|
| **Upper-profile density** | 29 fixed slots | Available observed densities above the target; **not necessarily 29 measured bins** | Up to 29 values in 100 m slots; normalize by observed-upper VID, then cube root. Unavailable slots are missing. |
| **Target gap** | 1 | How far below the lowest observed bin to predict | Distance in metres: 100–500 |
| **Target height above minimum terrain** | 1 | Height of the target-bin bottom relative to the station's terrain reference | `target_agl_m = target_height_m − asl.min` (metres) |
| **Solar time** | 2 | When the observation falls within the night | Sine and cosine of sunset-to-sunrise position |
| **Day of year** | 2 | When the observation falls within the annual cycle | Sine and cosine of day of year |
| **Total** | **35** | | |

Here, `asl.min` comes from `metadata/radarInfo.csv` and is the station's minimum-terrain elevation above sea level. The feature is the target height **minus** `asl.min`, not `asl.min` itself or height above the radar antenna. It is an AGL proxy, not the exact height above terrain at every sampled location.

**Why a fixed width of 29 slots?** The source has 30 bins (bottoms at 0–2,900 m ASL), but at least one is withheld as a target. Thus 29 is the **maximum possible observed input length**, not the number observed in every example. For k=3, at most 27 bins remain; a higher sampled edge or missing measurements can leave fewer. Available bins are aligned relative to the lowest observed bin in the same 29-column layout. Unused slots and gaps stay **missing, not zero**. The feature matrix always has 35 columns, even when fewer density values are observed.

**Not input features:** radar identity, coordinates, year, k, absolute ASL height, or VID itself. VID is used for normalization, training weights, and conversion back to density units. Withheld target densities never enter the input profile or its normalization denominator.

## Earlier model comparison and tuning

**Transform comparison:** two depth-6, VID-weighted models, up to 5,000 rounds. Cube root improved overall raw-density R² (**0.709 vs 0.680**), but untransformed targets had better normalized R² (**0.554 vs 0.524**). Stopping used equally averaged per-k normalized RMSE, patience 150.

**Tuning:** eight cube-root models, up to 10,000 rounds. Early stopping used **unweighted k=3 raw-density RMSE**, patience **300**, with no minimum improvement threshold. The lowest stopping score selected the model before its reporting evaluation.

| Depth | Training weights | Min child weight | L2 (`lambda`) | Best round | Stopping density RMSE ↓ (birds/km³) |
|---:|---|---:|---:|---:|---:|
| 6 | VID | 5 | 1 | 9,999 | 63.222 |
| 6 | None | 5 | 1 | 9,992 | 68.118 |
| 8 | VID | 5 | 1 | 9,999 | 62.632 |
| 8 | None | 5 | 1 | 9,944 | 66.493 |
| 10 | VID | 5 | 1 | 6,216 | 62.516 |
| 10 | None | 5 | 1 | 9,871 | 65.380 |
| 8 | VID | 20 | 5 | 8,115 | 62.600 |
| **10** | **VID** | **20** | **5** | **3,564** | <span class="best">62.511</span> |

**How stopping RMSE is calculated:** cube and clip the model prediction to recover normalized density, then multiply by that row's observed-upper VID to recover density in birds/km³. Compute `sqrt(mean((predicted_density − observed_density)²))` over all k=3 stopping-validation target rows. Every row has equal weight in this mean; there is **no additional VID weighting of the validation metric**. The “Training weights” column applies only to fitting. Thus every trial is compared using the same raw-density metric; these numbers are not RMSE of integrated VID.

### Shared model settings

**Learning rate = 0.05** (XGBoost `eta`), fixed throughout training for both transform-comparison models, all eight tuning trials, and the crossed site/year model. It was not tuned and no learning-rate decay schedule was used. This parameter scales each new tree's contribution to the prediction.

| Setting | Value |
|---|---|
| **Learning rate (`eta`)** | **0.05, constant** |
| Objective | Squared error (`reg:squarederror`) |
| Row subsampling (`subsample`) | 0.8 |
| Feature subsampling per tree (`colsample_bytree`) | 0.8 |
| Histogram bins (`max_bin`) | 256 |
| L1 penalty (`alpha`); minimum split loss (`gamma`) | 0; 0 |
| Initial prediction (`base_score`) | 0.5 |
| Training seed | 20260903 |
| Implementation | Python XGBoost 3.2.0; GPU `hist`; 32 CPU threads |

Depth, training weights, minimum child weight, and L2 penalty vary as listed in the tuning table. The crossed model uses the selected depth-10 configuration.

VID weights prioritize higher-intensity training rows. `min_child_weight` limits low-support leaves; for squared error it measures summed row weights, not independent nights. `lambda` penalizes large leaf values. The stronger-regularization trials changed both parameters together.

### Earlier reporting results

| k | Extended depth-6 control R² | Selected depth-10 R² |
|---:|---:|---:|
| 1 | 0.9163 | <span class="best">0.9203</span> |
| 2 | 0.8340 | <span class="best">0.8366</span> |
| 3 | <span class="best">0.7425</span> | 0.7424 |
| 4 | 0.6616 | <span class="best">0.6631</span> |
| 5 | 0.6101 | <span class="best">0.6149</span> |

These are **reporting raw-density R²**. Only the selected model and control received new reporting evaluations. Tuning modestly helped other depths but **did not improve k=3 over the control or reach 0.8**. Repeated development use of validation means these are not final test results.


Green bold values in the earlier tables identify the lowest stopping RMSE or the higher reporting R² within each k. Selection used stopping validation, not the highlighted reporting scores.

## Crossed radar-site and year experiment

### Why this additional experiment was needed

A held-out year at familiar radar stations answers the next-year transfer question but cannot show whether a model relies on site-specific patterns. Excluding entire radar sites tests transfer to locations that provided no training examples. Crossing the two exclusions separates these questions rather than mixing them into one test score.

**Seen sites** means radar locations whose historical observations enter training. **Unseen sites** means locations excluded from training and early stopping in every year. Seen-site test observations are still held out; “seen” refers to the location, not reuse of the evaluated rows.

### What the evaluation categories mean

| Category | Which radar sites? | Which years? |
|---|---|---|
| **Historical reporting** | Sites included in training, but separate held-out reporting nights | 2013–2024 |
| **Temporal test** | Sites included in training | 2025 |
| **Spatial test** | 21 sites excluded from training entirely | 2013–2024 |
| **Spatial + temporal** | Those same 21 excluded sites | 2025 |

**How spatial + temporal was done:** evaluate the model on **2025 observations from the 21 held-out radars**. Training excluded both those radars in every year and all 2025 observations at every radar, so neither their locations nor that year contributed to fitting. This is the intersection of the site and year exclusions, evaluated using the same fitted model.

**k** is the number of consecutive 100 m layers withheld below the observed profile; each k score pools the target layers in that block.

The 2025 block contains both migration seasons. Historical radar-night assignments are preserved for seen sites. For unseen sites, all former historical training/validation rows enter the spatial test. All 2025 rows remain excluded from fitting and round selection.

### Site selection and sample sizes

**21 of 141 radars (14.89%, nearest whole-site allocation to 15%)** were selected before examining this experiment's errors. Five longitude bands crossed with latitude below/above 37° define ten geographic strata. Allocate held-out sites proportionally, with at least one per stratum, and select within each stratum by a deterministic hash using seed 20260929. Only historical participation and station geography determine the selection.

The complete frozen list and coordinates are in [the site manifest](metadata/conus-crossed-sites.csv). This tests scattered sites within the network; it is not a spatially buffered exclusion of an entire region.

| Cell | Period | Contributing sites | Expanded target rows |
|---|---|---:|---:|
| Training | 2013–2024 | 120 | 29,822,840 |
| Stopping validation | 2013–2024 | 120 | 1,692,390 |
| Historical reporting | 2013–2024 | 119 | 1,702,383 |
| Temporal test | 2025 | 119 | 2,881,618 |
| Spatial test | 2013–2024 | 21 | 6,400,798 |
| Spatial + temporal test | 2025 | 21 | 592,939 |

A site may contribute no eligible examples to a particular cell or k. Therefore counts of contributing sites can be smaller than the assigned site count. Rows include repeated profile contributions across sampled edges, k, and target depth; they are not independent observations.

### Fitting and evaluation

The previously selected configuration was fixed: cube-root normalized density slots and targets, observed-upper VID training weights, **learning rate 0.05 (`eta`, constant)**, depth 10, min_child_weight 20, and lambda 5. All other preparation and model settings above were retained. Early stopping minimized unweighted k=3 raw-density RMSE on historical stopping nights at seen sites, with a 10,000-round ceiling and patience 300.

The new model selected **3,633 rounds**, with stopping RMSE **62.6721 birds/km³**. Early stopping ended at 3,933 rounds. Fitting took **474.95 seconds (7.9 minutes)**, excluding data preparation, matrix construction, and reporting. The selected model and iteration were saved before test scoring. None of the test results below was used to adjust this model.

![Figure 1. Two views of the same stopping-validation history, with a constant learning rate of 0.05 (`eta`). A: rounds 1–3,933 on an RMSE scale of 60–150 birds/km³. B: the same values from rounds 100–3,933, with the vertical scale expanded to 62.5–68 birds/km³ (the shaded band in A). The different vertical scales make later improvements look larger in B; these are not different models or metrics. RMSE is 147.94 at round 1, 67.48 at round 100, and 62.67 at the selected round 3,633 (dashed line). Both panels show unweighted k=3 raw-density RMSE on historical stopping nights at seen sites, not training loss or test error.](docs/figures/stopping-curve.png)

## Results of the crossed experiment

**Reading the tables:** R² is higher-is-better; RMSE is lower-is-better; bias is prediction minus observation, so closer to zero is better. Green bold values mark the best numeric score **across evaluation cells at the same k**, including historical reporting. “All” pools the k=1–5 rows rather than averaging the five k scores. Comparisons use unrounded values. Highlights are descriptive and are not significance tests: these evaluation populations differ.

### Raw-density R² overview

![Figure 2. Raw-density R² by withheld block size k. Each point pools all target depths in that block. Performance declines with greater extrapolation depth; the four curves evaluate the same fitted model on different site/year populations. Connecting lines are visual guides.](docs/figures/density-r2-by-k.png)

| k | Historical reporting | Temporal test | Spatial test | Spatial + temporal |
|---|---:|---:|---:|---:|
| All | 0.7164 | <span class="best">0.7171</span> | 0.6854 | 0.7057 |
| 1 | <span class="best">0.9200</span> | 0.9188 | 0.8989 | 0.9035 |
| 2 | <span class="best">0.8433</span> | 0.8362 | 0.8149 | 0.8264 |
| 3 | 0.7477 | <span class="best">0.7528</span> | 0.7201 | 0.7460 |
| 4 | 0.6627 | <span class="best">0.6687</span> | 0.6328 | 0.6593 |
| 5 | <span class="best">0.6107</span> | 0.6100 | 0.5781 | 0.6036 |

### Complete raw-density scores

RMSE and bias are in birds/km³.

| Evaluation | k | Rows | R² | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Historical reporting | All | 1,702,383 | 0.7164 | 65.12 | -4.66 |
| Historical reporting | 1 | 166,852 | <span class="best">0.9200</span> | <span class="best">32.35</span> | -0.88 |
| Historical reporting | 2 | 281,260 | <span class="best">0.8433</span> | <span class="best">47.28</span> | -2.44 |
| Historical reporting | 3 | 363,537 | 0.7477 | 61.46 | -3.61 |
| Historical reporting | 4 | 423,804 | 0.6627 | 71.92 | -5.83 |
| Historical reporting | 5 | 466,930 | <span class="best">0.6107</span> | <span class="best">78.13</span> | -7.10 |
| Temporal test | All | 2,881,618 | <span class="best">0.7171</span> | 66.28 | <span class="best">+0.37</span> |
| Temporal test | 1 | 270,911 | 0.9188 | 33.45 | +3.89 |
| Temporal test | 2 | 469,486 | 0.8362 | 49.55 | +2.83 |
| Temporal test | 3 | 616,206 | <span class="best">0.7528</span> | 62.08 | <span class="best">+1.31</span> |
| Temporal test | 4 | 724,900 | <span class="best">0.6687</span> | 72.41 | -0.58 |
| Temporal test | 5 | 800,115 | 0.6100 | 79.27 | <span class="best">-2.14</span> |
| Spatial test | All | 6,400,798 | 0.6854 | 67.49 | -4.90 |
| Spatial test | 1 | 623,993 | 0.8989 | 35.28 | <span class="best">+0.76</span> |
| Spatial test | 2 | 1,060,222 | 0.8149 | 50.00 | <span class="best">-1.72</span> |
| Spatial test | 3 | 1,372,455 | 0.7201 | 63.17 | -3.32 |
| Spatial test | 4 | 1,595,468 | 0.6328 | 74.10 | -6.79 |
| Spatial test | 5 | 1,748,660 | 0.5781 | 81.07 | -8.36 |
| Spatial + temporal | All | 592,939 | 0.7057 | <span class="best">65.06</span> | +1.05 |
| Spatial + temporal | 1 | 54,159 | 0.9035 | 34.89 | +5.12 |
| Spatial + temporal | 2 | 95,660 | 0.8264 | 48.33 | +4.14 |
| Spatial + temporal | 3 | 126,972 | 0.7460 | <span class="best">59.70</span> | +2.52 |
| Spatial + temporal | 4 | 150,108 | 0.6593 | <span class="best">70.84</span> | <span class="best">-0.03</span> |
| Spatial + temporal | 5 | 166,040 | 0.6036 | 78.16 | -2.22 |

### Complete normalized-density scores

Both targets and decoded predictions are divided by observed-upper VID. These scores describe errors relative to the observed upper-profile intensity; they are not errors in the cube-root-transformed space.

| Evaluation | k | Rows | R² | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Historical reporting | All | 1,702,383 | 0.5542 | 2.742 | -0.354 |
| Historical reporting | 1 | 166,852 | <span class="best">0.8274</span> | 1.071 | -0.083 |
| Historical reporting | 2 | 281,260 | <span class="best">0.7132</span> | 1.715 | -0.188 |
| Historical reporting | 3 | 363,537 | 0.6102 | 2.389 | -0.290 |
| Historical reporting | 4 | 423,804 | 0.5168 | 3.040 | -0.444 |
| Historical reporting | 5 | 466,930 | 0.4829 | 3.521 | -0.518 |
| Temporal test | All | 2,881,618 | <span class="best">0.5630</span> | 2.627 | -0.144 |
| Temporal test | 1 | 270,911 | 0.8136 | 1.018 | +0.101 |
| Temporal test | 2 | 469,486 | 0.7064 | 1.638 | <span class="best">+0.027</span> |
| Temporal test | 3 | 616,206 | <span class="best">0.6142</span> | 2.270 | -0.079 |
| Temporal test | 4 | 724,900 | <span class="best">0.5397</span> | 2.877 | -0.218 |
| Temporal test | 5 | 800,115 | <span class="best">0.4955</span> | 3.385 | -0.309 |
| Spatial test | All | 6,400,798 | 0.5172 | 2.767 | -0.365 |
| Spatial test | 1 | 623,993 | 0.7856 | 1.129 | <span class="best">-0.023</span> |
| Spatial test | 2 | 1,060,222 | 0.6795 | 1.719 | -0.154 |
| Spatial test | 3 | 1,372,455 | 0.5724 | 2.375 | -0.265 |
| Spatial test | 4 | 1,595,468 | 0.4833 | 3.038 | -0.483 |
| Spatial test | 5 | 1,748,660 | 0.4466 | 3.596 | -0.587 |
| Spatial + temporal | All | 592,939 | 0.5223 | <span class="best">2.425</span> | <span class="best">-0.098</span> |
| Spatial + temporal | 1 | 54,159 | 0.7913 | <span class="best">0.970</span> | +0.139 |
| Spatial + temporal | 2 | 95,660 | 0.6651 | <span class="best">1.524</span> | +0.083 |
| Spatial + temporal | 3 | 126,972 | 0.5676 | <span class="best">2.073</span> | <span class="best">-0.008</span> |
| Spatial + temporal | 4 | 150,108 | 0.4977 | <span class="best">2.623</span> | <span class="best">-0.167</span> |
| Spatial + temporal | 5 | 166,040 | 0.4565 | <span class="best">3.138</span> | <span class="best">-0.287</span> |

### Equal-site uncertainty at k=3

Pooled metrics give more influence to sites with more rows. To complement them, equal-site RMSE is the square root of mean per-site MSE, giving each radar equal weight. Percentile 95% intervals use 2,000 whole-radar bootstrap resamples, seed 20260929.

![Figure 3. Equal-site k=3 raw-density RMSE with percentile 95% whole-radar bootstrap intervals. Lower values indicate smaller errors. These intervals condition on this fitted model and the observed years; they do not capture nearby-site dependence, between-year variability, or refitting uncertainty.](docs/figures/k3-site-uncertainty.png)

| Evaluation | Sites | Equal-site RMSE (birds/km³) | 95% site-bootstrap interval |
|---|---:|---:|---|
| Historical reporting | 119 | 60.18 | 55.34–65.37 |
| Temporal test | 118 | 61.11 | 56.46–65.94 |
| Spatial test | 21 | 60.88 | 53.23–68.01 |
| Spatial + temporal | 21 | <span class="best">57.05</span> | 49.74–64.02 |

Full metrics, including k/gap groups, normalized and density scales, underprediction fractions, and equal-site uncertainty, are provided in [comparison.csv](summaries/crossed/comparison.csv). The [per-radar table](summaries/crossed/all_by_site.csv) provides each radar's overall, k, and k/gap scores. [Cohort counts](summaries/crossed/cohort_counts.csv) record eligibility by role, radar, year, and k.

## Interpretation and limitations

**Temporal transfer:** k=3 raw-density R² is 0.7528 in 2025 versus 0.7477 on historical reporting nights. This supports transfer of the profile-extrapolation relationship to that later year at familiar sites; it does not establish robustness across all future years.

**Spatial transfer:** historical unseen-site R² is modestly lower across k=1–5. At k=3 it is 0.7201, with pooled RMSE 63.17 versus 61.46 on seen-site historical reporting. The difference is consistent with a spatial transfer cost, but the site and observation populations also differ.

**Combined transfer:** unseen sites in 2025 give k=3 R² 0.7460 and RMSE 59.70, compared with 0.7528 and 62.08 at seen sites in 2025. The lower combined-test RMSE does not imply unseen sites are intrinsically easier: response variance and intensity distributions differ across cells. R² and RMSE should be read together.

**Deeper extrapolation remains harder:** scores decline as k increases, and historical bias becomes more negative. The earlier tuning search did not resolve the deep-target difficulty or reach the k=3 R² target of 0.8. A modest change in score across the old and new experiments cannot be attributed solely to model improvement because the cohorts differ.

**Limits on independence:** the earlier hyperparameter search included historical data from all sites, including the 21 now held out. The new fit excludes those sites, but the chosen hyperparameters were not selected independently of their historical data. This is therefore a retrospective spatial test, not a fully nested spatial validation. Internal stopping still uses the parent's scattered radar-night assignments. The 2025 observations were excluded from fitting and selection; they have now been evaluated and should not be described as an untouched test set in later development.

**Limits on uncertainty:** the site-bootstrap intervals condition on the fitted model and observed years. They do not account for dependence between nearby radars, between-year variation, or refitting uncertainty. Overlapping intervals do not establish statistical equivalence. A future region-blocked or nested site/year evaluation would address a broader deployment question.

## Completed run and verification

Run **conus-crossed-cuberoot-20260929-01** completed successfully on September 29, 2026. The source split counts were reproduced exactly, fitting rejected any held-out-site or 2025 training rows, all 11 CONUS tests passed locally, and the four new crossed tests passed on the remote worker. The final S3 success marker reports exit code 0.

EC2 worker **i-0588fbdd8cf661105** was confirmed **terminated** after completion. Host and CloudWatch safeguards enforced a three-hour idle limit, with a separate 24-hour runtime guard. No worker needs to remain running to read or reuse the archived results.

## Where everything lives

**Source archive** — already hourly-thinned and VID ≥6.6:

```text
s3://vpts-extrapolation-863683271215/production/profile_sampled_archive_v1/
```

**Exact filtered, expanded, normalized model data:**

```text
s3://vpts-extrapolation-863683271215/training-runs/conus-edges-k1to5-paired-20260923-04/prepared/
```

Contains `train/`, `valid_stop/`, `valid_report/`, `test/`, configuration, feature manifest, night assignments, counts, and checksums.

**Earlier tuning results and models:**

```text
s3://vpts-extrapolation-863683271215/training-runs/conus-k3-search-20260923-01/artifacts/search/
```

Earlier selected model: `depth10_vid_regularized/model.ubj`; exact settings: `fit.json` in the same subfolder. Use **`iteration_range=(0, 3564)`** when predicting: the saved booster also contains patience rounds. Metrics, histories, predictions, and selection records are archived alongside it.

**Crossed experiment and current generalization model:**

```text
s3://vpts-extrapolation-863683271215/training-runs/conus-crossed-cuberoot-20260929-01/
```

Model: `artifacts/crossed/model.ubj`. Use **`iteration_range=(0, 3633)`** when predicting with this model; do not substitute the earlier model's 3,564 rounds. `artifacts/crossed/fit.json` records exact settings, selected iteration, and training time. Cell-specific predictions and metrics are stored beneath `artifacts/crossed/<cell>/`; split manifests and audits are under `artifacts/split/`.

Small metric tables, fitting records, and split summaries are included in `summaries/crossed/` in this handoff. Large datasets, model weights, and row predictions remain in S3. The [machine-readable run record](metadata/runs/conus-crossed-cuberoot-20260929-01.json) records the successful completion and terminated worker.

## Reuse

**Plain scripts and small result tables, without bundled training datasets.** Run these commands from this folder. Python 3.12, the AWS CLI, and credentials with bucket access are required. The reproduction commands below do not launch a cloud instance.

### 1. Set up

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# NVIDIA GPU / Linux only:
python -m pip install -r requirements-gpu.txt
```

Skip the GPU install on a Mac. Use `--device cpu` instead of `--device cuda` below; CPU results need not be bit-for-bit identical to AWS GPU results.

### 2. Get the exact model inputs

```bash
aws s3 sync \
  s3://vpts-extrapolation-863683271215/training-runs/conus-edges-k1to5-paired-20260923-04/prepared/ \
  prepared/ --exclude 'test/*'
```

**Alternatively, rebuild preparation from the archived source** using the actual AWS preparation script:

```bash
aws s3 sync \
  s3://vpts-extrapolation-863683271215/production/profile_sampled_archive_v1/ source-data/
python training/prepare-conus-paired.py --source source-data/ --output prepared/ \
  --config config/conus-paired-gpu.json --workers 4
```

Choose download **or** rebuild; rebuilding requires a fresh output directory. It prepares all years, including 2025 test rows. The original comparison and tuning scripts never open that test split; the crossed script evaluates it only after fitting. Source snapshot IDs and source-file checksums are checked/recorded. `prepared/manifest.json` records the feature names and cohort totals.

### 3. Reproduce training

Original cube-root versus untransformed comparison:

```bash
python training/train-conus-paired.py --prepared prepared/ --output results/paired/ --device cuda
```

Eight-trial tuning search (reuse the archived baseline predictions for its diagnosis):

```bash
aws s3 sync \
  s3://vpts-extrapolation-863683271215/training-runs/conus-edges-k1to5-paired-20260923-04/artifacts/full/cuberoot/predictions/ \
  baseline/predictions/
python training/search-conus-k3.py --prepared prepared/ --previous baseline/ \
  --output results/search/ --config config/conus-k3-search.json --device cuda
```

The two runs are independent: the search can use the downloaded baseline without rerunning the paired comparison. Use fresh result directories. The full runs are large; a `--profile-cap` preparation run is a software check only and must not be substituted for the full cohort.

### 4. Reproduce the crossed holdout

Download **all** frozen parent roles, including `test/`. The exclusion in the earlier download command was specific to reproducing the original development experiments.

```bash
aws s3 sync \
  s3://vpts-extrapolation-863683271215/training-runs/conus-edges-k1to5-paired-20260923-04/prepared/ \
  prepared/
python training/conus-crossed.py prepare --source prepared/ --output crossed-prepared/ \
  --config config/conus-crossed.json --sites metadata/conus-crossed-sites.csv
python training/conus-crossed.py fit --source crossed-prepared/ --output results/crossed/ \
  --config config/conus-crossed.json --device cuda
```

Run from this handoff folder, use fresh output directories, and keep the committed site list fixed. The trainer uses the existing shared helpers already included here. CPU execution is supported with `--device cpu` but need not reproduce GPU results bit for bit. This is a full-cohort fit, not a small laptop example.

For inspecting results without refitting, open `summaries/crossed/comparison.csv` or the HTML handoff. To retrieve the trained model:

```bash
aws s3 cp \
  s3://vpts-extrapolation-863683271215/training-runs/conus-crossed-cuberoot-20260929-01/artifacts/crossed/model.ubj \
  crossed-model.ubj
```

### Files to read

| File | Purpose |
|---|---|
| `training/prepare-conus-paired.py` | Filter source → select edges → normalize → save splits |
| `training/train-conus-paired.py` | Two transform fits and shared prediction/metric helpers |
| `training/search-conus-k3.py` | Diagnose → fit eight trials → select → report |
| `training/conus-crossed.py` | Repartition by frozen sites → retrain → score all three test cells |
| `metadata/conus-crossed-sites.csv` | Frozen 21-site holdout and geographic selection metadata |
| `summaries/crossed/` | Completed scores, fitting records, and cohort counts |
| `config/*.json` | Exact preparation and experiment settings |
| `metadata/radarInfo.csv` | Station geometry; not a training-observation dataset |
| `tests/` | Checks for seasons, sampling, normalization, and scoring |
| `scripts/`, `docs/` | Original AWS execution scripts and archived run notes |

Check the implementation with `python -m unittest discover -s tests`. **Both AWS bundles are fully unpacked.** The original preparation/tuning scripts, configurations, metadata, and tests remain unchanged; `source-checksums.json` records their SHA-256 values. The two original checksum manifests are retained separately in `metadata/`. Archived originals remain at `source/bundle.tgz` under the two run prefixes above. AWS used an A10G GPU and the recorded dependency versions; both workers terminated after completion.

Use the numbered commands above for manual reproduction. The archived `scripts/` files assume the original EC2 setup and include shutdown commands; they are not local setup scripts. Their additional AWS dependencies are recorded in each run's S3 `artifacts/requirements-resolved.txt`.

The crossed experiment is an additive update. Its included files have a separate `crossed-checksums.json`; the original `source-checksums.json` remains unchanged. The HTML is self-contained and can be sent on its own for reading; send this entire folder as well if the recipient needs linked CSV tables and reproducible scripts.

To regenerate this HTML after editing `README.md`, run `bash scripts/render-handoff.sh` from this folder. The renderer uses R with `rmarkdown`, `jsonlite`, `systemfonts`, and `ragg`, plus Pandoc (included with RStudio or installed separately). It draws the three figures directly from archived metrics and learning history in `summaries/crossed/`, then renders the README as a self-contained R Markdown HTML report. Open Sans is bundled in `docs/fonts/` and used for the report and plot labels. Fonts, styling, figures, and navigation assets are embedded in the HTML; no R session or network connection is needed to read it. The contents panel floats beside the text on desktop, with expandable subsections, and collapses above the text on smaller screens.
