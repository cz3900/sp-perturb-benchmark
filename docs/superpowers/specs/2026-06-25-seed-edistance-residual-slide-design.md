# Seed E-Distance Residual Slide Design

## Purpose

Create one technical PPT slide that explains why seed-level E-distance can be structurally inflated
when a mean-field seed predictor collapses its predictions to one point, and why adding
same-cell-type control residuals makes the E-distance comparison fair without changing the
predicted seed mean.

The slide is for a technical lab/methods discussion. It should make the failure mode visually
obvious before showing the formula.

## Scope

This design covers one 16:9 slide or one standalone figure panel. It does not define a full deck,
new benchmark behavior, or new metric implementation.

## Core Message

Seed observations are a real cloud of `N` perturbed cells. A mean-field seed model can emit the
same vector for every cell of the same cell type, so its predicted cloud collapses to one point or
a few type-level points. Energy distance compares distributions:

```text
E = 2 * cross(pred, obs) - spread(pred) - spread(obs)
```

When `spread(pred)` is near zero, E-distance is artificially high because the prediction has no
within-cloud spread to subtract. Adding a zero-mean control residual restores realistic
cell-to-cell variance while preserving the predicted mean.

## Slide Layout

Use a horizontal three-panel mechanism figure.

### Panel 1: Observed Truth

Title: `Observed perturbed seeds`

Visual:
- Green dashed ellipse.
- Green scatter cloud representing `N` real perturbed seed cells.
- Small badge: `N perturbed cells`.
- Optional diagnostic badge: `real cell-to-cell variance`.

Message:
- The observed seed target is a distribution, not a single point.
- Avoid using a fixed number such as `39` in the main visual; use `N` so the slide generalizes.

### Panel 2: Mean-Field Prediction

Title: `Mean-field seed prediction`

Visual:
- One orange point, or several orange points stacked tightly.
- A small label: `one vector per cell type`.
- Red annotation: `collapsed cloud: spread(pred) near zero`.

Message:
- The collapse is caused by the model's type-level mean prediction, not by matched sample size.
- If multiple cell types are present, there may be a few type-level points, but the prediction
  still lacks cell-level variance.

### Panel 3: Residual-Restored Prediction

Title: `Prediction + control residual`

Visual:
- Orange scatter cloud with similar spread to the control/observed biological variance.
- Mark the cloud center with a small cross or dot labeled `mean unchanged`.
- Green annotation: `fair E-distance`.

Formula:

```text
seed_pred_resid_i =
  seed_mean(cell_type_i) + (control_i - mean_control(cell_type_i))
```

Message:
- The residual restores variance, not mean accuracy.
- Residuals come from controls only, so they do not leak observed perturbed truth.

## Bottom Explanation Strip

Use two narrow callout strips under the three panels.

Problem:

```text
If prediction is a point cloud with spread(pred) near zero, E-distance can look worse than null even
when the predicted mean shift is useful.
```

Fix:

```text
Use residual-restored predictions for E-distance; keep raw mean predictions for PCC-delta and MSE.
```

## Speaker Notes

The seed target is a cloud of `N` real perturbed cells, so it has biological cell-to-cell variance.
TrivialSeed is mean-field: for cells with the same cell type, it emits essentially the same seed
vector. The predicted samples therefore collapse to one point, or to a small number of type-level
points.

Energy distance compares two clouds. It subtracts within-cloud spread from both sides. If the
prediction has no within-cloud spread, the prediction-side spread term is zero, so E-distance is
structurally inflated. That inflation is about missing variance, not necessarily wrong mean shift.
The null/control cloud has real variance, so it can appear better for the wrong reason.

The repair is to add a same-cell-type control residual to each predicted mean. The residual is
`control cell - same-type control mean`, so it is approximately zero-mean and does not move the
prediction center. It restores realistic cell-level spread and lets E-distance compare mean shifts
more fairly. PCC-delta and MSE should remain on the raw mean prediction because those metrics are
mean-based.

## Pitfalls To Avoid

- Do not imply that the matched sample count causes duplicated predictions. The model causes the
  duplicated predictions; matched-n only aligns comparison size.
- Do not claim residuals improve the predicted mean. They restore variance around the same mean.
- Do not draw residuals from observed perturbed cells. They must come from the control pool to avoid
  leakage.
- Do not overstate exact equality of variance. The goal is variance calibration, not perfect
  generative modeling.

## Visual Style

- Observed/ground truth: green.
- Prediction: orange.
- Failure/biased E: red.
- Explanatory arrows and formulas: neutral gray.
- Keep formulas sparse: the slide should lead with the visual mechanism.

## Acceptance Criteria

- A technical viewer can explain why the collapsed seed cloud makes E-distance unfair.
- The slide explicitly says `N perturbed cells` rather than anchoring on one dataset-specific count.
- The slide distinguishes mean preservation from variance restoration.
- The slide states that residuals come from controls only and therefore avoid leakage.
- The slide states that E uses residual-restored seed predictions while PCC-delta/MSE use raw seed
  predictions.
