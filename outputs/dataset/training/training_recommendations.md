# LoveDA Training-Oriented EDA Recommendations

## Normalization

Recommended train-only RGB normalization values:

| channel | train_mean | train_std |
| ------- | ---------- | --------- |
| R       | 77.101179  | 39.930312 |
| G       | 82.905089  | 34.184394 |
| B       | 78.997382  | 32.877234 |

## No-data label policy

- Treat class `0` as `ignore_index=0`.
- Exclude class `0` from loss weighting and metric aggregation.

## Class weighting

Suggested reference weights:

| class_id | class_name  | pixel_ratio | median_frequency_weight | inverse_sqrt_frequency_weight |
| -------- | ----------- | ----------- | ----------------------- | ----------------------------- |
| 1        | background  | 0.36749638  | 0.32198142              | 1.64958031                    |
| 2        | building    | 0.11832701  | 1.0                     | 2.90708719                    |
| 3        | road        | 0.05575285  | 2.12234906              | 4.23512724                    |
| 4        | water       | 0.06191798  | 1.91102833              | 4.01875582                    |
| 5        | barren      | 0.05395076  | 2.19324074              | 4.30527809                    |
| 6        | forest      | 0.15453716  | 0.76568645              | 2.54380337                    |
| 7        | agriculture | 0.18801786  | 0.6293392               | 2.30621849                    |

Practical recommendation:
- start with `CrossEntropyLoss(ignore_index=0)`
- then move to `CrossEntropy + Dice`
- only add class weights if minority classes still underperform

## Crop and tiling strategy

Observed:
- all images are fixed at `1024x1024`
- average train boundary density is `0.007069`
- roads and buildings show high component fragmentation relative to other classes

Recommendation:
- start with `512x512` random crops for training
- use class-aware crop sampling when the crop contains too much background/no-data
- use `50%` overlap during validation/inference tiling if whole-image inference is not feasible

## Domain-aware evaluation

- report metrics separately for `Urban` and `Rural`
- do not rely on one merged score only

## Sanity subset

A small overfit/debug subset was exported to:
- `outputs/dataset/training/sanity_subset.csv`

Use it first to validate:
- dataloader correctness
- label handling
- loss reduction
- basic model overfit behavior
