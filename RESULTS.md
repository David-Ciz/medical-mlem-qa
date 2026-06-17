# Current Results

Dataset: Kvasir-SEG  
Split: 700 train / 150 validation / 150 test  
Image size: 352

## Single Models

| Model | Test Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| U-Net | 0.598 | 0.466 | 0.649 | 0.690 |
| DeepLabV3 ResNet50 pretrained | 0.855 | 0.774 | 0.901 | 0.859 |
| FCN ResNet50 pretrained | 0.818 | 0.737 | 0.806 | 0.906 |
| LRASPP MobileNetV3-Large pretrained | 0.866 | 0.789 | 0.905 | 0.870 |

## Complementarity

Pairwise Dice complementarity on the test split:

| Pair | Model A Wins | Model B Wins | Oracle Dice | Oracle Gain |
|---|---:|---:|---:|---:|
| U-Net vs DeepLabV3 | 15 | 135 | 0.866 | +0.0108 |
| U-Net vs FCN ResNet50 | 21 | 129 | 0.837 | +0.0186 |
| DeepLabV3 vs FCN ResNet50 | 76 | 73 | 0.888 | +0.0332 |
| DeepLabV3 vs LRASPP MobileNetV3 | 74 | 75 | 0.893 | +0.0266 |
| FCN ResNet50 vs LRASPP MobileNetV3 | 70 | 79 | 0.898 | +0.0321 |

The strongest current setup is the three-model pool: DeepLabV3, FCN ResNet50, and LRASPP MobileNetV3. All three win a meaningful number of oracle cases.

## Static Ensemble

Mean-probability ensemble on DeepLabV3 + FCN ResNet50:

| Baseline | Test Dice |
|---|---:|
| Best single model | 0.855 |
| Mean-probability ensemble | 0.870 |
| Oracle top-1 | 0.888 |

The two-model static ensemble improves over the best single model by +0.015 Dice. The oracle gap leaves another +0.018 Dice for a QA-guided router or learned weighting method.

Mean-probability ensemble on DeepLabV3 + FCN ResNet50 + LRASPP MobileNetV3:

| Baseline | Test Dice |
|---|---:|
| Best single model | 0.866 |
| Mean-probability ensemble | 0.882 |
| Oracle top-1 | 0.909 |

The three-strong-model static ensemble improves over the best single model by +0.0158 Dice. The oracle gap is +0.0428 Dice, which is a better target for MLEM-QA than the original two-model setup.

Mean-probability ensemble on U-Net + DeepLabV3 + FCN ResNet50:

| Baseline | Test Dice |
|---|---:|
| Best single model | 0.855 |
| Mean-probability ensemble | 0.845 |
| Oracle top-1 | 0.894 |

The naive three-model average is worse because U-Net is much weaker. U-Net still contributes oracle wins on a few cases, but it needs learned gating rather than uniform averaging.

## QA Router

Prototype QA router:

- models: DeepLabV3 + FCN ResNet50 + LRASPP MobileNetV3
- train split for QA calibration: validation
- test split: test
- QA model: random forest regressor predicting per-model Dice from prediction-only features
- QA ensemble policies: top-1 routing, weighted averaging, within-margin subset averaging, and above-threshold subset averaging

| Method | Test Dice | Gain Over Best Single |
|---|---:|---:|
| Best single model | 0.866 | 0.000 |
| Static mean-probability ensemble | 0.882 | +0.0158 |
| QA routed top-1 | 0.868 | +0.0015 |
| QA weighted-all ensemble | 0.883 | +0.0169 |
| QA within-margin subset ensemble | 0.879 | +0.0128 |
| QA above-threshold subset ensemble | 0.878 | +0.0117 |
| Oracle routed top-1 | 0.909 | +0.0428 |

The QA weighted-all ensemble slightly beats the static three-model mean-probability ensemble and beats the best single model. This better matches the intended MLEM-QA behavior: estimate which models are good enough and combine them, rather than forcing a binary model-selection decision.

## Paper Caveat

The current QA router uses the validation split for calibration, and the same validation split was also used for checkpoint selection. This is acceptable for a prototype, but a publishable experiment should add an independent calibration split or use cross-validation for QA training.
