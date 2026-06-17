# Medical MLEM-QA Run Steps

## Setup

From the repository root:

```bash
cd /Users/davidciz/Work/innovaite/medical-mlem-qa
source .venv/bin/activate
```

The code auto-selects the best available device in this order:

```text
cuda -> mps -> cpu
```

## Current Pipeline

Download and preprocess Kvasir-SEG:

```bash
python -m mlem_qa_medseg.preprocess --params params.yaml
```

Train the current models:

```bash
python -m mlem_qa_medseg.train --params params.yaml --model unet
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3
python -m mlem_qa_medseg.train --params params.yaml --model fcn_resnet50
python -m mlem_qa_medseg.train --params params.yaml --model lraspp_mobilenet_v3_large
```

Additional model pool candidates:

```bash
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3_resnet101
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3_mobilenet_v3_large
python -m mlem_qa_medseg.train --params params.yaml --model fcn_resnet101
```

Evaluate models on the test split:

```bash
python -m mlem_qa_medseg.evaluate --params params.yaml --model unet --split test
python -m mlem_qa_medseg.evaluate --params params.yaml --model deeplabv3 --split test
python -m mlem_qa_medseg.evaluate --params params.yaml --model fcn_resnet50 --split test
python -m mlem_qa_medseg.evaluate --params params.yaml --model lraspp_mobilenet_v3_large --split test
```

Export test predictions:

```bash
python -m mlem_qa_medseg.export_predictions --params params.yaml --model unet --split test
python -m mlem_qa_medseg.export_predictions --params params.yaml --model deeplabv3 --split test
python -m mlem_qa_medseg.export_predictions --params params.yaml --model fcn_resnet50 --split test
python -m mlem_qa_medseg.export_predictions --params params.yaml --model lraspp_mobilenet_v3_large --split test
```

Run complementarity analysis:

```bash
python -m mlem_qa_medseg.complementarity --model-a unet --model-b deeplabv3 --split test --metric dice
python -m mlem_qa_medseg.complementarity --model-a deeplabv3 --model-b fcn_resnet50 --split test --metric dice
python -m mlem_qa_medseg.complementarity --model-a unet --model-b fcn_resnet50 --split test --metric dice
python -m mlem_qa_medseg.complementarity --model-a deeplabv3 --model-b lraspp_mobilenet_v3_large --split test --metric dice
python -m mlem_qa_medseg.complementarity --model-a fcn_resnet50 --model-b lraspp_mobilenet_v3_large --split test --metric dice
```

Run the static mean-probability ensemble and oracle reference:

```bash
python -m mlem_qa_medseg.ensemble --models unet deeplabv3 fcn_resnet50 --split test --threshold 0.5
python -m mlem_qa_medseg.ensemble --models deeplabv3 fcn_resnet50 --split test --threshold 0.5
python -m mlem_qa_medseg.ensemble --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --split test --threshold 0.5
```

Export validation predictions for QA calibration:

```bash
python -m mlem_qa_medseg.export_predictions --params params.yaml --model deeplabv3 --split val
python -m mlem_qa_medseg.export_predictions --params params.yaml --model fcn_resnet50 --split val
python -m mlem_qa_medseg.export_predictions --params params.yaml --model lraspp_mobilenet_v3_large --split val
```

Train and evaluate the first QA router:

```bash
python -m mlem_qa_medseg.qa_router --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --train-split val --test-split test --seed 42 --margin 0.03 --threshold 0.80
```

## Current Results

Test metrics from the first successful run:

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| U-Net | 0.598 | 0.466 | 0.649 | 0.690 |
| DeepLabV3 ResNet50 pretrained | 0.855 | 0.774 | 0.901 | 0.859 |
| FCN ResNet50 pretrained | 0.818 | 0.737 | 0.806 | 0.906 |
| LRASPP MobileNetV3-Large pretrained | 0.866 | 0.789 | 0.905 | 0.870 |

Complementarity:

| Metric | Value |
|---|---:|
| U-Net wins | 15 / 150 |
| DeepLabV3 wins | 135 / 150 |
| Oracle best-of-two Dice | 0.866 |
| Gain over best single model | +0.0108 Dice |

Strong-model complementarity:

| Metric | Value |
|---|---:|
| DeepLabV3 wins | 76 / 150 |
| FCN ResNet50 wins | 73 / 150 |
| Ties | 1 / 150 |
| Oracle best-of-two Dice | 0.888 |
| Gain over best single model | +0.0332 Dice |

Two-model DeepLabV3 + FCN baselines:

| Method | Test Dice |
|---|---:|
| Best single model | 0.855 |
| Mean-probability ensemble | 0.870 |
| QA routed top-1 | 0.865 |
| Oracle routed top-1 | 0.888 |

Three-strong-model baselines:

| Method | Test Dice |
|---|---:|
| Best single model | 0.866 |
| Mean-probability ensemble | 0.882 |
| QA weighted-all ensemble | 0.883 |
| Oracle routed top-1 | 0.909 |

## Interpretation

DeepLabV3 is a credible baseline. The current U-Net is too weak compared with DeepLabV3, but it still wins on 10% of test cases, so there is some complementarity.

FCN ResNet50 is weaker on average than DeepLabV3, but it wins on nearly half of the test cases. This is the current best pair for MLEM-QA routing/ensembling.

For a stronger paper setup, the next model candidates are:

- SegFormer
- UNet++ or FPN with a pretrained encoder
- PraNet-style model
- Polyp-PVT-style model

The current QA router is a prototype. It trains on the validation split, which was also used for checkpoint selection. For a paper, we need to add an independent QA calibration split or use cross-validation.

## Useful Files

- `params.yaml`: experiment configuration
- `models/unet/`: trained U-Net checkpoint and history
- `models/deeplabv3/`: trained DeepLabV3 checkpoint and history
- `models/deeplabv3_resnet101/`: optional trained DeepLabV3-ResNet101 checkpoint and history
- `models/deeplabv3_mobilenet_v3_large/`: optional trained DeepLabV3-MobileNetV3 checkpoint and history
- `models/fcn_resnet50/`: trained FCN-ResNet50 checkpoint and history
- `models/fcn_resnet101/`: optional trained FCN-ResNet101 checkpoint and history
- `models/lraspp_mobilenet_v3_large/`: optional trained LRASPP-MobileNetV3 checkpoint and history
- `predictions/unet/test/manifest.csv`: exported U-Net test predictions
- `predictions/deeplabv3/test/manifest.csv`: exported DeepLabV3 test predictions
- `predictions/fcn_resnet50/test/manifest.csv`: exported FCN-ResNet50 test predictions
- `reports/ensemble/`: static ensemble and oracle summaries
- `reports/qa_router/`: QA routing summaries
- `RESULTS.md`: current metrics and interpretation
- `src/mlem_qa_medseg/`: package source code
