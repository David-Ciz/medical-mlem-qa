# Medical MLEM-QA

Prototype medical segmentation workflow for testing MLEM-QA style model selection and model combination.

The current dataset is **Kvasir-SEG** polyp segmentation. The current useful model pool is:

- `deeplabv3`
- `fcn_resnet50`
- `lraspp_mobilenet_v3_large`

The prototype is not a strong paper result yet. Static averaging already performs about as well as the first QA-weighted ensemble. The practical value of the repo is that it provides a concrete medical workflow for dataset upload, containerization, Lexis/HPC execution, and QA/oracle analysis.

## Current State

The repo can:

- download and preprocess Kvasir-SEG;
- train a selected segmentation model using `--model`;
- evaluate a trained model;
- export prediction banks for validation/test splits;
- compute pairwise oracle headroom;
- compute static mean-probability ensembles;
- train a first QA regressor and evaluate QA-guided routing/weighted combination.

Latest local result summary:

| Method | Test Dice |
|---|---:|
| DeepLabV3 | 0.855 |
| FCN ResNet50 | 0.818 |
| LRASPP MobileNetV3 | 0.866 |
| Static 3-model mean ensemble | 0.882 |
| QA weighted-all ensemble | 0.883 |
| Oracle top-1 router | 0.909 |

See [RESULTS.md](RESULTS.md) for details.

## Repository Layout

```text
.
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── scripts
│   ├── run_qa_reports.sh
│   ├── train_model.sh
│   └── train_strong_models.sh
├── src/mlem_qa_medseg
│   ├── data.py
│   ├── ensemble.py
│   ├── evaluate.py
│   ├── export_predictions.py
│   ├── metrics.py
│   ├── models.py
│   ├── preprocess.py
│   ├── qa_router.py
│   └── train.py
└── RESULTS.md
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Apple Silicon, run from a normal shell so PyTorch can see MPS. The code selects device in this order:

```text
cuda -> mps -> cpu
```

## Data

Preprocess Kvasir-SEG:

```bash
python -m mlem_qa_medseg.preprocess --params params.yaml
```

For a mounted dataset, use a params file with an empty `data.url` and `data.raw_dir` pointing to a directory containing `images/` and `masks/`. A Lexis-oriented template is included:

```text
params.lexis.example.yaml
```

Generated data goes under:

```text
data/raw/kvasir-seg
data/processed/kvasir-seg
```

Data and generated artifacts are ignored by git and intended to be tracked through DVC or a Lexis dataset mount.

## Train One Model

```bash
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3
```

Available model names:

```text
unet
deeplabv3
deeplabv3_resnet101
deeplabv3_mobilenet_v3_large
fcn_resnet50
fcn_resnet101
lraspp_mobilenet_v3_large
```

The current recommended pool is:

```bash
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3
python -m mlem_qa_medseg.train --params params.yaml --model fcn_resnet50
python -m mlem_qa_medseg.train --params params.yaml --model lraspp_mobilenet_v3_large
```

Equivalent helper:

```bash
PARAMS_PATH=params.yaml scripts/train_strong_models.sh
```

## Evaluate And QA

For one model:

```bash
python -m mlem_qa_medseg.evaluate --params params.yaml --model deeplabv3 --split test
python -m mlem_qa_medseg.export_predictions --params params.yaml --model deeplabv3 --split test
```

For the current three-model pool:

```bash
PARAMS_PATH=params.yaml scripts/run_qa_reports.sh
```

That script evaluates each model, exports validation/test prediction banks, runs the static ensemble, and runs the QA router.

## DVC

DVC stages are configured for preprocessing, training, evaluation, prediction export, ensemble reporting, and QA reporting:

```bash
dvc repro
```

Useful targeted stages:

```bash
dvc repro train_deeplabv3 train_fcn_resnet50 train_lraspp_mobilenet_v3_large
dvc repro export_predictions export_qa_calibration_predictions ensemble qa_router
```

The data itself should not be committed to git. Add a DVC remote or replace local download with a Lexis dataset mount for shared execution.

## Container

A starter Dockerfile is included.

Build:

```bash
docker build -t mlem-qa-medseg:local .
```

Train one model:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/predictions:/app/predictions" \
  mlem-qa-medseg:local \
  mlem_qa_medseg.train --params params.yaml --model deeplabv3
```

For Lexis/HPC, this should be converted into the platform's expected container format and dataset mount convention. The intended runtime parameters are:

- dataset/config selection: `--params <params.yaml>`
- model selection: `--model <model_name>`
- action selection: train, evaluate, export predictions, ensemble, or QA router

## Handoff Notes

The GitLab issue for the Lexis/container workflow is prepared in:

```text
.gitlab/issue_templates/lexis_container_workflow.md
```

The issue is scoped for Monika to upload the dataset to Lexis, containerize this application, expose model/dataset parameters, and create workflows that train all three models and run oracle/QA reports.
