# Containerize Medical MLEM-QA Workflow And Run On Lexis

## Summary

Prepare the `medical-mlem-qa` prototype for Lexis/HPC execution.

The goal is not to prove a new QA method yet. The goal is to make a reproducible medical segmentation workflow that can:

- use a Lexis-hosted dataset;
- train one selected model via a runtime parameter;
- train the current three-model pool as a workflow;
- run oracle headroom, static ensemble, and QA reports as a follow-up workflow/task.

Suggested assignee: Monika

## Background

The repository currently works locally on Kvasir-SEG with these three useful models:

- `deeplabv3`
- `fcn_resnet50`
- `lraspp_mobilenet_v3_large`

Current prototype results:

| Method | Test Dice |
|---|---:|
| Best single model, LRASPP | 0.866 |
| Static 3-model mean ensemble | 0.882 |
| QA weighted-all ensemble | 0.883 |
| Oracle top-1 router | 0.909 |

The QA result is not dramatically better than static averaging, but the workflow is useful as a medical benchmark and as a practical MLEM-QA/Lexis integration task.

## Deliverables

### 1. Upload Dataset To Lexis

Upload Kvasir-SEG as a Lexis dataset.

Dataset requirements:

- include original `images/` and `masks/` directories;
- preserve dataset provenance and citation metadata;
- document dataset name, version, owner/project, and access permissions;
- expose the dataset in a way the training container can mount/read it.

Expected follow-up change in this repo:

- use or adapt `params.lexis.example.yaml` for Lexis-mounted data, for example:

```yaml
data:
  dataset: kvasir-seg
  raw_dir: /lexis/input/kvasir-seg
  processed_dir: /lexis/work/data/processed/kvasir-seg
```

### 2. Containerize The Application

Create a production-ready container from the current starter `Dockerfile`.

The container must support these runtime actions:

- preprocess data;
- train one model;
- evaluate one model;
- export prediction banks;
- run static ensemble/oracle report;
- run QA report.

The current Python entry points are:

```bash
python -m mlem_qa_medseg.preprocess --params params.yaml
python -m mlem_qa_medseg.train --params params.yaml --model deeplabv3
python -m mlem_qa_medseg.evaluate --params params.yaml --model deeplabv3 --split test
python -m mlem_qa_medseg.export_predictions --params params.yaml --model deeplabv3 --split test
python -m mlem_qa_medseg.ensemble --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --split test
python -m mlem_qa_medseg.qa_router --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --train-split val --test-split test
```

Container requirements:

- accept a params/config path at runtime;
- accept model name at runtime for training/evaluation/export;
- mount dataset, model output, reports, and prediction output directories;
- write artifacts outside the container filesystem;
- run without notebook/manual state;
- document image build and run commands;
- push the resulting image to the relevant GitLab/Lexis registry if applicable.

### 3. Runtime Parameters

Expose at least these parameters:

| Parameter | Purpose | Example |
|---|---|---|
| `PARAMS_PATH` or `--params` | Select dataset/config | `params.lexis.kvasir.yaml` |
| `MODEL_NAME` or `--model` | Select model to train/evaluate | `deeplabv3` |
| `SPLIT` or `--split` | Select split for evaluation/export | `test` |
| output paths | Place models/reports/predictions on mounted storage | `/lexis/work/models` |

The model parameter must support at least:

```text
deeplabv3
fcn_resnet50
lraspp_mobilenet_v3_large
```

### 4. Workflow: Train Three Models

Create a Lexis/HPC workflow that trains all three current models.

Required models:

```text
deeplabv3
fcn_resnet50
lraspp_mobilenet_v3_large
```

The workflow may run them sequentially or as parallel jobs, depending on available resources.

Each model task should run the same container with a different model parameter:

```bash
python -m mlem_qa_medseg.train --params <params> --model <model_name>
```

Expected outputs:

```text
models/deeplabv3/best.pt
models/fcn_resnet50/best.pt
models/lraspp_mobilenet_v3_large/best.pt
models/*/history.json
```

### 5. Workflow: Reports, Oracle, And QA

Create a second workflow/task that consumes the trained models and produces reports.

It should:

1. evaluate each trained model on the test split;
2. export validation and test prediction banks;
3. run pairwise complementarity/oracle headroom;
4. run static ensemble report;
5. run QA router/weighted ensemble report.

Useful existing helper:

```bash
PARAMS_PATH=<params> scripts/run_qa_reports.sh
```

Equivalent explicit commands:

```bash
python -m mlem_qa_medseg.evaluate --params <params> --model deeplabv3 --split test
python -m mlem_qa_medseg.evaluate --params <params> --model fcn_resnet50 --split test
python -m mlem_qa_medseg.evaluate --params <params> --model lraspp_mobilenet_v3_large --split test

python -m mlem_qa_medseg.export_predictions --params <params> --model deeplabv3 --split val
python -m mlem_qa_medseg.export_predictions --params <params> --model fcn_resnet50 --split val
python -m mlem_qa_medseg.export_predictions --params <params> --model lraspp_mobilenet_v3_large --split val

python -m mlem_qa_medseg.export_predictions --params <params> --model deeplabv3 --split test
python -m mlem_qa_medseg.export_predictions --params <params> --model fcn_resnet50 --split test
python -m mlem_qa_medseg.export_predictions --params <params> --model lraspp_mobilenet_v3_large --split test

python -m mlem_qa_medseg.ensemble --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --split test --threshold 0.5
python -m mlem_qa_medseg.qa_router --models deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large --train-split val --test-split test --seed 42 --margin 0.03 --threshold 0.80
```

Expected outputs:

```text
reports/deeplabv3/test_summary.json
reports/fcn_resnet50/test_summary.json
reports/lraspp_mobilenet_v3_large/test_summary.json
reports/ensemble/*
reports/qa_router/*
predictions/*/val/manifest.csv
predictions/*/test/manifest.csv
```

### 6. Documentation

Update `README.md` or add Lexis-specific docs with:

- dataset upload/mount instructions;
- container build instructions;
- example training command for one model;
- example command/workflow for all three models;
- example command/workflow for reports and QA;
- expected output paths;
- known limitations.

## Acceptance Criteria

- [ ] Kvasir-SEG is uploaded to Lexis as a reusable dataset with documented metadata.
- [ ] Container image builds reproducibly from the repo.
- [ ] Container can run preprocessing/training/evaluation from command-line parameters.
- [ ] Model selection is controlled by a runtime parameter.
- [ ] Dataset/config selection is controlled by a runtime parameter.
- [ ] Workflow trains `deeplabv3`, `fcn_resnet50`, and `lraspp_mobilenet_v3_large`.
- [ ] Workflow writes model artifacts to mounted/shared storage.
- [ ] Separate workflow/task generates oracle/static ensemble/QA reports.
- [ ] README or Lexis docs include copy-pasteable build/run commands.
- [ ] A small smoke test run is documented, including where outputs are written.

## Notes

The current QA method is intentionally simple. Do not overstate it as a research contribution yet. The main engineering contribution here is a reusable medical AI workflow that can later host stronger QA methods and additional datasets/models.
