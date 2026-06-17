#!/usr/bin/env sh
set -eu

PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
MODELS="deeplabv3 fcn_resnet50 lraspp_mobilenet_v3_large"

for MODEL_NAME in $MODELS; do
    python -m mlem_qa_medseg.evaluate --params "$PARAMS_PATH" --model "$MODEL_NAME" --split test
    python -m mlem_qa_medseg.export_predictions --params "$PARAMS_PATH" --model "$MODEL_NAME" --split test
    python -m mlem_qa_medseg.export_predictions --params "$PARAMS_PATH" --model "$MODEL_NAME" --split val
done

python -m mlem_qa_medseg.ensemble --models $MODELS --split test --threshold 0.5
python -m mlem_qa_medseg.qa_router --models $MODELS --train-split val --test-split test --seed 42 --margin 0.03 --threshold 0.80
