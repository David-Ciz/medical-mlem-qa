#!/usr/bin/env sh
set -eu

PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to one of: deeplabv3, fcn_resnet50, lraspp_mobilenet_v3_large}"

python -m mlem_qa_medseg.train --params "$PARAMS_PATH" --model "$MODEL_NAME"
