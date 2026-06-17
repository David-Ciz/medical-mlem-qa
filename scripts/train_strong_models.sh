#!/usr/bin/env sh
set -eu

PARAMS_PATH="${PARAMS_PATH:-params.yaml}"

python -m mlem_qa_medseg.train --params "$PARAMS_PATH" --model deeplabv3
python -m mlem_qa_medseg.train --params "$PARAMS_PATH" --model fcn_resnet50
python -m mlem_qa_medseg.train --params "$PARAMS_PATH" --model lraspp_mobilenet_v3_large
