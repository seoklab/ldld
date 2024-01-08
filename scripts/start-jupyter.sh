#!/bin/bash

set -euo pipefail

srun_args=(--gpus 1 -p gpu-micro.q)

if [[ $# -ge 1 ]]; then
	srun_args+=(-w "$1")
fi

srun "${srun_args[@]}" bash -c \
	'jupyter notebook --ip "$(dig +short "$(hostname)")" --no-browser'
