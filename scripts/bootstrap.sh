#!/bin/bash

if [[ $(uname -s) != *Linux* ]]; then
	echo "This script is only supported on Linux."
	exit 1
fi

if ! command -v curl &>/dev/null; then
	echo >&2 "curl is required but not installed. Aborting."
	exit 1
fi

if type conda &>/dev/null; then
	echo "conda is already installed! using it..."

	. "$(dirname "${CONDA_EXE}")/activate" base
else
	echo "conda not found. installing..."

	set -euo pipefail

	unset PYTHONPATH
	conda_prefix="$HOME/miniforge3"

	# Download miniforge3 and install
	tmpd="$(mktemp -d)"
	installer="${tmpd}/miniforge3.sh"
	curl -o "$installer" -fsSL \
		"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
	chmod +x "$installer"
	"$installer" -bfup "${conda_prefix}"

	# Cleanup
	rm -f "$installer"
	rmdir "$tmpd"

	set +eu

	# Init conda
	. "${conda_prefix}/bin/activate" base
	conda init --all
fi

set -euo pipefail

# Update and create environment
if conda env list | grep -Eq "^ldld\b"; then
	echo "ldld is already installed. Skipping."
else
	conda update -yq conda -n base
	conda env create -f "$(dirname "$0")/../environment.yml" \
		-n ldld --no-default-packages

	cat >>"${CONDA_PREFIX}/envs/ldld/.condarc" <<-EOM
		channels:
		  - nvidia
		  - dglteam
		  - pytorch
		  - conda-forge
		  - defaults
	EOM

	cat >>"${CONDA_PREFIX}/envs/ldld/conda-meta/pinned" <<-EOM
		python 3.10.*
		pytorch 1.13.*
		pytorch-cuda 11.6.*
		cuda 11.6.*
	EOM
fi

set +e

# Check that the environment is working
. "${CONDA_PREFIX}/bin/activate" ldld
