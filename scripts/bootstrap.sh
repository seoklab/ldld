#!/bin/bash

set -i

. /etc/profile
. ~/.profile
. /etc/bash.bashrc
. ~/.bashrc

set -e

if ! [[ $(uname -s) =~ .*Linux.* ]]; then
	echo "This script is only supported on Linux."
	exit 1
fi

command -v curl &>/dev/null || {
	echo >&2 "curl is required but not installed. Aborting."
	exit 1
}

if type conda &>/dev/null; then
	echo "conda is already installed! using it..."

	. "$(dirname "${CONDA_EXE}")/activate"
else
	echo "conda not found. installing..."

	set -u

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

	set +u

	# Init conda
	. "${conda_prefix}/bin/activate"
	conda init --all
fi

# Update and create environment
if conda env list | grep -Eq "^ldld\b"; then
	echo "ldld is already installed. Skipping."
else
	conda update -yq conda -n base
	conda env create -f "$(dirname "$0")/../environment.yml" \
		-n ldld --no-default-packages -q

	cat >>"${CONDA_PREFIX}/envs/ldld/.condarc" <<-EOM
		channels:
		  - dglteam
		  - pytorch-lts
		  - conda-forge
		  - defaults
	EOM

	cat >>"${CONDA_PREFIX}/envs/ldld/conda-meta/pinned" <<-EOM
		python 3.9.*
		pytorch 1.8.*
		cudatoolkit 11.1.*
	EOM
fi

# Check that the environment is working
. "${CONDA_PREFIX}/bin/activate" ldld
# Install jupyter notebook
if ! ( pip list --format=freeze | cut -d'=' -f 1 | grep -q '^jupyter$' ); then
	pip install -U jupyter
fi
