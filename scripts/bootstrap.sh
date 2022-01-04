#!/bin/bash

set -eu

if ! [[ $(uname -s) =~ .*Linux.* ]]; then
	echo "This script is only supported on Linux."
	exit 1
fi

command -v curl &>/dev/null || {
	echo >&2 "curl is required but not installed. Aborting."
	exit 1
}

if type conda &>/dev/null; then
	if conda env list | grep -Eq "^pytorch\b"; then
		echo "pytorch is already installed. Skipping."
		exit 0
	fi
	echo "conda is already installed! using it..."
	. "$CONDA_PREFIX/bin/activate"
else
	echo "conda not found. installing..."

	unset PYTHONPATH
	user_shell="$(basename "$SHELL")"
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

	# Init conda
	. "${conda_prefix}/bin/activate"
	conda init "${user_shell}"
	. "${HOME}/.${user_shell}rc"
fi

# Update and create environment
conda update -yq conda -n base
conda env create -f "$(dirname "$0")/../environment.yml" \
	-n pytorch --no-default-packages

# Check that the environment is working
set +u
conda activate pytorch
command -v python
