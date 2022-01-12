#!/bin/bash

set -euo pipefail

script_dir="$(dirname "${0}")/../seoklab/scripts"
if [[ ! -d "$script_dir" ]]; then
  git submodule update --init --recursive &>/dev/null
fi

"$script_dir/config-ssh.sh"
"$script_dir/init-jupyter.sh"
