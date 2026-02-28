#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

conda env create --prefix "${repo_root}/.conda" --file environment.yml
echo "Conda env created at ${repo_root}/.conda"
echo "Activate with: conda activate ${repo_root}/.conda"
