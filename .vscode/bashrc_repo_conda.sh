# Load user shell setup first.
if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi

# Ensure conda function is available.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Force this workspace's environment as final active env.
if command -v conda >/dev/null 2>&1; then
  conda deactivate >/dev/null 2>&1 || true
  if [ -n "$REPO_CONDA_PREFIX" ]; then
    conda activate "$REPO_CONDA_PREFIX" >/dev/null 2>&1 || true
  fi
fi
