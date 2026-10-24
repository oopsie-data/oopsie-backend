# Shared setup of the oopsie jobs; sourced with BACKEND set to the staged oopsie-backend checkout.
#
# Data lives under the shared Corral root, which TACC does not intend for computation. Code, tools,
# caches and the episodes that jobs read heavily (QA, training, imports) are staged into $SCRATCH;
# the canonical copies of the code are the git remotes. Git uses the runner's SSH key. Each path can
# be overridden from the environment.
ROOT=${OOPSIE_ROOT:-/corral/utexas/IRI26024}
DATA_DIR=${OOPSIE_DATA_DIR:-$ROOT/oopsiedata}
INDEX_DIR=${OOPSIE_INDEX_DIR:-$ROOT/episode_index}
REPORTS_DIR=${OOPSIE_REPORTS_DIR:-$ROOT/reports}
LOG_DIR=${OOPSIE_LOG_DIR:-$ROOT/logs}
TOKENS=${OOPSIE_TOKENS:-$DATA_DIR/tokens.yaml}

: "${SCRATCH:?SCRATCH is not set}"
STAGE=${OOPSIE_STAGE:-$SCRATCH/oopsie}
EPISODE_STAGE=${OOPSIE_EPISODE_STAGE:-$STAGE/episodes}
DOCS=${OOPSIE_DOCS:-$STAGE/oopsie-docs}
DOCS_REMOTE=${OOPSIE_DOCS_REMOTE:-git@github.com:oopsie-data/oopsie-docs.git}
DOCS_BRANCH=${OOPSIE_DOCS_BRANCH:-main}
STATS_REL=docs/assets/data/stats.json
GIT_AUTHOR=(-c user.name="oopsie-daily" -c user.email="${OOPSIE_GIT_EMAIL:-$USER@tacc.utexas.edu}")
UV=${UV:-$STAGE/tools/bin/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$STAGE/tools/uv-python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$STAGE/uv-cache}
export HF_HOME=${HF_HOME:-$STAGE/hf_home}

if [[ ! -x "$UV" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(dirname "$UV")" UV_NO_MODIFY_PATH=1 sh
fi
if [[ ! -x "$UV" ]]; then
  echo "error: could not install uv into $(dirname "$UV")" >&2
  exit 1
fi
if [[ ! -f "$BACKEND/pyproject.toml" ]]; then
  echo "error: no oopsie-backend checkout at $BACKEND; set OOPSIE_BACKEND" >&2
  exit 1
fi
# The Hugging Face token is the `master` entry's hf_token in tokens.yaml, unless HF_TOKEN is set.
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "$TOKENS" ]]; then
    HF_TOKEN="$(awk '/^master:/{m=1; next} /^[^[:space:]]/{m=0} m && $1 == "hf_token:" {print $2}' "$TOKENS")"
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "error: set HF_TOKEN or put master.hf_token in $TOKENS (mode 640, group G-829069)" >&2
    exit 1
  fi
  export HF_TOKEN
fi

mkdir -p "$LOG_DIR" "$DATA_DIR" "$INDEX_DIR" "$REPORTS_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$EPISODE_STAGE"

cd "$BACKEND"

run_step() {
  local name=$1
  shift
  local step_start=$SECONDS
  echo "=== $(date -Is) START: ${name} ==="
  local rc=0
  "$@" || rc=$?
  if (( rc != 0 )); then
    echo "=== $(date -Is) FAILED: ${name} exited ${rc} after $((SECONDS - step_start))s ==="
    return "$rc"
  fi
  echo "=== $(date -Is) SUCCESS: ${name} completed in $((SECONDS - step_start))s ==="
}

# Inputs of `oopsie pipeline run` for this layout; append further --set options as needed.
PIPELINE_SETTINGS=(
  --config configs/pipeline.yaml
  --set data_dir="$DATA_DIR"
  --set index_path="$INDEX_DIR/episode_index.parquet"
  --set cache_dir="$INDEX_DIR/pipeline_cache"
  --set reports_dir="$REPORTS_DIR"
  --set stage_dir="$EPISODE_STAGE"
  --set training_dir="$ROOT/training"
  --set training_work_dir="$STAGE/training"
)
