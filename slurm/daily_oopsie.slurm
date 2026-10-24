#!/bin/bash
#SBATCH -J oopsie-daily
#SBATCH -A IRI23005
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00:00
#SBATCH -o /corral/utexas/IRI26024/logs/oopsie-daily-%j.log
#SBATCH -e /corral/utexas/IRI26024/logs/oopsie-daily-%j.log

set -euo pipefail
umask 007
SECONDS=0

export HF_HUB_DISABLE_XET=1

# Data lives under the shared Corral root. Code, tools and caches are staged into $SCRATCH for
# every run, as TACC asks for job activity; their canonical copies are the git remotes. The scron
# entry (daily.scron) fetches this repo's main into $STAGE/oopsie-backend before starting this
# script. Git uses the runner's SSH key. Each path can be overridden from the environment.
ROOT=${OOPSIE_ROOT:-/corral/utexas/IRI26024}
DATA_DIR=${OOPSIE_DATA_DIR:-$ROOT/oopsiedata}
INDEX_DIR=${OOPSIE_INDEX_DIR:-$ROOT/episode_index}
REPORTS_DIR=${OOPSIE_REPORTS_DIR:-$ROOT/reports}
LOG_DIR=${OOPSIE_LOG_DIR:-$ROOT/logs}
TOKENS=${OOPSIE_TOKENS:-$DATA_DIR/tokens.yaml}

: "${SCRATCH:?SCRATCH is not set}"
STAGE=${OOPSIE_STAGE:-$SCRATCH/oopsie}
BACKEND=${OOPSIE_BACKEND:-$STAGE/oopsie-backend}
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
  echo "error: no oopsie-backend checkout at $BACKEND; daily.scron stages it, or set OOPSIE_BACKEND" >&2
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

mkdir -p "$LOG_DIR" "$DATA_DIR" "$INDEX_DIR" "$REPORTS_DIR" "$UV_CACHE_DIR" "$HF_HOME"

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

# Stage the docs site that receives stats.json. Without it, stats.json goes to the reports
# directory and is not published.
stage_docs() {
  if [[ -d "$DOCS/.git" ]]; then
    git -C "$DOCS" fetch -q --depth 1 origin "$DOCS_BRANCH" && git -C "$DOCS" reset -q --hard FETCH_HEAD
  else
    git clone -q --depth 1 --branch "$DOCS_BRANCH" "$DOCS_REMOTE" "$DOCS"
  fi
}
docs_rc=0
run_step stage_docs stage_docs || docs_rc=$?
if (( docs_rc == 0 )); then
  STATS_OUT=$DOCS/$STATS_REL
else
  STATS_OUT=$REPORTS_DIR/stats.json
fi

# Pin numpy before 2.4's x86-v2 baseline requirement. Need this for running on vm-small.
run_step setup_numpy bash -c "\"$UV\" add 'numpy<2.4' && \"$UV\" sync --reinstall-package numpy"

# Datasets that still fail after retries are skipped, and the sync exits non-zero.
# Keep going so stats cover everything that did sync; the job is marked failed at the end.
sync_rc=0
run_step sync_hf_org "$UV" run python -m oopsie_backend.ingest.sync \
  --org OopsieData-Submissions \
  --local-dir "$DATA_DIR" \
  --group G-829069 \
  --verbose || sync_rc=$?

# Note: This step is kept as a safety net, but is not needed anymore.
# run_step fix_data_permissions bash -c "
#   find \"$DATA_DIR\" -user \"\$USER\" ! -group G-829069 -exec chgrp G-829069 {} +
#   find \"$DATA_DIR\" -user \"\$USER\" -type d -exec chmod 2770 {} +
#   find \"$DATA_DIR\" -user \"\$USER\" -type f -exec chmod 660 {} +
#   echo \"Set group G-829069 and permissions (dirs 770, files 660) under $DATA_DIR\"
# "

# Index, duplicate checks, QA, critical flags and stats (configs/pipeline.yaml), recomputing only
# what changed. The stats leave out duplicate copies and critical episodes, so they run last.
pipeline_rc=0
run_step pipeline "$UV" run oopsie pipeline run nightly --config configs/pipeline.yaml --explain \
  --set data_dir="$DATA_DIR" \
  --set index_path="$INDEX_DIR/episode_index.parquet" \
  --set cache_dir="$INDEX_DIR/pipeline_cache" \
  --set stats_docs_output="$STATS_OUT" \
  --set reports_dir="$REPORTS_DIR" || pipeline_rc=$?

# Publish the website stats: commit stats.json to oopsie-docs when it changed.
publish_stats() {
  git -C "$DOCS" add "$STATS_REL"
  if git -C "$DOCS" diff --cached --quiet; then
    echo "stats.json unchanged"
    return 0
  fi
  git "${GIT_AUTHOR[@]}" -C "$DOCS" commit -q -m "Update dashboard stats $(date -I)"
  git "${GIT_AUTHOR[@]}" -C "$DOCS" pull -q --rebase origin "$DOCS_BRANCH"
  git -C "$DOCS" push -q origin "HEAD:$DOCS_BRANCH"
}
publish_rc=0
if (( pipeline_rc == 0 && docs_rc == 0 )); then
  run_step publish_stats publish_stats || publish_rc=$?
fi

# The index, pipeline cache and reports are shared with the group, like the synced data.
for dir in "$INDEX_DIR" "$REPORTS_DIR"; do
  chgrp -R G-829069 "$dir"
  find "$dir" -type d -exec chmod 2770 {} +
  find "$dir" -type f -exec chmod 660 {} +
done

chmod g+s "$LOG_DIR"
chgrp G-829069 "$LOG_DIR"/*.log
chmod g+r "$LOG_DIR"/*.log

if (( sync_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but sync_hf_org exited ${sync_rc}; see failed datasets above (job ${SLURM_JOB_ID}) ==="
  exit "$sync_rc"
fi
if (( pipeline_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but the pipeline exited ${pipeline_rc} (job ${SLURM_JOB_ID}) ==="
  exit "$pipeline_rc"
fi
if (( docs_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but staging oopsie-docs exited ${docs_rc}; stats.json is in $REPORTS_DIR (job ${SLURM_JOB_ID}) ==="
  exit "$docs_rc"
fi
if (( publish_rc != 0 )); then
  echo "=== $(date -Is) FAILED: oopsie-daily finished in ${SECONDS}s, but publishing stats.json exited ${publish_rc} (job ${SLURM_JOB_ID}) ==="
  exit "$publish_rc"
fi

echo "=== $(date -Is) SUCCESS: oopsie-daily completed in ${SECONDS}s (job ${SLURM_JOB_ID}) ==="
