#!/bin/zsh
# Local end-to-end trial runner: env server + entry + scorer, one seed at a time.
# usage: ./run-trial.sh <trial-dir> [seed]
#   <trial-dir> holds input/env/{providers,workload}.json, input/prompt.txt, reference/{ratings,catalog}.json
set -euo pipefail

KIT=/Volumes/Madhouse/makefun/free-model-aggregator-01-kit
ENTRY=/Volumes/Madhouse/makefun/aggregator
NODE=/Volumes/Madhouse/makefun/toolbox/node22/bin/node
TRIAL=${1:?trial dir}
SEED=${2:-local-seed-1}
PORT=$(( 20000 + RANDOM % 20000 ))
WORK=$ENTRY/.trials/$(basename "$TRIAL")-$SEED
mkdir -p "$WORK"

echo "== trial $(basename "$TRIAL") seed=$SEED env=:$PORT"
MAKEFUN_ENV_DATA_DIR="$TRIAL/input/env" \
MAKEFUN_STATE_DIR="$WORK/state" \
MAKEFUN_ENV_HOST=127.0.0.1 MAKEFUN_ENV_PORT=$PORT \
"$NODE" "$KIT/program/env/server.mjs" >"$WORK/env.log" 2>&1 &
ENV_PID=$!
trap 'kill -TERM $ENV_PID 2>/dev/null || true' EXIT

for i in $(seq 1 50); do [ -f "$WORK/state/ready" ] && break; sleep 0.1; done
[ -f "$WORK/state/ready" ] || { echo "env never became ready"; tail "$WORK/env.log"; exit 1; }
echo "env ready: $(cat "$WORK/state/ready")"

(cd "$ENTRY" && MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
  MAKEFUN_PROMPT_FILE="$TRIAL/input/prompt.txt" \
  "$NODE" main.ts >"$WORK/entry.log" 2>&1) &
ENTRY_PID=$!

# wait for the entry to exit 0 (it exits the moment /work says done)
wait $ENTRY_PID || { echo "entry exited nonzero"; tail -40 "$WORK/entry.log"; exit 1; }
sleep 0.5
kill -TERM $ENV_PID
wait $ENV_PID 2>/dev/null || true

mkdir -p "$WORK/scores"
MAKEFUN_OUTPUT_DIR="$WORK/state/export" \
MAKEFUN_REFERENCE_DIR="$TRIAL/reference" \
MAKEFUN_SCORES_DIR="$WORK/scores" \
"$NODE" "$KIT/program/scorer/score.mjs" >"$WORK/score.log" 2>&1 || { echo "scorer failed"; cat "$WORK/score.log"; exit 1; }
cat "$WORK/scores/scores.json" 2>/dev/null || cat "$WORK/score.log"
echo
echo "entry log tail:"; tail -25 "$WORK/entry.log"
