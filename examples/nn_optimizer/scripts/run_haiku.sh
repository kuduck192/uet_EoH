#!/bin/bash
# Re-run the blind searches R1-R3 with Claude Haiku 5.5 (Anthropic API, effort "low") in place of the
# local Qwen3.8-27B. Sample budgets, population sizes, concurrency, timeouts and deadlines are those of
# the Qwen runs (runs/R*/config.json); outputs go to runs/R*_haiku[.log]. One search at a time on GPU 1,
# as for Qwen: concurrent searches would share the 16 CPU cores and push more candidates into the
# evaluation timeout.
#
#   screen -dmS eoh-haiku bash -c 'bash scripts/run_haiku.sh > runs/haiku/run_haiku.log 2>&1'
#   bash scripts/run_haiku.sh R3            # a subset; MODEL, EFFORT and TAG can be overridden
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$NNOPT_ROOT"
MODEL=${MODEL:-claude-haiku-5-5}
EFFORT=${EFFORT:-low}
TAG=${TAG:-haiku}

search() {
    local run=$1; shift
    echo "[$(date '+%F %T')] start $run ($MODEL, effort $EFFORT) -> runs/${run}_${TAG}"
    $PY runEoH.py --run "$run" --model "$MODEL" --effort "$EFFORT" --tag "$TAG" --samplers 12 "$@" \
        > "runs/${run}_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] end $run (exit $rc)"
}

for run in ${@:-R1 R2 R3}; do
    case $run in
        R1) search R1 --pop-size 10 --max-samples 1000 --evaluators 12 --timeout 240 --deadline-min 150 ;;
        R2) search R2 --pop-size 16 --max-samples 1000 --evaluators 8 --timeout 240 --deadline-min 150 ;;
        R3) search R3 --pop-size 16 --max-samples 400 --evaluators 14 --timeout 600 --deadline-min 180 ;;
        *) echo "unknown run: $run"; exit 1 ;;
    esac
done
