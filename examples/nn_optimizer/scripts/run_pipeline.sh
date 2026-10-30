#!/bin/bash
# Pipeline of the nn_optimizer project. Experiments run on GPU 1; the LLM server
# (scripts/serve_llm.sh) must be up on GPU 0 for the probe/smoke/search stages.
#
#   bash scripts/run_pipeline.sh data selftest calibrate probe search analysis
#   bash scripts/run_pipeline.sh search_r2 search_r3 polish validate r3 validate_r3 search_r4 r4 validate_r4 report
#
# "analysis" = motifs bench ablate ngd population (each can also be run alone; RUN=R2 for R2).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$NNOPT_ROOT"
RUN=${RUN:-R1}
A=analysis_out
BASE="baseline:sgd_momentum baseline:adam baseline:lion baseline:signum baseline:diag_ngd_emp"
BASE_NO_ADAM="baseline:sgd_momentum baseline:lion baseline:signum baseline:diag_ngd_emp"  # transfer.py adds Adam
MINIMAL="file:$A/ablate_${RUN}_1/minimal.py"
mkdir -p runs "$A"

stage_data() {
    NNOPT_DATA="$NNOPT_ROOT/data" CUDA_VISIBLE_DEVICES= $PY prepare_data.py
    mkdir -p "$NNOPT_DATA" && cp -u "$NNOPT_ROOT"/data/*.pt "$NNOPT_DATA"/
}
stage_selftest() { $PY selftest.py; }
stage_calibrate() {
    $PY prob.py --calibrate
    $PY prob.py --calibrate --steps-scale 0.25
}
stage_probe() {
    NNOPT_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY probe_llm.py -n 12 | tee runs/probe_nothink.log
    NNOPT_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY probe_llm.py -n 12 --thinking --temperature 1.0 --top-p 0.95 \
        --max-tokens 16384 | tee runs/probe_think.log
}
stage_smoke() {
    $PY runEoH.py --run "$RUN" --tag smoke --steps-scale 0.25 --pop-size 4 --max-samples 16 \
        --samplers 8 --evaluators 6 --timeout 300
}
stage_search() {
    # Thinking stays off: with it on, 12/12 probe answers hit 16k tokens and only 1/12 was valid.
    $PY runEoH.py --run "$RUN" --pop-size 10 --max-samples 1000 --samplers 12 --evaluators 12 \
        --timeout 240 --deadline-min 150 2>&1 | tee "runs/$RUN.log"
}
stage_search_r2() {
    # Designed after R1: robust fitness and at most 2 individuals per code skeleton.
    $PY runEoH.py --run R2 --pop-size 16 --max-samples 1000 --samplers 12 --evaluators 8 \
        --timeout 240 --deadline-min 150 2>&1 | tee runs/R2.log
}
stage_search_r3() {
    # Per-tensor rules that see T, validation loss, rotated ill-conditioned softmax (second-order pressure).
    $PY prob.py --calibrate --r3
    $PY runEoH.py --run R3 --pop-size 16 --max-samples 600 --samplers 12 --evaluators 8 \
        --timeout 900 --deadline-min 240 2>&1 | tee runs/R3.log
}
stage_motifs() { $PY analysis/motifs.py "$RUN"; }
stage_bench() {
    $PY prob.py --baselines
    $PY evaluation/transfer.py --rules $BASE_NO_ADAM "run:$RUN:3" --suite search --seeds 5 --seed-offset 1 \
        --workers 6 --out "$A/eval_search.json"
    $PY evaluation/transfer.py --rules $BASE_NO_ADAM "run:$RUN:3" --suite transfer --seeds 3 --seed-offset 1 \
        --workers 4 --out "$A/eval_transfer.json"
}
stage_ablate() { $PY analysis/ablate.py --rule "run:$RUN:1" --workers 8 --max-rounds 4; }
stage_ngd() {
    for task in softmax_illcond mlp_fmnist; do
        $PY analysis/alpha_fit.py --rules $BASE "run:$RUN:3" $MINIMAL --task "$task" --out "$A/alpha_$task.json"
    done
    $PY analysis/invariance.py --rules $BASE "run:$RUN:3" $MINIMAL --task softmax_illcond --out "$A/invariance.json"
    $PY analysis/ngd_alignment.py --rules $BASE "run:$RUN:3" $MINIMAL --out "$A/ngd_alignment.json"
}
stage_population() {
    # SUBSAMPLE=n analyses a fixed random subset of n valid samples (the same one in every script)
    SUB=${SUBSAMPLE:+--subsample $SUBSAMPLE}
    $PY analysis/alpha_fit.py --all-samples "$RUN" --task softmax_illcond --workers ${WORKERS:-6} $SUB \
        --out "$A/${RUN}_softmax_illcond_alpha.csv"
    for task in softmax_illcond tiny_fmnist; do
        $PY analysis/ngd_alignment.py --all-samples "$RUN" --task "$task" --workers ${WORKERS:-6} $SUB \
            --out "$A/${RUN}_${task}_alignment.csv"
    done
    $PY analysis/population.py "$RUN" --alpha "$A/${RUN}_softmax_illcond_alpha.csv" \
        --align "$A/${RUN}_softmax_illcond_alignment.csv" "$A/${RUN}_tiny_fmnist_alignment.csv"
}
stage_polish() {
    # Memetic refinement of the best rules: structural edits + (1+lambda)-ES on constants.
    $PY analysis/polish.py --rule run:R1:1 --workers 5
    $PY analysis/polish.py --rule sample:R1:402 --name R1@402 --workers 5
    $PY analysis/polish.py --rule run:R2:1 --workers 8
}
stage_validate() {
    # Fresh seeds and held-out tasks for everything added after the R1 analysis.
    MORE="sample:R1:402 run:R2:3"
    for f in "$A"/polish_*/*_polished.py; do MORE="$MORE file:$f"; done
    $PY evaluation/transfer.py --rules $MORE --suite search --seeds 5 --seed-offset 1 --workers 6 \
        --merge-into "$A/eval_search.json"
    $PY evaluation/transfer.py --rules $MORE --suite transfer --seeds 3 --seed-offset 1 --workers 4 \
        --merge-into "$A/eval_transfer.json"
    for task in softmax_illcond mlp_fmnist; do
        $PY analysis/alpha_fit.py --rules $MORE --task "$task" --out "$A/alpha_${task}_more.json"
    done
    $PY analysis/invariance.py --rules $MORE --task softmax_illcond --out "$A/invariance_more.json"
    $PY analysis/ngd_alignment.py --rules $MORE --no-full-ngd --out "$A/ngd_alignment_more.json"
}
R3BASE="baseline:adamw_cosine baseline:sgd_cosine baseline:lion_cosine baseline:muon_cosine baseline:shampoo_cosine"
stage_r3() {
    $PY analysis/motifs.py R3
    $PY prob.py --baselines --r3
    $PY analysis/polish.py --rule run:R3:1 --metric val --workers 8
}
evolved_rules() {  # top-3 evolved rules of a run (seeds skipped) plus its polished rules, as rule specs
    local rules="evolved:$1:3"
    for f in "$A"/polish_"$1"_*/*_polished.py; do [ -e "$f" ] && rules="$rules file:$f"; done
    echo "$rules"
}
stage_validate_r3() {
    # R3 rules vs best-practice and second-order baselines: fresh seeds and held-out tasks, validation loss.
    RULES=$(evolved_rules R3)
    $PY evaluation/transfer.py --rules $R3BASE $RULES --suite r3 --metric val --seeds 5 --seed-offset 1 \
        --workers ${WORKERS:-6} --out "$A/eval_r3.json"
    $PY evaluation/transfer.py --rules $R3BASE $RULES --suite r3_transfer --metric val --seeds 3 \
        --seed-offset 1 --workers ${WORKERS:-4} --out "$A/eval_r3_transfer.json"
    $PY analysis/ngd_alignment.py --rules baseline:adam_t baseline:adamw_cosine baseline:muon_cosine \
        baseline:shampoo_cosine $RULES --tasks softmax_illcond_rot tiny_fmnist --out "$A/ngd_alignment_r3.json"
}
stage_search_r4() {
    # Seeded (not blind): R3's setting, initial population = Shampoo, Muon and R3's best rules.
    $PY runEoH.py --run R4 --pop-size 16 --max-samples 250 --samplers 12 --evaluators 12 \
        --timeout 900 --deadline-min 150 2>&1 | tee runs/R4.log
}
stage_r4() {
    $PY analysis/motifs.py R4
    $PY analysis/polish.py --rule run:R4:1 --metric val --workers 8
}
stage_validate_r4() {
    RULES=$(evolved_rules R4)
    $PY evaluation/transfer.py --rules $RULES --suite r3 --metric val --seeds 5 --seed-offset 1 \
        --workers ${WORKERS:-6} --merge-into "$A/eval_r3.json"
    # Held-out tasks for R4's best only, without ResNet-18: R4's best has no size cap on its Kronecker
    # factors, so on ResNet-18 it would run eigh on 4608x4608 matrices every 5 steps (hours per run).
    # Own file (with its own Adam reference), merged into the R3 table by make_report.py.
    $PY evaluation/transfer.py --rules evolved:R4:1 --suite gpt_v,mlp_fmnist_long_v,cnn_cifar_bs512_v,softmax_rot_long_v \
        --metric val --seeds 3 --seed-offset 1 --workers ${WORKERS:-4} --out "$A/eval_r3_transfer-R4.json"
    $PY analysis/ngd_alignment.py --rules $RULES --no-full-ngd --tasks softmax_illcond_rot tiny_fmnist \
        --out "$A/ngd_alignment_r4.json"
}
stage_analysis() { stage_motifs; stage_bench; stage_ablate; stage_ngd; stage_population; }
stage_report() { $PY make_report.py; }

for stage in "$@"; do
    echo "=== $stage ($(date '+%F %T'))"
    "stage_$stage"
done
