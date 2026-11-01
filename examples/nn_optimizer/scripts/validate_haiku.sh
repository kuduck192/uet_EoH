#!/bin/bash
# Held-out validation of the Claude Haiku runs (runs/R*_haiku) with the protocol of the Qwen runs
# (run_pipeline.sh: stage_bench/stage_validate, stage_validate_r3, stage_step_cost). The Qwen eval tables are
# copied to analysis_out/haiku/ and the Haiku rules merged into the copies (transfer.py --merge-into): Adam's
# reference runs, the baselines and the Qwen rules are reused as they are in the report, and the originals stay
# untouched. baseline:adam is re-run as a determinism check (its rel to adam_ref must be ~0); the NGD analyses
# re-run one Qwen rule or baseline as an anchor against the Qwen files.
#
#   GPU=0 bash scripts/validate_haiku.sh elementwise   # R1/R2: search + transfer suites, alpha, invariance, NGD
#   GPU=1 bash scripts/validate_haiku.sh tensor        # R3: r3 + r3_transfer suites (validation loss), NGD
#   GPU=1 bash scripts/validate_haiku.sh step_cost     # wall-clock per step; only on an otherwise idle machine
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd "$NNOPT_ROOT"
A=analysis_out
H=$A/haiku
EW="run:R1_haiku:3 run:R2_haiku:3"
T3="evolved:R3_haiku:3"
mkdir -p "$H"

stage_elementwise() {
    for f in eval_search eval_transfer; do [ -e "$H/$f.json" ] || cp "$A/$f.json" "$H/"; done
    $PY evaluation/transfer.py --rules $EW baseline:adam --suite search --seeds 5 --seed-offset 1 --workers 6 \
        --merge-into "$H/eval_search.json"
    $PY evaluation/transfer.py --rules $EW --suite transfer --seeds 3 --seed-offset 1 --workers 4 \
        --merge-into "$H/eval_transfer.json"
    for task in softmax_illcond mlp_fmnist; do
        $PY analysis/alpha_fit.py --rules run:R1:1 $EW --task "$task" --out "$H/alpha_$task.json"
    done
    $PY analysis/invariance.py --rules run:R1:1 $EW --task softmax_illcond --out "$H/invariance.json"
    $PY analysis/ngd_alignment.py --rules run:R1:1 $EW --no-full-ngd --out "$H/ngd_alignment.json"
}
stage_tensor() {
    for f in eval_r3 eval_r3_transfer; do [ -e "$H/$f.json" ] || cp "$A/$f.json" "$H/"; done
    $PY evaluation/transfer.py --rules $T3 baseline:adam --suite r3 --metric val --seeds 5 --seed-offset 1 \
        --workers 6 --merge-into "$H/eval_r3.json"
    $PY evaluation/transfer.py --rules $T3 --suite r3_transfer --metric val --seeds 3 --seed-offset 1 \
        --workers 4 --merge-into "$H/eval_r3_transfer.json"
    $PY analysis/ngd_alignment.py --rules baseline:shampoo_cosine $T3 --no-full-ngd \
        --tasks softmax_illcond_rot tiny_fmnist --out "$H/ngd_alignment_r3.json"
}
stage_tolerant() {
    # Haiku's Shampoo variants run float32 eigh without the fallback of the hand-written baseline (baselines.py
    # keeps the old root when eigh fails), so they raise on NaN input, e.g. at tune_lr's lr = 1 and 10 probes
    # (the R3 search only probed up to 0.1), and transfer.py then drops the whole task. The *_tolerant copies
    # only turn an exception in step() into a NaN step, i.e. into the divergence the harness already scores
    # as "no progress from there on"; the update rule itself is unchanged.
    mkdir -p "$H/rules"
    $PY - "$H/rules" <<'EOF'
import os, sys
sys.path[:0] = ["analysis", "."]
from common import resolve_rules
WRAP = '''

# --- validation wrapper, not part of the evolved rule: an exception in step() counts as divergence ---
_EvolvedRule = UpdateRule


class UpdateRule(_EvolvedRule):
    def step(self, p, g, state, t, T, lr):
        try:
            return super().step(p, g, state, t, T, lr)
        except RuntimeError:  # incl. torch.linalg.LinAlgError from eigh on a NaN or degenerate matrix
            return torch.full_like(p, float("nan"))
'''
for r in resolve_rules(["evolved:R3_haiku:3"]):
    name = r["name"].replace("#", "_") + "_tolerant"
    with open(os.path.join(sys.argv[1], name + ".py"), "w") as f:
        f.write(r["code"].rstrip() + "\n" + WRAP)
EOF
    TOL=$(for f in "$H"/rules/R3_haiku_*_tolerant.py; do echo -n "file:$f "; done)
    $PY evaluation/transfer.py --rules $TOL --suite r3 --metric val --seeds 5 --seed-offset 1 --workers 6 \
        --merge-into "$H/eval_r3.json"
    $PY evaluation/transfer.py --rules $TOL --suite r3_transfer --metric val --seeds 3 --seed-offset 1 \
        --workers 4 --merge-into "$H/eval_r3_transfer.json"
    # ngd_alignment.py tunes lr on the same wide grid and does not catch exceptions either
    $PY analysis/ngd_alignment.py --rules baseline:shampoo_cosine $TOL --no-full-ngd \
        --tasks softmax_illcond_rot tiny_fmnist --out "$H/ngd_alignment_r3_tolerant.json"
}
stage_step_cost() {
    $PY analysis/step_cost.py --rules baseline:adam_t baseline:adamw_cosine baseline:muon_cosine \
        baseline:shampoo_cosine evolved:R3:1 evolved:R4:1 run:R1_haiku:1 run:R2_haiku:1 $T3 \
        --tasks mlp_fmnist_v cnn_cifar_v gpt_v --steps 60 --out "$H/step_cost.json"
}

for stage in "$@"; do
    echo "=== $stage (GPU $CUDA_VISIBLE_DEVICES, $(date '+%F %T'))"
    "stage_$stage"
    echo "=== $stage done ($(date '+%F %T'))"
done
