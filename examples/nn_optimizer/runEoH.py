"""Run one EoH optimizer search.

Runs of this project (both "blind": SGD-momentum template, no seed rules, inputs
(p, g, state, t, lr), and a task description that never mentions natural
gradients or the Fisher matrix):
    R1  fitness = mean over tasks of the score relative to Adam
    R2  robust fitness (mean + worst task's excess over Adam) and at most two
        individuals per code skeleton in the population (run with --pop-size 16)
    R3  as R2, but per-tensor rules that also receive the run length T (any tensor
        operation, incl. matrix functions), held-out validation loss, tasks of
        different lengths and the rotated ill-conditioned softmax problem
    R4  as R3, but *seeded* (not blind): the initial population is Shampoo, Muon
        and R3's best rules, so EoH starts from Kronecker / orthogonalised
        preconditioners and can recombine them with R3's schedule
The "expert" (Adam template + baseline seeds) and "fisher" (Fisher-aware)
variants are kept from the original design but were not run.

On the GPU server (LLM on GPU 0, evaluations on GPU 1; see scripts/):

    nohup bash scripts/serve_llm.sh > runs/vllm.log 2>&1 &
    python prepare_data.py && python prob.py --calibrate
    python runEoH.py --run R1 --model qwen3.8-27b --deadline-min 210

Outputs go to runs/<run>[_<tag>]/results/{pops,pops_best,samples,eval_log/,run_log.txt}.
To resume, start a new --tag with --continue-from <population json> and
--max-samples set to the remaining budget (EoH restarts its sample numbering).
"""

import argparse
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eoh', 'src'))

from collections import Counter

import eoh.eoh.eoh as eoh_core
from eoh import EoH, LLMConfig
from eoh.eoh.eoh import EOH
from baselines import baselines_for
from harness import code_skeleton
from prob import PROBE_LRS_R3, NNOptimizerSearch
from tasks import R3_SUITE, SEARCH_SUITE

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "R1": dict(use_fisher=False, template="sgd", seeded=False, aggregate="mean", skeleton_cap=None),
    "R2": dict(use_fisher=False, template="sgd", seeded=False, aggregate="robust", skeleton_cap=2),
    "R3": dict(use_fisher=False, template="tensor", seeded=False, aggregate="robust", skeleton_cap=2,
               metric="val", tasks=R3_SUITE, probe_lrs=PROBE_LRS_R3),
    "R4": dict(use_fisher=False, template="tensor", seeded=True, aggregate="robust", skeleton_cap=2,
               metric="val", tasks=R3_SUITE, probe_lrs=PROBE_LRS_R3,
               seeds=["baseline:shampoo_cosine", "baseline:muon_cosine", "run:R3:1", "sample:R3:300",
                      "file:analysis_out/polish_R3_1/R3_1_polished.py"]),
    "expert": dict(use_fisher=False, template="adam", seeded=True, aggregate="mean", skeleton_cap=None),
    "fisher": dict(use_fisher=True, template="adam", seeded=True, aggregate="mean", skeleton_cap=None),
}

# Descriptions for seed rules that have none (polished files).
SEED_ALGORITHMS = {
    "R3_1_polished": "Nesterov momentum divided by the root of a bias-corrected EMA of squared gradients, "
                     "with cosine decay of the learning rate to a 1.2% floor (R3's best rule after "
                     "tuning its constants by local search).",
}

# Flush sample records every 20 samples instead of 200, so a killed run loses little.
EOH._SAMPLE_BATCH = 20


def cap_population_per_skeleton(cap):
    """Make EoH keep at most `cap` individuals per code skeleton (rules that differ
    only in constants); remaining slots are filled by fitness only if too few
    distinct skeletons exist. Counters the population filling up with
    hyper-parameter clones of one rule."""
    original = eoh_core.population_management

    def manage(pop, size):
        ranked = original(pop, len(pop))  # valid, de-duplicated by objective, best first
        kept, overflow, count = [], [], Counter()
        for ind in ranked:
            sk = code_skeleton(ind["code"])
            (kept if count[sk] < cap else overflow).append(ind)
            count[sk] += 1
        return (kept + overflow)[:size]

    eoh_core.population_management = manage


def set_deadline(minutes):
    """Interrupt the search after `minutes`: EOH.run's finally-block then waits for
    running evaluations and flushes the sample buffer; periodic population
    checkpoints are already on disk."""
    def handler(signum, frame):
        raise KeyboardInterrupt(f"deadline of {minutes} min reached")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(minutes * 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, default="R1")
    ap.add_argument("--tag", default="", help="suffix for repeated or resumed runs, e.g. smoke")
    ap.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="model name as served by vLLM")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096, help="cap on generated tokens (incl. reasoning)")
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--samplers", type=int, default=12)
    ap.add_argument("--evaluators", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=480, help="seconds per evaluation")
    ap.add_argument("--deadline-min", type=float, default=0, help="stop the search after this many minutes")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="<1 for smoke tests")
    ap.add_argument("--tasks", nargs="+", default=None, help="default: the run's suite")
    ap.add_argument("--continue-from", default=None, help="population JSON to resume from")
    ap.add_argument("--continue-id", type=int, default=0)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set LLM_MODEL)")

    cfg = RUNS[args.run]
    out_dir = os.path.join(HERE, "runs", args.run + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    seed_path = os.path.join(out_dir, "seeds.json")
    if cfg["seeded"] and "seeds" in cfg:
        sys.path.append(os.path.join(HERE, "analysis"))
        from common import resolve_rules
        os.chdir(HERE)  # file: specs are relative to the project
        seeds = [dict(algorithm=r["algorithm"] or SEED_ALGORITHMS[r["name"]], name=r["name"],
                      code=r["code"].split("\n", 1)[1] if r["code"].startswith("# ") else r["code"])
                 for r in resolve_rules(cfg["seeds"])]  # (polished files start with a "# <trace>" line)
        with open(seed_path, "w") as f:
            json.dump(seeds, f, indent=2)
    elif cfg["seeded"]:
        seeds = [dict(algorithm=b["algorithm"], code=b["code"])
                 for b in baselines_for(cfg["use_fisher"]).values()]
        with open(seed_path, "w") as f:
            json.dump(seeds, f, indent=2)

    extra_body = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    llm = LLMConfig(api_endpoint=args.endpoint, api_key=args.api_key, model=args.model,
                    timeout=900 if args.thinking else 300, extra_body=extra_body)

    task = NNOptimizerSearch(
        use_fisher=cfg["use_fisher"],
        template=cfg["template"],
        tasks=args.tasks or cfg.get("tasks", SEARCH_SUITE),
        steps_scale=args.steps_scale,
        log_dir=os.path.join(out_dir, "results", "eval_log"),
        timeout=args.timeout,
        compute_reference=False,
        aggregate=cfg["aggregate"],
        metric=cfg.get("metric", "train"),
        **({"probe_lrs": cfg["probe_lrs"]} if "probe_lrs" in cfg else {}),
    )
    if cfg["skeleton_cap"]:
        cap_population_per_skeleton(cfg["skeleton_cap"])

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), **cfg, template_program=task.template_program,
                       task_description=task.task_description, reference=task.reference), f, indent=2)

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=args.pop_size,
        n_pop=max(1, args.max_samples // args.pop_size),
        max_sample_nums=args.max_samples,
        operators=['e1', 'e2', 'm1', 'm2', 'm3'],
        operator_weights=[1.0, 1.0, 1.0, 1.0, 0.5],
        num_samplers=args.samplers,
        num_evaluators=args.evaluators,
        output_dir=out_dir,
        use_seed=cfg["seeded"] and args.continue_from is None,
        seed_path=seed_path,
        use_continue=args.continue_from is not None,
        continue_path=args.continue_from or "",
        continue_id=args.continue_id,
    )
    if args.deadline_min:
        set_deadline(args.deadline_min)
    try:
        eoh.run()
    except KeyboardInterrupt as e:
        print(f"[runEoH] stopped: {e}", flush=True)


if __name__ == "__main__":
    main()
