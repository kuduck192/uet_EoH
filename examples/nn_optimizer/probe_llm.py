"""Probe an LLM endpoint with EoH's initialisation prompt before a long run:
latency, response length, and the fraction of responses that yield a
compilable, element-wise UpdateRule.

    NNOPT_DEVICE=cpu python probe_llm.py -n 12 [--thinking --temperature 1.0 --top-p 0.95 --max-tokens 16384]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eoh', 'src'))

from eoh.llm.api_general import InterfaceAPI
from eoh.eoh.evolution import Evolution
from harness import check_elementwise, compile_rule
from prob import make_description, make_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--template", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--fisher", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set LLM_MODEL)")

    body = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": args.thinking}}
    api = InterfaceAPI(args.endpoint, "EMPTY", args.model, timeout=900, extra_body=body)
    template = make_template(args.fisher, args.template)
    prompt = (f"{make_description(args.fisher)}\nFirst, describe your new algorithm and main steps in one "
              "sentence. The description must be inside a brace. Next, implement the following Python class:\n"
              f"```python\n{template.strip()}\n```\nDo not give additional explanations.")

    def one(_):
        t0 = time.time()
        resp = api.get_response(prompt)
        return time.time() - t0, resp

    with ThreadPoolExecutor(args.n) as pool:
        results = list(pool.map(one, range(args.n)))

    ok, no_code = 0, 0
    for i, (dt, resp) in enumerate(results):
        if resp is None:
            print(f"[{i}] {dt:.0f}s  no response")
            continue
        algorithm, code = Evolution._extract(None, resp)
        status = "no code"
        if algorithm and code:
            try:
                reason = check_elementwise(compile_rule(code[0]), args.fisher)
                status = "VALID" if reason is None else f"rejected: {reason}"
                ok += reason is None
            except Exception as e:
                status = f"compile error: {type(e).__name__}: {e}"
        else:
            no_code += 1
        print(f"[{i}] {dt:.0f}s  {len(resp)} chars  {status}\n     {(algorithm or ['-'])[0][:200]}")
    lat = [dt for dt, r in results if r is not None]
    print(f"\nthinking={args.thinking}  valid {ok}/{args.n}  no-code {no_code}  "
          f"median latency {np.median(lat) if lat else float('nan'):.0f}s  max {max(lat, default=float('nan')):.0f}s")


if __name__ == "__main__":
    main()
