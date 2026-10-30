#!/bin/bash
# Serve Qwen3.8-27B-FP8 with vLLM on GPU 0 (OpenAI-compatible API on 127.0.0.1:$LLM_PORT).
# No --reasoning-parser: the chat template opens <think> in the prompt and the
# patched EoH client strips everything up to </think>.
#   nohup bash scripts/serve_llm.sh > runs/vllm.log 2>&1 &
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES=0
export VLLM_CACHE_ROOT=/mnt/data/duclm3/uet_optimization/vllm-cache
# torchcodec in the vLLM venv is a CUDA 13 build that fails to load; see vllm_shims/.
export PYTHONPATH="$NNOPT_ROOT/scripts/vllm_shims${PYTHONPATH:+:$PYTHONPATH}"
# DeepGEMM JIT-compiles FP8 kernels and needs NVCC >= 12.9; the system toolkit is
# CUDA 12.2, so use vLLM's CUTLASS/Triton FP8 block kernels instead.
export VLLM_USE_DEEP_GEMM=0
# FlashInfer JIT-compiles its top-k/top-p sampler (ninja + nvcc); use vLLM's
# PyTorch sampler instead, and keep the venv's tools (ninja) on PATH for other JIT steps.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$VLLM_VENV/bin:$PATH"
exec "$VLLM_VENV/bin/vllm" serve "$QWEN_CKPT" \
    --served-model-name "$LLM_MODEL" \
    --host 127.0.0.1 --port "$LLM_PORT" \
    --language-model-only \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32
