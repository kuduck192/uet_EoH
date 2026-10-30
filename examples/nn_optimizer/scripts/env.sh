# Environment shared by the nn_optimizer scripts:  source scripts/env.sh
# GPU 0 serves the LLM (serve_llm.sh), GPU 1 runs every experiment.
NNOPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NNOPT_ROOT
export PY=/mnt/data/duclm3/venvs/pytorch-latest/bin/python
export VLLM_VENV=/home/duclm3/projects/uet_optimization/.venv
export QWEN_CKPT=/mnt/data/duclm3/crs_refactor/tdrl/models/Qwen3.8-27B-FP8/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
export LLM_MODEL=qwen3.8-27b
export LLM_PORT=8000
export LLM_ENDPOINT=http://127.0.0.1:$LLM_PORT
export HF_HOME=/mnt/data/duclm3/huggingface-cache
export NNOPT_DATA=/dev/shm/nnopt_data          # copy of $NNOPT_ROOT/data on tmpfs (NFS is slow)
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
