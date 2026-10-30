# Shadows torchcodec for the text-only vLLM server (serve_llm.sh puts this
# directory on PYTHONPATH). The installed torchcodec wheel is a CUDA 13 build
# (needs libnvrtc.so.13) while the driver only supports CUDA 12.9, so importing
# it raises OSError, which vLLM does not catch. vLLM treats ImportError as
# "audio/video decoding unavailable", which is all we need.
raise ImportError("torchcodec is disabled for the text-only vLLM server")
