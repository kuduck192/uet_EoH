# Claude backend for EoH: the Anthropic Messages API through the official SDK.
#
# Credentials come from the environment (ANTHROPIC_API_KEY, resolved by anthropic.Anthropic()).
# `params` holds extra request fields, e.g. {"max_tokens": 16000, "output_config": {"effort": "low"}}.
# Claude Haiku 5.5 and newer models reject non-default temperature/top_p/top_k, so none are sent.

import json
import logging
import threading
import time

import anthropic

logger = logging.getLogger('eoh')

# Errors that retrying the same request cannot fix.
_PERMANENT = (anthropic.BadRequestError, anthropic.AuthenticationError,
              anthropic.PermissionDeniedError, anthropic.NotFoundError)


class InterfaceAnthropic:
    def __init__(self, model_LLM, timeout=300, params=None, usage_log=None):
        self.model_LLM = model_LLM
        self.params = {"max_tokens": 16000, **(params or {})}
        self.usage_log = usage_log
        self._log_lock = threading.Lock()
        # The SDK itself retries connection errors, 408/409/429 and 5xx with exponential backoff.
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=5)

    def get_response(self, prompt_content: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                response = self.client.messages.create(
                    model=self.model_LLM,
                    messages=[{"role": "user", "content": prompt_content}],
                    **self.params,
                )
            except _PERMANENT as e:
                self._log({"error": f"{type(e).__name__}: {e.status_code}", "latency_s": time.time() - t0})
                logger.warning("Claude API rejected the request: %s", e)
                return None
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                self._log({"error": type(e).__name__, "latency_s": time.time() - t0})
                logger.debug("Claude API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                continue

            usage = response.usage
            self._log({
                "request_id": response._request_id,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_s": time.time() - t0,
            })
            # Thinking arrives in separate blocks; only the text blocks are the answer.
            content = "".join(block.text for block in response.content if block.type == "text").strip()
            if response.stop_reason == "refusal":
                logger.warning("Claude refused the request (%s)", response.stop_details)
            elif response.stop_reason == "max_tokens":
                logger.debug("Claude hit max_tokens=%d; the answer may be truncated", self.params["max_tokens"])
            if content:
                return content
            logger.debug("Claude returned no text (attempt %d/%d, stop_reason=%s)",
                         attempt + 1, max_retries, response.stop_reason)

        logger.warning("Claude API call failed after %d attempts (model=%s).", max_retries, self.model_LLM)
        return None

    def _log(self, record):
        """Append one JSON line per call, so the token cost of a run can be audited afterwards."""
        if not self.usage_log:
            return
        line = json.dumps({"t": time.time(), "model": self.model_LLM, **record})
        with self._log_lock, open(self.usage_log, "a") as f:
            f.write(line + "\n")
