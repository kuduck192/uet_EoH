# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import http.client
import json
import logging
import re
import time

logger = logging.getLogger('eoh')

# Reasoning models (Qwen3, DeepSeek-R1, ...) may emit their chain of thought
# inline. It must not reach code/description extraction, which would otherwise
# pick up code blocks or {braces} from the reasoning instead of the answer.
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_reasoning(content: str) -> str:
    content = _THINK_RE.sub('', content)
    # Unclosed tag (truncated reasoning) or a dangling close tag.
    if '<think>' in content:
        content = content[:content.find('<think>')]
    if '</think>' in content:
        content = content[content.rfind('</think>') + len('</think>'):]
    return content.strip()


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, timeout=60, extra_body=None):
        # api_endpoint is a bare host ("api.deepseek.com", HTTPS) or a URL with
        # an explicit scheme ("http://localhost:8000" for a local vLLM server).
        if api_endpoint.startswith('http://'):
            self._conn_cls = http.client.HTTPConnection
        else:
            self._conn_cls = http.client.HTTPSConnection
        self.api_endpoint = re.sub(r'^https?://', '', api_endpoint).rstrip('/')
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def get_response(self, prompt_content: str, max_retries: int = 5) -> str | None:
        payload = json.dumps({
            "model": self.model_LLM,
            "messages": [{"role": "user", "content": prompt_content}],
            **self.extra_body,
        })
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            conn = None
            try:
                conn = self._conn_cls(self.api_endpoint, timeout=self.timeout)
                conn.request("POST", "/v1/chat/completions", payload, headers)
                data = conn.getresponse().read()
                parsed = json.loads(data)
                choices = parsed.get("choices")
                if not choices:
                    error_msg = parsed.get("error", {}).get("message", str(parsed))
                    raise ValueError(f"API returned no choices: {error_msg}")
                content = _strip_reasoning(choices[0]["message"]["content"] or "")
                if not content:
                    raise ValueError("API returned empty content after stripping reasoning")
                return content
            except Exception as e:
                logger.debug("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            finally:
                if conn is not None:
                    conn.close()

        logger.warning("API call failed after %d attempts (endpoint=%s, model=%s).",
                       max_retries, self.api_endpoint, self.model_LLM)
        return None
