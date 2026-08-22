from __future__ import annotations

import time
from typing import Any, Iterable

import requests

from .common import parse_json_object


class ModelResponseError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class OllamaClient:
    def __init__(self, base_url: str, settings: dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/api/chat"
        self.settings = settings

    def available_models(self) -> set[str]:
        response = requests.get(
            self.base_url + "/api/tags",
            timeout=int(self.settings.get("preflight_timeout_seconds", 60)),
        )
        response.raise_for_status()
        payload = response.json()
        names: set[str] = set()
        for item in payload.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                names.add(str(name))
        return names

    def assert_models_available(self, models: Iterable[str]) -> None:
        requested = set(models)
        available = self.available_models()
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(
                "Ollama model(s) not installed: " + ", ".join(missing)
                + ". Available models: " + ", ".join(sorted(available))
            )

    def _thinking_setting(self, model: str) -> bool | str:
        explicit = self.settings.get("thinking_by_model", {})
        if isinstance(explicit, dict) and model in explicit:
            return explicit[model]
        lowered = model.lower()
        # Ollama's GPT-OSS integration accepts a reasoning level rather than false.
        if "gpt-oss" in lowered:
            return "low"
        # Qwen3 can be run with thinking disabled for stable structured output.
        if "qwen3" in lowered:
            return False
        return False

    def generate_json(
        self,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object matching the supplied JSON schema. "
                        "Do not use markdown fences or add prose outside the object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # A schema object is stricter and more reliable than format='json'.
            "format": response_schema,
            "think": self._thinking_setting(model),
            "keep_alive": self.settings.get("keep_alive", "30m"),
            "options": {
                "temperature": self.settings.get("temperature", 0),
                "seed": self.settings.get("seed", 42),
                "num_ctx": self.settings.get("num_ctx", 32768),
                "num_predict": self.settings.get("num_predict", 4200),
            },
        }
        started = time.time()
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=int(self.settings.get("timeout_seconds", 900)),
            )
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            diagnostics: dict[str, Any] = {"model": model, "endpoint": self.url}
            if "response" in locals():
                diagnostics["http_status"] = getattr(response, "status_code", None)
                diagnostics["http_text"] = getattr(response, "text", "")[:4000]
            raise ModelResponseError(f"Ollama request failed: {type(exc).__name__}: {exc}", diagnostics) from exc

        message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
        text = str(message.get("content") or "")
        thinking = str(message.get("thinking") or "")
        diagnostics = {
            "model": model,
            "endpoint": self.url,
            "done_reason": raw.get("done_reason"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "thinking_chars": len(thinking),
            "response_keys": sorted(raw.keys()),
            "message_keys": sorted(message.keys()),
            "raw_final_text": text[:12000],
        }
        if not text.strip():
            raise ModelResponseError("Ollama returned an empty final response.", diagnostics)
        try:
            obj = parse_json_object(text)
        except Exception as exc:
            raise ModelResponseError(
                f"Final response was not parseable JSON: {type(exc).__name__}: {exc}",
                diagnostics,
            ) from exc

        meta = {
            "latency_seconds": round(time.time() - started, 3),
            "done_reason": raw.get("done_reason"),
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "eval_count": raw.get("eval_count"),
            "thinking_chars": len(thinking),
            "raw_text": text,
        }
        return obj, meta
