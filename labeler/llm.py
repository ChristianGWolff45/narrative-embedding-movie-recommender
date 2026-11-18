from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from .prompts import PROMPT_VERSION, labeling_messages
from .schema import MovieLabel


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model did not return parseable JSON: {text[:300]!r}")


class LocalLabeler:
    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the project dependencies and ensure the CUDA-enabled "
                "PyTorch installation is visible to the project environment."
            ) from exc

        self.model_id = model_id
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def label(self, movie: dict) -> MovieLabel:
        messages = labeling_messages(movie)
        last_error: Exception | None = None
        for attempt in range(3):
            text = self._generate(messages)
            try:
                payload = _extract_json(text)
                payload["model_name"] = self.model_id
                payload["prompt_version"] = PROMPT_VERSION
                return MovieLabel.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < 2:
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "Correct the response to match the requested schema exactly. "
                                "Return one strict JSON object. The themes, tone, characters, "
                                "and narrative_structure fields must each be JSON arrays of "
                                "strings, not a string containing a list. The rationale field "
                                "must be a JSON string. Do not use Python list syntax."
                            ),
                        },
                    ]
        assert last_error is not None
        raise last_error


class OllamaLabeler:
    """Local Ollama chat backend for machines without a CUDA GPU."""

    def __init__(self, model_id: str = "qwen2.5:3b") -> None:
        import httpx

        self.model_id = model_id
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.client = httpx.Client(base_url=host, timeout=300.0)

    def _generate(self, messages: list[dict[str, str]]) -> str:
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model_id,
                "messages": messages,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 384},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def label(self, movie: dict) -> MovieLabel:
        messages = labeling_messages(movie)
        text = self._generate(messages)
        try:
            payload = _extract_json(text)
        except ValueError:
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": "Return one valid JSON object with the requested movie labels only.",
                }
            ]
            payload = _extract_json(self._generate(retry_messages))
        payload["model_name"] = self.model_id
        payload["prompt_version"] = PROMPT_VERSION
        return MovieLabel.model_validate(payload)
