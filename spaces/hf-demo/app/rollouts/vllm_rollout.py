"""vLLM rollout backend.

Talks to a vLLM server running with `--enable-prefix-caching` so the
rollout cost stays approximately constant in the prompt length once
the prefix cache warms up. The `Rollout` class is intentionally thin:
it speaks OpenAI-compatible chat completions.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class Rollout:
    text: str
    token_ids: list[int]
    log_probs: list[float] | None
    finish_reason: str | None


class VllmRollout:
    def __init__(self, url: str, model: str) -> None:
        self.url = url
        self.model = model
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def sample(
        self,
        prompt: str,
        *,
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        return_logprobs: bool = False,
    ) -> list[Rollout]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": return_logprobs,
            "top_logprobs": 1 if return_logprobs else None,
        }
        r = await self._http.post(f"{self.url}/chat/completions", json=body)
        r.raise_for_status()
        out = r.json()
        rollouts: list[Rollout] = []
        for ch in out["choices"]:
            rollouts.append(Rollout(
                text=ch["message"]["content"] or "",
                token_ids=[],
                log_probs=None,
                finish_reason=ch.get("finish_reason"),
            ))
        return rollouts
