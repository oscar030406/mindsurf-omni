"""One place that reaches an external judge, because credentials deserve one.

Two scripts now ask a model to rank or check replies, and both need the same
four things: where the key lives, how to retry a flaky endpoint, how to run a
few hundred calls without waiting for each, and how to record what judged
without recording what authenticated. Writing that twice means two places where
a key can end up somewhere it should not.

The key is read from a file under the user's home, or from the environment. Not
from the source, which is tracked and which the release gate rejects on the way
out. Not from a machine-wide variable, which hands it to every process on the
box. And not from inside the project directory even when gitignored --
verify_delivery scans the working tree rather than the index, because packaging
copies what is on disk rather than what git would ship, and it caught exactly
that.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_CREDENTIALS = Path.home() / ".mindsurf" / "judge.json"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def load_credentials(path: Path | None = None) -> dict[str, str]:
    """The local file first, then the environment. Placeholders count as absent."""
    settings: dict[str, str] = {}
    source = path or DEFAULT_CREDENTIALS
    if source.exists():
        blob = json.loads(source.read_text(encoding="utf-8"))
        settings = {k: v for k, v in blob.items() if isinstance(v, str) and v.strip()}
    for field, variable in (
        ("api_key", "JUDGE_API_KEY"),
        ("base_url", "JUDGE_BASE_URL"),
        ("model", "JUDGE_MODEL"),
    ):
        settings.setdefault(field, os.environ.get(variable, ""))
    if settings.get("api_key", "").startswith("<"):
        settings.pop("api_key")
    return {key: value for key, value in settings.items() if value}


class Judge:
    """An OpenAI-compatible chat endpoint, asked one short question at a time.

    Temperature is zero and the reply budget is a handful of tokens: every
    caller here wants a label, and a judge given room to explain itself spends
    it arguing toward whichever label it mentioned first.
    """

    def __init__(
        self,
        credentials: Path | None = None,
        workers: int = 8,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = load_credentials(credentials)
        if not settings.get("api_key"):
            raise SystemExit(
                f"没有判官凭据。把 key 填进 {credentials or DEFAULT_CREDENTIALS} 的 api_key "
                "字段，或者设 JUDGE_API_KEY 环境变量。两种都不进版本控制。"
            )
        self.key = settings["api_key"]
        self.base_url = settings.get("base_url") or DEFAULT_BASE_URL
        # An override exists so one task can use a different model from another
        # against the same endpoint. Writing evaluation probes and judging
        # replies should not be the same model where it can be avoided.
        self.model = model or settings.get("model") or DEFAULT_MODEL
        self.workers = workers
        # Two minutes is plenty for a one-word verdict and nowhere near enough for
        # a model asked to write a hundred lines. A caller that wants prose has
        # to raise this, and the first one that forgot lost a whole run to a
        # read timeout.
        self.timeout = timeout

    def provenance(self, prompt_template: str, **extra: Any) -> dict[str, Any]:
        """What judged, never what authenticated."""
        import hashlib

        return {
            "model": self.model,
            "endpoint": self.base_url,
            "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
            **extra,
        }

    def ask(self, client: Any, question: str, max_tokens: int = 8) -> str:
        for attempt in range(4):
            try:
                reply = client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": question}],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    },
                )
                reply.raise_for_status()
                return str(reply.json()["choices"][0]["message"]["content"]).strip()
            except Exception:  # a flaky endpoint should not lose the other calls
                if attempt == 3:
                    raise
        return ""

    def run(
        self,
        items: list[Any],
        question_of: Callable[[Any], str],
        label: str = "判",
        max_tokens: int = 8,
    ) -> list[str]:
        """Answers in the order the items came in, several calls in flight.

        ``max_tokens`` defaults to a label's worth because that is what judging
        needs, and a judge given room to explain itself argues toward whichever
        label it said first. A caller that wants prose has to say so -- and a
        caller that forgets gets eight tokens, which fails loudly rather than
        returning something shaped right and truncated.
        """
        import httpx

        answers: list[str] = [""] * len(items)
        with (
            httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.key}"},
            ) as client,
            concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool,
        ):
            futures = {
                pool.submit(self.ask, client, question_of(item), max_tokens): index
                for index, item in enumerate(items)
            }
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                answers[futures[future]] = future.result()
                if done % 50 == 0:
                    print(f"  {label} {done}/{len(items)}", flush=True)
        return answers
