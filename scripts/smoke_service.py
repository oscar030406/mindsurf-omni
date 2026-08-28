"""Drive the service the way the backend will, and check what comes back.

The unit tests exercise handlers directly; this goes over the wire against a
running container. The two catch different things -- a route that works in
process can still fail on serialisation, on a header the framework drops, or
on a WebSocket frame that never arrives.

Run it against the container before handing the service to anyone::

    python scripts/smoke_service.py --base http://localhost:8000

Exits non-zero on the first failure, and says which check failed rather than
printing a stack trace at the caller.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed.append(name)
        else:
            self.failed.append((name, detail))

    def report(self) -> int:
        for name in self.passed:
            print(f"  通过  {name}")
        for name, detail in self.failed:
            print(f"  失败  {name}: {detail}")
        print(f"\n{len(self.passed)} 通过, {len(self.failed)} 失败")
        return 1 if self.failed else 0


def silence(seconds: float, rate: int = 16_000) -> bytes:
    return struct.pack(f"<{int(rate * seconds)}h", *([0] * int(rate * seconds)))


def check_models(client: httpx.Client, results: Results) -> str | None:
    response = client.get("/v1/models")
    results.check("GET /v1/models 有响应", response.status_code == 200, str(response.status_code))
    if response.status_code != 200:
        return None

    data = response.json().get("data", [])
    if not data:
        print("  提示  没有配置引擎，只检查未配置时的行为")
        return None

    model = data[0]
    results.check(
        "报告当前路径",
        model.get("path") in {"native", "cascade"},
        f"path={model.get('path')!r}",
    )
    # The restriction is inherited from the training data and carried by every
    # derivative; a response that omits it lets a caller use the model for
    # months without meeting it.
    results.check(
        "响应带许可",
        "licence" in model and "commercial_use_permitted" in model,
        f"keys={sorted(model)}",
    )
    return str(model.get("path"))


def check_health(client: httpx.Client, results: Results) -> None:
    """Readiness must name what is not ready, not just report a status."""
    response = client.get("/health")
    results.check(
        "GET /health 有响应",
        response.status_code in {200, 503},
        str(response.status_code),
    )
    if response.status_code not in {200, 503}:
        return
    body = response.json()
    results.check(
        "健康报告含状态与未就绪清单",
        "status" in body and "not_ready" in body,
        f"keys={sorted(body)}",
    )
    # An empty report reading as healthy is the failure mode of every check
    # that only pings the process.
    results.check(
        "空报告不算就绪",
        not (body.get("status") == "ready" and not body.get("components")),
        "报告为空却称 ready",
    )


def check_unconfigured(client: httpx.Client, results: Results) -> None:
    """With no engine, endpoints must refuse with a reason, not fake success."""
    for path in ("/v1/token-spec",):
        response = client.get(path)
        results.check(
            f"未配置时 {path} 返回 503",
            response.status_code == 503,
            str(response.status_code),
        )
        if response.status_code == 503:
            results.check(
                f"{path} 的 503 说明原因",
                bool(response.json().get("detail")),
                "detail 为空",
            )


def check_chat(client: httpx.Client, results: Results) -> None:
    body = {"messages": [{"role": "user", "content": "你好"}]}
    response = client.post("/v1/chat/completions", json=body)
    results.check(
        "POST /v1/chat/completions", response.status_code == 200, str(response.status_code)
    )
    if response.status_code != 200:
        return
    payload = response.json()
    results.check(
        "OpenAI 形状",
        payload.get("object") == "chat.completion" and bool(payload.get("choices")),
        json.dumps(payload)[:120],
    )

    started = time.perf_counter()
    first_delta_ms: float | None = None
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: ") and first_delta_ms is None and "[DONE]" not in line:
                first_delta_ms = (time.perf_counter() - started) * 1000
    results.check("流式返回增量", first_delta_ms is not None, "没有收到 data 行")
    if first_delta_ms is not None:
        print(f"  信息  首个增量 {first_delta_ms:.0f} ms")


def check_transcription(client: httpx.Client, results: Results) -> None:
    response = client.post("/v1/audio/transcriptions", content=silence(1.0))
    results.check(
        "POST /v1/audio/transcriptions", response.status_code == 200, str(response.status_code)
    )
    empty = client.post("/v1/audio/transcriptions", content=b"")
    results.check("空音频被拒绝", empty.status_code == 400, str(empty.status_code))


def check_speech(client: httpx.Client, results: Results) -> None:
    response = client.post("/v1/audio/speech", json={"input": "你好"})
    results.check("POST /v1/audio/speech", response.status_code == 200, str(response.status_code))
    if response.status_code != 200:
        return
    # Without these a client has to guess how to play the bytes, and a wrong
    # guess is a changed pitch rather than an obvious failure.
    results.check(
        "声明采样率与编码",
        "x-sample-rate" in response.headers and "x-encoding" in response.headers,
        f"headers={sorted(response.headers)}",
    )


def check_realtime(base: str, results: Results) -> None:
    try:
        from websockets.sync.client import connect
    except ImportError:
        # Counted as a failure, not skipped. "Could not check" and "checked and
        # it works" are different answers, and printing a note while returning
        # zero turns the first into the second -- a green smoke run that never
        # touched the realtime path at all. A sister project lost a fleet to
        # the same shape: a catch branch that treated blocked playback as
        # playback finished, so every device went silent and nothing reported.
        results.check(
            "WS /v1/realtime",
            False,
            "websockets 未安装，实时链路未检查——这不是通过",
        )
        return

    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    url = f"{scheme}://{parsed.netloc}/v1/realtime"

    try:
        with connect(url, open_timeout=5) as socket:
            first = json.loads(socket.recv(timeout=5))
            results.check(
                "WS 连接后收到 session.created 或 error",
                first.get("type") in {"session.created", "error"},
                str(first)[:120],
            )
            if first.get("type") != "session.created":
                return
            # The client must not have to guess the rates.
            results.check(
                "session.created 带采样率",
                "input_sample_rate" in first and "output_sample_rate" in first,
                str(first)[:120],
            )
            socket.send(json.dumps({"type": "response.make_coffee"}))
            reply = json.loads(socket.recv(timeout=5))
            # Silence here would leave a client waiting forever.
            results.check(
                "未知事件收到 error 而非静默",
                reply.get("type") == "error",
                str(reply)[:120],
            )
    except Exception as error:  # noqa: BLE001 - a smoke test reports, it does not raise
        results.check("WS /v1/realtime", False, f"{type(error).__name__}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    print(f"冒烟测试 {args.base}")
    results = Results()

    with httpx.Client(base_url=args.base, timeout=args.timeout) as client:
        try:
            path = check_models(client, results)
        except httpx.ConnectError as error:
            print(f"连不上 {args.base}: {error}")
            raise SystemExit(2) from error

        check_health(client, results)

        if path is None:
            check_unconfigured(client, results)
        else:
            print(f"  信息  当前路径 {path}")
            check_chat(client, results)
            check_transcription(client, results)
            check_speech(client, results)

    check_realtime(args.base, results)
    sys.exit(results.report())


if __name__ == "__main__":
    main()
