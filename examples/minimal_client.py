"""Dictate a wav file two ways, and read the finished note aloud.

Both entry points on the same recording, so the output can be compared. There is
no client here in the product sense -- no hotkey, no text box, no clipboard.
This is the smallest thing that proves the service works from outside.

    python examples/minimal_client.py turn.wav

The file has to be a real recording. Digital silence demonstrates nothing: the
service answers "nothing was heard", which is correct and uninteresting.
"""

from __future__ import annotations

import base64
import json
import sys
import wave
from pathlib import Path

import httpx
from websockets.sync.client import connect

BASE = "http://localhost:8000"
APPEND_MS = 100


def read(path: Path) -> tuple[bytes, int]:
    """The samples, refusing a file the service would only transcribe badly."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(
                f"{path} 要是单声道 16 位，收到的是 "
                f"{handle.getnchannels()} 声道 {handle.getsampwidth() * 8} 位"
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


def whole_file(pcm: bytes, rate: int) -> str:
    """Record first, send afterwards. Nothing appears while the user talks."""
    print("\n--- 整段送 POST /v1/audio/transcriptions")
    answer = httpx.post(
        f"{BASE}/v1/audio/transcriptions",
        content=wav(pcm, rate),
        headers={"content-type": "audio/wav"},
        timeout=180,
    )
    answer.raise_for_status()
    got = answer.json()
    print(f"    转写 {got['text']}")
    print(f"    润色 {got.get('polished')}")
    return got.get("polished") or got["text"]


def wav(pcm: bytes, rate: int) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def while_speaking(pcm: bytes, rate: int) -> None:
    """Send as it is recorded. Preview appears while the user is still talking."""
    print("\n--- 边录边送 WS /v1/realtime")
    with connect(BASE.replace("http", "ws") + "/v1/realtime") as socket:
        opened = json.loads(socket.recv())
        assert opened["type"] == "session.created"
        # The rate the service declares, not one written into this file.
        rate = opened.get("input_sample_rate", rate)
        step = int(rate * APPEND_MS / 1000) * 2

        preview = ""
        for at in range(0, len(pcm), step):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[at : at + step]).decode(),
                    }
                )
            )
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))

        while True:
            event = json.loads(socket.recv())
            kind = event["type"]
            if kind.endswith("transcription.delta"):
                # Additions only. Append; never re-render what was already shown.
                preview += event["delta"]
            elif kind.endswith("transcription.completed"):
                print(f"    转写 {event['transcript']}")
            elif kind == "response.text.delta":
                print(f"    润色 {event['delta']}")
            elif kind == "error":
                print(f"    错误 {event['error']['message']}")
                break
            elif kind == "response.done":
                break
        print(f"    预览 {preview}")


def read_aloud(text: str) -> None:
    """The speaker button. Off unless the deployment set MINDSURF_TTS=edge."""
    print("\n--- 朗读 POST /v1/audio/speech")
    answer = httpx.post(
        f"{BASE}/v1/audio/speech", json={"input": text, "response_format": "wav"}, timeout=180
    )
    if answer.status_code != 200:
        print(f"    {answer.status_code} {answer.text[:160]}")
        return
    # The rate the service declares, not one written into this file. Assuming it
    # is how a client ends up playing 24 kHz audio at 16 kHz.
    rate = answer.headers.get("x-sample-rate", "?")
    Path("read_aloud.wav").write_bytes(answer.content)
    print(f"    写到 read_aloud.wav（{len(answer.content)} 字节 @ {rate} Hz）")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法: python examples/minimal_client.py turn.wav")

    models = httpx.get(f"{BASE}/v1/models", timeout=10).json()["data"]
    # First thing a backend author should see: the weights are not commercial.
    if not models[0].get("commercial_use_permitted", False):
        print("许可：当前权重不可商用（GET /v1/licence 有完整链条）")
    named = [c["name"] for c in models[0]["components"]]
    print(f"组件 {', '.join(named)}")
    if "polish-tagger" not in named:
        print("  注意：第二条臂没接，口语词清除会明显变差")

    pcm, rate = read(Path(sys.argv[1]))
    kept = whole_file(pcm, rate)
    while_speaking(pcm, rate)
    if kept:
        read_aloud(kept)


if __name__ == "__main__":
    main()
