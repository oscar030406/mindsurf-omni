"""A working client, short enough to read in one sitting.

Copy it, point it at the service, delete what you do not need. It uses only
httpx and websockets, so it drops into a backend without pulling in this
project.

Every call is an OpenAI-shaped one, which is the point: a client already
written against that API needs almost none of this file.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any

import httpx

BASE = "http://localhost:8000"


def check_what_is_serving() -> dict[str, Any]:
    """Always do this first.

    It reports which path is answering and whether the output may be used
    commercially -- both change what you may do with the result.
    """
    models = httpx.get(f"{BASE}/v1/models", timeout=10).json()["data"]
    if not models:
        raise SystemExit("no engine configured; the service will answer 503")
    model: dict[str, Any] = models[0]
    print(f"path={model['path']}  commercial={model['commercial_use_permitted']}")
    return model


def ask_streaming(prompt: str) -> str:
    """Prefer streaming. The first token arrives long before the last."""
    text = ""
    with httpx.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}], "stream": True},
        timeout=60,
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            piece = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
            text += piece
            print(piece, end="", flush=True)
    print()
    return text


def speak(text: str, emotion: str = "neutral") -> bytes:
    """Emotion is its own field. Putting it in the text gets it read aloud."""
    response = httpx.post(
        f"{BASE}/v1/audio/speech",
        json={"input": text, "emotion": emotion},
        timeout=60,
    )
    response.raise_for_status()
    # Read the rate from the header rather than assuming it: playing 24 kHz
    # audio at 16 kHz does not fail, it changes the pitch.
    rate = int(response.headers.get("x-sample-rate", 24_000))
    print(f"{len(response.content)} bytes at {rate} Hz")
    return response.content


def converse(pcm: bytes) -> None:
    """One realtime turn: speech in, text and speech out, both streamed."""
    from websockets.sync.client import connect

    with connect(BASE.replace("http", "ws") + "/v1/realtime") as socket:
        session = json.loads(socket.recv())
        print(f"session up, input {session['input_sample_rate']} Hz")

        # Send in chunks, the way a microphone produces them.
        for start in range(0, len(pcm), 3200):  # 100 ms at 16 kHz
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[start : start + 3200]).decode(),
                    }
                )
            )
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))

        while True:
            event = json.loads(socket.recv())
            kind = event["type"]
            if kind == "response.text.delta":
                print(event["delta"], end="", flush=True)
            elif kind == "response.audio.delta":
                # Play this immediately. Buffering until the end throws away
                # the reason the audio is streamed at all.
                base64.b64decode(event["audio"])
            elif kind == "error":
                print("error:", event["error"]["message"])
                break
            elif kind == "response.done":
                # dropped_turns > 0 means history was shortened to fit.
                print("context:", event.get("context"))
                break


if __name__ == "__main__":
    check_what_is_serving()
    reply = ask_streaming("今天天气怎么样")
    speak(reply, emotion="happy")
    converse(struct.pack("<16000h", *([0] * 16000)))
