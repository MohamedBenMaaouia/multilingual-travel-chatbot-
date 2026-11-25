"""Tiny client for the local Ollama server. Every LLM task (travel filter, answers, translation) uses it.

    from src.llm.ollama_client import chat
    text, seconds = chat("mistral:7b-instruct", [{"role": "user", "content": "Hello"}])

If the Ollama server is not running, the first call starts it automatically (`ollama serve`, in the background).
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache

# 127.0.0.1, not "localhost": on Windows "localhost" can resolve to IPv6 and miss the server
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_URL = OLLAMA_BASE + "/api/chat"


def _server_is_up():
    try:
        urllib.request.urlopen(OLLAMA_BASE + "/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@lru_cache(maxsize=1)                      # check (and start) the server only once per Python session
def ensure_ollama_running(wait_seconds=30):
    if _server_is_up():
        return
    flags = 0
    if sys.platform == "win32":            # detach it, so it keeps running after this Python process ends
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed or not on PATH. Install it from https://ollama.com") from None
    for _ in range(wait_seconds * 2):
        if _server_is_up():
            return
        time.sleep(0.5)
    raise RuntimeError("Could not start the Ollama server. Start the Ollama app (or run `ollama serve`) and retry.")


def chat(model, messages, *, temperature=0.0, max_tokens=200, think=None, keep_alive="15m", seed=42, timeout=300,
         format=None):
    """Send a list of {"role", "content"} messages and return (answer_text, seconds_taken).

    temperature=0  -> always the same answer for the same input (reproducible)
    max_tokens     -> hard limit on how much the model may write
    think          -> only for "thinking" models; leave None for mistral
    keep_alive     -> keep the model loaded in memory between calls (only the first call is slow)
    format         -> optional JSON schema: Ollama forces the answer to match it (e.g. a fixed list of labels)
    """
    ensure_ollama_running()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }
    if think is not None:
        payload["think"] = think
    if format is not None:
        payload["format"] = format
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data["message"]["content"].strip(), time.perf_counter() - start


def chat_stream(model, messages, *, temperature=0.0, max_tokens=200, keep_alive="15m", seed=42, total_timeout=60):
    """Same as chat(), but yields the answer piece by piece WHILE Mistral writes it (for the UI).

    Ollama sends one JSON line per piece: {"message": {"content": "..."}, "done": false}, the last one has done=true.
    total_timeout is NOT a speed limit: it only stops a call that is really stuck (normal answers take ~5-10 s).
    """
    ensure_ollama_running()
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=total_timeout) as response:
        for line in response:
            if time.perf_counter() - start > total_timeout:
                raise TimeoutError(f"no complete answer after {total_timeout} s")
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                return
