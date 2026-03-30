"""
modules/ollama_client.py
------------------------
Thin wrapper around the Ollama local API.
Ollama exposes an OpenAI-compatible HTTP endpoint at http://localhost:11434
"""

import json
import requests
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"


def check_ollama() -> tuple[bool, str]:
    """Return (is_running, message)."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            return True, "Ollama is running"
        return False, f"Ollama returned status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return False, str(e)


def list_models() -> list[str]:
    """Return list of locally available model names."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        return []
    except Exception:
        return []


def chat(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    timeout: int = 120,
) -> str:
    """
    Send a prompt to Ollama and return the response text.

    Uses /api/generate (non-streaming) for simplicity.

    Args:
        model:       Ollama model name, e.g. "llama3", "mistral"
        prompt:      User prompt
        system:      Optional system message
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens:  Maximum tokens to generate
        timeout:     Request timeout in seconds

    Returns:
        Generated text string

    Raises:
        RuntimeError on API failure
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama API error {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama request timed out. "
            "The model may be loading — try again in a few seconds."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it's running: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama client error: {e}")


def parse_json_response(text: str) -> dict | list:
    """
    Safely parse a JSON response from the LLM.
    Handles common issues: markdown code fences, leading text.
    """
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or (not text.startswith("```") and not line.startswith("```")):
                inner.append(line)
        text = "\n".join(inner).strip()

    # Try to find JSON object/array within the text
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Last resort: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON.\nRaw: {text[:500]}\nError: {e}")
