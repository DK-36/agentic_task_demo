"""Send the conversation to Llama running locally in Ollama."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "llama3.1:8b"
URL = "http://127.0.0.1:11434/api/chat"


class LocalLlama:
    def complete(self, messages):
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 512},
        }
        request = Request(URL, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}. Check that {MODEL} is installed."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Cannot reach local Ollama or the request timed out. Make sure Ollama is running.") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc
        try:
            if data.get("done_reason") == "length":
                raise RuntimeError("Llama's response exceeded the output limit.")
            content = data["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty response")
            return content
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Ollama returned no valid message content.") from exc
