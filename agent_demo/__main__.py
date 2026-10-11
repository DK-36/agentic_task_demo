"""Run local Llama and save the actual execution record."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import DEFAULT_QUESTION, run_agent
from .models import MODEL, LocalLlama

RUN_LOG_PATH = Path(__file__).resolve().parent.parent / "runs" / "latest.json"


def main():
    record = {
        "model": MODEL,
        "question": DEFAULT_QUESTION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "steps": [],
    }
    print(f"Local model: {MODEL}")
    print(f"Question: {DEFAULT_QUESTION}", flush=True)
    exit_code = 0
    try:
        answer = run_agent(LocalLlama(), steps=record["steps"])
        record.update(status="completed", answer=answer)
        print("\nFinal answer: " + answer)
    except KeyboardInterrupt:
        record.update(status="interrupted", error="Run interrupted by the user.")
        print("\nDemo interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        print(f"Demo failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            RUN_LOG_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Run record: {RUN_LOG_PATH}", flush=True)
        except OSError as exc:
            print(f"Cannot save the run record: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
