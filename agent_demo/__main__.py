"""Run local Llama and save the actual execution record."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_agent
from .config import load_config
from .models import MODEL, LocalLlama

RUN_LOG_PATH = Path(__file__).resolve().parent.parent / "runs" / "latest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a local Llama tool-use loop.")
    parser.add_argument("--prompt", type=Path, help="System prompt file containing {{tools}}")
    parser.add_argument("--question", type=Path, help="User question text file")
    parser.add_argument("--tools", type=Path, help="Tool definitions JSON file")
    args = parser.parse_args(argv)
    record = {
        "model": MODEL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "steps": [],
    }
    print(f"Local model: {MODEL}")
    exit_code = 0
    try:
        config = load_config(args.prompt, args.question, args.tools)
        record["config"] = config
        record["question"] = config["question"]
        print(f"Question: {config['question']}", flush=True)
        answer = run_agent(LocalLlama(), steps=record["steps"], config=config)
        record.update(status="completed", answer=answer)
        print("\nFinal answer: " + answer)
    except KeyboardInterrupt:
        record.update(status="interrupted", error="Run interrupted by the user.")
        print("\nDemo interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        record.update(status="failed", error=str(exc) or type(exc).__name__)
        print(f"Demo failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            stamp = datetime.fromisoformat(record["started_at"]).strftime("%Y%m%dT%H%M%S%fZ")
            archive_path = RUN_LOG_PATH.parent / f"run_{stamp}.json"
            archive_path.write_text(text, encoding="utf-8")
            RUN_LOG_PATH.write_text(text, encoding="utf-8")
            print(f"Archived record: {archive_path}", flush=True)
            print(f"Run record: {RUN_LOG_PATH}", flush=True)
        except OSError as exc:
            print(f"Cannot save the run record: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
