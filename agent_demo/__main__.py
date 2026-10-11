"""Run the default task with the local Llama model."""

import sys

from .agent import DEFAULT_QUESTION, run_agent
from .models import MODEL, LocalLlama


def main():
    print(f"Local model: {MODEL}")
    print(f"Question: {DEFAULT_QUESTION}", flush=True)
    try:
        answer = run_agent(LocalLlama())
    except (RuntimeError, ValueError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    print("\nFinal answer: " + answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
