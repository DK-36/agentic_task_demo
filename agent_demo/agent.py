"""The question -> model -> tool -> feedback loop."""

import json
import re

from copy import deepcopy

from .config import load_config
from .tools import execute_tool

class AgentError(RuntimeError):
    """The loop stopped without a final answer."""


def parse_decision(raw, trace=None):
    """Extract one tagged action without modifying the original model output."""
    trace = {} if trace is None else trace
    if raw.lstrip().startswith("{"):
        # Keep existing JSON-only prompt variants usable.
        decision = json.loads(raw)
    else:
        tags = re.findall(r"</?(?:tool|final)\b[^>]*>", raw)
        if tags == ["<tool>", "</tool>"]:
            match = re.fullmatch(r"\s*(.*?)<tool>(.*?)</tool>\s*", raw, re.DOTALL)
            if not match:
                raise ValueError("Return one complete tool block with no text after it.")
            reason, payload = match.groups()
            trace["tool_call_text"] = payload
            decision = json.loads(payload)
            if not isinstance(decision, dict):
                raise ValueError("The tool block must contain a JSON object.")
            if decision.get("action", "tool") != "tool":
                raise ValueError("A tool block cannot contain a final action.")
            decision["action"] = "tool"
            decision.setdefault("reason", reason.strip() or "Execute the selected tool.")
        elif tags == ["<final>", "</final>"]:
            match = re.search(r"<final>(.*?)</final>", raw, re.DOTALL)
            if not match:
                raise ValueError("Put the final answer inside one final block.")
            decision = {"action": "final", "answer": match.group(1).strip()}
        else:
            raise ValueError("Return one <tool>...</tool> block or one <final>...</final> block.")
    if not isinstance(decision, dict):
        raise ValueError("Return a JSON object for a tool call.")
    if decision.get("action") == "final":
        if not isinstance(decision.get("answer"), str) or not decision["answer"].strip():
            raise ValueError("A final action needs a nonempty answer.")
    elif decision.get("action") == "tool":
        for key in ("name", "reason"):
            if not isinstance(decision.get(key), str) or not decision[key].strip():
                raise ValueError(f"A tool action needs a nonempty {key}.")
        if not isinstance(decision.get("arguments"), dict):
            raise ValueError("Tool arguments must be an object.")
    else:
        raise ValueError("The action must be tool or final.")
    return decision


def run_agent(model, question=None, max_steps=8, steps=None, config=None):
    """Append actual steps to the supplied log, including incomplete steps."""
    steps = [] if steps is None else steps
    config = load_config() if config is None else config
    question = config["question"] if question is None else question
    allowed_tools = {spec["name"] for spec in config["tools"]}
    messages = [{"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        print(f"\nStep {step}: asking Llama...", flush=True)
        entry = {"step": step, "input_messages": deepcopy(messages)}
        steps.append(entry)
        raw = model.complete(messages, trace=entry)
        entry["model_response"] = raw
        messages.append({"role": "assistant", "content": raw})
        try:
            decision = parse_decision(raw, trace=entry)
            entry["decision"] = decision
        except (ValueError, TypeError) as exc:
            observation = {"ok": False, "error": f"Invalid action: {exc} Follow the output format in the system prompt."}
        else:
            if decision["action"] == "final":
                return decision["answer"]
            print("Reason: " + decision["reason"], flush=True)
            print(f"Tool: {decision['name']}({json.dumps(decision['arguments'])})", flush=True)
            observation = execute_tool(decision["name"], decision["arguments"], allowed_tools)
        entry["observation"] = observation
        print("Observation: " + json.dumps(observation), flush=True)
        # Keep both the model's action and the actual tool result for the next step.
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})
    raise AgentError(f"No final answer after {max_steps} model calls.")
