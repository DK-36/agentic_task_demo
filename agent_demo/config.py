"""Load editable prompt files and tool descriptions at run time."""

import json
from pathlib import Path

from tool_library.functions import REGISTRY

ROOT = Path(__file__).resolve().parent.parent


def load_config(prompt_path=None, question_path=None, tools_path=None):
    paths = {
        "prompt": Path(prompt_path) if prompt_path else ROOT / "prompt_library/system.txt",
        "question": Path(question_path) if question_path else ROOT / "prompt_library/question.txt",
        "tools": Path(tools_path) if tools_path else ROOT / "tool_library/definitions.json",
    }
    sources = {key: {"path": str(path.resolve()), "text": path.read_text(encoding="utf-8")}
               for key, path in paths.items()}
    template = sources["prompt"]["text"]
    question = sources["question"]["text"]
    if template.count("{{tools}}") != 1:
        raise ValueError("The system prompt must contain exactly one {{tools}} placeholder.")
    if not question.strip():
        raise ValueError("The question file must not be empty.")
    specs = json.loads(sources["tools"]["text"])
    if not isinstance(specs, list):
        raise ValueError("Tool definitions must be a JSON list.")
    names = []
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError("Each tool definition needs a name.")
        name = spec["name"]
        if name not in REGISTRY or name in names:
            raise ValueError(f"Unknown or duplicate tool definition: {name}")
        if not isinstance(spec.get("description"), str) or not isinstance(spec.get("parameters"), dict):
            raise ValueError(f"Tool {name} needs a description and parameters.")
        names.append(name)
    return {"sources": sources, "question": question, "tools": specs,
            "system_prompt": template.replace("{{tools}}", json.dumps(specs, ensure_ascii=False))}
