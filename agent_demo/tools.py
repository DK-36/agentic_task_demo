"""Dispatch only the tools enabled for this run."""

from tool_library.functions import REGISTRY


def execute_tool(name, arguments, allowed_tools=None):
    allowed = REGISTRY if allowed_tools is None else allowed_tools
    if name not in REGISTRY or name not in allowed:
        return {"ok": False, "error": "Unknown or disabled tool: " + str(name)}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Arguments must be a JSON object."}
    try:
        return {"ok": True, "data": REGISTRY[name](**arguments)}
    except (TypeError, ValueError, OverflowError) as exc:
        return {"ok": False, "error": str(exc)}
