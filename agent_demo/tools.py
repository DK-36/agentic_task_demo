"""Small local tools; all prices are fictional demo data."""

import math


PRICES = {"notebook": 12.5, "pen": 18.0, "eraser": 2.0}

TOOL_SPECS = [
    {
        "name": "get_price",
        "description": "Look up the unit price in CNY in a fictional store. A notebook is a paper notebook.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string", "enum": list(PRICES)}},
            "required": ["item"],
            "additionalProperties": False,
        },
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers, for example a unit price by a quantity.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
]


def get_price(item):
    if not isinstance(item, str) or item not in PRICES:
        raise ValueError("Unknown item. Choose from: " + ", ".join(PRICES))
    return {"item": item, "unit_price": PRICES[item], "currency": "CNY", "source": "fictional_demo_store"}


def multiply(a, b):
    for value in (a, b):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Arguments must be finite numbers with absolute values at most 1e12.")
    return {"result": round(a * b, 6)}


# Only these functions can run. Never evaluate generated Python or shell code.
REGISTRY = {"get_price": get_price, "multiply": multiply}


def execute_tool(name, arguments):
    if name not in REGISTRY:
        return {"ok": False, "error": "Unknown tool. Use get_price or multiply."}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Arguments must be a JSON object."}
    try:
        return {"ok": True, "data": REGISTRY[name](**arguments)}
    except (TypeError, ValueError, OverflowError) as exc:
        return {"ok": False, "error": str(exc)}
