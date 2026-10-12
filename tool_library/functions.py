"""Local tools; all prices are fictional demo data."""

import math

PRICES = {'notebook': 12.5, 'pen': 18.0, 'eraser': 2.0}


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

