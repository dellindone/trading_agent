"""Helpers for converting broker symbol formats."""


def fyers_to_groww(symbol: str) -> str:
    """Convert a FYERS symbol to Groww style by stripping exchange prefixes."""
    if not symbol:
        return symbol

    for prefix in ("NSE:", "BSE:"):
        if symbol.startswith(prefix):
            return symbol[len(prefix) :]
    return symbol
