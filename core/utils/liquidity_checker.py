"""Liquidity checks for option contracts."""

MAX_SPREAD = {
    "NIFTY": 1.0,
    "BANKNIFTY": 2.0,
    "FINNIFTY": 1.5,
    "MIDCPNIFTY": 1.5,
    "SENSEX": 1.5,
    "BANKEX": 2.0,
}

MIN_OI = {
    "NIFTY": 10000,
    "BANKNIFTY": 5000,
    "FINNIFTY": 3000,
    "MIDCPNIFTY": 2500,
    "SENSEX": 2500,
    "BANKEX": 2000,
}


def is_liquid(bid: float, ask: float, oi: int, instrument: str) -> bool:
    """Return True when spread and open-interest pass the liquidity gate."""
    if bid <= 0 or ask <= 0 or ask < bid:
        return False

    normalized_instrument = instrument.upper()
    spread = ask - bid
    max_spread = MAX_SPREAD.get(normalized_instrument, MAX_SPREAD["NIFTY"])
    min_oi = MIN_OI.get(normalized_instrument, MIN_OI["NIFTY"])

    return spread <= max_spread and oi >= min_oi
