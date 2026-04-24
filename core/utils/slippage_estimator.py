"""Simple slippage estimation by instrument and option moneyness."""

SLIPPAGE_PER_UNIT = {
    "NIFTY": {"ATM": 1.0, "ITM": 2.0},
    "BANKNIFTY": {"ATM": 1.0, "ITM": 2.0},
    "FINNIFTY": {"ATM": 1.0, "ITM": 2.0},
    "MIDCPNIFTY": {"ATM": 1.0, "ITM": 2.0},
    "SENSEX": {"ATM": 1.5, "ITM": 3.0},
    "BANKEX": {"ATM": 1.5, "ITM": 3.0},
}


def estimate_slippage(
    instrument: str,
    quantity: int,
    option_type: str = "ATM",
) -> float:
    """Return total slippage in rupees for the given quantity."""
    normalized_instrument = instrument.upper()
    normalized_option_type = option_type.upper()

    instrument_slippage = SLIPPAGE_PER_UNIT.get(
        normalized_instrument,
        SLIPPAGE_PER_UNIT["NIFTY"],
    )
    per_unit = instrument_slippage.get(
        normalized_option_type,
        instrument_slippage["ATM"],
    )
    return round(per_unit * max(quantity, 0), 2)
