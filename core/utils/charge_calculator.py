"""Trading charge estimation helpers."""

from core.utils.slippage_estimator import estimate_slippage

BROKERAGE = 40.0
EXCHANGE_CHARGE = 10.0
GST_RATE = 0.18
STT_RATE = 0.0015  # 0.15% per master plan (post Budget 2024)
STAMP_DUTY_RATE = 0.00003


def calculate_charges(
    premium: float,
    lot_size: int,
    lots: int,
    instrument: str,
    side: str = "SELL",
) -> dict[str, float]:
    """Estimate execution charges for an options trade.

    Assumptions:
    - `premium` is per-unit option premium.
    - Brokerage is flat at Rs. 40 for the round trip.
    - Exchange charge is flat at Rs. 10.
    - STT and stamp duty are estimated using standard option-premium rates.
    - Slippage uses the instrument ATM profile unless modeled elsewhere.
    """
    quantity = max(lot_size, 0) * max(lots, 0)
    turnover = max(premium, 0.0) * quantity

    stt = round(turnover * STT_RATE, 2) if side.upper() == "SELL" else 0.0
    brokerage = BROKERAGE if quantity else 0.0
    exchange = EXCHANGE_CHARGE if quantity else 0.0
    gst = round((brokerage + exchange) * GST_RATE, 2)
    stamp_duty = round(turnover * STAMP_DUTY_RATE, 2)
    slippage = estimate_slippage(instrument=instrument, quantity=quantity)
    total = round(stt + brokerage + exchange + gst + stamp_duty + slippage, 2)

    return {
        "stt": stt,
        "brokerage": round(brokerage, 2),
        "exchange": round(exchange, 2),
        "gst": gst,
        "stamp_duty": stamp_duty,
        "slippage": slippage,
        "total": total,
    }
