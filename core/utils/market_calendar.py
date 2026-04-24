"""Indian trading calendar utilities."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytz

from config.instruments import EXPIRY_WEEKDAY

IST = pytz.timezone("Asia/Kolkata")
MARKET_CLOSE = time(hour=15, minute=30)

# Source basis:
# 2022-2024: NSE trading holiday calendars used for historical training windows.
# 2025 F&O holidays: NSE circular dated December 13, 2024.
# 2026 F&O holidays: NSE circular dated December 12, 2025.
NSE_HOLIDAYS = {
    date(2022, 1, 26),
    date(2022, 3, 1),
    date(2022, 3, 18),
    date(2022, 4, 14),
    date(2022, 4, 15),
    date(2022, 5, 3),
    date(2022, 8, 9),
    date(2022, 8, 15),
    date(2022, 8, 31),
    date(2022, 10, 5),
    date(2022, 10, 24),
    date(2022, 10, 26),
    date(2022, 11, 8),
    date(2023, 1, 26),
    date(2023, 3, 7),
    date(2023, 3, 30),
    date(2023, 4, 4),
    date(2023, 4, 7),
    date(2023, 4, 14),
    date(2023, 5, 1),
    date(2023, 6, 29),
    date(2023, 8, 15),
    date(2023, 9, 19),
    date(2023, 10, 2),
    date(2023, 10, 24),
    date(2023, 11, 14),
    date(2023, 11, 27),
    date(2023, 12, 25),
    date(2024, 1, 22),
    date(2024, 1, 26),
    date(2024, 3, 8),
    date(2024, 3, 25),
    date(2024, 3, 29),
    date(2024, 4, 11),
    date(2024, 4, 17),
    date(2024, 5, 1),
    date(2024, 6, 17),
    date(2024, 7, 17),
    date(2024, 8, 15),
    date(2024, 10, 2),
    date(2024, 11, 1),
    date(2024, 11, 15),
    date(2024, 12, 25),
    date(2025, 2, 26),
    date(2025, 3, 14),
    date(2025, 3, 31),
    date(2025, 4, 10),
    date(2025, 4, 14),
    date(2025, 4, 18),
    date(2025, 5, 1),
    date(2025, 8, 15),
    date(2025, 8, 27),
    date(2025, 10, 2),
    date(2025, 10, 21),
    date(2025, 10, 22),
    date(2025, 11, 5),
    date(2025, 12, 25),
    date(2026, 1, 26),
    date(2026, 3, 3),
    date(2026, 3, 26),
    date(2026, 3, 31),
    date(2026, 4, 3),
    date(2026, 4, 14),
    date(2026, 5, 1),
    date(2026, 5, 28),
    date(2026, 6, 26),
    date(2026, 9, 14),
    date(2026, 10, 2),
    date(2026, 10, 20),
    date(2026, 11, 10),
    date(2026, 11, 24),
    date(2026, 12, 25),
}


def _coerce_to_ist_datetime(value: date | datetime | None) -> datetime:
    if value is None:
        return datetime.now(IST)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return IST.localize(value)
        return value.astimezone(IST)
    return IST.localize(datetime.combine(value, time.min))


def is_trading_day(value: date | datetime) -> bool:
    """Return True for Indian market trading days."""
    current_date = _coerce_to_ist_datetime(value).date()
    return current_date.weekday() < 5 and current_date not in NSE_HOLIDAYS


def _adjust_to_previous_trading_day(candidate: date) -> date:
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def next_expiry(instrument: str, from_date: date | datetime | None = None) -> date:
    """Return the next weekly expiry date adjusted for holidays/weekends."""
    normalized_instrument = instrument.upper()
    if normalized_instrument not in EXPIRY_WEEKDAY:
        raise ValueError(f"Unsupported instrument for expiry lookup: {instrument}")

    reference_dt = _coerce_to_ist_datetime(from_date)
    expiry_weekday = EXPIRY_WEEKDAY[normalized_instrument]

    days_ahead = (expiry_weekday - reference_dt.date().weekday()) % 7
    candidate = reference_dt.date() + timedelta(days=days_ahead)

    while True:
        adjusted = _adjust_to_previous_trading_day(candidate)
        adjusted_close = IST.localize(datetime.combine(adjusted, MARKET_CLOSE))
        if adjusted_close > reference_dt:
            return adjusted
        candidate += timedelta(days=7)


def days_to_next_expiry(instrument: str, from_date: date | datetime | None = None) -> int:
    """Return calendar days remaining until the next expiry."""
    reference_dt = _coerce_to_ist_datetime(from_date)
    expiry_date = next_expiry(instrument, reference_dt)
    return (expiry_date - reference_dt.date()).days


class _MarketCalendar:
    def days_to_next_expiry(self, instrument: str, from_date: date | datetime | None = None) -> int:
        return days_to_next_expiry(instrument, from_date)


market_calendar = _MarketCalendar()
