import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "nifty" / "raw"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fyers_client import fyers_client

INSTRUMENT_SYMBOLS = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
    "INDIAVIX": "NSE:INDIAVIX-INDEX",
}

RESOLUTIONS = ["5", "15", "60", "D"]
SESSION_START = "09:15"
SESSION_END = "15:30"
LOOKBACK_DAYS = 1460
INTRADAY_CHUNK_DAYS = 90
DAILY_CHUNK_DAYS = 365

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download multi-timeframe historical market data")
    parser.add_argument(
        "--start",
        dest="start_date",
        default=None,
        help="Inclusive start date (YYYY-MM-DD). Overrides --days when provided.",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        default=None,
        help="Inclusive end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=LOOKBACK_DAYS,
        help=f"Lookback window in days when --start is not provided (default: {LOOKBACK_DAYS}).",
    )
    return parser.parse_args()


def build_dataframe(candles: list, resolution: str) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

    if df.empty:
        empty_index = pd.DatetimeIndex([], name="timestamp")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=empty_index)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.set_index("timestamp").sort_index()

    if resolution != "D":
        df = df.between_time(SESSION_START, SESSION_END, inclusive="both")

    return df


def fetch_historical_candles(symbol: str, resolution: str, start_date: datetime.date, end_date: datetime.date) -> list:
    chunk_days = DAILY_CHUNK_DAYS if resolution == "D" else INTRADAY_CHUNK_DAYS

    candles: list = []
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
        chunk = fyers_client.get_historical(
            symbol=symbol,
            resolution=resolution,
            date_from=chunk_start.strftime("%Y-%m-%d"),
            date_to=chunk_end.strftime("%Y-%m-%d"),
        )
        candles.extend(chunk)
        chunk_start = chunk_end + timedelta(days=1)

    return candles


def save_history(instrument: str, resolution: str, date_from: str, date_to: str) -> int:
    symbol = INSTRUMENT_SYMBOLS[instrument]
    candles = fetch_historical_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=datetime.strptime(date_from, "%Y-%m-%d").date(),
        end_date=datetime.strptime(date_to, "%Y-%m-%d").date(),
    )
    df = build_dataframe(candles, resolution)
    if not df.empty:
        df = df[~df.index.duplicated(keep="last")]

    output_path = DATA_DIR / f"{instrument}_{resolution}.parquet"
    df.to_parquet(output_path)

    if df.empty:
        date_range = "empty"
        logger.warning(
            "Downloaded empty dataset for instrument=%s resolution=%s range=%s..%s path=%s",
            instrument,
            resolution,
            date_from,
            date_to,
            output_path,
        )
    else:
        start = df.index.min().strftime("%Y-%m-%d %H:%M:%S %Z")
        end = df.index.max().strftime("%Y-%m-%d %H:%M:%S %Z")
        date_range = f"{start} -> {end}"

    print(
        f"instrument={instrument} resolution={resolution} rows={len(df)} "
        f"date_range={date_range}"
    )
    return len(df)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.days < 1:
        raise ValueError("--days must be >= 1")

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else datetime.now().date()
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)

    if start_date > end_date:
        raise ValueError("--start must be on or before --end")

    date_from = start_date.strftime("%Y-%m-%d")
    date_to = end_date.strftime("%Y-%m-%d")

    failures: list[tuple[str, str]] = []
    empty_downloads: list[tuple[str, str]] = []

    for instrument in INSTRUMENT_SYMBOLS:
        instrument_resolutions = ["5"] if instrument == "INDIAVIX" else RESOLUTIONS
        for resolution in instrument_resolutions:
            try:
                row_count = save_history(instrument, resolution, date_from, date_to)
                if row_count == 0:
                    empty_downloads.append((instrument, resolution))
            except Exception as exc:
                failures.append((instrument, resolution))
                logger.exception(
                    "Failed to download history for instrument=%s resolution=%s: %s",
                    instrument,
                    resolution,
                    exc,
                )

    if empty_downloads:
        logger.warning("Empty downloads detected: %s", empty_downloads)
        logger.warning(
            "If daily index files are empty, verify Fyers historical response for resolution='D' and symbol mapping."
        )

    if failures:
        logger.error("Download failures detected: %s", failures)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
