"""Paper-capital tracking persisted to model_improver/data/capital.parquet."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapitalSnapshot:
    timestamp: datetime
    capital: float
    daily_pnl: float
    cumulative_pnl: float
    open_margin_used: float
    event: str


SNAPSHOT_COLUMNS = [
    "timestamp",
    "capital",
    "daily_pnl",
    "cumulative_pnl",
    "open_margin_used",
    "event",
]


class CapitalTracker:
    INITIAL_CAPITAL = 100_000.0

    def __init__(self, data_dir: Path | None = None, initial_capital: float | None = None, capital_path: Path | None = None) -> None:
        # Backward-compatible construction:
        # - preferred: CapitalTracker(data_dir=Path(...))
        # - legacy:   CapitalTracker(initial_capital=..., capital_path=Path(...))
        if capital_path is not None:
            self.capital_path = Path(capital_path)
        else:
            resolved_data_dir = Path(data_dir) if data_dir is not None else Path("model_improver/data")
            self.capital_path = resolved_data_dir / "capital.parquet"
        self.capital_path.parent.mkdir(parents=True, exist_ok=True)

        self.initial_capital = float(self.INITIAL_CAPITAL if initial_capital is None else initial_capital)
        self._reserved_margin: dict[str, float] = {}
        self._current_capital = self.initial_capital
        self._cumulative_pnl = 0.0

        history = self.load_history()
        if history.empty:
            self.snapshot(event="INIT")
        else:
            last = history.iloc[-1]
            self._current_capital = float(last["capital"])
            self._cumulative_pnl = float(last["cumulative_pnl"])

    def get_available_capital(self) -> float:
        return max(0.0, self._current_capital - self._open_margin_used())

    def get_current_capital(self) -> float:
        return float(self._current_capital)

    @property
    def current_capital(self) -> float:
        return self.get_current_capital()

    def reserve_margin(self, trade_id: str, amount: float) -> bool:
        margin = max(0.0, float(amount))
        if self.get_available_capital() < margin:
            logger.info("capital_reserve_failed trade_id=%s amount=%.2f available=%.2f", trade_id, margin, self.get_available_capital())
            return False
        self._reserved_margin[str(trade_id)] = margin
        self.snapshot(event="ENTRY")
        return True

    def release_margin(self, trade_id: str, pnl_net: float) -> None:
        self._reserved_margin.pop(str(trade_id), None)
        pnl = float(pnl_net)
        self._cumulative_pnl += pnl
        self._current_capital = self.initial_capital + self._cumulative_pnl
        self.snapshot(event="EXIT")

    # Backward compatibility for previous interface.
    def apply_realized_pnl(self, pnl: float, timestamp: datetime | None = None):
        self._cumulative_pnl += float(pnl)
        self._current_capital = self.initial_capital + self._cumulative_pnl
        self.snapshot(event="EXIT", timestamp=timestamp)
        return self.load_history().iloc[-1]

    def snapshot(self, event: str, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.utcnow()
        daily_series = self.daily_pnl_series()
        today = ts.date()
        daily_pnl = float(daily_series.get(today, 0.0))
        row = CapitalSnapshot(
            timestamp=ts,
            capital=float(self._current_capital),
            daily_pnl=daily_pnl,
            cumulative_pnl=float(self._cumulative_pnl),
            open_margin_used=float(self._open_margin_used()),
            event=str(event),
        )

        history = self.load_history()
        updated = pd.concat([history, pd.DataFrame([asdict(row)])], ignore_index=True)
        updated.to_parquet(self.capital_path, index=False, engine="pyarrow")

    def load_history(self) -> pd.DataFrame:
        if not self.capital_path.exists():
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        try:
            df = pd.read_parquet(self.capital_path, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(self.capital_path)
        for column in SNAPSHOT_COLUMNS:
            if column not in df.columns:
                df[column] = None
        return df[SNAPSHOT_COLUMNS].copy()

    def daily_pnl_series(self) -> pd.Series:
        history = self.load_history()
        if history.empty:
            return pd.Series(dtype=float)

        ts = pd.to_datetime(history["timestamp"], errors="coerce")
        cum = pd.to_numeric(history["cumulative_pnl"], errors="coerce").fillna(0.0)
        frame = pd.DataFrame({"date": ts.dt.date, "cum": cum}).dropna()
        if frame.empty:
            return pd.Series(dtype=float)
        end_of_day = frame.groupby("date", as_index=True)["cum"].last()
        return end_of_day.diff().fillna(end_of_day.iloc[0])

    def _open_margin_used(self) -> float:
        return float(sum(self._reserved_margin.values()))

