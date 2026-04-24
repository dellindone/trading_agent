"""Model promotion evaluator using accumulated shadow trading performance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PromotionReport:
    eligible: bool
    reason: str
    total_trades: int
    win_rate: float
    avg_rr: float
    sharpe_ratio: float
    max_drawdown_pct: float
    weeks_of_data: float
    recommendation: str


class ModelPromoter:
    MIN_WEEKS = 6
    MIN_TRADES = 50
    MIN_WIN_RATE = 0.52
    MIN_SHARPE = 0.8
    MAX_DRAWDOWN_PCT = 0.15

    def evaluate(self, trades_df: pd.DataFrame, capital_df: pd.DataFrame) -> PromotionReport:
        trades = trades_df.copy() if trades_df is not None else pd.DataFrame()
        capital = capital_df.copy() if capital_df is not None else pd.DataFrame()

        if trades.empty:
            return PromotionReport(
                eligible=False,
                reason="No shadow trades available.",
                total_trades=0,
                win_rate=0.0,
                avg_rr=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                weeks_of_data=0.0,
                recommendation="CONTINUE_SHADOW",
            )

        trades["timestamp_entry"] = pd.to_datetime(trades.get("timestamp_entry"), errors="coerce", utc=True)
        trades["timestamp_exit"] = pd.to_datetime(trades.get("timestamp_exit"), errors="coerce", utc=True)

        closed = trades[trades["timestamp_exit"].notna()].copy()
        total_trades = int(len(closed))
        if total_trades == 0:
            return PromotionReport(
                eligible=False,
                reason="No closed trades available yet.",
                total_trades=0,
                win_rate=0.0,
                avg_rr=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=self._max_drawdown(capital),
                weeks_of_data=self._weeks_of_data(trades),
                recommendation="CONTINUE_SHADOW",
            )

        pnl_net = pd.to_numeric(closed.get("pnl_net"), errors="coerce").fillna(0.0)
        wins = pnl_net > 0
        win_rate = float(wins.mean()) if total_trades else 0.0

        avg_rr = self._avg_rr_on_winners(closed)
        sharpe_ratio = self._daily_sharpe(closed)
        max_drawdown_pct = self._max_drawdown(capital)
        weeks_of_data = self._weeks_of_data(trades)

        failures: list[str] = []
        if weeks_of_data < self.MIN_WEEKS:
            failures.append(f"weeks_of_data {weeks_of_data:.2f} < {self.MIN_WEEKS}")
        if total_trades < self.MIN_TRADES:
            failures.append(f"total_trades {total_trades} < {self.MIN_TRADES}")
        if win_rate < self.MIN_WIN_RATE:
            failures.append(f"win_rate {win_rate:.3f} < {self.MIN_WIN_RATE}")
        if sharpe_ratio < self.MIN_SHARPE:
            failures.append(f"sharpe_ratio {sharpe_ratio:.3f} < {self.MIN_SHARPE}")
        if max_drawdown_pct > self.MAX_DRAWDOWN_PCT:
            failures.append(f"max_drawdown_pct {max_drawdown_pct:.3f} > {self.MAX_DRAWDOWN_PCT}")

        eligible = len(failures) == 0
        if eligible:
            recommendation = "PROMOTE"
            reason = "All promotion criteria passed."
        else:
            if weeks_of_data < self.MIN_WEEKS or total_trades < self.MIN_TRADES:
                recommendation = "CONTINUE_SHADOW"
            else:
                recommendation = "RETRAIN"
            reason = "; ".join(failures)

        return PromotionReport(
            eligible=eligible,
            reason=reason,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_rr=avg_rr,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            weeks_of_data=weeks_of_data,
            recommendation=recommendation,
        )

    def save_report(self, report: PromotionReport, output_dir: Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out / f"promotion_report_{ts}.json"
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    def _avg_rr_on_winners(self, closed: pd.DataFrame) -> float:
        if closed.empty:
            return 0.0

        entry = pd.to_numeric(closed.get("entry_premium"), errors="coerce")
        exit_ = pd.to_numeric(closed.get("exit_premium"), errors="coerce")
        sl = pd.to_numeric(closed.get("sl_price"), errors="coerce")
        pnl = pd.to_numeric(closed.get("pnl_net"), errors="coerce").fillna(0.0)

        risk_per_unit = (entry - sl).replace(0.0, np.nan)
        actual_gain_per_unit = (exit_ - entry)
        rr = actual_gain_per_unit / risk_per_unit
        winner_rr = rr[pnl > 0].replace([np.inf, -np.inf], np.nan).dropna()
        if winner_rr.empty:
            return 0.0
        return float(winner_rr.mean())

    def _daily_sharpe(self, closed: pd.DataFrame) -> float:
        if closed.empty:
            return 0.0

        exit_ts = pd.to_datetime(closed.get("timestamp_exit"), errors="coerce", utc=True)
        pnl = pd.to_numeric(closed.get("pnl_net"), errors="coerce").fillna(0.0)
        daily = pd.DataFrame({"date": exit_ts.dt.date, "pnl": pnl}).dropna()
        if daily.empty:
            return 0.0
        daily_pnl = daily.groupby("date", as_index=False)["pnl"].sum()["pnl"]
        if len(daily_pnl) < 2:
            return 0.0
        std = float(daily_pnl.std(ddof=0))
        if std <= 0:
            return 0.0
        mean = float(daily_pnl.mean())
        return float((mean / std) * np.sqrt(252.0))

    def _max_drawdown(self, capital: pd.DataFrame) -> float:
        if capital is None or capital.empty or "capital" not in capital.columns:
            return 0.0
        capital_series = pd.to_numeric(capital["capital"], errors="coerce").dropna()
        if capital_series.empty:
            return 0.0
        running_peak = capital_series.cummax()
        dd = (running_peak - capital_series) / running_peak.replace(0.0, np.nan)
        dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(dd.max())

    def _weeks_of_data(self, trades: pd.DataFrame) -> float:
        entry = pd.to_datetime(trades.get("timestamp_entry"), errors="coerce", utc=True)
        exit_ = pd.to_datetime(trades.get("timestamp_exit"), errors="coerce", utc=True)
        start = entry.min()
        end_candidates = [entry.max(), exit_.max()]
        end = max([x for x in end_candidates if pd.notna(x)], default=pd.NaT)
        if pd.isna(start) or pd.isna(end):
            return 0.0
        delta_days = (end - start).total_seconds() / 86400.0
        return max(0.0, float(delta_days / 7.0))

