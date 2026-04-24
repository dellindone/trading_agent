import logging

import pandas as pd

from core.data.multi_tf_builder import multi_tf_builder
from core.data.option_premium_history import build_premium_history
from core.features.candlestick import compute_candlestick_features
from core.features.option_features import compute_option_features
from core.features.pattern_context import compute_pattern_context
from core.features.regime import RegimeDetector
from core.features.vix_features import compute_vix_features

logger = logging.getLogger(__name__)


def _ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index, utc=True)
    elif normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    else:
        normalized.index = normalized.index.tz_convert("UTC")
    return normalized.sort_index()


def _suffix_columns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        column_name = str(column)
        if column_name.endswith(f"_{suffix}"):
            renamed[column] = column_name
        else:
            renamed[column] = f"{column_name}_{suffix}"
    return frame.rename(columns=renamed)


def _drop_base_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    base_columns = {"open", "high", "low", "close", "volume"}
    keep_columns = [column for column in frame.columns if str(column) not in base_columns]
    return frame.loc[:, keep_columns]


def _filter_date_range(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    filtered = frame
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        filtered = filtered.loc[filtered.index >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        filtered = filtered.loc[filtered.index <= end_ts]
    return filtered


class FeatureEngineer:
    def __init__(self) -> None:
        self.regime_detector = RegimeDetector()

    def _prepare_5m_frame(
        self,
        frame: pd.DataFrame,
        instrument: str,
        days_to_expiry: int = 7,
        risk_free_rate: float = 6.5,
    ) -> pd.DataFrame:
        featured = _ensure_utc_index(frame)
        featured = compute_candlestick_features(featured)
        featured = compute_pattern_context(featured)

        if "vix" in featured.columns:
            featured = compute_vix_features(featured)
            if "ce_premium" not in featured.columns or "pe_premium" not in featured.columns:
                premium_history = build_premium_history(
                    featured[["close", "vix"]],
                    days_to_expiry=days_to_expiry,
                    risk_free_rate=risk_free_rate,
                )
                featured = featured.join(premium_history)
            featured = compute_option_features(featured, instrument=instrument, tf="5m")
        else:
            logger.warning("Skipping option_features on 5m frame because vix column is missing")

        return featured

    def _prepare_15m_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        featured = _ensure_utc_index(frame)
        featured = self.regime_detector.detect(featured, tf="15m")
        featured = compute_candlestick_features(featured)
        featured = _drop_base_ohlcv(featured)
        return _suffix_columns(featured, "15m")

    def _prepare_60m_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        featured = _ensure_utc_index(frame)
        featured = self.regime_detector.detect(featured, tf="60m")
        featured = compute_candlestick_features(featured)
        featured = _drop_base_ohlcv(featured)
        return _suffix_columns(featured, "60m")

    def _prepare_daily_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        featured = _ensure_utc_index(frame)
        featured = self.regime_detector.detect(featured, tf="D")
        featured = _drop_base_ohlcv(featured)
        return _suffix_columns(featured, "D")

    def _merge_asof_features(self, left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> pd.DataFrame:
        if right_frame.empty:
            return left_frame

        left_reset = left_frame.sort_index().reset_index().rename(columns={"index": "timestamp"})
        right_reset = right_frame.sort_index().reset_index().rename(columns={"index": "timestamp"})

        merged = pd.merge_asof(
            left_reset,
            right_reset,
            on="timestamp",
            direction="backward",
        )
        return merged.set_index("timestamp").sort_index()

    def _build_from_frames(
        self,
        frames: dict[str, pd.DataFrame],
        instrument: str,
        days_to_expiry: int = 7,
        risk_free_rate: float = 6.5,
    ) -> pd.DataFrame:
        frame_5m = frames.get("5m")
        if frame_5m is None or frame_5m.empty:
            raise ValueError(f"Missing required 5m frame for instrument={instrument}")

        assembled = self._prepare_5m_frame(
            frame_5m,
            instrument=instrument,
            days_to_expiry=days_to_expiry,
            risk_free_rate=risk_free_rate,
        )

        frame_15m = frames.get("15m")
        if frame_15m is not None and not frame_15m.empty:
            assembled = self._merge_asof_features(assembled, self._prepare_15m_frame(frame_15m))
        else:
            logger.warning("Missing 15m frame for instrument=%s", instrument)

        frame_60m = frames.get("60m")
        if frame_60m is not None and not frame_60m.empty:
            assembled = self._merge_asof_features(assembled, self._prepare_60m_frame(frame_60m))
        else:
            logger.warning("Missing 60m frame for instrument=%s", instrument)

        frame_daily = frames.get("D")
        if frame_daily is not None and not frame_daily.empty:
            assembled = self._merge_asof_features(assembled, self._prepare_daily_frame(frame_daily))
        else:
            logger.warning("Missing D frame for instrument=%s", instrument)

        assembled["instrument"] = instrument.upper()
        return assembled.sort_index()

    def build(
        self,
        instrument: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames = multi_tf_builder.build(instrument)
        assembled = self._build_from_frames(frames, instrument=instrument)
        return _filter_date_range(assembled, start_date=start_date, end_date=end_date)


def build_feature_frame(
    frames: dict[str, pd.DataFrame],
    instrument: str,
    days_to_expiry: int = 7,
    risk_free_rate: float = 6.5,
) -> pd.DataFrame:
    return feature_engineer._build_from_frames(
        frames,
        instrument=instrument,
        days_to_expiry=days_to_expiry,
        risk_free_rate=risk_free_rate,
    )


feature_engineer = FeatureEngineer()
