import numpy as np
import pandas as pd


class DriftMonitor:
    Z_ALERT_THRESHOLD = 4.0   # |z| > 4 = severe outlier vs training dist

    def __init__(self, feature_stats: dict[str, dict]):
        # feature_stats: {feature_name: {"mean": float, "std": float}}
        self.feature_stats = feature_stats

    def check(self, row: pd.Series) -> list[str]:
        """Return list of feature names in severe drift (|z| > threshold)."""
        alerts = []
        for feat, stats in self.feature_stats.items():
            val = row.get(feat, np.nan)
            if pd.isna(val):
                continue
            std = float(stats.get("std", 0.0)) or 1e-10
            z = abs((float(val) - float(stats["mean"])) / std)
            if z > self.Z_ALERT_THRESHOLD:
                alerts.append(f"{feat}(z={z:.1f})")
        return alerts
