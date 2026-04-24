import logging

logger = logging.getLogger(__name__)

SPREAD_THRESHOLD_PCT = 2.0


class NSEBSEStrategy:
    def select(self, processed, atm, direction, mode="NORMAL") -> dict | None:
        try:
            candidates = []
            for item in processed:
                if item.get("strike"):
                    candidates.append(item)
            if not candidates:
                return None

            if direction == "BULLISH":
                itm = [c for c in candidates if c["strike"] < atm]
                itm = sorted(itm, key=lambda x: x["strike"], reverse=True)
            else:
                itm = [c for c in candidates if c["strike"] > atm]
                itm = sorted(itm, key=lambda x: x["strike"])

            normal_order = []
            if len(itm) >= 2:
                normal_order.append(itm[1])
            if len(itm) >= 1:
                normal_order.append(itm[0])

            atm_candidates = [c for c in candidates if c["strike"] == atm]
            atm_candidate = atm_candidates[0] if atm_candidates else None

            if mode == "SCALP":
                candidates_ordered = []
                if atm_candidate:
                    candidates_ordered.append(atm_candidate)
                if len(itm) >= 1:
                    candidates_ordered.append(itm[0])
                if len(itm) >= 2:
                    candidates_ordered.append(itm[1])
            else:
                candidates_ordered = list(normal_order)
                if atm_candidate:
                    candidates_ordered.append(atm_candidate)

            for target in candidates_ordered:
                if target["spread"] is None:
                    continue
                spread_pct = abs(target["spread"]) * 100
                if spread_pct < SPREAD_THRESHOLD_PCT:
                    logger.info(f"Selected {target['instrument']} with spread {spread_pct:.2f}%")
                    return target
                logger.warning(f"{target['instrument']} spread {spread_pct:.2f}% too wide, trying next")

            if candidates_ordered:
                best = candidates_ordered[0]
                logger.warning(f"No spread data available, falling back to {best['instrument']} by ITM order")
                return best
            return None
        except Exception as e:
            logger.error(f"NSEBSEStrategy error: {e}")
            return None


class StrikeSelector:
    def select(self, processed: list, atm: float, direction: str, mode: str = "NORMAL") -> dict | None:
        return NSEBSEStrategy().select(processed, atm, direction, mode)


strike_selector = StrikeSelector()
