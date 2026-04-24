"""Instrument metadata shared across the trading agent."""

LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "SENSEX": 20,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 75,
}

EXPIRY_WEEKDAY = {
    "NIFTY": 1,
    "BANKNIFTY": 1,
    "FINNIFTY": 1,
    "SENSEX": 3,
    "BANKEX": 3,
}

HAS_WEEKLY_EXPIRY = {
    "NIFTY": True,       # NSE - weekly expires Tuesday
    "BANKNIFTY": False,  # weekly discontinued Nov 2024
    "FINNIFTY": False,   # weekly discontinued Nov 2024
    "SENSEX": True,      # BSE - weekly expires Thursday
    "BANKEX": False,     # weekly discontinued Nov 2024
    "MIDCPNIFTY": False, # weekly discontinued Nov 2024
}

EXCHANGE = {
    "NIFTY": "NSE",
    "BANKNIFTY": "NSE",
    "FINNIFTY": "NSE",
    "MIDCPNIFTY": "NSE",
    "SENSEX": "BSE",
    "BANKEX": "BSE",
}

STRIKE_GAP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
    "BANKEX": 100,
}

FYERS_SYMBOL = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
    "BANKEX": "BSE:BANKEX-INDEX",
}

NSE_INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

BSE_INDEX_SYMBOLS = {"SENSEX", "BANKEX"}

MCX_SYMBOLS = set()
