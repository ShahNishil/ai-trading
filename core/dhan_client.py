import os
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from dhanhq import DhanContext, dhanhq


class DhanClient:
    """Wrapper around the official DhanHQ Python client."""

    def __init__(self, config: dict, use_paper_credentials: Optional[dict] = None):
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        self.config = config
        client_id = os.getenv("DHAN_CLIENT_ID", "")
        access_token = os.getenv("DHAN_ACCESS_TOKEN", "")

        placeholders = {"", "your_client_id_here", "your_access_token_here", "placeholder"}
        if not client_id or not access_token or client_id.lower() in placeholders or access_token.lower() in placeholders:
            raise ValueError(
                "DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in .env. "
                "Copy .env.example to .env and fill in your Dhan credentials. "
                "Without Dhan credentials the app will use Yahoo Finance (yfinance) free data for scanning/backtesting."
            )

        self.context = DhanContext(client_id, access_token)
        self.client = dhanhq(self.context)
        self.trading_mode = config.get("dhan", {}).get("trading_mode", "paper")

    # ------------------------------------------------------------------
    # Account / Portfolio
    # ------------------------------------------------------------------
    def get_fund_limit(self) -> dict:
        try:
            return self.client.get_fund_limit() or {}
        except Exception as e:
            return {"error": str(e)}

    def get_positions(self) -> list:
        try:
            resp = self.client.get_positions() or {}
            return resp.get("data", []) if isinstance(resp.get("data"), list) else []
        except Exception as e:
            return []

    def get_holdings(self) -> list:
        try:
            resp = self.client.get_holdings() or {}
            return resp.get("data", []) if isinstance(resp.get("data"), list) else []
        except Exception as e:
            return []

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "LIMIT",
        price: float = 0.0,
        product_type: str = "INTR",
        validity: str = "DAY",
        tag: str = "ai_trading",
    ) -> dict:
        return self.client.place_order(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            validity=validity,
            tag=tag,
        )

    def modify_order(self, order_id: str, order_type: str, quantity: int, price: float) -> dict:
        return self.client.modify_order(
            order_id=str(order_id),
            order_type=order_type,
            quantity=quantity,
            price=price,
        )

    def cancel_order(self, order_id: str) -> dict:
        return self.client.cancel_order(order_id=str(order_id))

    def get_order_list(self) -> list:
        try:
            resp = self.client.get_order_list() or {}
            return resp.get("data", []) if isinstance(resp.get("data"), list) else []
        except Exception as e:
            return []

    def get_trade_book(self) -> list:
        try:
            resp = self.client.get_trade_book() or {}
            return resp.get("data", []) if isinstance(resp.get("data"), list) else []
        except Exception as e:
            return []

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------
    def get_quote(self, securities: dict) -> dict:
        return self.client.quote_data(securities=securities)

    def get_ohlc(self, securities: dict) -> dict:
        return self.client.ohlc_data(securities=securities)

    def get_ltp(self, security_id: str, exchange: str = "NSE_EQ") -> Optional[float]:
        try:
            resp = self.client.intraday_data(securities={exchange: [str(security_id)]})
            if resp and resp.get("data"):
                key = f"{exchange}|{security_id}"
                if key in resp["data"]:
                    return resp["data"][key].get("last_price") or resp["data"][key].get("close")
            return None
        except Exception:
            return None

    def get_historical_daily(
        self,
        security_id: str,
        exchange_segment: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        from_date: str = "2024-01-01",
        to_date: str = "2025-12-31",
        expiry_code: int = 0,
        oi: bool = False,
    ) -> dict:
        return self.client.historical_daily_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
            expiry_code=expiry_code,
            oi=oi,
        )

    def get_historical_intraday(
        self,
        security_id: str,
        exchange_segment: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        from_date: str = "",
        to_date: str = "",
        interval: int = 15,
        oi: bool = False,
    ) -> dict:
        return self.client.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            oi=oi,
        )

    # ------------------------------------------------------------------
    # Instrument Lists
    # ------------------------------------------------------------------
    def fetch_security_list(self, mode: str = "compact") -> list:
        try:
            resp = self.client.fetch_security_list(mode)
            if resp and resp.get("data"):
                return resp["data"]
            return []
        except Exception as e:
            return []

    def get_market_status(self) -> Optional[str]:
        try:
            resp = self.client.get_market_status()
            if resp and resp.get("data"):
                return resp["data"].get("status")
            return None
        except Exception as e:
            return str(e)


# Module-level constants for readability
NSE = "NSE_EQ"
NSE_FNO = "NSE_FNO"
BSE = "BSE_EQ"
NSE_CURRENCY = "NSE_CURRENCY"
MCX_COMM = "MCX_COMM"
IDX_I = "IDX_I"