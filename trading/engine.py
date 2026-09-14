import json
from datetime import datetime

from core.dhan_client import DhanClient
from core.risk_manager import RiskManager
from data.cache import DataCache
from trading.order_manager import OrderManager
from trading.portfolio import Portfolio


class TradingEngine:
    """Unified engine for both paper and live trading."""

    def __init__(
        self,
        config: dict,
        dhan: DhanClient = None,
        cache: DataCache = None,
        mode: str = None,
    ):
        self.config = config
        self.mode = mode or config.get("dhan", {}).get("trading_mode", "paper")
        self.cache = cache or DataCache()
        self.dhan = dhan
        self.orders = OrderManager(self.cache, dhan=dhan, mode=self.mode)
        self.portfolio = Portfolio(self.cache, mode=self.mode)
        self.risk = RiskManager(config, portfolio=self.portfolio)

    # ------------------------------------------------------------------
    # Capital / account
    # ------------------------------------------------------------------
    def get_capital(self) -> float:
        if self.mode == "live" and self.dhan is not None:
            try:
                fund = self.dhan.get_fund_limit()
                data = fund.get("data", {}) if isinstance(fund.get("data"), dict) else {}
                if isinstance(data, dict):
                    return float(data.get("availableBalance", 0) or 0)
            except Exception:
                pass
        return self.portfolio.equity()

    def get_account_summary(self) -> dict:
        summary = {
            "mode": self.mode,
            "realized_pnl": self.portfolio.get_realized_pnl(),
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(),
            "open_positions": len(self.portfolio.get_open_positions()),
            "capital": self.get_capital(),
        }
        if self.mode == "live":
            summary["dhan_fund"] = self.dhan.get_fund_limit() if self.dhan else {}
        return summary

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------
    def enter_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        security_id: str = "",
        side: str = "BUY",
        strategy: str = "auto_ai",
        order_type: str = "MARKET",
        reason: str = "",
        exchange_segment: str = None,
        instrument_type: str = "EQUITY",
    ) -> dict:
        allowed, block_reason = self.risk.can_open_position(self.get_capital())
        if not allowed:
            return {"success": False, "error": block_reason}

        order = self.orders.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=entry_price,
            order_type=order_type,
            product_type=self.config.get("auto_trade", {}).get("product_type", "INTR"),
            strategy=strategy,
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
        )
        if order.get("status") == "ERROR":
            return {"success": False, "error": order.get("error", "Order failed")}

        filled_price = order.get("filled_price") or entry_price
        trade_id = self.portfolio.open_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=filled_price,
            strategy=strategy,
            entry_reason=reason,
            exchange_segment=exchange_segment or "NSE_EQ",
            instrument_type=instrument_type,
        )
        return {
            "success": True,
            "order": order,
            "trade_id": trade_id,
            "filled_price": filled_price,
        }

    # ------------------------------------------------------------------
    # Derivatives ({EX} futures / options)
    # ------------------------------------------------------------------
    def enter_option_structure(
        self,
        underlying: str,
        strategy_name: str,
        legs: list,
        num_lots: int,
        lot_size: int,
        premiums: dict = None,
        reason: str = "",
    ) -> dict:
        """Open a multi-leg option structure (paper or, with Dhan, live).

        legs: list of dicts {option_type, strike, side, security_id (optional),
                              instrument_type, trading_symbol}
        premiums: optional {leg_index: price} used for paper fills and sizing.
        """
        allowed, block_reason = self.risk.can_open_position(self.get_capital())
        if not allowed:
            return {"success": False, "error": block_reason}

        premiums = premiums or {}
        qty = lot_size * num_lots
        order_ids = []
        net_debit = 0.0
        failures = []
        for i, leg in enumerate(legs):
            side = leg["side"]
            prem = float(premiums.get(i) or leg.get("entry_premium") or 0)
            net_debit += (1.0 if side == "BUY" else -1.0) * prem
            instrument_type = leg.get("instrument_type", "OPTIDX" if underlying in ("NIFTY", "BANKNIFTY") else "OPTSTK")
            order = self.orders.place_order(
                symbol=leg.get("trading_symbol") or f"{underlying} {leg['option_type']} {leg['strike']:,.0f}",
                side=side,
                quantity=qty,
                price=prem,
                order_type="LIMIT",
                product_type=self.config.get("derivatives", {}).get("product_type", "INTR"),
                strategy=f"deriv_{strategy_name}",
                security_id=leg.get("security_id", ""),
                exchange_segment="NSE_FNO",
                instrument_type=instrument_type,
                tag="ai_derivatives",
            )
            if order.get("status") == "ERROR":
                failures.append({"leg": leg, "error": order.get("error")})
            else:
                order_ids.append(order.get("order_id"))

        if failures:
            return {"success": False, "error": f"Order failures: {failures}"}

        symbol = f"{underlying} {strategy_name}"
        leg_meta = json.dumps(
            [{"ot": l["option_type"], "strike": l["strike"], "side": l["side"]} for l in legs]
        )
        trade_id = self.portfolio.open_trade(
            symbol=symbol,
            side=strategy_name,
            quantity=qty,
            entry_price=round(net_debit / qty, 6) if qty else 0.0,
            strategy=f"deriv_{strategy_name}",
            entry_reason=reason or leg_meta,
            exchange_segment="NSE_FNO",
            instrument_type=strategy_name,
        )
        return {
            "success": True,
            "order_ids": order_ids,
            "trade_id": trade_id,
            "net_debit": round(net_debit, 2),
            "num_lots": num_lots,
            "lot_size": lot_size,
        }

    def enter_futures_position(
        self,
        underlying: str,
        quantity: int,
        entry_price: float,
        security_id: str = "",
        side: str = "BUY",
        reason: str = "",
        instrument_type: str = "FUTIDX",
    ) -> dict:
        """Open a futures position (paper or live via Dhan NSE_FNO)."""
        return self.enter_position(
            symbol=underlying,
            quantity=quantity,
            entry_price=entry_price,
            security_id=security_id,
            side=side,
            strategy="deriv_futures",
            order_type="MARKET",
            reason=reason,
            exchange_segment="NSE_FNO",
            instrument_type=instrument_type,
        )

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        security_id: str = "",
        side: str = "SELL",
        reason: str = "manual_exit",
        quantity: int = 0,
        symbol: str = "",
        exchange_segment: str = None,
        instrument_type: str = "EQUITY",
    ) -> dict:
        order = self.orders.place_order(
            symbol=symbol or "",
            side=side,
            quantity=quantity,
            price=exit_price,
            order_type="MARKET",
            product_type=self.config.get("auto_trade", {}).get("product_type", "INTR"),
            strategy="exit",
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type=instrument_type,
        )
        result = self.portfolio.close_trade(trade_id, exit_price or order.get("filled_price") or 0, reason)
        if result is None:
            return {"success": False, "error": "Trade not found"}
        result["success"] = True
        result["order"] = order
        return result

    def update_positions_with_prices(self, current_prices: dict):
        """Apply trailing stops and targets based on current prices."""
        for pos in self.portfolio.get_open_positions():
            symbol = pos["symbol"].upper()
            price = current_prices.get(symbol)
            if not price:
                continue
            entry = float(pos["entry_price"])
            qty = int(pos["quantity"])
            if pos.get("side", "BUY").upper() == "BUY":
                if price <= self.risk.trailing_stop(entry, price):
                    self.portfolio.close_trade(pos["trade_id"], price, "trailing_stop")
                elif price >= self.risk.compute_target(entry):
                    self.portfolio.close_trade(pos["trade_id"], price, "target_hit")