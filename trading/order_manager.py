import uuid
from datetime import datetime

from core.dhan_client import DhanClient
from data.cache import DataCache


class OrderManager:
    """Order lifecycle management for both paper and live trading."""

    def __init__(self, cache: DataCache, dhan: DhanClient = None, mode: str = "paper"):
        self.cache = cache
        self.dhan = dhan
        self.mode = mode

    @property
    def exchange_segment(self):
        return "NSE_EQ"

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
        product_type: str = "INTR",
        strategy: str = "auto_ai",
        security_id: str = "",
        tag: str = "ai_trading",
    ) -> dict:
        """Place an order. Returns order dict with order_id."""
        order_id = str(uuid.uuid4())[:12]
        order = {
            "order_id": order_id,
            "symbol": symbol.upper(),
            "side": side,
            "quantity": quantity,
            "price": price,
            "filled_price": 0,
            "status": "PLACED",
            "strategy": strategy,
            "order_type": order_type,
            "product_type": product_type,
            "created_at": datetime.now().isoformat(),
            "mode": self.mode,
        }

        if self.mode == "live" and self.dhan is not None and security_id:
            try:
                txn_type = "BUY" if side.upper() == "BUY" else "SELL"
                ot = "MARKET" if order_type.upper() == "MARKET" else "LIMIT"
                resp = self.dhan.place_order(
                    security_id=security_id,
                    exchange_segment=self.exchange_segment,
                    transaction_type=txn_type,
                    quantity=quantity,
                    order_type=ot,
                    price=price if ot == "LIMIT" else 0,
                    product_type=product_type,
                    tag=tag,
                )
                if resp and resp.get("data"):
                    order["order_id"] = str(resp["data"].get("orderId", order_id))
                    order["status"] = resp["data"].get("orderStatus", "PLACED").upper()
                    order["filled_price"] = resp["data"].get("tradedPrice", price)
                else:
                    order["status"] = "ERROR"
                    order["error"] = str(resp)
            except Exception as e:
                order["status"] = "ERROR"
                order["error"] = str(e)
        else:
            # Paper mode: immediate fill at market (price assumed = last price)
            order["status"] = "FILLED"
            order["filled_price"] = price

        self.cache.save_order(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if self.mode == "live" and self.dhan is not None:
            try:
                resp = self.dhan.cancel_order(order_id)
                return bool(resp and resp.get("status") == "success")
            except Exception:
                return False
        cur = self.cache._conn.cursor()
        cur.execute("UPDATE orders SET status='CANCELLED' WHERE order_id=?", (order_id,))
        self.cache._conn.commit()
        return True