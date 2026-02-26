"""
Binance WebSocket helpers for real-time crypto price data.
"""
import asyncio
import json
import time
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

BINANCE_WSS = "wss://stream.binance.com:9443"

# Asset to Binance symbol mapping
SYMBOLS = {
    "BTC": "btcusdt",
    "BTC_5M": "btcusdt",
    "ETH": "ethusdt",
    "ETH_5M": "ethusdt",
    "SOL": "solusdt",
    "SOL_5M": "solusdt",
    "XRP": "xrpusdt",
    "XRP_5M": "xrpusdt",
}


@dataclass
class PriceState:
    """Real-time price state for an asset."""
    asset: str
    price: float = 0.0
    last_update: Optional[datetime] = None
    history: List[float] = field(default_factory=list)
    max_history: int = 1000

    def update(self, price: float):
        self.price = price
        self.last_update = datetime.now(timezone.utc)
        self.history.append(price)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


class BinanceStreamer:
    """Stream real-time prices from Binance for multiple assets."""

    def __init__(self, assets: List[str] = None):
        """
        Initialize streamer.

        Args:
            assets: List of assets to track (e.g., ["BTC", "ETH", "SOL"])
        """
        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.states: Dict[str, PriceState] = {}
        self.running = False
        self.callbacks: List[Callable] = []

        for asset in self.assets:
            self.states[asset] = PriceState(asset=asset)

    def on_price(self, callback: Callable):
        """Register a callback for price updates."""
        self.callbacks.append(callback)

    def get_price(self, asset: str) -> float:
        """Get current price for an asset."""
        state = self.states.get(asset)
        return state.price if state else 0.0

    def get_history(self, asset: str, n: int = 100) -> List[float]:
        """Get price history for an asset."""
        state = self.states.get(asset)
        return state.history[-n:] if state else []

    async def stream(self):
        """Start streaming prices."""
        self.running = True

        # Build stream URL
        symbols = [SYMBOLS[a] for a in self.assets if a in SYMBOLS]
        streams = "/".join([f"{s}@trade" for s in symbols])
        url = f"{BINANCE_WSS}/stream?streams={streams}"

        print(f"Connecting to Binance WSS for {', '.join(self.assets)}...")

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    print("[WSS] Connected to Binance")

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(msg)

                            if "data" in data:
                                trade = data["data"]
                                symbol = trade["s"].upper()
                                price = float(trade["p"])

                                # Map to asset (update ALL matching assets, don't break)
                                for asset, sym in SYMBOLS.items():
                                    if sym.upper() == symbol:
                                        state = self.states.get(asset)
                                        if state:
                                            state.update(price)

                                            # Call callbacks
                                            for cb in self.callbacks:
                                                try:
                                                    cb(asset, price)
                                                except:
                                                    pass

                        except asyncio.TimeoutError:
                            pass
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                print(f"WSS error: {e}, reconnecting...")
                await asyncio.sleep(1)

    def stop(self):
        """Stop streaming."""
        self.running = False


async def get_current_prices(assets: List[str] = None) -> Dict[str, float]:
    """
    Get current prices for assets (one-shot, not streaming).

    Args:
        assets: List of assets (default: BTC, ETH, SOL)

    Returns:
        Dict mapping asset to price
    """
    if assets is None:
        assets = ["BTC", "ETH", "SOL"]

    prices = {}

    for asset in assets:
        symbol = SYMBOLS.get(asset)
        if not symbol:
            continue

        try:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                prices[asset] = float(data["price"])
        except:
            pass

    return prices


class BinanceKlineStreamer:
    """Stream real-time kline (candlestick) data from Binance WebSocket.

    Subscribes to 1m and 1s kline streams and calls a callback when each
    kline closes (is_final=True). This feeds IncrementalTF instances in
    real-time so prediction at the 5m boundary requires zero REST API calls.
    """

    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.running = False
        self._on_kline_1m = []   # callbacks: (ts, o, h, l, c, v)
        self._on_kline_1s = []
        self._warmup_done = False

    def on_kline_1m(self, callback: Callable):
        """Register callback for completed 1m klines."""
        self._on_kline_1m.append(callback)

    def on_kline_1s(self, callback: Callable):
        """Register callback for completed 1s klines."""
        self._on_kline_1s.append(callback)

    @property
    def is_warm(self):
        return self._warmup_done

    async def warmup_from_rest(self, n_1m: int = 800):
        """Fetch historical 1m + 1s klines via REST to seed IncrementalTF state.

        Called once at startup so the TFs have enough history before the first
        5m boundary. After this, WebSocket streams maintain the state.
        """
        import requests

        now = int(time.time())
        boundary = (now // 300) * 300

        # Fetch 1m klines
        all_1m = []
        end_ms = boundary * 1000
        start_ms = end_ms - n_1m * 60 * 1000
        t = start_ms
        while t < end_ms:
            try:
                resp = requests.get(
                    'https://api.binance.com/api/v3/klines',
                    params={'symbol': self.symbol.upper(), 'interval': '1m',
                            'startTime': t, 'limit': 1000},
                    timeout=10,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_1m.extend(batch)
                t = batch[-1][0] + 60000
            except Exception as e:
                print(f"[KlineStream] Warmup 1m fetch error: {e}")
                break

        # Feed historical 1m klines
        for k in all_1m:
            ts = int(k[0]) // 1000
            if ts >= boundary:
                continue
            row = (ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            for cb in self._on_kline_1m:
                try:
                    cb(*row)
                except Exception:
                    pass

        # Fetch 1s klines (last 1000 = ~16.7 min)
        try:
            resp = requests.get(
                'https://api.binance.com/api/v3/klines',
                params={'symbol': self.symbol.upper(), 'interval': '1s', 'limit': 1000},
                timeout=10,
            )
            resp.raise_for_status()
            for k in resp.json():
                ts = int(k[0]) // 1000
                if ts >= boundary:
                    continue
                row = (ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                for cb in self._on_kline_1s:
                    try:
                        cb(*row)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[KlineStream] Warmup 1s fetch error (non-fatal): {e}")

        self._warmup_done = True
        print(f"[KlineStream] Warmup complete: {len(all_1m)} 1m klines fed")

    async def stream(self):
        """Start streaming kline data via WebSocket."""
        self.running = True

        # Subscribe to both 1m and 1s klines
        streams = f"{self.symbol}@kline_1m/{self.symbol}@kline_1s"
        url = f"{BINANCE_WSS}/stream?streams={streams}"

        print(f"[KlineStream] Connecting to {self.symbol} kline streams (1m + 1s)...")

        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    print("[KlineStream] Connected")

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                            data = json.loads(msg)

                            if "data" not in data:
                                continue

                            kline = data["data"].get("k")
                            if not kline:
                                continue

                            # Only process closed (final) klines
                            if not kline.get("x", False):
                                continue

                            ts = int(kline["t"]) // 1000
                            o = float(kline["o"])
                            h = float(kline["h"])
                            l = float(kline["l"])
                            c = float(kline["c"])
                            v = float(kline["v"])
                            interval = kline.get("i", "")

                            if interval == "1m":
                                for cb in self._on_kline_1m:
                                    try:
                                        cb(ts, o, h, l, c, v)
                                    except Exception:
                                        pass
                            elif interval == "1s":
                                for cb in self._on_kline_1s:
                                    try:
                                        cb(ts, o, h, l, c, v)
                                    except Exception:
                                        pass

                        except asyncio.TimeoutError:
                            pass
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                if self.running:
                    print(f"[KlineStream] Error: {e}, reconnecting in 2s...")
                    await asyncio.sleep(2)

    def stop(self):
        self.running = False


if __name__ == "__main__":
    import asyncio

    async def test():
        prices = await get_current_prices(["BTC", "ETH", "SOL", "XRP"])
        print("Current prices:")
        for asset, price in prices.items():
            print(f"  {asset}: ${price:,.2f}")

    asyncio.run(test())
