"""
Polymarket API helpers for 15-min up/down markets.
Finds BTC, ETH, SOL, XRP markets using slug pattern.
"""
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# 15-min assets (slug pattern: {asset}-updown-15m-{timestamp})
ASSETS_15M = ["btc", "eth", "sol", "xrp"]


@dataclass
class Market:
    """Active 15-min prediction market."""
    condition_id: str
    question: str
    asset: str
    end_time: datetime
    token_up: str
    token_down: str
    price_up: float = 0.5
    price_down: float = 0.5
    slug: str = ""


def get_market_from_clob(condition_id: str) -> Optional[Dict]:
    """Get market details from CLOB API including token IDs."""
    url = f"{CLOB_API}/markets/{condition_id}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return None
    return resp.json()


def get_15m_markets(assets: List[str] = None) -> List[Market]:
    """
    Get currently active 15-min up/down markets.

    Uses slug pattern: {asset}-updown-15m-{timestamp}

    Args:
        assets: List of assets (default: btc, eth, sol, xrp)

    Returns:
        List of active Market objects sorted by end time.
    """
    if assets is None:
        assets = ASSETS_15M
    else:
        assets = [a.lower() for a in assets]

    markets = []
    now = datetime.now(timezone.utc)
    current_ts = int(now.timestamp())

    # Round to 15-min boundary (900 seconds)
    window_start = (current_ts // 900) * 900

    # Check current and next 3 windows
    timestamps = [window_start + (i * 900) for i in range(4)]

    for asset in assets:
        for ts in timestamps:
            slug = f"{asset}-updown-15m-{ts}"

            try:
                url = f"{GAMMA_API}/events?slug={slug}"
                resp = requests.get(url, timeout=5)

                if resp.status_code != 200:
                    continue

                events = resp.json()
                if not events:
                    continue

                e = events[0]
                end_str = e.get("endDate", "")

                if not end_str:
                    continue

                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

                if end_dt <= now:
                    continue

                # Get condition ID
                condition_id = None
                for m in e.get("markets", []):
                    condition_id = m.get("conditionId")
                    if condition_id:
                        break

                if not condition_id:
                    continue

                # Get token IDs from CLOB
                clob_data = get_market_from_clob(condition_id)
                if not clob_data:
                    continue

                if not clob_data.get("active") or clob_data.get("closed"):
                    continue

                tokens = clob_data.get("tokens", [])
                token_up = None
                token_down = None
                price_up = 0.5
                price_down = 0.5

                for t in tokens:
                    outcome = t.get("outcome", "").lower()
                    if outcome == "up":
                        token_up = t.get("token_id")
                        price_up = t.get("price", 0.5)
                    elif outcome == "down":
                        token_down = t.get("token_id")
                        price_down = t.get("price", 0.5)

                if not token_up or not token_down:
                    continue

                market = Market(
                    condition_id=condition_id,
                    question=clob_data.get("question", ""),
                    asset=asset.upper(),
                    end_time=end_dt,
                    token_up=token_up,
                    token_down=token_down,
                    price_up=price_up,
                    price_down=price_down,
                    slug=slug,
                )
                markets.append(market)
                break  # Got next market for this asset

            except Exception:
                continue

    # Sort by end time
    markets.sort(key=lambda m: m.end_time)
    return markets


def get_5m_markets(assets: List[str] = None) -> List[Market]:
    """Get currently active 5-min up/down markets (same logic as 15m, different slug/window)."""
    if assets is None:
        assets = ["btc"]
    else:
        assets = [a.lower() for a in assets]

    markets = []
    now = datetime.now(timezone.utc)
    current_ts = int(now.timestamp())

    # Round to 5-min boundary (300 seconds)
    window_start = (current_ts // 300) * 300

    # Check current and next 3 windows
    timestamps = [window_start + (i * 300) for i in range(4)]

    for asset in assets:
        for ts in timestamps:
            slug = f"{asset}-updown-5m-{ts}"
            try:
                url = f"{GAMMA_API}/events?slug={slug}"
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
                events = resp.json()
                if not events:
                    continue
                e = events[0]
                end_str = e.get("endDate", "")
                if not end_str:
                    continue
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                if end_dt <= now:
                    continue
                condition_id = None
                for m in e.get("markets", []):
                    condition_id = m.get("conditionId")
                    if condition_id:
                        break
                if not condition_id:
                    continue
                clob_data = get_market_from_clob(condition_id)
                if not clob_data:
                    continue
                if not clob_data.get("active") or clob_data.get("closed"):
                    continue
                tokens = clob_data.get("tokens", [])
                token_up = None
                token_down = None
                price_up = 0.5
                price_down = 0.5
                for t in tokens:
                    outcome = t.get("outcome", "").lower()
                    if outcome == "up":
                        token_up = t.get("token_id")
                        price_up = t.get("price", 0.5)
                    elif outcome == "down":
                        token_down = t.get("token_id")
                        price_down = t.get("price", 0.5)
                if not token_up or not token_down:
                    continue
                market = Market(
                    condition_id=condition_id,
                    question=clob_data.get("question", ""),
                    asset=asset.upper() + "_5M",
                    end_time=end_dt,
                    token_up=token_up,
                    token_down=token_down,
                    price_up=price_up,
                    price_down=price_down,
                    slug=slug,
                )
                markets.append(market)
                break
            except Exception:
                continue

    markets.sort(key=lambda m: m.end_time)
    return markets


def get_next_market(asset: str) -> Optional[Market]:
    """Get the next closing 15-min market for a specific asset."""
    markets = get_15m_markets(assets=[asset])
    return markets[0] if markets else None


def get_5m_market_by_slot(asset: str, slot_ts: int) -> Optional[Market]:
    """Fetch a specific 5m market by slot timestamp using Gamma slug API.
    Much faster than scanning multiple event endpoints.
    """
    import json as _json
    slug = f"{asset}-updown-5m-{slot_ts}"
    url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            return None
        data = resp.json()
        raw_ids = data.get("clobTokenIds")
        if raw_ids is None:
            return None
        ids = raw_ids if isinstance(raw_ids, list) else _json.loads(raw_ids)
        if len(ids) < 2:
            return None
        end_str = data.get("endDate", "")
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else None
        return Market(
            condition_id=data.get("conditionId", ""),
            question=data.get("question", slug),
            asset=asset.upper(),
            end_time=end_dt or datetime.now(timezone.utc),
            token_up=ids[0],
            token_down=ids[1],
            slug=slug,
        )
    except Exception:
        return None


def get_5m_current_and_next(asset: str = "btc") -> List[Market]:
    """Get current and next 5m markets using direct Gamma slug lookup.
    Returns [current, next] — up to 2 markets. Fast (2 HTTP calls).
    """
    import time as _time
    now = int(_time.time())
    current_slot = (now // 300) * 300
    next_slot = current_slot + 300

    markets = []
    for slot in [current_slot, next_slot]:
        m = get_5m_market_by_slot(asset, slot)
        if m:
            markets.append(m)
    return markets


def get_15m_market_by_slot(asset: str, slot_ts: int) -> Optional[Market]:
    """Fetch a specific 15m market by slot timestamp using Gamma slug API.
    Much faster than scanning multiple event endpoints.
    """
    import json as _json
    slug = f"{asset}-updown-15m-{slot_ts}"
    url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            return None
        data = resp.json()
        raw_ids = data.get("clobTokenIds")
        if raw_ids is None:
            return None
        ids = raw_ids if isinstance(raw_ids, list) else _json.loads(raw_ids)
        if len(ids) < 2:
            return None
        end_str = data.get("endDate", "")
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else None
        return Market(
            condition_id=data.get("conditionId", ""),
            question=data.get("question", slug),
            asset=asset.upper(),
            end_time=end_dt or datetime.now(timezone.utc),
            token_up=ids[0],
            token_down=ids[1],
            slug=slug,
        )
    except Exception:
        return None


def get_15m_current_and_next(asset: str = "btc") -> List[Market]:
    """Get current and next 15m markets using direct Gamma slug lookup.
    Returns [current, next] — up to 2 markets. Fast (2 HTTP calls).
    """
    import time as _time
    now = int(_time.time())
    current_slot = (now // 900) * 900
    next_slot = current_slot + 900

    markets = []
    for slot in [current_slot, next_slot]:
        m = get_15m_market_by_slot(asset, slot)
        if m:
            markets.append(m)
    return markets


def resolve_via_gamma(slug: str) -> Optional[str]:
    """Resolve a market outcome using the Gamma API.

    Checks if the market identified by slug is closed, then reads outcomePrices
    to determine the winner. Index 0 = UP, index 1 = DOWN.

    Returns 'UP' or 'DOWN' if resolved, None if still open or error.
    """
    import json as _json
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets/slug/{slug}",
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("closed"):
            return None
        raw_prices = data.get("outcomePrices")
        if not raw_prices:
            return None
        prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        if len(prices) < 2:
            return None
        return "UP" if float(prices[0]) > 0.5 else "DOWN"
    except Exception:
        return None


# Backwards compat
get_active_markets = get_15m_markets


if __name__ == "__main__":
    print("=" * 60)
    print("15-MIN UP/DOWN MARKETS")
    print("=" * 60)

    markets = get_15m_markets()
    now = datetime.now(timezone.utc)

    if not markets:
        print("\nNo active 15-min markets found!")
    else:
        for m in markets:
            mins_left = (m.end_time - now).total_seconds() / 60
            print(f"\n{m.asset} 15m")
            print(f"  {m.question}")
            print(f"  Closes in: {mins_left:.1f} min")
            print(f"  UP: {m.price_up:.3f} | DOWN: {m.price_down:.3f}")
            print(f"  Condition: {m.condition_id}")
            print(f"  Token UP: {m.token_up}")
            print(f"  Token DOWN: {m.token_down}")
