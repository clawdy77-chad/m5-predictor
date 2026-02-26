#!/usr/bin/env python3
"""
Verify paper trader results by replaying its exact logged klines.

Reads the kline JSONL log produced by paper_trader.py, runs the identical
v24 engine + model pipeline, and compares predictions trade-by-trade.
Results must be 100% identical since both paths process the same data.

Usage:
    python bot/verify_trades.py <kline_log.jsonl> [trades.json]

    If trades.json is omitted, fetches from the live paper trader API.
"""
import sys, os, json, pickle
import numpy as np
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))

from v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_klines(path):
    klines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                klines.append(json.loads(line))
    return klines


def load_paper_trades(path_or_url):
    if path_or_url.startswith('http'):
        import requests
        return requests.get(path_or_url).json()
    with open(path_or_url) as f:
        return json.load(f)


def replay(klines, model_path):
    """Replay klines through the v24 engine and model.
    
    Returns list of dicts with m5_ts (boundary timestamp), direction, confidence,
    and resolves each trade when the next M5 completes — identical to paper_trader.py.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    feature_names = model_data['feature_names']

    tfs = {}
    for key, sec in TF_SECS.items():
        tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

    trades = []
    pending = None
    prev_m5 = 0

    for ts, o, h, l, c, v in klines:
        for key in ['m1', 'm3', 'm5']:
            tfs[key].feed_1m(ts, o, h, l, c, v)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
            tfs['10s'].feed_sub(*sub)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
            tfs['30s'].feed_sub(*sub)

        m5_count = tfs['m5'].n
        if m5_count > prev_m5 and m5_count > 50:
            # Resolve previous
            if pending is not None:
                actual_dir = 'UP' if tfs['m5'].closes[-1] > tfs['m5'].opens[-1] else 'DOWN'
                won = pending['direction'] == actual_dir
                pending['actual'] = actual_dir
                pending['result'] = 'WIN' if won else 'LOSS'
                trades.append(pending)
                pending = None

            # New prediction
            features = extract_features(tfs, target_tf='m5')
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(X)[0]
            conf = float(proba[1])
            direction = 'UP' if conf >= 0.5 else 'DOWN'
            disp_conf = conf if direction == 'UP' else 1.0 - conf

            m5_ts = tfs['m5'].timestamps[-1]
            pending = {
                'm5_ts': datetime.fromtimestamp(m5_ts, tz=timezone.utc).isoformat(),
                'direction': direction,
                'confidence': disp_conf,
            }
            prev_m5 = m5_count

        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

    return trades


def compare(paper_trades, replay_trades):
    """Compare by aligning replay M5 boundary timestamps to paper trade times.
    
    Paper trader logs wall-clock time (~1min after M5 boundary).
    Replay uses exact M5 boundary timestamp.
    We match by finding the replay trade whose m5_ts is within 5min before
    the paper trade timestamp.
    """
    from datetime import timedelta

    print(f"\nAligning {len(paper_trades)} paper trades to {len(replay_trades)} replay trades...")

    # Build lookup: m5_ts -> replay trade
    replay_by_ts = {}
    for rt in replay_trades:
        replay_by_ts[rt['m5_ts']] = rt

    # For each paper trade, find matching replay trade
    # Paper trade ts = M5 candle CLOSE time (= m5_ts + 5min)
    # Replay m5_ts = M5 candle OPEN time
    # So: m5_ts = paper_ts - 5min
    matched = []
    for pt in paper_trades:
        pt_dt = datetime.fromisoformat(pt['ts'])
        m5_prediction_open = pt_dt - timedelta(minutes=5)
        m5_key = m5_prediction_open.isoformat()

        rt = replay_by_ts.get(m5_key)
        if rt is None:
            # Try nearby offsets
            for offset in [-1, 1, -2, 2]:
                alt = m5_prediction_open + timedelta(minutes=offset * 5)
                rt = replay_by_ts.get(alt.isoformat())
                if rt:
                    break

        matched.append((pt, rt))

    # Print comparison
    print(f"\n{'='*95}")
    print(f"{'#':>3} | {'Paper Dir':>9} {'Conf':>8} {'Result':>6} | {'Replay Dir':>10} {'Conf':>8} {'Result':>6} | {'Match':>5}")
    print(f"{'='*95}")

    total = 0
    full_matches = 0
    dir_matches = 0

    for i, (pt, rt) in enumerate(matched):
        total += 1
        if rt is None:
            print(f"{i+1:>3} | {pt['direction']:>9} {pt['confidence']:>8.4f} {pt['result']:>6} | {'???':>10} {'???':>8} {'???':>6} | {'MISS':>5}")
            continue

        dir_match = pt['direction'] == rt['direction']
        res_match = pt['result'] == rt['result']
        conf_match = abs(pt['confidence'] - rt['confidence']) < 0.0001
        full = dir_match and res_match and conf_match

        if dir_match:
            dir_matches += 1
        if full:
            full_matches += 1

        flag = 'OK' if full else ('DIR' if dir_match else 'FAIL')

        print(f"{i+1:>3} | {pt['direction']:>9} {pt['confidence']:>8.4f} {pt['result']:>6} | "
              f"{rt['direction']:>10} {rt['confidence']:>8.4f} {rt['result']:>6} | {flag:>5}")

    print(f"{'='*95}")
    print(f"\nResults:")
    print(f"  Trades compared: {total}")
    print(f"  Direction matches: {dir_matches}/{total} ({dir_matches/total*100:.1f}%)")
    print(f"  Full matches: {full_matches}/{total} ({full_matches/total*100:.1f}%)")

    paper_wins = sum(1 for t in paper_trades if t['result'] == 'WIN')
    replay_wins = sum(1 for _, rt in matched if rt and rt['result'] == 'WIN')
    print(f"  Paper WR:  {paper_wins/len(paper_trades)*100:.1f}% ({paper_wins}/{len(paper_trades)})")
    print(f"  Replay WR: {replay_wins}/{total}")

    if full_matches == total:
        print(f"\n  ✓ PERFECT MATCH — 100% identical")
    else:
        print(f"\n  ✗ {total - full_matches} mismatches found")

    return full_matches == total


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python bot/verify_trades.py <kline_log.jsonl> [trades.json]")
        sys.exit(1)

    kline_path = sys.argv[1]
    
    # Auto-detect model from kline log filename
    MODEL_MAP = {
        'btcusdt': 'btc_54_model.pkl',
        'ethusdt': 'eth_model.pkl',
        'xrpusdt': 'xrp_model.pkl',
        'solusdt': 'sol_model.pkl',
    }
    fname = os.path.basename(kline_path).lower()
    model_file = 'btc_54_model.pkl'  # default
    for sym, mf in MODEL_MAP.items():
        if sym in fname:
            model_file = mf
            break
    model_path = os.path.join(_ROOT, 'model', model_file)
    print(f"Auto-detected model: {model_file}")

    print(f"Loading klines from: {kline_path}")
    klines = load_klines(kline_path)
    print(f"  {len(klines)} klines")

    if len(sys.argv) >= 3:
        trades_source = sys.argv[2]
    else:
        trades_source = 'http://localhost:8099/api/trades'
    print(f"Loading paper trades from: {trades_source}")
    paper_trades = load_paper_trades(trades_source)
    print(f"  {len(paper_trades)} trades")

    print(f"\nReplaying {len(klines)} klines through v24 engine...")
    replay_trades = replay(klines, model_path)
    print(f"  {len(replay_trades)} trades from replay")

    compare(paper_trades, replay_trades)
