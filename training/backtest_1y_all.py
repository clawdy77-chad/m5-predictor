#!/usr/bin/env python3
"""
1-year backtest for ALL assets: BTC, ETH, XRP, SOL.
"""
import sys, os, json, time, pickle
import numpy as np
from datetime import datetime, timezone, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))

from v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs
from training.train_final_model import fetch_binance_klines

import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(_ROOT, 'model')

WARMUP_DAYS = 7
TEST_DAYS = 392
TOTAL_DAYS = WARMUP_DAYS + TEST_DAYS

BET_SIZE = 5.0
LIMIT_PRICE = 0.51
START_BALANCE = 100.0

ASSETS = {
    'BTCUSDT': 'btc_54_model.pkl',
    'ETHUSDT': 'eth_model.pkl',
    'XRPUSDT': 'xrp_model.pkl',
    'SOLUSDT': 'sol_model.pkl',
}


def load_model(model_file):
    path = os.path.join(MODEL_DIR, model_file)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['model'], d['feature_names']


def run_pipeline(klines, model, feature_names, test_start_ts):
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
            if pending is not None:
                actual_dir = 'UP' if tfs['m5'].closes[-1] > tfs['m5'].opens[-1] else 'DOWN'
                won = pending['direction'] == actual_dir

                if pending['m5_ts'] >= test_start_ts:
                    trades.append({
                        'ts': datetime.fromtimestamp(pending['m5_ts'] + 300, tz=timezone.utc).isoformat(),
                        'direction': pending['direction'],
                        'confidence': round(pending['confidence'], 4),
                        'actual': actual_dir,
                        'result': 'WIN' if won else 'LOSS',
                    })
                pending = None

            features = extract_features(tfs, target_tf='m5')
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(X)[0]
            conf = float(proba[1])
            direction = 'UP' if conf >= 0.5 else 'DOWN'
            disp_conf = conf if direction == 'UP' else 1.0 - conf

            m5_ts = tfs['m5'].timestamps[-1]
            pending = {
                'direction': direction,
                'confidence': disp_conf,
                'm5_ts': m5_ts,
            }
            prev_m5 = m5_count

        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

    return trades


def main():
    t0 = time.time()
    all_results = {}

    for symbol, model_file in ASSETS.items():
        name = symbol.replace('USDT', '')
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        model, feature_names = load_model(model_file)
        print(f"  Fetching {TOTAL_DAYS} days of {symbol} 1m klines...")
        klines = fetch_binance_klines(symbol, days=TOTAL_DAYS)
        print(f"  Klines: {len(klines)}")

        last_ts = klines[-1][0]
        test_start_ts = last_ts - TEST_DAYS * 86400

        print(f"  Running pipeline...")
        trades = run_pipeline(klines, model, feature_names, test_start_ts)
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        losses = len(trades) - wins
        print(f"  Trades: {len(trades)} | WR: {wins/len(trades)*100:.2f}% | {wins}W/{losses}L")

        all_results[name] = trades

        # Monthly breakdown
        months = {}
        for t in trades:
            m = t['ts'][:7]
            months.setdefault(m, []).append(t)

        print(f"\n  {'Month':<10} {'Trades':>7} {'WR':>7}")
        print(f"  {'-'*28}")
        for m in sorted(months):
            d = months[m]
            w = sum(1 for t in d if t['result'] == 'WIN')
            print(f"  {m:<10} {len(d):>7} {w/len(d)*100:>6.1f}%")

    # Combined summary
    print(f"\n{'='*60}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*60}")

    bet = BET_SIZE
    win_payout = bet / LIMIT_PRICE - bet

    print(f"\n  {'Asset':<6} {'Trades':>8} {'WR':>7} {'$50 PnL':>12} {'$5 from $100':>14} {'Min Bal':>9}")
    print(f"  {'-'*58}")

    grand_trades = []
    for name in ['BTC', 'ETH', 'XRP', 'SOL']:
        trades = all_results[name]
        grand_trades.extend(trades)
        w = sum(1 for t in trades if t['result'] == 'WIN')
        l = len(trades) - w
        pnl_50 = w * (50/0.51 - 50) - l * 50

        # Simulate $100 at $5
        bal = 100.0
        min_bal = 100.0
        busted = False
        for t in trades:
            if bal < bet:
                busted = True
                break
            if t['result'] == 'WIN':
                bal += win_payout
            else:
                bal -= bet
            if bal < min_bal:
                min_bal = bal

        bal_str = 'BUST' if busted else f'${bal:,.0f}'
        print(f"  {name:<6} {len(trades):>8} {w/len(trades)*100:>6.1f}% ${pnl_50:>+10,.0f}  {bal_str:>13} ${min_bal:>7.2f}")

    # Grand total
    w = sum(1 for t in grand_trades if t['result'] == 'WIN')
    l = len(grand_trades) - w
    pnl_50 = w * (50/0.51 - 50) - l * 50
    print(f"  {'ALL':<6} {len(grand_trades):>8} {w/len(grand_trades)*100:>6.1f}% ${pnl_50:>+10,.0f}")

    # Monthly combined WR
    print(f"\n  Monthly Combined WR:")
    months = {}
    for t in grand_trades:
        m = t['ts'][:7]
        months.setdefault(m, {'w': 0, 'l': 0})
        months[m]['w' if t['result'] == 'WIN' else 'l'] += 1
    
    print(f"  {'Month':<10} {'Trades':>7} {'WR':>7}")
    print(f"  {'-'*28}")
    for m in sorted(months):
        d = months[m]
        total = d['w'] + d['l']
        print(f"  {m:<10} {total:>7} {d['w']/total*100:>6.1f}%")

    # Save
    out = os.path.join(MODEL_DIR, 'backtest_1y_all.json')
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"\nSaved to {out}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
