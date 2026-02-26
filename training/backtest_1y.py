#!/usr/bin/env python3
"""
1-year BTC backtest: generate trades, then simulate $100 start at $5 bets.
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
TEST_DAYS = 392  # 365 + 27 to cover full Feb 2025
TOTAL_DAYS = WARMUP_DAYS + TEST_DAYS

BET_SIZE_SIM = 5.0
LIMIT_PRICE = 0.51
START_BALANCE = 100.0


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


def simulate(trades, bet, start_bal):
    bal = start_bal
    min_bal = start_bal
    max_bal = start_bal
    busted = False
    win_payout = bet / LIMIT_PRICE - bet

    daily = {}
    for t in trades:
        day = t['ts'][:10]
        if bal < bet:
            busted = True
            bust_day = day
            break
        if t['result'] == 'WIN':
            bal += win_payout
        else:
            bal -= bet
        if bal < min_bal:
            min_bal = bal
        if bal > max_bal:
            max_bal = bal
        daily.setdefault(day, {'start': bal, 'w': 0, 'l': 0})
        if t['result'] == 'WIN':
            daily[day]['w'] += 1
        else:
            daily[day]['l'] += 1
        daily[day]['end'] = bal

    return bal, min_bal, max_bal, busted, daily


def main():
    t0 = time.time()
    print(f"Fetching {TOTAL_DAYS} days of BTCUSDT 1m klines...")
    model, feature_names = load_model('btc_54_model.pkl')

    klines = fetch_binance_klines('BTCUSDT', days=TOTAL_DAYS)
    print(f"Klines: {len(klines)} ({TOTAL_DAYS} days)")

    last_ts = klines[-1][0]
    test_start_ts = last_ts - TEST_DAYS * 86400
    print(f"Test period: {datetime.fromtimestamp(test_start_ts, tz=timezone.utc).date()} to {datetime.fromtimestamp(last_ts, tz=timezone.utc).date()}")

    print("Running pipeline...")
    trades = run_pipeline(klines, model, feature_names, test_start_ts)
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    print(f"\nTrades: {len(trades)} | WR: {wins/len(trades)*100:.2f}% | {wins}W/{losses}L")

    # Monthly breakdown
    months = {}
    for t in trades:
        m = t['ts'][:7]
        months.setdefault(m, {'w': 0, 'l': 0})
        months[m]['w' if t['result'] == 'WIN' else 'l'] += 1

    print(f"\n{'Month':<10} {'Trades':>7} {'WR':>7} {'W':>5} {'L':>5}")
    print("-" * 40)
    for m in sorted(months):
        d = months[m]
        total = d['w'] + d['l']
        print(f"{m:<10} {total:>7} {d['w']/total*100:>6.1f}% {d['w']:>5} {d['l']:>5}")

    # Simulate $100 at $5 bets
    print(f"\n{'='*50}")
    print(f"SIMULATION: ${START_BALANCE} start, BTC only")
    print(f"{'='*50}")
    for bet in [2, 3, 5]:
        bal, min_bal, max_bal, busted, daily = simulate(trades, bet, START_BALANCE)
        if busted:
            print(f"\n${bet} bets: BUSTED (min ${min_bal:.2f})")
        else:
            print(f"\n${bet} bets: ${bal:,.0f} (min ${min_bal:.2f}, max ${max_bal:,.0f})")
            # Find worst streak
            days_sorted = sorted(daily.keys())
            worst_day_pnl = float('inf')
            for day in days_sorted:
                d = daily[day]
                day_pnl = d['w'] * (bet / LIMIT_PRICE - bet) - d['l'] * bet
                if day_pnl < worst_day_pnl:
                    worst_day_pnl = day_pnl
                    worst_day = day
            print(f"  Worst day: {worst_day} (${worst_day_pnl:+,.0f})")

    # Save trades for analysis
    out = os.path.join(MODEL_DIR, 'backtest_1y_btc.json')
    with open(out, 'w') as f:
        json.dump(trades, f)
    print(f"\nSaved {len(trades)} trades to {out}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
