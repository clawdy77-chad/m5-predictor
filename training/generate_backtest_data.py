#!/usr/bin/env python3
"""
Generate full trade-by-trade backtest data for all 4 assets over 30 days.
Saves to model/backtest_trades.json for the dashboard.
"""
import sys, os, json, time, pickle
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))

from v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs
from training.train_final_model import fetch_binance_klines

import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(_ROOT, 'model')

ASSETS = {
    'BTCUSDT': 'btc_54_model.pkl',
    'ETHUSDT': 'eth_model.pkl',
    'XRPUSDT': 'xrp_model.pkl',
    'SOLUSDT': 'sol_model.pkl',
}

WARMUP_DAYS = 7
TEST_DAYS = 30
TOTAL_DAYS = WARMUP_DAYS + TEST_DAYS

BET_SIZE = 50.0
LIMIT_PRICE = 0.51


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
    equity = 1000.0

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
                pnl = BET_SIZE * ((1.0 - LIMIT_PRICE) / LIMIT_PRICE) if won else -BET_SIZE

                if pending['m5_ts'] >= test_start_ts:
                    equity += pnl
                    trades.append({
                        'ts': datetime.fromtimestamp(pending['m5_ts'] + 300, tz=timezone.utc).isoformat(),
                        'direction': pending['direction'],
                        'confidence': round(pending['confidence'], 4),
                        'actual': actual_dir,
                        'result': 'WIN' if won else 'LOSS',
                        'pnl': round(pnl, 2),
                        'equity': round(equity, 2),
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
    t_start = time.time()
    all_data = {}

    for symbol, model_file in ASSETS.items():
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        try:
            model, feature_names = load_model(model_file)
            print(f"  Model: {model_file}")

            klines = fetch_binance_klines(symbol, days=TOTAL_DAYS)
            print(f"  Klines: {len(klines)}")

            last_ts = klines[-1][0]
            test_start_ts = last_ts - TEST_DAYS * 86400
            print(f"  Test start: {datetime.fromtimestamp(test_start_ts, tz=timezone.utc).date()}")

            trades = run_pipeline(klines, model, feature_names, test_start_ts)
            wins = sum(1 for t in trades if t['result'] == 'WIN')
            print(f"  Trades: {len(trades)} | WR: {wins/len(trades)*100:.1f}% | PnL: ${sum(t['pnl'] for t in trades):+,.2f}")

            all_data[symbol] = trades
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    out_path = os.path.join(MODEL_DIR, 'backtest_trades.json')
    with open(out_path, 'w') as f:
        json.dump(all_data, f)
    print(f"\nSaved {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
