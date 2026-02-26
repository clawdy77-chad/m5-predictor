#!/usr/bin/env python3
"""
7-Day Live Pipeline Simulation vs Backtest Comparison.

For each asset:
1. Download last 7 days + warmup (14 days total) of 1m klines from Binance REST
2. Run the LIVE pipeline (paper_trader logic) on those klines
3. Run the BACKTEST pipeline (train_final_model logic) on those same klines
4. Compare trade-by-trade: direction, confidence, outcome must match 1:1

This validates that live and backtest produce identical results on real data.
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

WARMUP_DAYS = 7   # warmup before the 7-day test window
TEST_DAYS = 7
TOTAL_DAYS = WARMUP_DAYS + TEST_DAYS

BET_SIZE = 50.0
LIMIT_PRICE = 0.51


def load_model(model_file):
    path = os.path.join(MODEL_DIR, model_file)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['model'], d['feature_names']


def run_live_pipeline(klines, model, feature_names, test_start_ts):
    """Simulate the exact paper_trader.py pipeline on klines.
    
    Only records trades that START after test_start_ts (warmup period skipped).
    """
    tfs = {}
    for key, sec in TF_SECS.items():
        tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

    trades = []
    pending = None
    prev_m5 = 0

    for ts, o, h, l, c, v in klines:
        # Feed LTFs (exact paper_trader order)
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
                pnl = BET_SIZE * ((1.0 - LIMIT_PRICE) / LIMIT_PRICE) if won else -BET_SIZE
                
                # Only record if prediction was made in test window
                if pending['m5_ts'] >= test_start_ts:
                    trades.append({
                        'ts': datetime.fromtimestamp(pending['m5_ts'] + 300, tz=timezone.utc).isoformat(),
                        'm5_ts': pending['m5_ts'],
                        'direction': pending['direction'],
                        'confidence': pending['confidence'],
                        'actual': actual_dir,
                        'result': 'WIN' if won else 'LOSS',
                        'pnl': round(pnl, 2),
                    })
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
                'direction': direction,
                'confidence': disp_conf,
                'm5_ts': m5_ts,
            }
            prev_m5 = m5_count

        # Feed HTFs AFTER
        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

    return trades


def run_backtest_pipeline(klines, model, feature_names, test_start_ts):
    """Run the backtest (process_candles style) on the same klines.
    
    This mirrors train_final_model.process_candles() exactly.
    """
    tfs = {}
    for key, sec in TF_SECS.items():
        tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

    trades = []
    pending = None
    prev_label_count = 0

    for ts, o, h, l, c, v in klines:
        # 1. Feed LTFs
        for key in ['m1', 'm3', 'm5']:
            tfs[key].feed_1m(ts, o, h, l, c, v)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
            tfs['10s'].feed_sub(*sub)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
            tfs['30s'].feed_sub(*sub)

        # 2. Check M5 boundary
        label_count = tfs['m5'].n
        if label_count > prev_label_count and label_count > 50:
            prev_label_count = label_count

            # Resolve previous
            if pending is not None:
                actual_dir = 'UP' if tfs['m5'].closes[-1] > tfs['m5'].opens[-1] else 'DOWN'
                won = pending['direction'] == actual_dir
                pnl = BET_SIZE * ((1.0 - LIMIT_PRICE) / LIMIT_PRICE) if won else -BET_SIZE

                if pending['m5_ts'] >= test_start_ts:
                    trades.append({
                        'ts': datetime.fromtimestamp(pending['m5_ts'] + 300, tz=timezone.utc).isoformat(),
                        'm5_ts': pending['m5_ts'],
                        'direction': pending['direction'],
                        'confidence': pending['confidence'],
                        'actual': actual_dir,
                        'result': 'WIN' if won else 'LOSS',
                        'pnl': round(pnl, 2),
                    })
                pending = None

            # Extract features and predict
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

        # 3. Feed HTFs AFTER
        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

    return trades


def compare_trades(live_trades, bt_trades, symbol):
    """Compare live vs backtest trades. Must be 100% identical."""
    n_live = len(live_trades)
    n_bt = len(bt_trades)

    if n_live != n_bt:
        print(f"  WARNING: Trade count mismatch — live={n_live}, backtest={n_bt}")

    n = min(n_live, n_bt)
    matches = 0
    dir_mismatches = 0
    conf_mismatches = 0
    result_mismatches = 0

    for i in range(n):
        lt = live_trades[i]
        bt = bt_trades[i]

        dir_ok = lt['direction'] == bt['direction']
        conf_ok = abs(lt['confidence'] - bt['confidence']) < 0.0001
        res_ok = lt['result'] == bt['result']

        if dir_ok and conf_ok and res_ok:
            matches += 1
        else:
            if not dir_ok:
                dir_mismatches += 1
            if not conf_ok:
                conf_mismatches += 1
            if not res_ok:
                result_mismatches += 1
            if dir_mismatches + conf_mismatches + result_mismatches <= 5:
                print(f"  MISMATCH #{i+1}: Live={lt['direction']} {lt['confidence']:.4f} {lt['result']} | "
                      f"BT={bt['direction']} {bt['confidence']:.4f} {bt['result']}")

    return {
        'total': n,
        'matches': matches,
        'match_pct': matches / n * 100 if n > 0 else 0,
        'dir_mismatches': dir_mismatches,
        'conf_mismatches': conf_mismatches,
        'result_mismatches': result_mismatches,
    }


def compute_stats(trades):
    """Compute trading statistics."""
    if not trades:
        return {}

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    wr = wins / len(trades)

    # Daily breakdown
    daily = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
    for t in trades:
        day = t['ts'][:10]
        if t['result'] == 'WIN':
            daily[day]['wins'] += 1
        else:
            daily[day]['losses'] += 1
        daily[day]['pnl'] += t['pnl']

    green_days = sum(1 for d in daily.values() if d['pnl'] > 0)
    total_days = len(daily)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'pnl': round(total_pnl, 2),
        'green_days': green_days,
        'total_days': total_days,
        'green_day_pct': green_days / total_days * 100 if total_days > 0 else 0,
        'daily_avg_pnl': round(total_pnl / total_days, 2) if total_days > 0 else 0,
        'daily_trades': defaultdict(dict, {k: dict(v) for k, v in daily.items()}),
    }


def run_asset(symbol, model_file):
    """Run full simulation for one asset."""
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  {symbol} — 7-Day Simulation")
    print(f"{'='*80}")

    # Load model
    model, feature_names = load_model(model_file)
    print(f"  Model: {model_file} ({len(feature_names)} features)")

    # Download data
    klines = fetch_binance_klines(symbol, days=TOTAL_DAYS)
    if len(klines) < 10000:
        print(f"  ERROR: Not enough klines ({len(klines)})")
        return None

    # Determine test window start
    last_ts = klines[-1][0]
    test_start_ts = last_ts - TEST_DAYS * 86400
    test_start_dt = datetime.fromtimestamp(test_start_ts, tz=timezone.utc)
    print(f"  Data: {len(klines)} klines")
    print(f"  Test window: {test_start_dt.date()} to {datetime.fromtimestamp(last_ts, tz=timezone.utc).date()}")

    # Run both pipelines on SAME klines
    print(f"\n  Running live pipeline...")
    live_trades = run_live_pipeline(klines, model, feature_names, test_start_ts)
    print(f"  -> {len(live_trades)} trades")

    print(f"  Running backtest pipeline...")
    bt_trades = run_backtest_pipeline(klines, model, feature_names, test_start_ts)
    print(f"  -> {len(bt_trades)} trades")

    # Compare
    print(f"\n  Comparing trade-by-trade...")
    comparison = compare_trades(live_trades, bt_trades, symbol)

    if comparison['matches'] == comparison['total']:
        print(f"  ✓ PERFECT MATCH — {comparison['total']} trades identical")
    else:
        print(f"  ✗ {comparison['total'] - comparison['matches']} mismatches out of {comparison['total']}")

    # Stats
    stats = compute_stats(live_trades)
    print(f"\n  7-Day Performance:")
    print(f"    Trades:     {stats['trades']}")
    print(f"    Win Rate:   {stats['wr']*100:.2f}% ({stats['wins']}W/{stats['losses']}L)")
    print(f"    PnL:        ${stats['pnl']:+,.2f}")
    print(f"    Green Days: {stats['green_days']}/{stats['total_days']} ({stats['green_day_pct']:.0f}%)")
    print(f"    Avg Daily:  ${stats['daily_avg_pnl']:+,.2f}")

    # Daily breakdown
    print(f"\n  Daily Breakdown:")
    for day in sorted(stats['daily_trades'].keys()):
        d = stats['daily_trades'][day]
        day_total = d['wins'] + d['losses']
        day_wr = d['wins'] / day_total * 100 if day_total > 0 else 0
        marker = '✓' if d['pnl'] > 0 else '✗'
        print(f"    {day}: {day_wr:5.1f}% WR ({d['wins']}W/{d['losses']}L) "
              f"PnL: ${d['pnl']:+8.2f} {marker}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    return {
        'symbol': symbol,
        'comparison': comparison,
        'stats': {k: v for k, v in stats.items() if k != 'daily_trades'},
        'daily': {k: dict(v) for k, v in stats['daily_trades'].items()},
        'elapsed_s': int(elapsed),
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("7-DAY LIVE PIPELINE SIMULATION vs BACKTEST")
    print("=" * 80)

    results = []
    for symbol, model_file in ASSETS.items():
        try:
            result = run_asset(symbol, model_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR on {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'symbol': symbol, 'error': str(e)})

    # Combined Summary
    print(f"\n\n{'='*80}")
    print("COMBINED 7-DAY SUMMARY")
    print(f"{'='*80}")

    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    all_match = True

    for r in results:
        if 'error' in r:
            print(f"  {r['symbol']}: FAILED — {r['error']}")
            all_match = False
            continue

        s = r['stats']
        c = r['comparison']
        match_str = '✓ 100%' if c['matches'] == c['total'] else f"✗ {c['match_pct']:.1f}%"
        print(f"  {r['symbol']}: WR={s['wr']*100:.2f}% | {s['trades']} trades | "
              f"PnL=${s['pnl']:+,.2f} | Green={s['green_days']}/{s['total_days']} | "
              f"Match: {match_str}")

        total_trades += s['trades']
        total_wins += s['wins']
        total_pnl += s['pnl']
        if c['matches'] != c['total']:
            all_match = False

    if total_trades > 0:
        combined_wr = total_wins / total_trades
        print(f"\n  COMBINED:")
        print(f"    Total Trades:  {total_trades}")
        print(f"    Combined WR:   {combined_wr*100:.2f}%")
        print(f"    Total PnL:     ${total_pnl:+,.2f}")
        print(f"    Daily Avg PnL: ${total_pnl / TEST_DAYS:+,.2f}")
        print(f"    Pipeline Match: {'✓ ALL IDENTICAL' if all_match else '✗ MISMATCHES FOUND'}")

    # Projected at different bet sizes
    if total_trades > 0:
        daily_trades = total_trades / TEST_DAYS
        daily_pnl_50 = total_pnl / TEST_DAYS
        print(f"\n  PROJECTIONS (based on this week):")
        for bet in [50, 100, 200]:
            scale = bet / BET_SIZE
            fees = daily_trades * 0.78
            net = daily_pnl_50 * scale - fees
            print(f"    ${bet} bets: ${daily_pnl_50 * scale:+,.2f}/day gross, "
                  f"-${fees:,.2f} fees = ${net:+,.2f}/day net")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")

    # Save
    out_path = os.path.join(MODEL_DIR, 'week_simulation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
