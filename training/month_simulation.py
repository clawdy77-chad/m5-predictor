#!/usr/bin/env python3
"""
30-Day Out-of-Sample Validation for All 4 Trading Models.

Steps:
1. Retrain all 4 models using 180 days of data ending ~30 days ago (around Jan 26, 2026)
2. Save temporary models to model/temp_30d/ (do NOT overwrite production models)
3. Run live-vs-backtest comparison on 30-day window (Jan 26 - Feb 25, 2026)
4. Use 7 days warmup before the test window
5. Compare trade-by-trade: direction, confidence, outcome must match 1:1
6. Report per-asset and combined stats

This validates that our models work correctly on truly out-of-sample data.
"""
import sys, os, json, time, pickle
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))

from v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs
from training.train_final_model import (
    process_candles, walk_forward_oos, _filter_sub1min
)

import warnings
warnings.filterwarnings('ignore')
# Suppress NumPy 2.x compatibility warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Handle NumPy 2.x compatibility
try:
    import numpy as np
    # Suppress the specific NumPy 2.x warnings
    np.warnings = warnings  # Make sure warnings are suppressed
except ImportError:
    pass

# Directories
MODEL_DIR = os.path.join(_ROOT, 'model')
TEMP_MODEL_DIR = os.path.join(MODEL_DIR, 'temp_30d')

# Assets to validate
ASSETS = {
    'BTCUSDT': 'btc_54_model.pkl',
    'ETHUSDT': 'eth_model.pkl', 
    'XRPUSDT': 'xrp_model.pkl',
    'SOLUSDT': 'sol_model.pkl',
}

# Simulation parameters
TRAINING_DAYS = 180      # Days of data to train on
WARMUP_DAYS = 7          # Warmup before test window  
TEST_DAYS = 30           # Test window length
CUTOFF_DAYS_AGO = 30     # Days ago when training data should end

BET_SIZE = 50.0
LIMIT_PRICE = 0.51

DAY_SEC = 86400


def fetch_historical_klines(symbol, days, end_days_ago=0):
    """Download 1m klines from Binance ending N days ago."""
    print(f"  Fetching {days} days of {symbol} ending {end_days_ago} days ago...", flush=True)
    t0 = time.time()
    
    # Calculate end time (N days ago from now)
    now = int(time.time())
    end_time_ms = (now - end_days_ago * DAY_SEC) * 1000
    target_start_ms = end_time_ms - days * DAY_SEC * 1000
    
    print(f"    Target date range: {datetime.fromtimestamp(target_start_ms//1000, tz=timezone.utc).date()} to {datetime.fromtimestamp(end_time_ms//1000, tz=timezone.utc).date()}", flush=True)
    
    all_klines = []
    current_end = end_time_ms
    batch_count = 0
    max_batches = days * 2  # Safety limit: ~2 batches per day max
    
    while current_end > target_start_ms and batch_count < max_batches:
        batch_count += 1
        url = (f'https://api.binance.com/api/v3/klines?symbol={symbol}'
               f'&interval=1m&limit=1000&endTime={current_end}')
        
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                print(f"    No more data available (batch {batch_count})", flush=True)
                break
                
            all_klines = batch + all_klines
            current_end = batch[0][0] - 1
            
            if batch_count % 50 == 0:  # Progress every 50 batches
                current_dt = datetime.fromtimestamp(batch[0][0]//1000, tz=timezone.utc)
                print(f"    Progress: batch {batch_count}, reached {current_dt.date()}, {len(all_klines)} candles", flush=True)
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"    Error in batch {batch_count}: {e}", flush=True)
            if batch_count < 5:  # Retry early errors
                time.sleep(1)
                continue
            else:
                print(f"    Too many errors, stopping with {len(all_klines)} candles", flush=True)
                break
    
    # Convert to our format
    rows_1m = []
    for k in all_klines:
        rows_1m.append((
            int(k[0]) // 1000,  # timestamp in seconds
            float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
        ))
    
    elapsed = time.time() - t0
    print(f"    -> {len(rows_1m):,} candles in {elapsed:.1f}s")
    if rows_1m:
        start_dt = datetime.fromtimestamp(rows_1m[0][0], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(rows_1m[-1][0], tz=timezone.utc)
        print(f"    -> Range: {start_dt.date()} to {end_dt.date()}")
    
    return rows_1m


def train_temp_model(symbol, training_data):
    """Train a model using the same methodology as train_final_model.py."""
    print(f"  Training temporary model for {symbol}...")
    t0 = time.time()
    
    # Handle NumPy 2.x compatibility issues with LightGBM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ['PYTHONWARNINGS'] = 'ignore'
        try:
            import lightgbm as lgb
        except ImportError as e:
            print(f"    Warning: LightGBM import issue (likely NumPy 2.x): {e}")
            # Try alternative import approach
            import sys
            for warning_type in [UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning]:
                warnings.filterwarnings('ignore', category=warning_type)
            import lightgbm as lgb
    
    # Process candles into M5 samples
    data = process_candles(training_data, target_tf='m5')
    if not data or len(data) < 5000:
        raise ValueError(f"Not enough processed data: {len(data) if data else 0} samples")
    
    # Run walk-forward to get baseline WR
    preds, labels, confs, timestamps, features, fnames = walk_forward_oos(data, backtest_days=90)
    if len(preds) < 100:
        raise ValueError(f"Not enough OOS predictions: {len(preds)}")
    
    base_wr = (preds == labels).mean()
    print(f"    Walk-forward WR: {base_wr*100:.2f}%")
    
    # Train final model
    fnames_all = sorted(data[0][1].keys())
    _, fnames_clean = _filter_sub1min(fnames_all)
    all_X = np.array([[d[1].get(f, 0) for f in fnames_clean] for d in data], dtype=np.float32)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    all_y = np.array([d[2] for d in data])
    
    # Feature selection
    pre_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    pre_model.fit(all_X, all_y)
    top_k = 25
    top_idx = np.argsort(pre_model.feature_importances_)[::-1][:top_k]
    final_fnames = [fnames_clean[i] for i in top_idx]
    all_X = all_X[:, top_idx]
    
    # Train model with same hyperparameters
    val_size = min(len(all_y) // 8, 2016)
    model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        max_depth=2, num_leaves=4, learning_rate=0.05, n_estimators=300,
        min_child_samples=200, subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, is_unbalance=True,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    
    if val_size >= 10:
        model.fit(
            all_X[:-val_size], all_y[:-val_size],
            eval_set=[(all_X[-val_size:], all_y[-val_size:])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        model.fit(all_X, all_y)
    
    elapsed = time.time() - t0
    print(f"    -> Trained in {elapsed:.0f}s ({len(final_fnames)} features, {len(all_y)} samples)")
    
    # Save to temp directory
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    pair_name = symbol.replace('USDT', '').lower()
    if pair_name == 'btc':
        temp_filename = 'btc_54_model.pkl'  # Match production filename
    else:
        temp_filename = f'{pair_name}_model.pkl'
    
    temp_path = os.path.join(TEMP_MODEL_DIR, temp_filename)
    
    trained_at = datetime.now(timezone.utc).isoformat()
    model_data = {
        'model': model,
        'feature_names': final_fnames,
        'config': {
            'type': 'v24_fractal_hybrid',
            'model': 'lgbm_baseline',
            'symbol': symbol,
            'n_features': len(final_fnames),
            'samples': len(all_y),
            'train_days': TRAINING_DAYS,
            'trained_at': trained_at,
            'walk_forward_wr': float(round(base_wr, 4)),
            'validation_model': True,
            'params': {
                'max_depth': 2, 'num_leaves': 4, 'learning_rate': 0.05,
                'n_estimators': 300, 'min_child_samples': 200,
                'subsample': 0.7, 'colsample_bytree': 0.5,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
            },
        }
    }
    
    with open(temp_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"    -> Saved: {temp_path}")
    
    return model, final_fnames, base_wr


def load_temp_model(symbol):
    """Load temporary model for validation."""
    pair_name = symbol.replace('USDT', '').lower()
    if pair_name == 'btc':
        temp_filename = 'btc_54_model.pkl'
    else:
        temp_filename = f'{pair_name}_model.pkl'
    
    temp_path = os.path.join(TEMP_MODEL_DIR, temp_filename)
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Temp model not found: {temp_path}")
    
    with open(temp_path, 'rb') as f:
        d = pickle.load(f)
    return d['model'], d['feature_names']


def run_live_pipeline(klines, model, feature_names, test_start_ts):
    """Simulate exact paper_trader.py pipeline on klines."""
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
    """Run backtest pipeline (process_candles style) on same klines."""
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


def validate_asset(symbol):
    """Run complete 30-day validation for one asset."""
    print(f"\n{'='*90}")
    print(f"  {symbol} — 30-Day Out-of-Sample Validation")
    print(f"{'='*90}")
    
    t0 = time.time()
    
    # Step 1: Download training data (ending CUTOFF_DAYS_AGO days ago)
    print(f"\n[1/4] Downloading training data...")
    training_data = fetch_historical_klines(symbol, TRAINING_DAYS, CUTOFF_DAYS_AGO)
    if len(training_data) < 50000:
        raise ValueError(f"Not enough training data: {len(training_data)} candles")
    
    # Step 2: Train temporary model
    print(f"\n[2/4] Training temporary model...")
    model, feature_names, base_wr = train_temp_model(symbol, training_data)
    
    # Step 3: Download validation data (warmup + test window)
    print(f"\n[3/4] Downloading validation data...")
    total_val_days = WARMUP_DAYS + TEST_DAYS
    val_data = fetch_historical_klines(symbol, total_val_days, 0)  # ending now
    if len(val_data) < 20000:
        raise ValueError(f"Not enough validation data: {len(val_data)} candles")
    
    # Determine test window start (after warmup)
    last_ts = val_data[-1][0]
    test_start_ts = last_ts - TEST_DAYS * DAY_SEC
    test_start_dt = datetime.fromtimestamp(test_start_ts, tz=timezone.utc)
    test_end_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
    
    print(f"  Validation data: {len(val_data)} klines")
    print(f"  Test window: {test_start_dt.date()} to {test_end_dt.date()}")
    print(f"  Training WR: {base_wr*100:.2f}%")
    
    # Step 4: Run pipelines and compare
    print(f"\n[4/4] Running validation pipelines...")
    print(f"  Running live pipeline...")
    live_trades = run_live_pipeline(val_data, model, feature_names, test_start_ts)
    print(f"  -> {len(live_trades)} trades")
    
    print(f"  Running backtest pipeline...")
    bt_trades = run_backtest_pipeline(val_data, model, feature_names, test_start_ts)
    print(f"  -> {len(bt_trades)} trades")
    
    # Compare pipelines
    print(f"\n  Comparing trade-by-trade...")
    comparison = compare_trades(live_trades, bt_trades, symbol)
    
    if comparison['matches'] == comparison['total']:
        print(f"  ✓ PERFECT MATCH — {comparison['total']} trades identical")
    else:
        print(f"  ✗ {comparison['total'] - comparison['matches']} mismatches out of {comparison['total']}")
    
    # Compute stats
    stats = compute_stats(live_trades)
    print(f"\n  30-Day Performance:")
    print(f"    Trades:     {stats['trades']}")
    print(f"    Win Rate:   {stats['wr']*100:.2f}% ({stats['wins']}W/{stats['losses']}L)")
    print(f"    PnL:        ${stats['pnl']:+,.2f}")
    print(f"    Green Days: {stats['green_days']}/{stats['total_days']} ({stats['green_day_pct']:.0f}%)")
    print(f"    Avg Daily:  ${stats['daily_avg_pnl']:+,.2f}")
    
    # Show daily breakdown (first and last 5 days)
    sorted_days = sorted(stats['daily_trades'].keys())
    if len(sorted_days) > 10:
        print(f"\n  Daily Breakdown (first 5 days):")
        for day in sorted_days[:5]:
            d = stats['daily_trades'][day]
            day_total = d['wins'] + d['losses']
            day_wr = d['wins'] / day_total * 100 if day_total > 0 else 0
            marker = '✓' if d['pnl'] > 0 else '✗'
            print(f"    {day}: {day_wr:5.1f}% WR ({d['wins']}W/{d['losses']}L) "
                  f"PnL: ${d['pnl']:+8.2f} {marker}")
        
        print(f"  ... (showing first/last 5 of {len(sorted_days)} days)")
        print(f"\n  Daily Breakdown (last 5 days):")
        for day in sorted_days[-5:]:
            d = stats['daily_trades'][day]
            day_total = d['wins'] + d['losses']
            day_wr = d['wins'] / day_total * 100 if day_total > 0 else 0
            marker = '✓' if d['pnl'] > 0 else '✗'
            print(f"    {day}: {day_wr:5.1f}% WR ({d['wins']}W/{d['losses']}L) "
                  f"PnL: ${d['pnl']:+8.2f} {marker}")
    else:
        print(f"\n  Daily Breakdown:")
        for day in sorted_days:
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
        'training_wr': float(round(base_wr, 4)),
        'comparison': comparison,
        'stats': {k: v for k, v in stats.items() if k != 'daily_trades'},
        'daily': {k: dict(v) for k, v in stats['daily_trades'].items()},
        'elapsed_s': int(elapsed),
    }


def main():
    """Run 30-day validation for all assets."""
    print("=" * 90, flush=True)
    print("30-DAY OUT-OF-SAMPLE VALIDATION FOR ALL TRADING MODELS", flush=True)
    print("=" * 90, flush=True)
    print(f"Training period: {TRAINING_DAYS} days ending {CUTOFF_DAYS_AGO} days ago", flush=True)
    print(f"Test period: {TEST_DAYS} days ending today", flush=True)
    print(f"Warmup: {WARMUP_DAYS} days", flush=True)
    print(f"Temporary models saved to: {TEMP_MODEL_DIR}", flush=True)
    
    t_start = time.time()
    results = []
    
    for symbol in ASSETS.keys():
        try:
            result = validate_asset(symbol)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR validating {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'symbol': symbol, 'error': str(e)})
    
    # Combined Summary
    print(f"\n\n{'='*90}")
    print("COMBINED 30-DAY VALIDATION SUMMARY")
    print(f"{'='*90}")
    
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    all_match = True
    successful_assets = []
    
    for r in results:
        if 'error' in r:
            print(f"  {r['symbol']}: FAILED — {r['error']}")
            all_match = False
            continue
        
        s = r['stats']
        c = r['comparison']
        match_str = '✓ 100%' if c['matches'] == c['total'] else f"✗ {c['match_pct']:.1f}%"
        
        print(f"  {r['symbol']}:")
        print(f"    Training WR: {r['training_wr']*100:.2f}%")
        print(f"    Test WR:     {s['wr']*100:.2f}% | {s['trades']} trades | "
              f"PnL: ${s['pnl']:+,.2f}")
        print(f"    Green Days:  {s['green_days']}/{s['total_days']} ({s['green_day_pct']:.0f}%)")
        print(f"    Pipeline:    {match_str}")
        print()
        
        total_trades += s['trades']
        total_wins += s['wins']
        total_pnl += s['pnl']
        successful_assets.append(r['symbol'])
        
        if c['matches'] != c['total']:
            all_match = False
    
    if total_trades > 0:
        combined_wr = total_wins / total_trades
        print(f"  COMBINED RESULTS ({len(successful_assets)} assets):")
        print(f"    Total Trades:   {total_trades}")
        print(f"    Combined WR:    {combined_wr*100:.2f}%")
        print(f"    Total PnL:      ${total_pnl:+,.2f}")
        print(f"    Daily Avg PnL:  ${total_pnl / TEST_DAYS:+,.2f}")
        print(f"    Pipeline Match: {'✓ ALL IDENTICAL' if all_match else '✗ MISMATCHES FOUND'}")
        
        # Projections at different bet sizes
        daily_trades = total_trades / TEST_DAYS
        daily_pnl_50 = total_pnl / TEST_DAYS
        print(f"\n  PROJECTIONS (based on this 30-day period):")
        for bet in [50, 100, 200]:
            scale = bet / BET_SIZE
            fees = daily_trades * 0.78
            net = daily_pnl_50 * scale - fees
            print(f"    ${bet} bets: ${daily_pnl_50 * scale:+,.2f}/day gross, "
                  f"-${fees:,.2f} fees = ${net:+,.2f}/day net")
    
    elapsed = time.time() - t_start
    print(f"\n  Total validation time: {elapsed:.0f}s")
    
    # Save results
    output_path = os.path.join(MODEL_DIR, 'month_simulation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Results saved: {output_path}")
    
    # Cleanup temp directory (comment out if you want to keep models)
    print(f"\n  Temporary models saved in: {TEMP_MODEL_DIR}")
    print(f"  (Clean up manually when no longer needed)")
    
    return results


if __name__ == '__main__':
    main()