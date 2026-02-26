#!/usr/bin/env python3
"""
4-Phase Sequential Hyperparameter Search for all assets.

Per TRAINING_NEW_PAIR.md:
  Phase 1: Depth / Leaves / Learning Rate
  Phase 2: Regularization (best from Phase 1)
  Phase 3: Sampling (best from Phase 2)
  Phase 4: Ensemble Size (best from Phase 3)

Runs walk-forward OOS evaluation for each config. Best config per phase
carries forward to the next phase.

Usage:
    python training/hyperparam_search.py
    python training/hyperparam_search.py --symbol ETHUSDT
"""
import sys, os, time, json, pickle, itertools
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'model'))

from v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs
from training.train_final_model import fetch_binance_klines, process_candles, _filter_sub1min

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DAY_SEC = 86400
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT']
DAYS = 180
BACKTEST_DAYS = 90

# Walk-forward config
TRAIN_SEC = 53 * DAY_SEC
VAL_SEC = 7 * DAY_SEC
TEST_SEC = 7 * DAY_SEC
STEP_SEC = 7 * DAY_SEC
EMBARGO_SEC = 7 * DAY_SEC


def walk_forward_eval(all_X, all_y, all_ts, params, backtest_start, last_ts):
    """Run walk-forward and return OOS win rate."""
    window_start = max(backtest_start, all_ts[0] + TRAIN_SEC + VAL_SEC + EMBARGO_SEC)

    oos_preds = []
    oos_labels = []

    while window_start + TEST_SEC <= last_ts:
        embargo_start = window_start - EMBARGO_SEC
        train_start_w = embargo_start - VAL_SEC - TRAIN_SEC
        val_start_w = embargo_start - VAL_SEC

        train_mask = (all_ts >= train_start_w) & (all_ts < val_start_w)
        val_mask = (all_ts >= val_start_w) & (all_ts < embargo_start)
        test_mask = (all_ts >= window_start) & (all_ts < window_start + TEST_SEC)

        X_tr, y_tr = all_X[train_mask], all_y[train_mask]
        X_val, y_val = all_X[val_mask], all_y[val_mask]
        X_te, y_te = all_X[test_mask], all_y[test_mask]

        if len(X_tr) < 100 or len(X_te) < 10:
            window_start += STEP_SEC
            continue

        model = lgb.LGBMClassifier(
            objective='binary', metric='binary_logloss',
            is_unbalance=True, verbose=-1, random_state=42, n_jobs=-1,
            **params,
        )
        if len(X_val) >= 10:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                                 lgb.log_evaluation(period=0)])
        else:
            model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        oos_preds.extend(preds)
        oos_labels.extend(y_te)

        window_start += STEP_SEC

    if not oos_preds:
        return 0.0, 0
    preds = np.array(oos_preds)
    labels = np.array(oos_labels)
    wr = (preds == labels).mean()
    return wr, len(preds)


def prepare_data(symbol):
    """Download and prepare data for a symbol."""
    rows_1m = fetch_binance_klines(symbol, days=DAYS)
    if len(rows_1m) < 50000:
        print(f"ERROR: Not enough data for {symbol}")
        return None

    data = process_candles(rows_1m, target_tf='m5')
    if not data:
        return None

    fnames_all = sorted(data[0][1].keys())
    keep_idx, fnames = _filter_sub1min(fnames_all)

    all_ts = np.array([d[0] for d in data])
    all_X = np.array([[d[1].get(f, 0) for f in fnames] for d in data], dtype=np.float32)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    all_y = np.array([d[2] for d in data])

    last_ts = all_ts[-1]
    backtest_start = last_ts - BACKTEST_DAYS * DAY_SEC

    # Feature selection on pre-backtest data
    pre_mask = all_ts < backtest_start
    if pre_mask.sum() > 500:
        pre_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
        pre_model.fit(all_X[pre_mask], all_y[pre_mask])
        top_idx = np.argsort(pre_model.feature_importances_)[::-1][:25]
        fnames = [fnames[i] for i in top_idx]
        all_X = all_X[:, top_idx]

    return all_X, all_y, all_ts, fnames, backtest_start, last_ts


def run_search(symbol):
    """Run 4-phase hyperparameter search for a symbol."""
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  HYPERPARAMETER SEARCH: {symbol}")
    print(f"{'='*80}")

    result = prepare_data(symbol)
    if result is None:
        return None
    all_X, all_y, all_ts, fnames, backtest_start, last_ts = result

    # Baseline
    baseline_params = {
        'max_depth': 2, 'num_leaves': 4, 'learning_rate': 0.05,
        'n_estimators': 300, 'min_child_samples': 200,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.0, 'reg_lambda': 5.0,
    }
    baseline_wr, baseline_n = walk_forward_eval(all_X, all_y, all_ts, baseline_params, backtest_start, last_ts)
    print(f"\n  Baseline: WR={baseline_wr*100:.2f}% ({baseline_n} trades)")

    best_params = dict(baseline_params)
    best_wr = baseline_wr

    # ── Phase 1: Depth / Leaves / Learning Rate ──
    print(f"\n  Phase 1: Depth / Leaves / Learning Rate")
    phase1_results = []
    for md, nl, lr in itertools.product([2, 3, 4], [4, 6, 8], [0.01, 0.03, 0.05, 0.1]):
        if nl > 2**md:
            continue  # num_leaves can't exceed 2^max_depth
        params = dict(best_params)
        params.update({'max_depth': md, 'num_leaves': nl, 'learning_rate': lr})
        wr, n = walk_forward_eval(all_X, all_y, all_ts, params, backtest_start, last_ts)
        phase1_results.append((wr, n, md, nl, lr))
        marker = ' ***' if wr > best_wr else ''
        print(f"    depth={md} leaves={nl} lr={lr}: WR={wr*100:.2f}% ({n}){marker}")

    phase1_results.sort(key=lambda x: -x[0])
    if phase1_results[0][0] > best_wr:
        best_wr = phase1_results[0][0]
        best_params['max_depth'] = phase1_results[0][2]
        best_params['num_leaves'] = phase1_results[0][3]
        best_params['learning_rate'] = phase1_results[0][4]
    print(f"  Phase 1 best: depth={best_params['max_depth']} leaves={best_params['num_leaves']} "
          f"lr={best_params['learning_rate']} WR={best_wr*100:.2f}%")

    # ── Phase 2: Regularization ──
    print(f"\n  Phase 2: Regularization")
    phase2_results = []
    for mcs, ra, rl in itertools.product([100, 200, 300, 500], [0.1, 1.0, 5.0], [0.5, 2.0, 5.0, 10.0]):
        params = dict(best_params)
        params.update({'min_child_samples': mcs, 'reg_alpha': ra, 'reg_lambda': rl})
        wr, n = walk_forward_eval(all_X, all_y, all_ts, params, backtest_start, last_ts)
        phase2_results.append((wr, n, mcs, ra, rl))
        marker = ' ***' if wr > best_wr else ''
        print(f"    mcs={mcs} alpha={ra} lambda={rl}: WR={wr*100:.2f}% ({n}){marker}")

    phase2_results.sort(key=lambda x: -x[0])
    if phase2_results[0][0] > best_wr:
        best_wr = phase2_results[0][0]
        best_params['min_child_samples'] = phase2_results[0][2]
        best_params['reg_alpha'] = phase2_results[0][3]
        best_params['reg_lambda'] = phase2_results[0][4]
    print(f"  Phase 2 best: mcs={best_params['min_child_samples']} alpha={best_params['reg_alpha']} "
          f"lambda={best_params['reg_lambda']} WR={best_wr*100:.2f}%")

    # ── Phase 3: Sampling ──
    print(f"\n  Phase 3: Sampling")
    phase3_results = []
    for ss, cs in itertools.product([0.5, 0.6, 0.7, 0.8], [0.3, 0.5, 0.7]):
        params = dict(best_params)
        params.update({'subsample': ss, 'colsample_bytree': cs})
        wr, n = walk_forward_eval(all_X, all_y, all_ts, params, backtest_start, last_ts)
        phase3_results.append((wr, n, ss, cs))
        marker = ' ***' if wr > best_wr else ''
        print(f"    subsample={ss} colsample={cs}: WR={wr*100:.2f}% ({n}){marker}")

    phase3_results.sort(key=lambda x: -x[0])
    if phase3_results[0][0] > best_wr:
        best_wr = phase3_results[0][0]
        best_params['subsample'] = phase3_results[0][2]
        best_params['colsample_bytree'] = phase3_results[0][3]
    print(f"  Phase 3 best: subsample={best_params['subsample']} "
          f"colsample={best_params['colsample_bytree']} WR={best_wr*100:.2f}%")

    # ── Phase 4: Ensemble Size ──
    print(f"\n  Phase 4: Ensemble Size")
    phase4_results = []
    for ne in [100, 200, 300, 500, 800]:
        params = dict(best_params)
        params.update({'n_estimators': ne})
        wr, n = walk_forward_eval(all_X, all_y, all_ts, params, backtest_start, last_ts)
        phase4_results.append((wr, n, ne))
        marker = ' ***' if wr > best_wr else ''
        print(f"    n_estimators={ne}: WR={wr*100:.2f}% ({n}){marker}")

    phase4_results.sort(key=lambda x: -x[0])
    if phase4_results[0][0] > best_wr:
        best_wr = phase4_results[0][0]
        best_params['n_estimators'] = phase4_results[0][2]
    print(f"  Phase 4 best: n_estimators={best_params['n_estimators']} WR={best_wr*100:.2f}%")

    elapsed = time.time() - t0
    improvement = (best_wr - baseline_wr) * 100

    print(f"\n  {'='*60}")
    print(f"  {symbol} FINAL RESULT")
    print(f"  {'='*60}")
    print(f"  Baseline WR:  {baseline_wr*100:.2f}%")
    print(f"  Best WR:      {best_wr*100:.2f}% ({improvement:+.2f}pp)")
    print(f"  Best params:  {json.dumps(best_params, indent=2)}")
    print(f"  Time:         {elapsed:.0f}s")

    return {
        'symbol': symbol,
        'baseline_wr': float(baseline_wr),
        'best_wr': float(best_wr),
        'improvement_pp': float(improvement),
        'best_params': best_params,
        'baseline_params': baseline_params,
        'elapsed_s': int(elapsed),
        'n_trades': baseline_n,
    }


def main():
    symbols = SYMBOLS
    # Allow single symbol via CLI
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--symbol' and i < len(sys.argv) - 1:
            symbols = [sys.argv[i + 1].upper()]

    print("=" * 80)
    print(f"4-PHASE HYPERPARAMETER SEARCH: {', '.join(symbols)}")
    print("=" * 80)

    results = []
    for symbol in symbols:
        try:
            result = run_search(symbol)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR on {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'symbol': symbol, 'error': str(e)})

    # Summary
    print(f"\n\n{'='*80}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*80}")
    for r in results:
        if 'error' in r:
            print(f"  {r['symbol']}: FAILED — {r['error']}")
        else:
            print(f"  {r['symbol']}: {r['baseline_wr']*100:.2f}% -> {r['best_wr']*100:.2f}% "
                  f"({r['improvement_pp']:+.2f}pp) | {r['elapsed_s']}s")

    # Save results
    out_path = os.path.join(_ROOT, 'model', 'hyperparam_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
