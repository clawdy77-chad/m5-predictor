#!/usr/bin/env python3
"""
Train models for multiple pairs using the exact same methodology as BTC.
Outputs: model/{symbol}_model.pkl + model/{symbol}_flip_rules.json per pair.
"""
import sys, os, time, json, pickle
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from training.train_final_model import (
    fetch_binance_klines,
    process_candles,
    walk_forward_oos,
    mine_flip_rules,
    create_interaction_features,
    serialize_flip_rules,
    _filter_sub1min,
)
from model.v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs

MODEL_DIR = os.path.join(_ROOT, 'model')

PAIRS = ['ETHUSDT', 'XRPUSDT', 'SOLUSDT']
DAYS = 180
BACKTEST_DAYS = 90


def train_pair(symbol):
    """Train a single pair end-to-end."""
    import lightgbm as lgb
    from datetime import datetime, timezone

    t_start = time.time()
    pair_name = symbol.replace('USDT', '').lower()
    model_path = os.path.join(MODEL_DIR, f'{pair_name}_model.pkl')
    flip_path = os.path.join(MODEL_DIR, f'{pair_name}_flip_rules.json')

    print(f"\n{'='*80}")
    print(f"  TRAINING: {symbol}")
    print(f"{'='*80}")

    # Step 1: Download
    rows_1m = fetch_binance_klines(symbol, days=DAYS)
    if len(rows_1m) < 50000:
        print(f"ERROR: Not enough data for {symbol} ({len(rows_1m)} candles)")
        return None

    # Step 2: Process
    data = process_candles(rows_1m, target_tf='m5')
    if not data:
        print(f"ERROR: No M5 samples for {symbol}")
        return None

    # Step 3: Walk-forward OOS
    preds, labels, confs, timestamps, features, fnames = walk_forward_oos(data, backtest_days=BACKTEST_DAYS)
    if len(preds) < 100:
        print(f"ERROR: Too few OOS predictions for {symbol}")
        return None

    base_wr = (preds == labels).mean()

    # Step 4: Train final model
    print(f"\n[4/6] Training final model for {symbol}...")
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

    trained_at = datetime.now(timezone.utc).isoformat()
    d = {
        'model': model, 'feature_names': final_fnames,
        'config': {
            'type': 'v24_fractal_hybrid',
            'model': 'lgbm_baseline',
            'symbol': symbol,
            'n_features': len(final_fnames),
            'samples': len(all_y),
            'train_days': DAYS,
            'trained_at': trained_at,
            'walk_forward_wr': float(round(base_wr, 4)),
            'params': {
                'max_depth': 2, 'num_leaves': 4, 'learning_rate': 0.05,
                'n_estimators': 300, 'min_child_samples': 200,
                'subsample': 0.7, 'colsample_bytree': 0.5,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
            },
        }
    }
    with open(model_path, 'wb') as f:
        pickle.dump(d, f)
    print(f"  Saved: {model_path}")

    # Step 5: Mine flip rules
    selected, all_fnames, composite_mask, stats = mine_flip_rules(
        preds, labels, confs, features, fnames, timestamps)

    # Step 6: Serialize flip rules
    # Temporarily override the global path
    import training.train_final_model as tfm
    orig_path = tfm.FLIP_RULES_PATH
    tfm.FLIP_RULES_PATH = flip_path
    serialize_flip_rules(selected, all_fnames, stats)
    tfm.FLIP_RULES_PATH = orig_path

    elapsed = time.time() - t_start
    print(f"\n  {symbol} DONE in {elapsed:.0f}s")
    print(f"  Base WR: {stats['base_wr']*100:.2f}% -> Flip WR: {stats['flip_wr']*100:.2f}%")
    print(f"  Rules: {stats['n_rules']} | Flipped: {stats['n_flipped']}/{stats['n_total']}")

    return {
        'symbol': symbol,
        'base_wr': stats['base_wr'],
        'flip_wr': stats['flip_wr'],
        'n_rules': stats['n_rules'],
        'n_trades': stats['n_total'],
        'model_path': model_path,
        'flip_path': flip_path,
        'elapsed_s': int(elapsed),
    }


def main():
    print("=" * 80)
    print("MULTI-PAIR TRAINING: ETH, XRP, SOL")
    print("=" * 80)

    results = []
    for symbol in PAIRS:
        try:
            result = train_pair(symbol)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'symbol': symbol, 'error': str(e)})

    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for r in results:
        if 'error' in r:
            print(f"  {r['symbol']}: FAILED — {r['error']}")
        else:
            print(f"  {r['symbol']}: Base WR {r['base_wr']*100:.2f}% -> Flip WR {r['flip_wr']*100:.2f}% | "
                  f"{r['n_rules']} rules | {r['n_trades']:,} trades | {r['elapsed_s']}s")

    # Save summary
    summary_path = os.path.join(MODEL_DIR, 'multi_pair_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
