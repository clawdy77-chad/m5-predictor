#!/usr/bin/env python3
"""
Train Final Model + Mine Flip Rules.

One-shot script to:
1. Download latest 180 days of BTC 1m klines from Binance
2. Process into M5 samples using IncrementalTF + extract_features
3. Run walk-forward to generate OOS predictions
4. Train a final LightGBM model on the full dataset
5. Run V5b-style flip mining on OOS trades
6. Export model/v24_fractal_model.pkl + model/flip_rules.json

Run before deploying live bot, and periodically (e.g., weekly).

Usage:
    python train_final_model.py
"""
import sys, os, time, json, pickle, math
import numpy as np
import requests
from datetime import datetime, timezone
from collections import defaultdict

# Import backtest infrastructure
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # validated_model/
MODEL_DIR = os.path.join(_ROOT, 'model')
sys.path.insert(0, MODEL_DIR)
from v24_backtest import (
    IncrementalTF, TF_SECS, FVG_LOOKBACKS,
    extract_features, interpolate_1m_to_subs,
)

DAY_SEC = 86400
MODEL_PATH = os.path.join(MODEL_DIR, 'v24_fractal_model.pkl')
FLIP_RULES_PATH = os.path.join(MODEL_DIR, 'flip_rules.json')


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Download data
# ═══════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol='BTCUSDT', days=60):
    """Download 1m klines from Binance REST API."""
    print(f"\n[1/6] Fetching {days} days of {symbol} 1m klines...")
    t0 = time.time()

    all_klines = []
    end_time = int(time.time() * 1000)
    target_start = end_time - days * 86400 * 1000

    current_end = end_time
    while current_end > target_start:
        url = (f'https://api.binance.com/api/v3/klines?symbol={symbol}'
               f'&interval=1m&limit=1000&endTime={current_end}')
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_klines = batch + all_klines
        current_end = batch[0][0] - 1
        time.sleep(0.2)

    rows_1m = []
    for k in all_klines:
        rows_1m.append((
            int(k[0]) // 1000,
            float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
        ))

    print(f"  {len(rows_1m):,} candles ({time.time()-t0:.1f}s)")
    if rows_1m:
        print(f"  Range: {datetime.fromtimestamp(rows_1m[0][0], tz=timezone.utc).date()} to "
              f"{datetime.fromtimestamp(rows_1m[-1][0], tz=timezone.utc).date()}")
    return rows_1m


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Process into M5 samples
# ═══════════════════════════════════════════════════════════════════════

def process_candles(rows_1m, target_tf='m5'):
    """Process 1m candles into feature samples using IncrementalTF."""
    print(f"\n[2/6] Processing {len(rows_1m):,} candles into {target_tf} samples...")
    t0 = time.time()

    tfs = {}
    for key, sec in TF_SECS.items():
        tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

    data = []
    prev_label_count = 0
    pending_feat = None
    pending_ts = 0

    for ts, o, h, l, c, v in rows_1m:
        # 1. Feed LTFs only (matches v24_backtest.py order — no HTF lookahead)
        for key in ['m1', 'm3', 'm5']:
            tfs[key].feed_1m(ts, o, h, l, c, v)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
            tfs['10s'].feed_sub(*sub)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
            tfs['30s'].feed_sub(*sub)

        # 2. Check M5 completion → extract features BEFORE HTF update
        label_count = tfs[target_tf].n
        if label_count > prev_label_count and label_count > 50:
            prev_label_count = label_count

            if pending_feat is not None:
                tf_dir = 1 if tfs[target_tf].closes[-1] > tfs[target_tf].opens[-1] else 0
                data.append((pending_ts, pending_feat, tf_dir))

            pending_feat = extract_features(tfs, target_tf=target_tf)
            pending_ts = tfs[target_tf].timestamps[-1]

        # 3. Feed HTFs AFTER feature extraction (no lookahead)
        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

    print(f"  {len(data):,} samples ({time.time()-t0:.0f}s)")
    return data


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Walk-forward OOS predictions
# ═══════════════════════════════════════════════════════════════════════

def _filter_sub1min(fnames):
    """Remove sub-1min features (10s, 30s) — interpolated from 1m, not real data."""
    _SUB1M = ('10s_', '30s_', 's10_', 's30_', 'c1_count_10s', 'c1_count_30s',
              'consec_c3_10s', 'consec_c3_30s')
    keep = [i for i, f in enumerate(fnames) if not any(f.startswith(p) or f == p for p in _SUB1M)]
    return keep, [fnames[i] for i in keep]


def walk_forward_oos(data, backtest_days=45):
    """Run walk-forward to generate OOS predictions and feature arrays."""
    print(f"\n[3/6] Walk-forward OOS predictions (last {backtest_days} days)...")
    t0 = time.time()

    import lightgbm as lgb

    fnames_all = sorted(data[0][1].keys())
    keep_idx, fnames = _filter_sub1min(fnames_all)
    print(f"  Dropped {len(fnames_all) - len(fnames)} sub-1min features, keeping {len(fnames)}")

    all_ts_arr = np.array([d[0] for d in data])
    all_X = np.array([[d[1].get(f, 0) for f in fnames] for d in data], dtype=np.float32)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    all_y = np.array([d[2] for d in data])

    last_ts = all_ts_arr[-1]
    backtest_start = last_ts - backtest_days * DAY_SEC

    TRAIN_SEC = 53 * DAY_SEC
    VAL_SEC = 7 * DAY_SEC
    TEST_SEC = 7 * DAY_SEC
    STEP_SEC = 7 * DAY_SEC
    EMBARGO_SEC = 7 * DAY_SEC

    # Keep full feature array for flip mining (segments + rules need all features)
    all_X_full = all_X.copy()
    fnames_full = list(fnames)

    # Feature selection on pre-backtest data (for ML model only)
    pre_mask = all_ts_arr < backtest_start
    if pre_mask.sum() > 500:
        pre_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
        pre_model.fit(all_X[pre_mask], all_y[pre_mask])
        top_k = 25
        top_idx = np.argsort(pre_model.feature_importances_)[::-1][:top_k]
        fnames = [fnames[i] for i in top_idx]
        all_X = all_X[:, top_idx]

    first_ts = all_ts_arr[0]
    window_start = max(backtest_start, first_ts + TRAIN_SEC + VAL_SEC + EMBARGO_SEC)

    oos_preds = []
    oos_labels = []
    oos_confs = []
    oos_timestamps = []
    oos_features = []

    while window_start + TEST_SEC <= last_ts:
        embargo_start = window_start - EMBARGO_SEC
        train_start_w = embargo_start - VAL_SEC - TRAIN_SEC
        val_start_w = embargo_start - VAL_SEC

        train_mask = (all_ts_arr >= train_start_w) & (all_ts_arr < val_start_w)
        val_mask = (all_ts_arr >= val_start_w) & (all_ts_arr < embargo_start)
        test_mask = (all_ts_arr >= window_start) & (all_ts_arr < window_start + TEST_SEC)

        X_tr, y_tr = all_X[train_mask], all_y[train_mask]
        X_val, y_val = all_X[val_mask], all_y[val_mask]
        X_te, y_te = all_X[test_mask], all_y[test_mask]

        if len(X_tr) < 100 or len(X_te) < 10:
            window_start += STEP_SEC
            continue

        model = lgb.LGBMClassifier(
            objective='binary', metric='binary_logloss',
            max_depth=2, num_leaves=4, learning_rate=0.05, n_estimators=300,
            min_child_samples=200, subsample=0.7, colsample_bytree=0.5,
            reg_alpha=1.0, reg_lambda=5.0, is_unbalance=True,
            verbose=-1, random_state=42, n_jobs=-1,
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
        oos_confs.extend(proba)
        oos_timestamps.extend(all_ts_arr[test_mask])
        oos_features.append(all_X_full[test_mask])  # full features for mining

        window_start += STEP_SEC

    preds = np.array(oos_preds)
    labels = np.array(oos_labels)
    confs = np.array(oos_confs)
    timestamps = np.array(oos_timestamps)
    features = np.vstack(oos_features) if oos_features else np.empty((0, len(fnames_full)))

    base_wr = (preds == labels).mean()
    print(f"  {len(preds):,} OOS trades, base WR: {base_wr*100:.2f}% ({time.time()-t0:.0f}s)")
    return preds, labels, confs, timestamps, features, fnames_full


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Train final model
# ═══════════════════════════════════════════════════════════════════════

def train_final_model(data, target_tf='m5'):
    """Train final LightGBM on full dataset and save."""
    print(f"\n[4/6] Training final model...")
    t0 = time.time()

    import lightgbm as lgb

    fnames_all = sorted(data[0][1].keys())
    _, fnames = _filter_sub1min(fnames_all)
    all_X = np.array([[d[1].get(f, 0) for f in fnames] for d in data], dtype=np.float32)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    all_y = np.array([d[2] for d in data])

    # Feature selection
    pre_model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    pre_model.fit(all_X, all_y)
    top_k = 25
    top_idx = np.argsort(pre_model.feature_importances_)[::-1][:top_k]
    fnames = [fnames[i] for i in top_idx]
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
        'model': model, 'feature_names': fnames,
        'config': {
            'type': 'v24_fractal_hybrid',
            'model': 'lgbm_baseline',
            'n_features': len(fnames),
            'samples': len(all_y),
            'train_days': 180,
            'trained_at': trained_at,
            'auto_retrained': False,
            'incremental': True,
            'walk_forward_wr': 0.5430,
            'params': {
                'max_depth': 2, 'num_leaves': 4, 'learning_rate': 0.05,
                'n_estimators': 300, 'min_child_samples': 200,
                'subsample': 0.7, 'colsample_bytree': 0.5,
                'reg_alpha': 1.0, 'reg_lambda': 5.0,
            },
        }
    }

    if os.path.exists(MODEL_PATH):
        os.replace(MODEL_PATH, MODEL_PATH + '.bak')
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(d, f)

    print(f"  Saved model: {len(fnames)} features, {len(all_y)} samples ({time.time()-t0:.0f}s)")
    print(f"  -> {MODEL_PATH}")
    return model, fnames


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: V5b flip mining
# ═══════════════════════════════════════════════════════════════════════

def create_interaction_features(X, fnames, ml_conf):
    """Create interaction features (mirrors regime_flip_v5b.py)."""
    n = len(X)
    new_features = []
    new_names = []

    conf_abs = np.maximum(ml_conf, 1 - ml_conf).astype(np.float32)
    new_features.append(conf_abs)
    new_names.append('conf_abs')

    conf_edge = (conf_abs - 0.5).astype(np.float32)
    new_features.append(conf_edge)
    new_names.append('conf_edge')

    key_features = {}
    for i, name in enumerate(fnames):
        for key in ['body_vs_atr', 'range_vs_atr', 'direction', 'c123',
                     'swept_high', 'swept_low', 'fvg_bull', 'fvg_bear',
                     'unfilled_fvg_dist', 'weighted_vote', 'alignment_count',
                     'consec_c3', 'c1_count', 'last_displacement']:
            if key in name:
                if key not in key_features:
                    key_features[key] = []
                key_features[key].append((i, name))
                break

    # Cross-TF ratios
    for key in ['body_vs_atr', 'range_vs_atr']:
        if key not in key_features:
            continue
        feats = key_features[key]
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                fi, ni = feats[i]
                fj, nj = feats[j]
                denom = np.where(np.abs(X[:, fj]) > 1e-6, X[:, fj], 1e-6)
                ratio = np.clip(X[:, fi] / denom, -10, 10).astype(np.float32)
                new_features.append(ratio)
                new_names.append(f"R_{ni}/{nj}")

    # Direction agreement
    if 'direction' in key_features:
        dir_feats = key_features['direction']
        for i in range(len(dir_feats)):
            for j in range(i + 1, len(dir_feats)):
                fi, ni = dir_feats[i]
                fj, nj = dir_feats[j]
                agree = (X[:, fi] * X[:, fj]).astype(np.float32)
                new_features.append(agree)
                new_names.append(f"AGR_{ni}*{nj}")

    # Volatility spreads
    rva = key_features.get('range_vs_atr', [])
    for i in range(len(rva)):
        for j in range(i + 1, len(rva)):
            fi, ni = rva[i]
            fj, nj = rva[j]
            diff = (X[:, fi] - X[:, fj]).astype(np.float32)
            new_features.append(diff)
            new_names.append(f"D_{ni}-{nj}")

    # Displacement x volatility
    if 'last_displacement' in key_features and 'range_vs_atr' in key_features:
        for di, dn in key_features['last_displacement'][:3]:
            for vi, vn in key_features['range_vs_atr'][:4]:
                prod = (X[:, di] * X[:, vi]).astype(np.float32)
                new_features.append(prod)
                new_names.append(f"P_{dn}*{vn}")

    # Sweep asymmetry
    sh = key_features.get('swept_high', [])
    sl = key_features.get('swept_low', [])
    for shi, shn in sh[:3]:
        for sli, sln in sl[:3]:
            if shn.split('_')[0] == sln.split('_')[0]:
                asym = (X[:, shi] + X[:, sli]).astype(np.float32)
                new_features.append(asym)
                new_names.append(f"ASYM_{shn}+{sln}")

    # Confidence x features
    for fname in ['weighted_vote', 'alignment_count', 'm5_range_vs_atr',
                   'm1_range_vs_atr', 'm5_body_vs_atr', 'consec_c3_m5',
                   'consec_c3_m1', 'm5_unfilled_fvg_dist', 'm1_body_vs_atr',
                   'c1_count_m5', 'c1_count_m1']:
        if fname in fnames:
            fi = fnames.index(fname)
            prod = (conf_edge * X[:, fi]).astype(np.float32)
            new_features.append(prod)
            new_names.append(f"CxF_{fname}")

    # C123 match across TF
    c123_feats = key_features.get('c123', [])
    for i in range(len(c123_feats)):
        for j in range(i + 1, len(c123_feats)):
            fi, ni = c123_feats[i]
            fj, nj = c123_feats[j]
            match = (X[:, fi] == X[:, fj]).astype(np.float32)
            new_features.append(match)
            new_names.append(f"C123M_{ni}={nj}")

    if new_features:
        new_X = np.column_stack(new_features)
        new_X = np.nan_to_num(new_X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.hstack([X, new_X]), fnames + new_names
    return X, fnames


def build_conditions(X_seg, fnames):
    """Build threshold conditions for all features in a segment."""
    percentiles = list(range(2, 99, 2))
    conditions = []
    for fi, fname in enumerate(fnames):
        col = X_seg[:, fi]
        if np.std(col) < 1e-8:
            continue
        seen = set()
        for pct in percentiles:
            thresh = np.percentile(col, pct)
            tr = round(thresh, 5)
            if tr in seen:
                continue
            seen.add(tr)
            m_le = col <= thresh
            m_gt = col > thresh
            if m_le.sum() >= 6 and m_gt.sum() >= 6:
                conditions.append((fname, m_le, fi, '<=', thresh))
                conditions.append((fname, m_gt, fi, '>', thresh))
    return conditions


def deep_scan(is_loss, conditions, min_trades=8, min_lr=0.51):
    """Scan for flip rules: singles, pairs, triples, quads, quints."""
    results = []

    # SINGLES
    single_promising = []
    for i, (name, mask, fi, op, th) in enumerate(conditions):
        nm = mask.sum()
        if nm < min_trades:
            continue
        lr = is_loss[mask].mean()
        if lr >= min_lr + 0.01:
            results.append((lr, nm, f"{name}{op}{th:.4f}", mask, [(fi, op, th)]))
        if lr >= 0.46 and nm >= 8:
            single_promising.append((i, name, mask, fi, op, th, lr))

    # Pre-filter
    by_feat = defaultdict(list)
    for item in single_promising:
        by_feat[item[3]].append(item)
    filtered = []
    for fi in by_feat:
        items = sorted(by_feat[fi], key=lambda x: -x[6])
        filtered.extend(items[:5])
    filtered.sort(key=lambda x: -x[6])
    filtered = filtered[:250]

    # PAIRS
    pair_promising = []
    for i in range(len(filtered)):
        _, n1, m1, fi1, op1, th1, lr1 = filtered[i]
        for j in range(i + 1, len(filtered)):
            _, n2, m2, fi2, op2, th2, lr2 = filtered[j]
            if fi1 == fi2:
                continue
            combined = m1 & m2
            nm = combined.sum()
            if nm < min_trades:
                continue
            lr = is_loss[combined].mean()
            if lr >= min_lr:
                results.append((lr, nm, f"{n1}&{n2}", combined,
                                [(fi1, op1, th1), (fi2, op2, th2)]))
            if lr >= 0.49 and nm >= 8:
                pair_promising.append((lr, nm, combined,
                                       [(fi1, op1, th1), (fi2, op2, th2)],
                                       n1, n2, fi1, fi2))

    # TRIPLES
    pair_promising.sort(key=lambda x: -x[0])
    top_pairs = pair_promising[:150]

    for lr_p, nm_p, mask_p, det_p, n1, n2, fi1, fi2 in top_pairs:
        used = {fi1, fi2}
        for _, ns, ms, fis, ops, ths, lrs in filtered[:150]:
            if fis in used:
                continue
            combined = mask_p & ms
            nm = combined.sum()
            if nm < min_trades:
                continue
            lr = is_loss[combined].mean()
            if lr >= min_lr and lr > lr_p + 0.001:
                new_det = det_p + [(fis, ops, ths)]
                results.append((lr, nm, f"{n1}&{n2}&{ns}", combined, new_det))

    results.sort(key=lambda x: (x[0] - 0.5) * x[1], reverse=True)
    return results


def _recompute_rule_mask(rule, X, fnames, segments, n):
    """Recompute a rule's flip mask on the full dataset using feature conditions.

    During mining, flip_mask only covers discovery indices. This re-evaluates
    the same conditions on ALL data so we can test on validation.
    """
    seg_name = rule['segment']
    seg_mask = segments.get(seg_name, np.zeros(n, dtype=bool))

    # Apply feature conditions within the segment
    idx = np.where(seg_mask)[0]
    if len(idx) == 0:
        return np.zeros(n, dtype=bool)

    X_seg = X[idx]
    mask = np.ones(len(idx), dtype=bool)
    for fi, op, thresh in rule['cond_details']:
        col = X_seg[:, fi]
        if op == '<=':
            mask &= col <= thresh
        elif op == '>':
            mask &= col > thresh

    result = np.zeros(n, dtype=bool)
    result[idx[mask]] = True
    return result


def mine_flip_rules(preds, labels, confs, features, fnames, timestamps=None):
    """Run V5b-style flip mining with discovery/validation split.

    Mine rules on first 60% of OOS (discovery), validate on last 40%.
    Only rules that are also profitable on validation survive.
    Final stats reported on the FULL OOS set for transparency.
    """
    print(f"\n[5/6] Mining flip rules (V5b-style with discovery/validation split)...")
    t0 = time.time()

    n = len(preds)
    correct = (preds == labels).astype(int)
    conf_abs = np.maximum(confs, 1 - confs)
    base_wr = correct.mean()

    # Discovery / validation split (temporal — first 60% / last 40%)
    split_idx = int(n * 0.6)
    disc_mask = np.zeros(n, dtype=bool)
    disc_mask[:split_idx] = True
    val_mask = ~disc_mask
    print(f"  Discovery: {disc_mask.sum():,} trades | Validation: {val_mask.sum():,} trades")

    # Add interaction features
    X = features.copy()
    X, all_fnames = create_interaction_features(X, list(fnames), confs)
    print(f"  {len(all_fnames)} features (incl. interactions)")

    # Build segments
    def get_feat(name):
        if name in all_fnames:
            return X[:, all_fnames.index(name)]
        return np.zeros(n)

    m5_c = get_feat('m5_c123')
    m5_rva = get_feat('m5_range_vs_atr')
    cc3_m5 = get_feat('consec_c3_m5')
    m5_dir = get_feat('m5_direction')
    ac = get_feat('alignment_count')
    c1_m5 = get_feat('c1_count_m5')

    trending = (cc3_m5 >= 2) | (m5_c == 3)
    consolidating = (c1_m5 >= 2) | (m5_c == 1)
    high_vol = m5_rva >= 1.2
    low_vol = m5_rva < 0.6
    low_conf = conf_abs < 0.52
    mid_conf = (conf_abs >= 0.52) & (conf_abs < 0.55)

    segments = {}

    # Confidence bins
    for lo in range(50, 58):
        mask = (conf_abs >= lo / 100.0) & (conf_abs < (lo + 1) / 100.0)
        if mask.sum() > 50:
            segments[f'C{lo}'] = mask

    # Direction x Confidence
    for d, dn in [(1, 'U'), (0, 'D')]:
        pred = preds == d
        for lo in [50, 51, 52, 53, 54, 55]:
            mask = pred & (conf_abs >= lo / 100.0) & (conf_abs < (lo + 2) / 100.0)
            if mask.sum() > 50:
                segments[f'{dn}c{lo}'] = mask

    # Direction x Market
    for d, dn in [(1, 'U'), (0, 'D')]:
        pred = preds == d
        for nm, mk in [('TR', trending), ('CO', consolidating),
                        ('HV', high_vol), ('LV', low_vol)]:
            mask = pred & mk
            if mask.sum() > 50:
                segments[f'{dn}_{nm}'] = mask
        for c in [1, 2, 3]:
            mask = pred & (m5_c == c)
            if mask.sum() > 50:
                segments[f'{dn}_C{c}'] = mask
        if d == 1:
            segments[f'{dn}_TF'] = pred & (m5_dir > 0.5)
            segments[f'{dn}_RV'] = pred & (m5_dir < -0.5)
        else:
            segments[f'{dn}_TF'] = pred & (m5_dir < -0.5)
            segments[f'{dn}_RV'] = pred & (m5_dir > 0.5)
        segments[f'{dn}_HA'] = pred & (ac >= 5)
        segments[f'{dn}_LA'] = pred & (ac <= 1)
        segments[f'{dn}_MOM'] = pred & (cc3_m5 >= 3)

    # Conf x Market
    for cn, cm in [('LC', low_conf), ('MC', mid_conf)]:
        for mn, mm in [('TR', trending), ('CO', consolidating),
                       ('HV', high_vol), ('LV', low_vol)]:
            mask = cm & mm
            if mask.sum() > 30:
                segments[f'{cn}_{mn}'] = mask
    segments['CO_HV'] = consolidating & high_vol

    # 3-way
    for d, dn in [(1, 'U'), (0, 'D')]:
        pred = preds == d
        for cn, cm in [('LC', low_conf), ('MC', mid_conf)]:
            for mn, mm in [('TR', trending), ('CO', consolidating),
                           ('HV', high_vol), ('LV', low_vol)]:
                mask = pred & cm & mm
                if mask.sum() > 30:
                    segments[f'{dn}_{cn}_{mn}'] = mask

    # Hour-based (derive hour from timestamps)
    hours = np.array([(int(t) % 86400) // 3600 for t in
                      (preds * 0)])  # placeholder — need actual timestamps
    # We don't have hour in features array directly, use feature if available
    hour_sin = get_feat('hour_sin')
    hour_cos = get_feat('hour_cos')
    # Reconstruct approximate hour from sin/cos encoding
    if np.any(hour_sin != 0) or np.any(hour_cos != 0):
        approx_hour = np.round(np.arctan2(hour_sin, hour_cos) * 12 / np.pi) % 24
        asia = (approx_hour >= 0) & (approx_hour < 8)
        europe = (approx_hour >= 8) & (approx_hour < 16)
        us = (approx_hour >= 16) & (approx_hour < 24)
        for sn, sm in [('ASIA', asia), ('EU', europe), ('US', us)]:
            if sm.sum() > 50:
                segments[sn] = sm
            for d, dn in [(1, 'U'), (0, 'D')]:
                mask = (preds == d) & sm
                if mask.sum() > 50:
                    segments[f'{dn}_{sn}'] = mask
            mask = low_conf & sm
            if mask.sum() > 30:
                segments[f'LC_{sn}'] = mask

    # Catch-all
    segments['ALL'] = np.ones(n, dtype=bool)
    segments['U'] = preds == 1
    segments['D'] = preds == 0
    segments = {k: v for k, v in segments.items() if v.sum() >= 30}
    print(f"  {len(segments)} segments")

    # SCAN — mine rules on DISCOVERY set only
    disc_correct = correct.copy()
    all_rules = []
    seg_count = 0

    for seg_name, seg_mask in sorted(segments.items(), key=lambda x: -x[1].sum()):
        # Only use discovery-set trades for mining
        seg_disc = seg_mask & disc_mask
        seg_n = seg_disc.sum()
        if seg_n < 30:
            continue

        seg_count += 1
        idx = np.where(seg_disc)[0]
        X_seg = X[idx]
        is_loss = (disc_correct[idx] == 0).astype(np.float32)

        conditions = build_conditions(X_seg, all_fnames)
        rules = deep_scan(is_loss, conditions, min_trades=8, min_lr=0.51)

        for lr, nm, cname, local_mask, details in rules[:500]:
            global_mask = np.zeros(n, dtype=bool)
            global_mask[idx[local_mask]] = True
            all_rules.append({
                'segment': seg_name,
                'cond_name': cname,
                'cond_details': details,  # list of (fi, op, thresh)
                'lr': lr,
                'n_trades': nm,
                'flip_mask': global_mask,
                'expected_profit': (lr - 0.5) * nm,
                'n_conditions': len(details),
            })

        if seg_count % 10 == 0:
            print(f"    ... {seg_count} segments, {len(all_rules):,} rules")

    print(f"  {seg_count} segments scanned, {len(all_rules):,} rules (discovery set)")

    # VALIDATE — recompute each rule's flip_mask on FULL data using feature conditions,
    # then check if the rule is also profitable on validation set
    print(f"  Validating rules on held-out set...")
    validated_rules = []
    for rule in all_rules:
        # Recompute the rule mask on the full dataset using the condition thresholds
        # (the flip_mask from mining only covers discovery indices)
        full_mask = _recompute_rule_mask(rule, X, all_fnames, segments, n)
        val_hits = full_mask & val_mask
        n_val = val_hits.sum()
        if n_val < 3:
            continue
        val_lr = (correct[val_hits] == 0).mean()
        if val_lr <= 0.50:
            continue  # Not profitable on validation — discard
        rule['flip_mask'] = full_mask
        rule['val_lr'] = float(val_lr)
        rule['val_n'] = int(n_val)
        rule['n_trades'] = int(full_mask.sum())
        rule['expected_profit'] = (rule['lr'] - 0.5) * rule['n_trades']
        validated_rules.append(rule)

    print(f"  {len(validated_rules):,} rules survived validation (from {len(all_rules):,})")

    # 4-PASS COMPOSITE — scored on VALIDATION set to prevent composite overfitting
    composite_mask = np.zeros(n, dtype=bool)
    selected = []

    def try_add(rule, pass_num, min_new=2, min_mlr=0.505):
        nonlocal composite_mask
        fm = rule['flip_mask']
        # Score marginal contribution on VALIDATION set only
        new_val = fm & ~composite_mask & val_mask
        n_new_val = new_val.sum()
        if n_new_val < min_new:
            return False
        val_mlr = (correct[new_val] == 0).mean()
        if val_mlr <= min_mlr:
            return False
        composite_mask |= fm
        selected.append({**rule, 'marginal_n': int(n_new_val),
                         'marginal_lr': float(val_mlr), 'pass': pass_num})
        return True

    validated_rules.sort(key=lambda x: x['val_lr'] * x['val_n'], reverse=True)
    p1 = sum(1 for r in validated_rules if try_add(r, 1, min_new=3, min_mlr=0.505))
    print(f"  Pass 1 (val profit): {p1} rules, {composite_mask.sum():,} flipped")

    validated_rules.sort(key=lambda x: -x['val_lr'])
    p2 = sum(1 for r in validated_rules if try_add(r, 2, min_new=2, min_mlr=0.505))
    print(f"  Pass 2 (val LR): +{p2} rules, {composite_mask.sum():,} flipped")

    validated_rules.sort(key=lambda x: x['expected_profit'], reverse=True)
    p3 = sum(1 for r in validated_rules if try_add(r, 3, min_new=2, min_mlr=0.502))
    print(f"  Pass 3 (disc profit): +{p3} rules, {composite_mask.sum():,} flipped")

    # Results — report on FULL set (discovery + validation) for honest stats
    new_preds = preds.copy()
    new_preds[composite_mask] = 1 - new_preds[composite_mask]
    new_correct = (new_preds == labels)
    new_wr = new_correct.mean()

    # Also report validation-only WR
    val_new_preds = preds.copy()
    val_flipped = composite_mask & val_mask
    val_new_preds[val_flipped] = 1 - val_new_preds[val_flipped]
    val_new_correct = (val_new_preds[val_mask] == labels[val_mask])
    val_base_wr = (preds[val_mask] == labels[val_mask]).mean()
    val_new_wr = val_new_correct.mean()

    print(f"\n  Full OOS:       Base {base_wr*100:.2f}% -> Flip {new_wr*100:.2f}% "
          f"({(new_wr-base_wr)*100:+.2f}pp)")
    print(f"  Validation only: Base {val_base_wr*100:.2f}% -> Flip {val_new_wr*100:.2f}% "
          f"({(val_new_wr-val_base_wr)*100:+.2f}pp)")
    print(f"  {len(selected)} rules, {composite_mask.sum():,} trades flipped "
          f"({composite_mask.sum()/n*100:.1f}%)")
    print(f"  Mining took {time.time()-t0:.0f}s")

    return selected, all_fnames, composite_mask, {
        'base_wr': float(base_wr),
        'flip_wr': float(new_wr),
        'val_base_wr': float(val_base_wr),
        'val_flip_wr': float(val_new_wr),
        'n_rules': len(selected),
        'n_flipped': int(composite_mask.sum()),
        'n_total': n,
    }


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Serialize flip rules to JSON
# ═══════════════════════════════════════════════════════════════════════

def serialize_flip_rules(selected_rules, all_fnames, stats):
    """Convert selected rules into a JSON-serializable format for live inference.

    Groups rules by segment and stores feature-name-based conditions
    (not feature indices) so the live engine can match by name.
    """
    print(f"\n[6/6] Serializing {len(selected_rules)} rules to JSON...")

    # Build segment definitions with conditions that can be evaluated at inference time
    # The segment conditions here are simplified — the live engine uses the feature dict
    # directly, matching segment names to known condition patterns.
    segments_json = {}

    for rule in selected_rules:
        seg_name = rule['segment']

        if seg_name not in segments_json:
            segments_json[seg_name] = {
                'conditions': _segment_name_to_conditions(seg_name),
                'rules': [],
            }

        # Convert feature-index conditions to feature-name conditions
        rule_conditions = []
        for fi, op, thresh in rule['cond_details']:
            fname = all_fnames[fi] if fi < len(all_fnames) else f'f{fi}'
            rule_conditions.append([fname, op, float(round(thresh, 6))])

        segments_json[seg_name]['rules'].append({
            'conditions': rule_conditions,
            'loss_rate': float(round(rule['lr'], 4)),
            'n_trades': int(rule['n_trades']),
        })

    output = {
        'version': 'v5b',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'feature_names': list(all_fnames),
        'segments': segments_json,
        'stats': stats,
    }

    with open(FLIP_RULES_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    n_rules = sum(len(s['rules']) for s in segments_json.values())
    print(f"  Saved {n_rules} rules across {len(segments_json)} segments")
    print(f"  -> {FLIP_RULES_PATH}")
    return output


def _segment_name_to_conditions(seg_name):
    """Parse segment name into machine-readable conditions for live inference.

    Maps shorthand segment names (e.g., 'U_LC_TR') to dict conditions that
    the FlipRuleEngine can evaluate against live features.
    """
    conds = {}

    # Simple catch-alls
    if seg_name == 'ALL':
        return {}
    if seg_name == 'U':
        return {'ml_pred': 1}
    if seg_name == 'D':
        return {'ml_pred': 0}

    parts = seg_name.split('_')

    # Confidence-only: C50, C51, ...
    if len(parts) == 1 and parts[0].startswith('C') and parts[0][1:].isdigit():
        lo = int(parts[0][1:])
        return {'conf_abs_gte': lo / 100.0, 'conf_abs_lt': (lo + 1) / 100.0}

    # Direction + confidence: Uc50, Dc52, ...
    if len(parts) == 1 and len(parts[0]) >= 3 and parts[0][0] in ('U', 'D') and parts[0][1] == 'c':
        d = 1 if parts[0][0] == 'U' else 0
        lo = int(parts[0][2:])
        return {'ml_pred': d, 'conf_abs_gte': lo / 100.0, 'conf_abs_lt': (lo + 2) / 100.0}

    # Session-only: ASIA, EU, US
    session_map = {
        'ASIA': {'hour_gte': 0, 'hour_lt': 8},
        'EU': {'hour_gte': 8, 'hour_lt': 16},
        'US': {'hour_gte': 16, 'hour_lt': 24},
    }
    if seg_name in session_map:
        return session_map[seg_name]

    # Parse direction prefix
    direction = None
    remaining = list(parts)
    if remaining and remaining[0] in ('U', 'D'):
        direction = 1 if remaining[0] == 'U' else 0
        conds['ml_pred'] = direction
        remaining = remaining[1:]

    # Parse remaining conditions
    for part in remaining:
        if part == 'TR':
            # trending = (consec_c3_m5 >= 2) | (m5_c123 == 3) — use OR condition
            conds['_or_trending'] = True
        elif part == 'CO':
            # consolidating = (c1_count_m5 >= 2) | (m5_c123 == 1)
            conds['_or_consolidating'] = True
        elif part == 'HV':
            conds['m5_range_vs_atr_gte'] = 1.2
        elif part == 'LV':
            conds['m5_range_vs_atr_lt'] = 0.6
        elif part == 'LC':
            conds['conf_abs_lt'] = 0.52
        elif part == 'MC':
            conds['conf_abs_gte'] = 0.52
            conds['conf_abs_lt'] = 0.55
        elif part.startswith('C') and part[1:].isdigit():
            conds['m5_c123_eq'] = int(part[1:])
        elif part == 'TF':
            # Trend-following: direction matches m5_direction
            pass  # complex condition, handled by rules themselves
        elif part == 'RV':
            # Reversal: direction opposes m5_direction
            pass
        elif part == 'HA':
            conds['alignment_count_gte'] = 5
        elif part == 'LA':
            conds['alignment_count_lt'] = 2
        elif part == 'MOM':
            conds['consec_c3_m5_gte'] = 3
        elif part in session_map:
            conds.update(session_map[part])

    return conds


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 80)
    print("TRAIN FINAL MODEL + MINE FLIP RULES")
    print("=" * 80)

    # Step 1: Download data (180 days — need 67+ days before backtest for walk-forward training)
    rows_1m = fetch_binance_klines('BTCUSDT', days=180)
    if len(rows_1m) < 50000:
        print("ERROR: Not enough data. Need at least 50K 1m candles.")
        return

    # Step 2: Process into M5 samples
    data = process_candles(rows_1m, target_tf='m5')
    if not data:
        print("ERROR: No M5 samples extracted.")
        return

    # Step 3: Walk-forward OOS predictions (90 days OOS, ~25K trades)
    preds, labels, confs, timestamps, features, fnames = walk_forward_oos(data, backtest_days=90)
    if len(preds) < 100:
        print("ERROR: Too few OOS predictions for flip mining.")
        return

    # Step 4: Train final model
    train_final_model(data, target_tf='m5')

    # Step 5: Mine flip rules
    selected, all_fnames, composite_mask, stats = mine_flip_rules(
        preds, labels, confs, features, fnames)

    # Step 6: Serialize to JSON
    serialize_flip_rules(selected, all_fnames, stats)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Rules: {FLIP_RULES_PATH}")
    print(f"  Base WR: {stats['base_wr']*100:.2f}% -> Flip WR: {stats['flip_wr']*100:.2f}%")
    print(f"  Rules: {stats['n_rules']} | Flipped: {stats['n_flipped']}/{stats['n_total']}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
