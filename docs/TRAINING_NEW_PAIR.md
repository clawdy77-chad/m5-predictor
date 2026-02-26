# Training a New Pair

Guide for adapting this system to assets other than BTC (e.g., ETH, SOL, XRP).

## Prerequisites

- 180+ days of 1-minute klines for the target pair
- Python environment with `lightgbm`, `numpy`, `scikit-learn`
- The `model/v24_backtest.py` engine (asset-agnostic)

## Step 1: Download Data

```python
from training.train_final_model import fetch_binance_klines

# Download 180 days of 1m klines
candles = fetch_binance_klines('ETHUSDT', days=180)
# Returns: list of [timestamp, open, high, low, close, volume]
```

For longer backtests, use `days=1095` (3 years). More data = more reliable WR estimates.

## Step 2: Process Candles

```python
from model.v24_backtest import IncrementalTF, TF_SECS, FVG_LOOKBACKS, extract_features, interpolate_1m_to_subs

# Initialize timeframe states
tfs = {}
for key, sec in TF_SECS.items():
    tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

# Feed candles and extract features at M5 boundaries
# (see train_final_model.py process_candles() for the full implementation)
```

The feature extraction is fully asset-agnostic — it operates on OHLCV candles with no BTC-specific logic. ATR and body/range ratios auto-normalize to the asset's volatility.

## Step 3: Walk-Forward Evaluation (Baseline)

Start with the baseline parameters:

```python
params = {
    'max_depth': 2,
    'num_leaves': 4,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
}
```

Walk-forward config:
- **train_days**: 53
- **val_days**: 7
- **embargo_days**: 7
- **test_days**: 7

Run `train_final_model.py` (edit the symbol at the top) and record the baseline WR.

## Step 4: Hyperparameter Grid Search

The 4-phase sequential search that proved most effective:

### Phase 1: Depth / Leaves / Learning Rate
```
max_depth:      [2, 3, 4]
num_leaves:     [4, 6, 8]
learning_rate:  [0.01, 0.03, 0.05, 0.1]
```

### Phase 2: Regularization (use best from Phase 1)
```
min_child_samples:  [100, 200, 300, 500]
reg_alpha:          [0.1, 1.0, 5.0]
reg_lambda:         [0.5, 2.0, 5.0, 10.0]
```

### Phase 3: Sampling (use best from Phase 2)
```
subsample:          [0.5, 0.6, 0.7, 0.8]
colsample_bytree:   [0.3, 0.5, 0.7]
```

### Phase 4: Ensemble Size (use best from Phase 3)
```
n_estimators:       [100, 200, 300, 500, 800]
```

For BTC, the optimal was identical to baseline except: `lr=0.03`, `min_child=300`, `reg_lambda=2.0` — a +0.38pp improvement.

## Step 5: Methodology Experiments (Optional)

If hyperparams don't yield enough improvement, try:

| Experiment | What to Try | BTC Finding |
|-----------|-------------|-------------|
| Train window | 45, 53, 60, 90 days | 60d gave +0.30pp on baseline |
| Decay weighting | exp(-lambda * age_days), lambda=0.005-0.02 | +0.16-0.20pp on baseline |
| Embargo length | 3, 5, 7 days | 3d gave +0.09pp |
| Feature count | Top 15, 20, 25, 30, ALL | 25 was optimal for BTC |

**Warning**: On BTC, improvements did NOT stack. The best single change (hyperparams) beat all combinations. Test stacking carefully.

## Step 6: Combined Verification

Take your best config and run the full walk-forward on all available data. Check:

1. **Overall WR** > 51% (minimum viable)
2. **H1 vs H2 stability**: Both halves > 51%
3. **Trade count** > 200K for 3-year data (ensure you're trading every candle)
4. **Monthly consistency**: No month below 50% ideally

## Step 7: Mine Flip Rules

Once you have a validated model, run the flip rule mining (V5b) in `train_final_model.py`. This can add additional edge by flipping incorrect predictions in specific feature regimes.

## Validation Criteria

Before deploying a new pair:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Overall WR | 51% | 53% | 54%+ |
| H1 WR | 51% | 53% | 54%+ |
| H2 WR | 51% | 53% | 54%+ |
| Trade count (3yr) | 200K | 250K | 258K |
| Green days % | 80% | 90% | 92%+ |
| Green weeks % | 90% | 95% | 99%+ |

## Anti-Patterns (What to Skip)

Based on 245 experiments on BTC:

1. **Don't try alternative models** — CatBoost, XGBoost, ensembles, stacking all underperform LightGBM
2. **Don't add interaction features in bulk** — Adding many at once degrades performance
3. **Don't use mutual information feature selection** — Terrible results (-2.58pp on BTC)
4. **Don't tune the prediction threshold** — 0.50 is optimal, any deviation hurts
5. **Don't use high decay rates** (>0.03) — Causes instability
6. **Don't use very short train windows** (<45 days) — Not enough data
7. **Don't expect improvements to stack** — Test combinations explicitly
