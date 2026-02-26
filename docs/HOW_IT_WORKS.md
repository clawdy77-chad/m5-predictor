# How It Works

## Overview

The system predicts whether each BTC 5-minute candle will close UP (close > open) or DOWN. It runs continuously, making 288 predictions per day. The signal is small (54.30% WR) but extremely stable across time, and is monetized on Polymarket's binary crypto markets at ~$0.50 per share.

## 1. Feature Extraction — IncrementalTF

The core engine is `IncrementalTF` in `model/v24_backtest.py`. It processes raw 1-minute candles into a multi-timeframe feature set.

### Timeframe Aggregation

Raw 1m candles are interpolated into 8 timeframes via `interpolate_1m_to_subs()`:

| Timeframe | Seconds | Purpose |
|-----------|--------:|---------|
| 10s | 10 | Microstructure noise |
| 30s | 30 | Short-term momentum |
| m1 | 60 | Base resolution |
| m5 | 300 | **Target timeframe** |
| m15 | 900 | Structure |
| m30 | 1800 | Structure |
| h1 | 3600 | Bias |
| h4 | 14400 | Higher-timeframe bias |

Each timeframe maintains its own `IncrementalTF` instance tracking:

- **C1/C2/C3 candle classification**: Based on body-to-range ratio and direction
- **Fair Value Gaps (FVGs)**: Unfilled price gaps, tracked with configurable lookback
- **ATR**: Average true range for normalization
- **Swing detection**: Local highs/lows with lookback=3

### Feature Vector

`extract_features()` produces ~52 candidate features, of which the top 25 by importance are used:

- `weighted_vote` — Multi-TF fractal alignment score (see below)
- `alignment_count` — How many fractals agree on direction
- `m5_range_vs_atr`, `m1_range_vs_atr` — Volatility ratios
- `m5_body_vs_atr`, `m1_body_vs_atr` — Body size ratios
- `consec_c3_m5`, `consec_c3_m1` — Consecutive C3 (indecision) candle counts
- `m5_unfilled_fvg_dist` — Distance to nearest unfilled FVG
- `c1_count_m5`, `c1_count_m1` — C1 (strong directional) candle counts
- Per-fractal scores and various cross-TF ratios

## 2. Fractal Structure

Six fractal definitions (F5a through F5f) define multi-timeframe alignment patterns:

| Fractal | Bias TF | Structure TF | Entry TF | Weight |
|---------|---------|--------------|----------|-------:|
| F5a | h1 | m15 | m5 | 2.0 |
| F5b | h4 | h1 | m15 | 3.0 |
| F5c | m30 | m15 | m5 | 1.5 |
| F5d | h1 | m30 | m5 | 2.5 |
| F5e | h4 | m30 | m15 | 3.0 |
| F5f | h1 | m5 | m1 | 1.0 |

Each fractal checks if the bias, structure, and entry timeframes agree on direction. The `weighted_vote` feature is the weight-sum of aligned fractals, and `alignment_count` is how many fractals agree.

## 3. Model

LightGBM binary classifier with deliberately shallow trees:

```python
max_depth=2        # Very shallow — prevents overfitting
num_leaves=4       # Matches max_depth=2
learning_rate=0.05 # Moderate
n_estimators=300   # Moderate ensemble size
min_child_samples=200  # High — requires significant evidence per leaf
subsample=0.7      # 70% row sampling
colsample_bytree=0.5  # 50% feature sampling
reg_alpha=1.0      # L1 regularization
reg_lambda=5.0     # L2 regularization (strong)
```

The key insight is that shallow trees with heavy regularization prevent the model from memorizing market regimes. The signal is weak but universal — deep trees or complex models overfit to recent patterns and fail out-of-sample.

## 4. Walk-Forward Evaluation

The model is validated using strict temporal walk-forward:

```
|--- 53d train ---|--- 7d val ---|--- 7d embargo ---|--- 7d test ---|
                                                      ↑ OOS predictions
```

- **Train**: 53 days of data to fit the model
- **Validation**: 7 days for early stopping / hyperparameter selection
- **Embargo**: 7 days of dead zone to prevent information leakage
- **Test**: 7 days of truly out-of-sample predictions

The window slides forward by 7 days each step. All reported WR figures are from the test windows only — the model never sees test data during training.

Over 3 years (2023-02 to 2026-02), this produces 258,048 OOS trades with 54.30% WR.

### Stability

| Period | WR% | Trades |
|--------|----:|-------:|
| First half | 54.25 | 129,024 |
| Second half | 54.36 | 129,024 |
| 2023 | 55.13 | 32,853 |
| 2024 | 54.00 | 105,408 |
| 2025 | 54.34 | 105,120 |
| 2026 | 54.37 | 14,667 |

No degradation over time. The worst month (2024-03) still achieved 53.1%.

## 5. Flip Rules (V5b)

After walk-forward generates OOS predictions, `train_final_model.py` mines "flip rules" — conditions where the model's prediction should be reversed.

### How flip mining works

1. **Discovery split** (first 60% of OOS trades): Mine candidate rules
2. **Validation split** (last 40%): Verify rules generalize
3. Rules are defined as: "When the model predicts direction X, confidence is in range [a,b], and a feature or confidence-x-feature interaction crosses a threshold, flip to direction Y"
4. Only rules that improve WR on both splits survive

The current model has **167 rules across 62 segments**, stored in `model/flip_rules.json`.

### Inference

`bot/flip_engine.py` applies rules in <0.1ms per prediction:
- Segment conditions are pre-compiled to fast-check tuples
- Interaction features are computed from a precomputed recipe (zero string scanning)
- Segments are indexed by `ml_pred` direction (halves search space)

## 6. Live Trading Flow

1. `BinanceKlineStreamer` (WebSocket) streams 1m + 1s candles
2. `StreamingV24Predictor` maintains `IncrementalTF` state continuously
3. At each M5 boundary: `extract_features()` -> `model.predict_proba()` -> `FlipRuleEngine.apply()`
4. Direction + confidence -> place order on Polymarket CLOB
5. Resolve trade when next M5 candle closes (check via Binance REST)
6. Dashboard at `http://localhost:8088` shows live stats

### Timing

- **T-150s**: Refresh Polymarket market list
- **T-5s**: Burst-poll target market slug at 40ms intervals
- **T+0s**: Candle closes -> predict (~2ms) -> place bet
- **T+300s**: Resolve trade via Binance kline
