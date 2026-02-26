# M5 Crypto Direction Predictor

LightGBM-based system that predicts the direction (UP/DOWN) of 5-minute crypto candles. Designed for Polymarket binary options trading.

## Performance

**1-Year Backtest (Feb 2025 – Feb 2026) — 451,577 trades:**

| Asset | Trades | Win Rate | Annual PnL ($50 bets) |
|-------|--------|----------|-----------------------|
| BTC | 112,894 | 55.2% | +$470,202 |
| ETH | 112,894 | 55.3% | +$475,692 |
| SOL | 112,894 | 55.1% | +$455,104 |
| XRP | 112,895 | 53.8% | +$312,211 |
| **Combined** | **451,577** | **54.9%** | **+$1,713,209** |

- **13 consecutive profitable months** (54.3% – 56.7% WR)
- **94% green days** (BTC/SOL), 93% (ETH), 87% (XRP)
- Average daily PnL: +$120/asset at $5 bets
- Best day: +$413, Worst day: -$146

## How It Works

1. **Feature Extraction**: 8-timeframe incremental pipeline (10s → h4) computing fractal alignment, FVG distances, ATR ratios, and candle classification (25 features)
2. **Model**: LightGBM with shallow trees (`max_depth=2`), heavy regularization — prevents regime overfitting
3. **Walk-Forward Validation**: 53d train / 7d val / 7d embargo / 7d test sliding window
4. **Flip Rules**: Post-hoc rule mining that reverses predictions under specific conditions (167 rules, validated on held-out split)

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full technical breakdown.

## Project Structure

```
├── bot/                    # Live trading & monitoring
│   ├── paper_trader.py         # BTC paper trader (port 8088)
│   ├── paper_trader_alt.py     # ETH/XRP/SOL paper trader (port 8089)
│   ├── paper_trader_multi.py   # Multi-asset paper trader
│   ├── backtest_dashboard.py   # Interactive backtest dashboard (port 8090)
│   ├── dashboard.html          # Dashboard frontend
│   ├── verify_trades.py        # Live vs backtest determinism verification
│   └── v24_backtest.py         # Core feature extraction engine
├── models/                 # Trained model files
│   ├── btc_54_model.pkl        # BTC LightGBM model
│   ├── eth_model.pkl           # ETH model
│   ├── sol_model.pkl           # SOL model
│   ├── xrp_model.pkl           # XRP model
│   └── *_flip_rules.json       # Per-asset flip rules
├── training/               # Training & evaluation scripts
│   ├── train_final_model.py    # Main training pipeline
│   ├── train_multi_pair.py     # Multi-asset training
│   ├── backtest_1y.py          # 1-year BTC backtest
│   ├── backtest_1y_all.py      # 1-year all-asset backtest
│   ├── generate_backtest_data.py # 30-day backtest data generation
│   ├── week_simulation.py      # Weekly simulation
│   └── month_simulation.py     # Monthly simulation
├── helpers/                # Utilities
│   ├── binance_wss.py          # Binance WebSocket streamer
│   ├── binance_futures.py      # Binance REST API
│   ├── polymarket_api.py       # Polymarket CLOB client
│   └── training_logger.py      # Training metrics logger
├── deploy/                 # Systemd services & scripts
├── docs/                   # Documentation
│   ├── HOW_IT_WORKS.md         # Full technical documentation
│   └── TRAINING_NEW_PAIR.md    # Guide for adding new assets
├── validation/             # Historical validation data
├── data/                   # Runtime data (logs, klines)
└── requirements.txt
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Paper Trading (BTC)

```bash
python bot/paper_trader.py --port 8088
```

Dashboard at `http://localhost:8088`

### Run Backtest Dashboard

```bash
python bot/backtest_dashboard.py
```

Dashboard at `http://localhost:8090`

### Train a New Model

```bash
python training/train_final_model.py --symbol BTCUSDT --days 180
```

See [docs/TRAINING_NEW_PAIR.md](docs/TRAINING_NEW_PAIR.md) for adding new assets.

## Key Design Decisions

- **Shallow trees + heavy regularization** over deep/complex models — the signal is weak but universal
- **100% deterministic pipeline** — live and backtest produce identical trades (verified on 8,056+ trades)
- **No hyperparameter tuning** — baseline params are robust enough, tuning risks overfitting
- **Walk-forward validation only** — no random splits, no lookahead bias
- **Model doesn't need frequent retraining** — 1-year backtest shows stable WR with no degradation

## Bet Structure (Polymarket)

- Limit price: $0.51 (implied ~50/50 odds)
- Win payout: $0.49 per $0.51 risked (96% return)
- Loss: -$0.51 per share
- At $50 bets: win +$48.04, lose -$50.00
- Edge comes from WR > 51% on a 50/50 market

## Requirements

- Python 3.10+
- LightGBM, scikit-learn, NumPy, Flask
- Binance API access (no key needed for public klines)
- Polymarket account + API key (for live trading only)
