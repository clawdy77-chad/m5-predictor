#!/usr/bin/env python3
"""
Paper Trader — Live forward-test of a validated model with zero real money.

Streams Binance 1m+1s klines via WebSocket, feeds IncrementalTF state identically
to the backtest, predicts at each M5 boundary, then resolves the trade when the
next M5 candle closes. Tracks WR, streaks, PnL simulation, and logs everything.

Usage:
    python paper_trader.py
    python paper_trader.py --asset ethusdt
    python paper_trader.py --model path/to/model.pkl
    python paper_trader.py --bet-size 50

Dashboard: http://localhost:8099
"""
import asyncio
import json
import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime, timezone
from collections import deque

# ─── CLI Args ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # validated_model/
MODEL_PATH = os.path.join(_ROOT, 'model', 'btc_54_model.pkl')
ASSET = 'btcusdt'
BET_SIZE = 50.0       # simulated bet size per trade
LIMIT_PRICE = 0.51    # Polymarket-style odds for PnL calc
PORT = 8099

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--asset' and i < len(sys.argv) - 1:
        ASSET = sys.argv[i + 1].lower()
    elif arg == '--model' and i < len(sys.argv) - 1:
        MODEL_PATH = sys.argv[i + 1]
    elif arg == '--bet-size' and i < len(sys.argv) - 1:
        BET_SIZE = float(sys.argv[i + 1])
    elif arg == '--port' and i < len(sys.argv) - 1:
        PORT = int(sys.argv[i + 1])

# ─── Imports (after path setup) ──────────────────────────────────────
sys.path.insert(0, _ROOT)
from helpers import BinanceKlineStreamer
import importlib.util

# ─── Load backtest machinery ─────────────────────────────────────────
backtest_path = os.path.join(_ROOT, 'model', 'v24_backtest.py')
spec = importlib.util.spec_from_file_location('v24_backtest', backtest_path)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
IncrementalTF = bt.IncrementalTF
extract_features = bt.extract_features
interpolate_1m_to_subs = bt.interpolate_1m_to_subs
TF_SECS = bt.TF_SECS
FVG_LOOKBACKS = bt.FVG_LOOKBACKS

# ─── Load model ──────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
feature_names = model_data['feature_names']
config = model_data.get('config', {})
print(f"Loaded model: {config.get('model', '?')} | {len(feature_names)} features | "
      f"WF WR: {config.get('walk_forward_wr', '?')}%")

# ─── State ────────────────────────────────────────────────────────────
import threading
lock = threading.Lock()

tfs = {}
for key, sec in TF_SECS.items():
    tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

m5_count = 0
boundary_features = None
bf_ts = 0.0
bf_m5_ts = 0

# Pending prediction waiting for resolution
pending = None  # dict: {direction, confidence, ts, m5_count}

# Trade log
STARTING_EQUITY = 1000.0
trades = deque(maxlen=10000)
equity_curve = [STARTING_EQUITY]  # equity after each trade
wins = 0
losses = 0
total_pnl = 0.0
current_streak = 0
max_win_streak = 0
max_lose_streak = 0
session_start = None

log_lines = deque(maxlen=500)

# ─── Kline logging for reproducibility ────────────────────────────────
KLINE_LOG_DIR = os.path.join(_ROOT, 'logs')
os.makedirs(KLINE_LOG_DIR, exist_ok=True)
_kline_log_path = os.path.join(KLINE_LOG_DIR, f'klines_{ASSET}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.jsonl')
_kline_log_file = open(_kline_log_path, 'a')

def _log_kline(ts, o, h, l, c, v):
    """Append every 1m kline to a JSONL file for exact replay."""
    _kline_log_file.write(json.dumps([ts, o, h, l, c, v]) + '\n')
    _kline_log_file.flush()

def log(msg):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)


def on_1m_kline(ts, o, h, l, c, v):
    global m5_count, boundary_features, bf_ts, bf_m5_ts, pending

    # Log raw kline for reproducibility
    _log_kline(ts, o, h, l, c, v)

    with lock:
        prev_m5 = tfs['m5'].n

        # Feed LTFs first (match backtest order)
        for key in ['m1', 'm3', 'm5']:
            tfs[key].feed_1m(ts, o, h, l, c, v)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
            tfs['10s'].feed_sub(*sub)
        for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
            tfs['30s'].feed_sub(*sub)

        m5_count = tfs['m5'].n

        # M5 boundary just crossed
        if m5_count > prev_m5 and m5_count > 50:
            # 1. Resolve pending trade from previous candle
            if pending is not None:
                actual_dir = 'UP' if tfs['m5'].closes[-1] > tfs['m5'].opens[-1] else 'DOWN'
                resolve_trade(pending, actual_dir)
                pending = None

            # 2. Snapshot features for new prediction
            boundary_features = extract_features(tfs, target_tf='m5')
            bf_ts = time.monotonic()
            bf_m5_ts = tfs['m5'].timestamps[-1]  # M5 candle open timestamp

        # Feed HTFs AFTER snapshot
        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)


def on_1s_kline(ts, o, h, l, c, v):
    """1s klines are no longer fed to TFs — they caused divergence from backtest.
    Kept as a no-op so the streamer callback doesn't error."""
    pass


def resolve_trade(trade, actual_dir):
    global wins, losses, total_pnl, current_streak, max_win_streak, max_lose_streak

    won = trade['direction'] == actual_dir
    pnl = BET_SIZE * ((1.0 - LIMIT_PRICE) / LIMIT_PRICE) if won else -BET_SIZE

    if won:
        wins += 1
        current_streak = max(1, current_streak + 1)
        max_win_streak = max(max_win_streak, current_streak)
    else:
        losses += 1
        current_streak = min(-1, current_streak - 1)
        max_lose_streak = max(max_lose_streak, abs(current_streak))

    total_pnl += pnl
    equity_curve.append(STARTING_EQUITY + total_pnl)
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0

    result = 'WIN' if won else 'LOSS'
    streak_str = f"+{current_streak}" if current_streak > 0 else str(current_streak)

    # Use M5 candle close time (open + 5min) as the prediction timestamp
    # This is deterministic and matches the backtest exactly
    pred_ts = datetime.fromtimestamp(trade['m5_ts'] + 300, tz=timezone.utc).isoformat()
    trades.append({
        'ts': pred_ts,
        'direction': trade['direction'],
        'confidence': trade['confidence'],
        'actual': actual_dir,
        'result': result,
        'pnl': round(pnl, 2),
    })

    log(f"[{result}] Predicted {trade['direction']} ({trade['confidence']:.1%}) | "
        f"Actual {actual_dir} | PnL: ${pnl:+.2f} | "
        f"Equity: ${STARTING_EQUITY + total_pnl:.2f} | "
        f"{wr:.1f}% WR ({wins}W/{losses}L)")


def predict():
    """Run model prediction on current boundary features."""
    global pending

    with lock:
        if boundary_features is None:
            return
        features = boundary_features
        m5_ts = bf_m5_ts

    X = np.array([[features.get(f, 0) for f in feature_names]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    proba = model.predict_proba(X)[0]
    conf = float(proba[1])
    direction = 'UP' if conf >= 0.5 else 'DOWN'
    disp_conf = conf if direction == 'UP' else 1.0 - conf

    with lock:
        pending = {
            'direction': direction,
            'confidence': disp_conf,
            'ts': time.time(),
            'm5_ts': m5_ts,  # M5 candle open timestamp for deterministic logging
            'm5_count': m5_count,
        }

    log(f"[PREDICT] {direction} ({disp_conf:.1%}) | raw_p1={conf:.4f} | M5 #{m5_count}")


# ─── Flask Dashboard ─────────────────────────────────────────────────
from flask import Flask, Response
app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<title>Paper Trader</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; color: #fff; font-family: -apple-system, 'Segoe UI', sans-serif; }
  .container { max-width: 720px; margin: 0 auto; padding: 48px 24px; }
  .header { margin-bottom: 48px; }
  .header h1 { font-size: 14px; font-weight: 400; color: #666; letter-spacing: 0.05em; text-transform: uppercase; }
  .equity { margin-bottom: 48px; }
  .equity-value { font-size: 48px; font-weight: 300; letter-spacing: -0.02em; }
  .equity-label { font-size: 12px; color: #666; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .stats { display: flex; gap: 48px; margin-bottom: 48px; }
  .stat .label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .stat .value { font-size: 24px; font-weight: 300; }
  .chart-container { margin-bottom: 24px; }
  .chart-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  canvas { width: 100%; height: 200px; display: block; }
  .dim { color: #444; }
</style>
</head><body>
<div class="container">
  <div class="header">
    <h1>Paper Trader &middot; BTC</h1>
  </div>
  <div class="equity">
    <div class="equity-label">Equity</div>
    <div class="equity-value" id="equity">$--</div>
  </div>
  <div class="stats">
    <div class="stat">
      <div class="label">Win Rate</div>
      <div class="value" id="wr">--%</div>
    </div>
    <div class="stat">
      <div class="label">PnL</div>
      <div class="value" id="pnl">$--</div>
    </div>
    <div class="stat">
      <div class="label">Trades</div>
      <div class="value" id="trades">0</div>
    </div>
  </div>
  <div class="chart-container">
    <div class="chart-label">Equity Curve</div>
    <canvas id="chart"></canvas>
  </div>
</div>
<script>
const STARTING = __STARTING_EQUITY__;

function draw(curve) {
  const canvas = document.getElementById('chart');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  ctx.clearRect(0, 0, W, H);

  if (curve.length < 2) {
    ctx.fillStyle = '#333';
    ctx.font = '13px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for trades...', W / 2, H / 2);
    return;
  }

  const min = Math.min(...curve);
  const max = Math.max(...curve);
  const pad = Math.max((max - min) * 0.1, 1);
  const lo = min - pad, hi = max + pad;

  // Starting equity line
  const startY = H - ((STARTING - lo) / (hi - lo)) * H;
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, startY);
  ctx.lineTo(W, startY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Equity line
  const last = curve[curve.length - 1];
  ctx.strokeStyle = last >= STARTING ? '#fff' : '#666';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < curve.length; i++) {
    const x = (i / (curve.length - 1)) * W;
    const y = H - ((curve[i] - lo) / (hi - lo)) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current value dot
  const lastX = W;
  const lastY = H - ((last - lo) / (hi - lo)) * H;
  ctx.fillStyle = last >= STARTING ? '#fff' : '#666';
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
  ctx.fill();

  // Y-axis labels
  ctx.fillStyle = '#444';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('$' + hi.toFixed(0), 4, 12);
  ctx.fillText('$' + lo.toFixed(0), 4, H - 4);
}

async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    const eq = (STARTING + d.pnl);
    document.getElementById('equity').textContent = '$' + eq.toFixed(2);
    document.getElementById('wr').textContent = d.total > 0 ? (d.wr * 100).toFixed(1) + '%' : '--%';
    document.getElementById('pnl').textContent = (d.pnl >= 0 ? '+' : '') + '$' + d.pnl.toFixed(2);
    document.getElementById('trades').textContent = d.total;
    draw(d.equity_curve);
  } catch(e) {}
}

refresh();
setInterval(refresh, 5000);
window.addEventListener('resize', refresh);
</script>
</body></html>"""

@app.route('/')
def dashboard():
    html = DASHBOARD_HTML.replace('__STARTING_EQUITY__', f'{STARTING_EQUITY:.2f}')
    return Response(html, content_type='text/html')

@app.route('/api/stats')
def api_stats():
    total = wins + losses
    return json.dumps({
        'wr': wins / total if total > 0 else 0,
        'wins': wins, 'losses': losses, 'total': total,
        'pnl': round(total_pnl, 2),
        'equity_curve': list(equity_curve),
    })

@app.route('/api/trades')
def api_trades():
    return json.dumps(list(trades))

@app.route('/api/kline_log')
def api_kline_log():
    return json.dumps({'path': _kline_log_path})


# ─── Main Loop ────────────────────────────────────────────────────────
async def main():
    global session_start

    print(f"\n{'=' * 60}")
    print(f"  PAPER TRADER — {ASSET.upper()}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Bet size: ${BET_SIZE:.2f} | Dashboard: http://localhost:{PORT}")
    print(f"{'=' * 60}\n")

    session_start = datetime.now(timezone.utc)

    # Start Flask in background thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    log(f"Dashboard running at http://localhost:{PORT}")

    # Start kline streamer
    streamer = BinanceKlineStreamer(symbol=ASSET)
    streamer.on_kline_1m(on_1m_kline)
    streamer.on_kline_1s(on_1s_kline)

    log("Warming up with historical klines...")
    await streamer.warmup_from_rest(800)
    log(f"Warmup done. M5 count: {m5_count}. Waiting for first M5 boundary...")

    # Start WebSocket stream
    stream_task = asyncio.create_task(streamer.stream())

    # Main loop: check for new M5 boundaries and predict
    last_predicted_m5 = m5_count

    while True:
        await asyncio.sleep(1)

        current_m5 = m5_count
        if current_m5 > last_predicted_m5 and current_m5 > 50 and boundary_features is not None:
            last_predicted_m5 = current_m5
            predict()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        print(f"\n\n{'=' * 50}")
        print(f"  SESSION SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Trades:  {total}")
        print(f"  WR:      {wr:.1f}% ({wins}W / {losses}L)")
        print(f"  PnL:     ${total_pnl:+.2f}")
        print(f"  Streaks: +{max_win_streak} / -{max_lose_streak}")
        print(f"{'=' * 50}")
