#!/usr/bin/env python3
"""
Multi-Asset Paper Trader — Runs 4 independent paper traders (BTC, ETH, XRP, SOL)
in a single process with a combined dashboard.
"""
import asyncio
import json
import os
import sys
import time
import pickle
import numpy as np
import threading
from datetime import datetime, timezone
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from helpers import BinanceKlineStreamer
import importlib.util

backtest_path = os.path.join(_ROOT, 'model', 'v24_backtest.py')
spec = importlib.util.spec_from_file_location('v24_backtest', backtest_path)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
IncrementalTF = bt.IncrementalTF
extract_features = bt.extract_features
interpolate_1m_to_subs = bt.interpolate_1m_to_subs
TF_SECS = bt.TF_SECS
FVG_LOOKBACKS = bt.FVG_LOOKBACKS

BET_SIZE = 50.0
LIMIT_PRICE = 0.51
PORT = 8089

ASSETS = {
    'ethusdt': os.path.join(_ROOT, 'model', 'eth_model.pkl'),
    'xrpusdt': os.path.join(_ROOT, 'model', 'xrp_model.pkl'),
    'solusdt': os.path.join(_ROOT, 'model', 'sol_model.pkl'),
}

# Seed trades from previous session (first 9 trades per asset)
SEED_TRADES = {
    'ethusdt': [
        {'ts':'2026-02-25T11:35:00+00:00','direction':'DOWN','confidence':0.55,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:40:00+00:00','direction':'DOWN','confidence':0.54,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T11:45:00+00:00','direction':'DOWN','confidence':0.53,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T11:50:00+00:00','direction':'DOWN','confidence':0.52,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:55:00+00:00','direction':'UP','confidence':0.51,'actual':'UP','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:00:00+00:00','direction':'DOWN','confidence':0.53,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:05:00+00:00','direction':'UP','confidence':0.52,'actual':'UP','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:10:00+00:00','direction':'DOWN','confidence':0.51,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T12:15:00+00:00','direction':'UP','confidence':0.52,'actual':'UP','result':'WIN','pnl':48.04},
    ],
    'xrpusdt': [
        {'ts':'2026-02-25T11:35:00+00:00','direction':'DOWN','confidence':0.54,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:40:00+00:00','direction':'DOWN','confidence':0.53,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:45:00+00:00','direction':'DOWN','confidence':0.52,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:50:00+00:00','direction':'DOWN','confidence':0.51,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:55:00+00:00','direction':'DOWN','confidence':0.53,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T12:00:00+00:00','direction':'DOWN','confidence':0.52,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:05:00+00:00','direction':'UP','confidence':0.51,'actual':'DOWN','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T12:10:00+00:00','direction':'UP','confidence':0.54,'actual':'UP','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:15:00+00:00','direction':'DOWN','confidence':0.52,'actual':'UP','result':'LOSS','pnl':-50.0},
    ],
    'solusdt': [
        {'ts':'2026-02-25T11:35:00+00:00','direction':'DOWN','confidence':0.53,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:40:00+00:00','direction':'DOWN','confidence':0.52,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T11:45:00+00:00','direction':'DOWN','confidence':0.54,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:50:00+00:00','direction':'DOWN','confidence':0.51,'actual':'UP','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T11:55:00+00:00','direction':'UP','confidence':0.52,'actual':'UP','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:00:00+00:00','direction':'DOWN','confidence':0.53,'actual':'DOWN','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:05:00+00:00','direction':'UP','confidence':0.51,'actual':'UP','result':'WIN','pnl':48.04},
        {'ts':'2026-02-25T12:10:00+00:00','direction':'UP','confidence':0.52,'actual':'DOWN','result':'LOSS','pnl':-50.0},
        {'ts':'2026-02-25T12:15:00+00:00','direction':'UP','confidence':0.53,'actual':'UP','result':'WIN','pnl':48.04},
    ],
}

# ─── Per-asset state ──────────────────────────────────────────────────
class AssetTrader:
    def __init__(self, symbol, model_path):
        self.symbol = symbol
        self.lock = threading.Lock()
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        config = model_data.get('config', {})
        print(f"[{symbol.upper()}] Loaded model: {len(self.feature_names)} features | "
              f"WF WR: {config.get('walk_forward_wr', '?')}")
        
        # TF state
        self.tfs = {}
        for key, sec in TF_SECS.items():
            self.tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)
        
        self.m5_count = 0
        self.boundary_features = None
        self.bf_m5_ts = 0
        self.pending = None
        
        # Trade log — seed with historical trades if available
        self.trades = deque(maxlen=10000)
        self.equity_curve = [1000.0]
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        
        seed = SEED_TRADES.get(symbol, [])
        for st in seed:
            won = st['result'] == 'WIN'
            if won:
                self.wins += 1
            else:
                self.losses += 1
            self.total_pnl += st['pnl']
            self.equity_curve.append(1000.0 + self.total_pnl)
            self.trades.append(st)
        if seed:
            total = self.wins + self.losses
            wr = self.wins / total * 100 if total > 0 else 0
            print(f"[{symbol.upper()}] Seeded {len(seed)} trades: {wr:.1f}% WR ({self.wins}W/{self.losses}L) | ${self.total_pnl:+.2f}")
        
        # Kline logging
        log_dir = os.path.join(_ROOT, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self._kline_log_path = os.path.join(
            log_dir, f'klines_{symbol}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.jsonl')
        self._kline_log_file = open(self._kline_log_path, 'a')
    
    def log_kline(self, ts, o, h, l, c, v):
        self._kline_log_file.write(json.dumps([ts, o, h, l, c, v]) + '\n')
        self._kline_log_file.flush()
    
    def on_1m_kline(self, ts, o, h, l, c, v):
        self.log_kline(ts, o, h, l, c, v)
        
        with self.lock:
            prev_m5 = self.tfs['m5'].n
            
            for key in ['m1', 'm3', 'm5']:
                self.tfs[key].feed_1m(ts, o, h, l, c, v)
            for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
                self.tfs['10s'].feed_sub(*sub)
            for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
                self.tfs['30s'].feed_sub(*sub)
            
            self.m5_count = self.tfs['m5'].n
            
            if self.m5_count > prev_m5 and self.m5_count > 50:
                if self.pending is not None:
                    actual_dir = 'UP' if self.tfs['m5'].closes[-1] > self.tfs['m5'].opens[-1] else 'DOWN'
                    self.resolve_trade(self.pending, actual_dir)
                    self.pending = None
                
                self.boundary_features = extract_features(self.tfs, target_tf='m5')
                self.bf_m5_ts = self.tfs['m5'].timestamps[-1]
            
            for key in ['m15', 'm30', 'h1', 'h4']:
                self.tfs[key].feed_1m(ts, o, h, l, c, v)
    
    def predict(self):
        with self.lock:
            if self.boundary_features is None:
                return False
            features = self.boundary_features
            m5_ts = self.bf_m5_ts
            current_m5 = self.m5_count
        
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(X)[0]
        conf = float(proba[1])
        direction = 'UP' if conf >= 0.5 else 'DOWN'
        disp_conf = conf if direction == 'UP' else 1.0 - conf
        
        with self.lock:
            self.pending = {
                'direction': direction,
                'confidence': disp_conf,
                'm5_ts': m5_ts,
                'm5_count': current_m5,
            }
            self.boundary_features = None
        
        ts_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
        print(f"[{ts_str}] [{self.symbol.upper()}] {direction} ({disp_conf:.1%}) | M5 #{current_m5}")
        return True
    
    def resolve_trade(self, trade, actual_dir):
        won = trade['direction'] == actual_dir
        pnl = BET_SIZE * ((1.0 - LIMIT_PRICE) / LIMIT_PRICE) if won else -BET_SIZE
        
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        self.total_pnl += pnl
        self.equity_curve.append(1000.0 + self.total_pnl)
        total = self.wins + self.losses
        wr = self.wins / total * 100 if total > 0 else 0
        
        pred_ts = datetime.fromtimestamp(trade['m5_ts'] + 300, tz=timezone.utc).isoformat()
        self.trades.append({
            'ts': pred_ts,
            'direction': trade['direction'],
            'confidence': trade['confidence'],
            'actual': actual_dir,
            'result': 'WIN' if won else 'LOSS',
            'pnl': round(pnl, 2),
        })
        
        result = 'WIN' if won else 'LOSS'
        ts_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
        print(f"[{ts_str}] [{self.symbol.upper()}] [{result}] {trade['direction']} ({trade['confidence']:.1%}) | "
              f"Actual {actual_dir} | ${pnl:+.2f} | Eq ${1000+self.total_pnl:.2f} | {wr:.1f}% ({self.wins}W/{self.losses}L)")
    
    def get_stats(self):
        total = self.wins + self.losses
        return {
            'symbol': self.symbol,
            'wr': self.wins / total if total > 0 else 0,
            'wins': self.wins,
            'losses': self.losses,
            'total': total,
            'pnl': round(self.total_pnl, 2),
            'equity_curve': list(self.equity_curve),
            'kline_log': self._kline_log_path,
        }


# ─── Initialize traders ──────────────────────────────────────────────
traders = {}
for symbol, model_path in ASSETS.items():
    traders[symbol] = AssetTrader(symbol, model_path)


# ─── Flask Dashboard ─────────────────────────────────────────────────
from flask import Flask, Response
app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<title>Paper Trader — ETH/XRP/SOL</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; color: #fff; font-family: -apple-system, 'Segoe UI', sans-serif; }
  .container { max-width: 800px; margin: 0 auto; padding: 32px 24px; }
  .header h1 { font-size: 14px; font-weight: 400; color: #666; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 32px; }
  .combined { margin-bottom: 48px; }
  .combined .equity-value { font-size: 48px; font-weight: 300; letter-spacing: -0.02em; }
  .combined .label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .combined .stats { display: flex; gap: 48px; margin-top: 16px; }
  .combined .stat .value { font-size: 24px; font-weight: 300; }
  .assets { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .asset { border: 1px solid #222; padding: 20px; border-radius: 4px; }
  .asset .name { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .asset .wr { font-size: 28px; font-weight: 300; }
  .asset .detail { font-size: 12px; color: #555; margin-top: 4px; }
  .dim { color: #444; }
</style>
</head><body>
<div class="container">
  <div class="header"><h1>Paper Trader &middot; Multi-Asset</h1></div>
  <div class="combined">
    <div class="label">Combined Equity</div>
    <div class="equity-value" id="equity">$--</div>
    <div class="stats">
      <div class="stat"><div class="label">Win Rate</div><div class="value" id="wr">--%</div></div>
      <div class="stat"><div class="label">PnL</div><div class="value" id="pnl">$--</div></div>
      <div class="stat"><div class="label">Trades</div><div class="value" id="trades">0</div></div>
    </div>
  </div>
  <div class="assets" id="assets"></div>
</div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    const eq = 4000 + d.combined.pnl;
    document.getElementById('equity').textContent = '$' + eq.toFixed(2);
    document.getElementById('wr').textContent = d.combined.total > 0 ? (d.combined.wr * 100).toFixed(1) + '%' : '--%';
    document.getElementById('pnl').textContent = (d.combined.pnl >= 0 ? '+' : '') + '$' + d.combined.pnl.toFixed(2);
    document.getElementById('trades').textContent = d.combined.total;
    
    let html = '';
    for (const [sym, s] of Object.entries(d.assets)) {
      const wr = s.total > 0 ? (s.wr * 100).toFixed(1) + '%' : '--';
      const pnl = (s.pnl >= 0 ? '+' : '') + '$' + s.pnl.toFixed(2);
      html += '<div class="asset"><div class="name">' + sym.toUpperCase().replace('USDT','') + '</div>'
            + '<div class="wr">' + wr + '</div>'
            + '<div class="detail">' + s.wins + 'W/' + s.losses + 'L &middot; ' + pnl + '</div></div>';
    }
    document.getElementById('assets').innerHTML = html;
  } catch(e) {}
}
refresh();
setInterval(refresh, 5000);
</script>
</body></html>"""

@app.route('/')
def dashboard():
    return Response(DASHBOARD_HTML, content_type='text/html')

@app.route('/api/stats')
def api_stats():
    assets = {}
    combined_wins = 0
    combined_losses = 0
    combined_pnl = 0.0
    
    for symbol, trader in traders.items():
        s = trader.get_stats()
        assets[symbol] = s
        combined_wins += s['wins']
        combined_losses += s['losses']
        combined_pnl += s['pnl']
    
    total = combined_wins + combined_losses
    return json.dumps({
        'assets': assets,
        'combined': {
            'wr': combined_wins / total if total > 0 else 0,
            'wins': combined_wins,
            'losses': combined_losses,
            'total': total,
            'pnl': round(combined_pnl, 2),
        }
    })

@app.route('/api/trades')
def api_trades():
    all_trades = []
    for symbol, trader in traders.items():
        for t in trader.trades:
            all_trades.append({**t, 'symbol': symbol})
    all_trades.sort(key=lambda x: x['ts'])
    return json.dumps(all_trades)

@app.route('/api/trades/<symbol>')
def api_trades_symbol(symbol):
    trader = traders.get(symbol.lower())
    if not trader:
        return json.dumps([])
    return json.dumps(list(trader.trades))

@app.route('/api/kline_log')
def api_kline_log():
    logs = {}
    for symbol, trader in traders.items():
        logs[symbol] = trader._kline_log_path
    return json.dumps(logs)


# ─── Main Loop ────────────────────────────────────────────────────────
async def main():
    print(f"\n{'='*60}")
    print(f"  ALT PAPER TRADER (ETH/XRP/SOL)")
    print(f"  Assets: {', '.join(s.upper() for s in ASSETS.keys())}")
    print(f"  Bet size: ${BET_SIZE:.2f} | Dashboard: http://localhost:{PORT}")
    print(f"{'='*60}\n")
    
    # Start Flask
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"Dashboard running at http://localhost:{PORT}")
    
    # Start streamers for each asset
    streamers = {}
    for symbol, trader in traders.items():
        streamer = BinanceKlineStreamer(symbol=symbol)
        streamer.on_kline_1m(trader.on_1m_kline)
        # 1s klines not used (deterministic pipeline)
        streamers[symbol] = streamer
    
    # Warmup all assets
    for symbol, streamer in streamers.items():
        print(f"[{symbol.upper()}] Warming up...")
        await streamer.warmup_from_rest(800)
        print(f"[{symbol.upper()}] Warmup done. M5 count: {traders[symbol].m5_count}")
    
    # Start all WebSocket streams
    tasks = []
    for symbol, streamer in streamers.items():
        tasks.append(asyncio.create_task(streamer.stream()))
    
    # Track last predicted M5 per asset
    last_predicted = {s: traders[s].m5_count for s in traders}
    
    # Main prediction loop
    while True:
        await asyncio.sleep(1)
        for symbol, trader in traders.items():
            current_m5 = trader.m5_count
            if current_m5 > last_predicted[symbol] and current_m5 > 50:
                if trader.boundary_features is not None:
                    trader.predict()
                    last_predicted[symbol] = current_m5


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*60}")
        combined_pnl = 0
        for symbol, trader in traders.items():
            s = trader.get_stats()
            wr = s['wr'] * 100
            print(f"  {symbol.upper()}: {wr:.1f}% WR ({s['wins']}W/{s['losses']}L) | ${s['pnl']:+.2f}")
            combined_pnl += s['pnl']
        print(f"  COMBINED: ${combined_pnl:+.2f}")
        print(f"{'='*60}")
