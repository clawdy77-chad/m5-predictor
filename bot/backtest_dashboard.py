#!/usr/bin/env python3
"""Backtest Dashboard — regenerates backtest data periodically, serves latest on refresh."""
import os, sys, json, time, threading, subprocess
from flask import Flask, Response, send_file

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, 'model', 'backtest_1y_all.json')
GENERATE_SCRIPT = os.path.join(_ROOT, 'training', 'backtest_1y_all.py')
REGEN_INTERVAL = 21600  # 6 hours (1y backtest takes ~35min)

app = Flask(__name__)

def regenerate():
    """Run the backtest data generation script in the background."""
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Regenerating backtest data...")
            t0 = time.time()
            subprocess.run([sys.executable, GENERATE_SCRIPT], cwd=_ROOT,
                          capture_output=True, timeout=600)
            print(f"[{time.strftime('%H:%M:%S')}] Regeneration done in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Regeneration error: {e}")
        time.sleep(REGEN_INTERVAL)

@app.route('/')
def dashboard():
    return send_file(os.path.join(os.path.dirname(__file__), 'dashboard.html'))

@app.route('/api/trades')
def trades():
    return send_file(DATA_PATH, mimetype='application/json')

@app.route('/api/last_updated')
def last_updated():
    try:
        mtime = os.path.getmtime(DATA_PATH)
        return Response(json.dumps({'ts': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(mtime))}),
                       content_type='application/json')
    except:
        return Response(json.dumps({'ts': 'unknown'}), content_type='application/json')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    # Start background regeneration thread
    t = threading.Thread(target=regenerate, daemon=True)
    t.start()
    print(f"Backtest Dashboard at http://0.0.0.0:{port} (regenerating every {REGEN_INTERVAL//60}min)")
    app.run(host='0.0.0.0', port=port, debug=False)
