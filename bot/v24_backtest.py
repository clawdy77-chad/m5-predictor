#!/usr/bin/env python3
"""
V24 Walk-Forward Backtest — Zero Look-Ahead, Incremental Processing.

Processes 1m candles ONE BY ONE in chronological order. At each M5 boundary,
features are extracted from ONLY the data that has arrived so far. All TF
candles, C1/C2/C3 classifications, FVGs, and swings are maintained incrementally.

Swings are only confirmed after lb=3 bars of look-forward data arrives — exactly
as in live trading. No precomputation on future data.

Usage:
    python model/v24_backtest.py
    python model/v24_backtest.py --days 90
"""
import csv, sys, numpy as np, math, time, pickle, os
import glob as globmod
from collections import defaultdict, deque
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DAY_SEC = 86400

# CLI args
BACKTEST_DAYS = 30
ASSET = 'btcusdt'
TARGET_TF = 'm5'

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--days' and i < len(sys.argv) - 1:
        BACKTEST_DAYS = int(sys.argv[i + 1])
    elif arg == '--asset' and i < len(sys.argv) - 1:
        ASSET = sys.argv[i + 1].lower()
    elif arg == '--target-tf' and i < len(sys.argv) - 1:
        TARGET_TF = sys.argv[i + 1].lower()

# Target TF seconds for labeling
TARGET_TF_SECS = {'m5': 300, 'm15': 900}
TARGET_TF_SEC = TARGET_TF_SECS.get(TARGET_TF, 300)


# ═══════════════════════════════════════════════════════════════════════
# INCREMENTAL TIMEFRAME — processes candles one by one, zero look-ahead
# ═══════════════════════════════════════════════════════════════════════

class IncrementalTF:
    """Maintains running state for a single timeframe, updated candle-by-candle.

    Call feed_1m() with each 1m candle. When a TF candle completes, all internal
    state (C1/C2/C3, FVG, ATR, swings) is updated using ONLY past data.
    """

    def __init__(self, tf_sec, fvg_lookback=5, swing_lb=3, atr_period=14, c1_threshold=0.5):
        self.tf_sec = tf_sec
        self.fvg_lookback = fvg_lookback
        self.swing_lb = swing_lb
        self.c1_threshold = c1_threshold

        # Completed candles (only keep last N for memory efficiency)
        self.max_history = max(200, fvg_lookback + 50)
        self.highs = []
        self.lows = []
        self.opens = []
        self.closes = []
        self.volumes = []
        self.timestamps = []
        self.n = 0  # total completed candles ever (not just in buffer)

        # Forming candle
        self._forming_bucket = -1
        self._f_open = 0.0
        self._f_high = 0.0
        self._f_low = 0.0
        self._f_close = 0.0
        self._f_vol = 0.0
        self._has_forming = False

        # Running ATR
        self._atr_period = atr_period
        self._trs = deque(maxlen=atr_period)
        self._last_close = 0.0
        self.last_atr = 0.01

        # C1/C2/C3 state
        self._last_c2c3_dir = 0
        self.labels = []       # 1/2/3
        self.cls_dirs = []     # +1/-1/0
        # Track last N for feature extraction
        self._max_cls_history = 200

        # FVG tracking — bounded deques of active unfilled FVGs
        self._bull_fvgs = deque()  # (candle_index, gap_low, gap_high)
        self._bear_fvgs = deque()
        self.last_bull_fvg = False
        self.last_bear_fvg = False
        self.last_fvg_dist = 0.0

        # Swing tracking — confirmed with lb-bar delay
        self._swing_buffer_h = deque(maxlen=swing_lb * 2 + 10)
        self._swing_buffer_l = deque(maxlen=swing_lb * 2 + 10)
        self._swing_buffer_idx = deque(maxlen=swing_lb * 2 + 10)
        self.last_swing_high = None  # most recent confirmed swing high value
        self.last_swing_low = None

    def feed_1m(self, ts, o, h, l, c, v):
        """Feed a 1m candle. Returns True if a TF candle just completed."""
        bucket = (ts // self.tf_sec) * self.tf_sec

        if bucket != self._forming_bucket:
            completed = False
            if self._has_forming:
                self._complete_candle()
                completed = True

            self._forming_bucket = bucket
            self._f_open = o
            self._f_high = h
            self._f_low = l
            self._f_close = c
            self._f_vol = v
            self._has_forming = True
            return completed
        else:
            self._f_high = max(self._f_high, h)
            self._f_low = min(self._f_low, l)
            self._f_close = c
            self._f_vol += v
            return False

    def feed_sub(self, ts, o, h, l, c, v):
        """Feed a sub-minute candle directly (for 10s/30s TFs built from 1s or interpolated)."""
        bucket = (ts // self.tf_sec) * self.tf_sec

        if bucket != self._forming_bucket:
            completed = False
            if self._has_forming:
                self._complete_candle()
                completed = True
            self._forming_bucket = bucket
            self._f_open = o
            self._f_high = h
            self._f_low = l
            self._f_close = c
            self._f_vol = v
            self._has_forming = True
            return completed
        else:
            self._f_high = max(self._f_high, h)
            self._f_low = min(self._f_low, l)
            self._f_close = c
            self._f_vol += v
            return False

    @property
    def forming_candle(self):
        """Get the currently forming (incomplete) candle, or None."""
        if not self._has_forming:
            return None
        return (self._f_open, self._f_high, self._f_low, self._f_close)

    def _complete_candle(self):
        """Finalize forming candle -> update all incremental state."""
        o, h, l, c, v = self._f_open, self._f_high, self._f_low, self._f_close, self._f_vol
        ts = self._forming_bucket

        # Store candle
        self.highs.append(h)
        self.lows.append(l)
        self.opens.append(o)
        self.closes.append(c)
        self.volumes.append(v)
        self.timestamps.append(ts)
        idx = self.n
        self.n += 1

        # Trim history
        if len(self.highs) > self.max_history * 2:
            trim = len(self.highs) - self.max_history
            del self.highs[:trim]
            del self.lows[:trim]
            del self.opens[:trim]
            del self.closes[:trim]
            del self.volumes[:trim]
            del self.timestamps[:trim]

        # ATR update
        if self.n == 1:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self._last_close), abs(l - self._last_close))
        self._trs.append(tr)
        self.last_atr = max(sum(self._trs) / len(self._trs), 0.01)
        self._last_close = c

        # C1/C2/C3 classification
        rng = h - l
        direction = 1 if c > o else -1

        if rng < self.c1_threshold * self.last_atr:
            lbl, d = 1, 0
        elif self._last_c2c3_dir == 0 or direction != self._last_c2c3_dir:
            lbl, d = 2, direction
            self._last_c2c3_dir = direction
        else:
            lbl, d = 3, direction
            self._last_c2c3_dir = direction

        self.labels.append(lbl)
        self.cls_dirs.append(d)
        if len(self.labels) > self._max_cls_history * 2:
            trim = len(self.labels) - self._max_cls_history
            del self.labels[:trim]
            del self.cls_dirs[:trim]

        # FVG detection: check if new candle creates FVG with candles[-3] and candles[-1]
        if len(self.highs) >= 3:
            c0_h = self.highs[-3]
            c0_l = self.lows[-3]
            c2_h = h
            c2_l = l
            if c2_l > c0_h:  # Bullish FVG
                self._bull_fvgs.append((idx, c0_h, c2_l))
            if c2_h < c0_l:  # Bearish FVG
                self._bear_fvgs.append((idx, c2_h, c0_l))

        # Fill check and cleanup
        min_valid_idx = idx - self.fvg_lookback
        # Remove old + filled bull FVGs
        new_bull = deque()
        for fvg_idx, gap_lo, gap_hi in self._bull_fvgs:
            if fvg_idx < min_valid_idx:
                continue
            if l <= gap_lo:  # filled
                continue
            new_bull.append((fvg_idx, gap_lo, gap_hi))
        self._bull_fvgs = new_bull

        new_bear = deque()
        for fvg_idx, gap_lo, gap_hi in self._bear_fvgs:
            if fvg_idx < min_valid_idx:
                continue
            if h >= gap_hi:  # filled
                continue
            new_bear.append((fvg_idx, gap_lo, gap_hi))
        self._bear_fvgs = new_bear

        # Update last FVG state
        self.last_bull_fvg = len(self._bull_fvgs) > 0
        self.last_bear_fvg = len(self._bear_fvgs) > 0

        price = c
        atr = self.last_atr
        best_dist = float('inf')
        for fvgs in [self._bull_fvgs, self._bear_fvgs]:
            for _, gap_lo, gap_hi in fvgs:
                mid = (gap_lo + gap_hi) / 2
                d = (price - mid) / atr
                if abs(d) < abs(best_dist):
                    best_dist = d
        self.last_fvg_dist = best_dist if best_dist != float('inf') else 0.0

        # Swing detection: confirm swing at position that now has lb bars after it
        # We add (idx, high, low) to buffer, then check if the bar lb positions back is a swing
        self._swing_buffer_h.append(h)
        self._swing_buffer_l.append(l)
        self._swing_buffer_idx.append(idx)

        buf_len = len(self._swing_buffer_h)
        lb = self.swing_lb
        check_pos = buf_len - 1 - lb  # position in buffer to check
        if check_pos >= lb:
            is_high = True
            is_low = True
            ch = self._swing_buffer_h[check_pos]
            cl = self._swing_buffer_l[check_pos]
            for j in range(1, lb + 1):
                if ch < self._swing_buffer_h[check_pos - j] or ch < self._swing_buffer_h[check_pos + j]:
                    is_high = False
                if cl > self._swing_buffer_l[check_pos - j] or cl > self._swing_buffer_l[check_pos + j]:
                    is_low = False
                if not is_high and not is_low:
                    break
            if is_high:
                self.last_swing_high = ch
            if is_low:
                self.last_swing_low = cl

    def last_c2c3_dir(self):
        """Direction of the last C2 or C3 candle."""
        for i in range(len(self.cls_dirs) - 1, max(len(self.cls_dirs) - 50, -1), -1):
            if i < 0:
                break
            if self.labels[i] in (2, 3) and self.cls_dirs[i] != 0:
                return self.cls_dirs[i]
        return 0

    def consec_c3(self, max_lookback=30):
        """Count consecutive C3 candles from the end."""
        count = 0
        for i in range(len(self.labels) - 1, max(len(self.labels) - max_lookback - 1, -1), -1):
            if i < 0 or self.labels[i] != 3:
                break
            count += 1
        return count

    def c1_count(self, lookback=10):
        """Count C1 candles in last `lookback` bars."""
        start = max(0, len(self.labels) - lookback)
        return sum(1 for i in range(start, len(self.labels)) if self.labels[i] == 1)

    def bars_since_c2(self, max_lookback=100):
        """Bars since last C2 (displacement)."""
        count = 0
        for i in range(len(self.labels) - 1, max(len(self.labels) - max_lookback - 1, -1), -1):
            if i < 0 or self.labels[i] == 2:
                break
            count += 1
        return count


    def body_vs_atr(self):
        """Last candle body / ATR."""
        if not self.closes:
            return 0.0
        return abs(self.closes[-1] - self.opens[-1]) / self.last_atr

    def range_vs_atr(self):
        """Last candle range / ATR."""
        if not self.highs:
            return 0.0
        return (self.highs[-1] - self.lows[-1]) / self.last_atr


# ═══════════════════════════════════════════════════════════════════════
# FRACTAL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

FRACTALS = {
    'F5a': ('h1',  'm15', 'm5',  2.0),
    'F5b': ('h4',  'h1',  'm15', 3.0),
    'F5c': ('m30', 'm15', 'm5',  1.5),
    'F5d': ('m15', 'm5',  'm1',  1.5),
    'F5e': ('m5',  'm1',  '10s', 1.0),
    'F5f': ('m15', 'm3',  '30s', 1.0),
}
TOTAL_WEIGHT = sum(w for _, _, _, w in FRACTALS.values())

# 15m-optimized fractals: entry M5-M30, structure M15-H1, bias H1-H4
FRACTALS_15M = {
    'F15a': ('h4',  'h1',  'm30', 3.0),
    'F15b': ('h4',  'm30', 'm15', 3.0),
    'F15c': ('h1',  'm30', 'm15', 2.5),
    'F15d': ('h1',  'm15', 'm5',  2.0),
    'F15e': ('m30', 'm15', 'm5',  1.5),
    'F15f': ('h4',  'h1',  'm15', 2.5),
}
TOTAL_WEIGHT_15M = sum(w for _, _, _, w in FRACTALS_15M.values())


def evaluate_fractal(tfs, name, fractal_dict=None):
    """Evaluate fractal from current incremental TF states."""
    if fractal_dict is None:
        fractal_dict = FRACTALS
    bias_key, struct_key, entry_key, _ = fractal_dict[name]
    if bias_key not in tfs or struct_key not in tfs or entry_key not in tfs:
        return 0
    bias_dir = tfs[bias_key].last_c2c3_dir()
    if bias_dir == 0:
        return 0
    struct_dir = tfs[struct_key].last_c2c3_dir()
    if struct_dir != bias_dir:
        return 0
    entry_tf = tfs[entry_key]
    if bias_dir == 1 and entry_tf.last_bull_fvg:
        return 1
    if bias_dir == -1 and entry_tf.last_bear_fvg:
        return -1
    return 0


def extract_features(tfs, target_tf='m5'):
    """Extract features from current incremental TF state. Zero look-ahead.

    target_tf: 'm5' uses FRACTALS (5m-optimized), 'm15' uses FRACTALS_15M.
    """
    use_15m = (target_tf == 'm15')
    frac = FRACTALS_15M if use_15m else FRACTALS
    frac_weight = TOTAL_WEIGHT_15M if use_15m else TOTAL_WEIGHT

    f = {}

    # Per-TF C1/C2/C3 + direction
    for key in ['h4', 'h1', 'm30', 'm15', 'm5', 'm3', 'm1', '10s', '30s']:
        if key in tfs and tfs[key].n > 0:
            tf = tfs[key]
            f[f'{key}_c123'] = tf.labels[-1] if tf.labels else 0
            f[f'{key}_direction'] = tf.cls_dirs[-1] if tf.cls_dirs else 0
        else:
            f[f'{key}_c123'] = 0
            f[f'{key}_direction'] = 0

    # Per-TF FVG presence
    fvg_keys = ['m5', 'm15', 'h1', 'm1', 'm3', 'm30'] if use_15m else ['m5', 'm15', 'h1', 'm1', 'm3', '10s', '30s']
    for key in fvg_keys:
        if key in tfs and tfs[key].n > 0:
            f[f'{key}_fvg_bull'] = 1 if tfs[key].last_bull_fvg else 0
            f[f'{key}_fvg_bear'] = 1 if tfs[key].last_bear_fvg else 0
        else:
            f[f'{key}_fvg_bull'] = 0
            f[f'{key}_fvg_bear'] = 0

    # Fractal alignment
    for name in frac:
        f[f'{name.lower()}_aligned'] = evaluate_fractal(tfs, name, frac)

    f['alignment_count'] = sum(1 for name in frac if f[f'{name.lower()}_aligned'] != 0)
    f['weighted_vote'] = sum(
        f[f'{n.lower()}_aligned'] * w for n, (_, _, _, w) in frac.items()
    ) / frac_weight

    # Bias direction
    bias_dir = 0
    for key in ['h4', 'h1', 'm30', 'm15']:
        if key in tfs and tfs[key].n > 0:
            d = tfs[key].last_c2c3_dir()
            if d != 0:
                bias_dir = d
                break
    f['bias_direction'] = bias_dir

    # Body/range vs ATR
    atr_keys = ['m5', 'm1', 'm15', 'h1', 'm30'] if use_15m else ['m5', 'm1', 'm15', 'h1', '10s', '30s']
    for key in atr_keys:
        if key in tfs and tfs[key].n > 0:
            f[f'{key}_body_vs_atr'] = tfs[key].body_vs_atr()
            f[f'{key}_range_vs_atr'] = tfs[key].range_vs_atr()
        else:
            f[f'{key}_body_vs_atr'] = 0.0
            f[f'{key}_range_vs_atr'] = 0.0

    # Swing sweep distances (M5)
    if 'm5' in tfs and tfs['m5'].n > 7:
        tf = tfs['m5']
        atr = tf.last_atr
        f['m5_swept_high'] = (tf.highs[-1] - tf.last_swing_high) / atr if tf.last_swing_high is not None else 0.0
        f['m5_swept_low'] = (tf.last_swing_low - tf.lows[-1]) / atr if tf.last_swing_low is not None else 0.0
    else:
        f['m5_swept_high'] = 0.0
        f['m5_swept_low'] = 0.0

    # Nearest unfilled FVG (M5)
    f['m5_unfilled_fvg_dist'] = tfs['m5'].last_fvg_dist if 'm5' in tfs else 0.0

    # M15-specific features (only for 15m target)
    if use_15m:
        # M15 swing sweep distances
        if 'm15' in tfs and tfs['m15'].n > 7:
            tf = tfs['m15']
            atr = tf.last_atr
            f['m15_swept_high'] = (tf.highs[-1] - tf.last_swing_high) / atr if tf.last_swing_high is not None else 0.0
            f['m15_swept_low'] = (tf.last_swing_low - tf.lows[-1]) / atr if tf.last_swing_low is not None else 0.0
        else:
            f['m15_swept_high'] = 0.0
            f['m15_swept_low'] = 0.0

        # M15 nearest unfilled FVG
        f['m15_unfilled_fvg_dist'] = tfs['m15'].last_fvg_dist if 'm15' in tfs else 0.0

        # M15 consecutive C3 and C1 counts
        f['consec_c3_m15'] = tfs['m15'].consec_c3(12) if 'm15' in tfs else 0
        f['c1_count_m15'] = tfs['m15'].c1_count(6) if 'm15' in tfs else 0

        # M15 displacement recency
        f['m15_last_displacement'] = tfs['m15'].bars_since_c2() if 'm15' in tfs else 0

    # Consecutive C3 counts
    for key, lb, feat in [('m5', 12, 'consec_c3_m5'), ('m1', 20, 'consec_c3_m1'),
                           ('10s', 30, 'consec_c3_10s'), ('30s', 20, 'consec_c3_30s')]:
        f[feat] = tfs[key].consec_c3(lb) if key in tfs else 0

    # C1 counts
    for key, lb, feat in [('m5', 6, 'c1_count_m5'), ('m1', 10, 'c1_count_m1'),
                           ('10s', 18, 'c1_count_10s'), ('30s', 10, 'c1_count_30s')]:
        f[feat] = tfs[key].c1_count(lb) if key in tfs else 0

    # Time encoding
    ref_key = 'm15' if use_15m else 'm5'
    ts = tfs[ref_key].timestamps[-1] if ref_key in tfs and tfs[ref_key].timestamps else 0
    hour = (ts % 86400) / 3600
    f['hour_sin'] = math.sin(2 * math.pi * hour / 24)
    f['hour_cos'] = math.cos(2 * math.pi * hour / 24)
    f['day_of_week'] = datetime.fromtimestamp(ts, tz=timezone.utc).weekday() if ts > 0 else 0

    # Displacement recency
    for key, feat in [('m1', 'm1_last_displacement'), ('10s', 's10_last_displacement'), ('30s', 's30_last_displacement')]:
        f[feat] = tfs[key].bars_since_c2() if key in tfs else 0

    return f


def _snapshot_sub_state(tfs):
    """Capture sub-minute (10s/30s) feature values for lookahead-free override."""
    s = {}
    for key in ['10s', '30s']:
        if key in tfs and tfs[key].n > 0:
            tf = tfs[key]
            s[f'{key}_c123'] = tf.labels[-1] if tf.labels else 0
            s[f'{key}_direction'] = tf.cls_dirs[-1] if tf.cls_dirs else 0
            s[f'{key}_body_vs_atr'] = tf.body_vs_atr()
            s[f'{key}_range_vs_atr'] = tf.range_vs_atr()
            s[f'{key}_fvg_bull'] = 1 if tf.last_bull_fvg else 0
            s[f'{key}_fvg_bear'] = 1 if tf.last_bear_fvg else 0
        else:
            for suffix in ['_c123', '_direction']:
                s[f'{key}{suffix}'] = 0
            for suffix in ['_body_vs_atr', '_range_vs_atr']:
                s[f'{key}{suffix}'] = 0.0
            for suffix in ['_fvg_bull', '_fvg_bear']:
                s[f'{key}{suffix}'] = 0
    for key, lb in [('10s', 30), ('30s', 20)]:
        s[f'consec_c3_{key}'] = tfs[key].consec_c3(lb) if key in tfs else 0
    for key, lb in [('10s', 18), ('30s', 10)]:
        s[f'c1_count_{key}'] = tfs[key].c1_count(lb) if key in tfs else 0
    for key, feat in [('10s', 's10_last_displacement'), ('30s', 's30_last_displacement')]:
        s[feat] = tfs[key].bars_since_c2() if key in tfs else 0
    return s


# ═══════════════════════════════════════════════════════════════════════
# INTERPOLATE 1m -> sub-minute
# ═══════════════════════════════════════════════════════════════════════

def interpolate_1m_to_subs(o, h, l, c, v, ts, n_splits):
    """Split one 1m candle into n_splits sub-candles. Returns list of (ts, o, h, l, c, v)."""
    interval = 60 // n_splits
    if c >= o:
        path = [o, l, h, c]
    else:
        path = [o, h, l, c]

    total_len = sum(abs(path[j + 1] - path[j]) for j in range(3))
    if total_len < 0.001:
        return [(ts + k * interval, o, o, o, o, v / n_splits) for k in range(n_splits)]

    points = []
    cum = 0.0
    seg = 0
    for pi in range(n_splits + 1):
        target = (pi / n_splits) * total_len
        while seg < 2:
            seg_len = abs(path[seg + 1] - path[seg])
            if cum + seg_len >= target - 1e-10:
                frac = (target - cum) / seg_len if seg_len > 0 else 0
                points.append(path[seg] + frac * (path[seg + 1] - path[seg]))
                break
            cum += seg_len
            seg += 1
        else:
            points.append(path[-1])

    result = []
    vp = v / n_splits
    for k in range(n_splits):
        p1, p2 = points[k], points[k + 1]
        result.append((ts + k * interval, p1, max(p1, p2), min(p1, p2), p2, vp))
    return result


# ═══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS (available on import)
# ═══════════════════════════════════════════════════════════════════════

TF_SECS = {'10s': 10, '30s': 30, 'm1': 60, 'm3': 180, 'm5': 300,
            'm15': 900, 'm30': 1800, 'h1': 3600, 'h4': 14400}
FVG_LOOKBACKS = {'10s': 15, '30s': 15, 'm1': 10, 'm3': 8, 'm5': 5,
                  'm15': 5, 'm30': 5, 'h1': 5, 'h4': 5}


# ═══════════════════════════════════════════════════════════════════════
# MAIN — only runs when executed directly, not on import
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"V24 Backtest — {BACKTEST_DAYS}d walk-forward, incremental (zero look-ahead)")
    print(f"Asset: {ASSET} | Target TF: {TARGET_TF} ({TARGET_TF_SEC}s)")
    print("=" * 60)

    CONTEXT_DAYS = max(70, BACKTEST_DAYS + 60 + 10)
    print(f"\nLoading 1m data (last {CONTEXT_DAYS} days)...")
    t0 = time.time()

    data_file = os.path.join(DATA_DIR, f'{ASSET}_1m_klines_1year.csv')
    # Fallback for legacy BTC filename
    if not os.path.exists(data_file) and ASSET == 'btcusdt':
        data_file = os.path.join(DATA_DIR, 'btc_1m_klines_1year.csv')

    all_rows = []
    with open(data_file) as f:
        for r in csv.DictReader(f):
            all_rows.append((
                int(float(r['timestamp'])) // 1000,
                float(r['open']),
                float(r['high']),
                float(r['low']),
                float(r['close']),
                float(r['volume']),
            ))

    last_ts = all_rows[-1][0]
    cutoff = last_ts - CONTEXT_DAYS * DAY_SEC
    rows_1m = [(ts, o, h, l, c, v) for ts, o, h, l, c, v in all_rows if ts >= cutoff]
    del all_rows
    print(f"  {len(rows_1m)} 1m candles in {time.time()-t0:.1f}s")

    # Load 1s data if available
    rows_1s_by_minute = defaultdict(list)  # minute_bucket -> list of (ts, o, h, l, c, v)
    s1_start = s1_end = 0
    s1_files = sorted(globmod.glob(os.path.join(DATA_DIR, 'btc_1s_klines_*.csv')))
    if s1_files:
        print(f"Loading 1s data from {os.path.basename(s1_files[-1])}...")
        t0 = time.time()
        count = 0
        with open(s1_files[-1]) as f:
            for r in csv.DictReader(f):
                ts = int(float(r['timestamp'])) // 1000
                minute_bucket = (ts // 60) * 60
                rows_1s_by_minute[minute_bucket].append((
                    ts, float(r['open']), float(r['high']),
                    float(r['low']), float(r['close']), float(r.get('volume', 0)),
                ))
                count += 1
                if count == 1:
                    s1_start = ts
                s1_end = ts
        print(f"  {count} 1s candles in {time.time()-t0:.1f}s")
        print(f"  Range: {datetime.fromtimestamp(s1_start, tz=timezone.utc)} to "
              f"{datetime.fromtimestamp(s1_end, tz=timezone.utc)}")
    else:
        print("  No 1s data — sub-minute uses interpolation only")


    # ═══════════════════════════════════════════════════════════════════════
    # INCREMENTAL PROCESSING
    # ═══════════════════════════════════════════════════════════════════════

    print("\nProcessing candles incrementally...")
    t0 = time.time()

    # Initialize TF objects

    tfs = {}
    for key, sec in TF_SECS.items():
        tfs[key] = IncrementalTF(sec, fvg_lookback=FVG_LOOKBACKS[key], swing_lb=3)

    # Tracking
    backtest_start = last_ts - BACKTEST_DAYS * DAY_SEC
    train_start = backtest_start - 60 * DAY_SEC  # 60 days context for training
    feature_start = train_start

    # Determine which TF to use for labeling (m5 or m15)
    label_tf_key = TARGET_TF  # 'm5' or 'm15'
    print(f"  Labeling with {label_tf_key} candle direction")

    data = []  # (ts, features_dict, label)
    prev_label_count = 0
    pending_feat = None  # features saved when target TF[i] completes, labeled by TF[i+1]
    pending_ts = 0
    prev_sub_state = None  # sub-minute snapshot from previous M5 boundary

    for row_idx, (ts, o, h, l, c, v) in enumerate(rows_1m):
        # 1. Feed to TFs <= target TF only (no lookahead from HTFs)
        for key in ['m1', 'm3', 'm5']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

        # 2. Feed sub-minute (also <= M5, no lookahead)
        if ts in rows_1s_by_minute:
            for s_ts, s_o, s_h, s_l, s_c, s_v in rows_1s_by_minute[ts]:
                tfs['10s'].feed_sub(s_ts, s_o, s_h, s_l, s_c, s_v)
                tfs['30s'].feed_sub(s_ts, s_o, s_h, s_l, s_c, s_v)
        else:
            for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 6):
                tfs['10s'].feed_sub(*sub)
            for sub in interpolate_1m_to_subs(o, h, l, c, v, ts, 2):
                tfs['30s'].feed_sub(*sub)

        # 3. Check if target TF candle just completed — extract features BEFORE HTF update
        label_count = tfs[label_tf_key].n
        if label_count > prev_label_count and label_count > 50:
            prev_label_count = label_count
            label_ts = tfs[label_tf_key].timestamps[-1]

            # Label the PREVIOUS pending features with THIS candle's direction
            if pending_feat is not None and pending_ts >= feature_start:
                tf_dir = 1 if tfs[label_tf_key].closes[-1] > tfs[label_tf_key].opens[-1] else 0
                data.append((pending_ts, pending_feat, tf_dir))

            # Save features NOW — HTFs still reflect pre-boundary state (matches live)
            pending_feat = extract_features(tfs, target_tf=TARGET_TF)
            # Override sub-minute features with PREVIOUS M5 boundary snapshot
            # to prevent interpolated sub-minute data from within this M5 leaking
            if prev_sub_state is not None:
                pending_feat.update(prev_sub_state)
            # Snapshot current sub-minute state for next M5
            prev_sub_state = _snapshot_sub_state(tfs)
            pending_ts = label_ts

        # 4. NOW feed HTFs (m15, m30, h1, h4) — after feature extraction
        for key in ['m15', 'm30', 'h1', 'h4']:
            tfs[key].feed_1m(ts, o, h, l, c, v)

        if (row_idx + 1) % 20000 == 0:
            print(f"  {row_idx + 1}/{len(rows_1m)} 1m candles processed, "
                  f"{len(data)} {label_tf_key} samples ({time.time()-t0:.0f}s)")

    print(f"\n  {len(data)} {label_tf_key} samples extracted in {time.time()-t0:.1f}s")

    if not data:
        print("ERROR: No data.")
        sys.exit(1)

    fnames = sorted(data[0][1].keys())
    print(f"  {len(fnames)} features")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: PURE RULES
    # ═══════════════════════════════════════════════════════════════════════

    active_fractals = FRACTALS_15M if TARGET_TF == 'm15' else FRACTALS
    fractal_names = sorted(active_fractals.keys())

    print(f"\n{'=' * 60}")
    print(f"PHASE 1: Pure V24 Rules ({len(active_fractals)} fractals — {fractal_names[0]}-{fractal_names[-1]})")
    print('=' * 60)

    # Re-run pure rules on the test window only (can't use cached since we
    # need per-sample fractal evaluation; but features already encode alignment)
    rule_correct = rule_total = 0
    fractal_hits = defaultdict(lambda: {'correct': 0, 'total': 0})

    for ts_val, feat, label in data:
        if ts_val < backtest_start:
            continue

        wv = feat['weighted_vote']
        if wv == 0:
            # Fallback
            bd = feat['bias_direction']
            if bd == 0:
                continue
            direction = bd
        else:
            direction = 1 if wv > 0 else -1

        rule_total += 1
        pred = 1 if direction == 1 else 0
        if pred == label:
            rule_correct += 1

        # Per-fractal
        for name in active_fractals:
            d = feat[f'{name.lower()}_aligned']
            if d != 0:
                fractal_hits[name]['total'] += 1
                if (1 if d == 1 else 0) == label:
                    fractal_hits[name]['correct'] += 1

    rule_wr = rule_correct / rule_total * 100 if rule_total > 0 else 0
    test_samples = sum(1 for ts_val, _, _ in data if ts_val >= backtest_start)
    print(f"\nPure rules: {rule_wr:.1f}% WR on {rule_total} trades "
          f"(coverage: {rule_total / max(1, test_samples) * 100:.0f}%)")

    print(f"\nPer-fractal accuracy:")
    for name in fractal_names:
        bias_key, struct_key, entry_key, weight = active_fractals[name]
        d = fractal_hits.get(name, {'correct': 0, 'total': 0})
        if d['total'] > 0:
            wr = d['correct'] / d['total'] * 100
            print(f"  {name} ({bias_key:>3s}/{struct_key:>3s}/{entry_key:>3s} w={weight}): "
                  f"{wr:.1f}% WR ({d['total']} signals)")
        else:
            print(f"  {name} ({bias_key:>3s}/{struct_key:>3s}/{entry_key:>3s} w={weight}): no signals")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: WALK-FORWARD LIGHTGBM
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("PHASE 2: V24 + LightGBM Walk-Forward")
    print('=' * 60)

    TRAIN_DAYS = 53
    VAL_DAYS = 7
    TEST_DAYS = 7
    STEP_DAYS = 7
    TRAIN_SEC = TRAIN_DAYS * DAY_SEC
    VAL_SEC = VAL_DAYS * DAY_SEC
    TEST_SEC = TEST_DAYS * DAY_SEC
    STEP_SEC = STEP_DAYS * DAY_SEC
    EMBARGO_DAYS = 7
    EMBARGO_SEC = EMBARGO_DAYS * DAY_SEC

    all_ts_arr = np.array([d[0] for d in data])
    all_X = np.array([[d[1].get(f, 0) for f in fnames] for d in data], dtype=np.float32)
    all_X = np.nan_to_num(all_X, nan=0.0, posinf=0.0, neginf=0.0)
    all_y = np.array([d[2] for d in data])

    # Feature selection: pre-fit on data before backtest to keep top 15 features
    pre_mask = all_ts_arr < backtest_start
    if pre_mask.sum() > 500:
        import lightgbm as _lgb_pre
        pre_model = _lgb_pre.LGBMClassifier(n_estimators=100, verbose=-1)
        pre_model.fit(all_X[pre_mask], all_y[pre_mask])
        top_k = 25
        top_idx = np.argsort(pre_model.feature_importances_)[::-1][:top_k]
        fnames = [fnames[i] for i in top_idx]
        all_X = all_X[:, top_idx]
        print(f"  Feature selection: kept top {top_k} features: {fnames}")

    first_data_ts = data[0][0]
    last_data_ts = data[-1][0]

    window_start = max(backtest_start, first_data_ts + TRAIN_SEC + VAL_SEC + EMBARGO_SEC)

    print(f"Train {TRAIN_DAYS}d + Val {VAL_DAYS}d + Test {TEST_DAYS}d, step {STEP_DAYS}d")
    print(f"Test: {datetime.fromtimestamp(window_start, tz=timezone.utc).date()} to "
          f"{datetime.fromtimestamp(last_data_ts, tz=timezone.utc).date()}")

    try:
        import lightgbm as lgb
        USE_LGBM = True
        print("Using LightGBM")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        USE_LGBM = False
        print("Using sklearn GBM")

    all_preds = []
    all_labels_wf = []
    all_confs = []
    all_test_X = []  # feature vectors for every test trade
    all_test_ts = []  # timestamps for every test trade
    weekly_results = []
    best_model = None

    while window_start + TEST_SEC <= last_data_ts:
        embargo_start = window_start - EMBARGO_SEC
        train_start_w = embargo_start - VAL_SEC - TRAIN_SEC
        val_start_w = embargo_start - VAL_SEC

        train_mask = (all_ts_arr >= train_start_w) & (all_ts_arr < val_start_w)
        val_mask = (all_ts_arr >= val_start_w) & (all_ts_arr < embargo_start)
        test_mask = (all_ts_arr >= window_start) & (all_ts_arr < window_start + TEST_SEC)
        # Data in [embargo_start, window_start) is intentionally unused

        X_tr, y_tr = all_X[train_mask], all_y[train_mask]
        X_val, y_val = all_X[val_mask], all_y[val_mask]
        X_te, y_te = all_X[test_mask], all_y[test_mask]

        if len(X_tr) < 100 or len(X_te) < 10:
            window_start += STEP_SEC
            continue

        if USE_LGBM:
            if TARGET_TF == 'm15':
                model = lgb.LGBMClassifier(
                    objective='binary', metric='binary_logloss',
                    max_depth=-1, num_leaves=20, learning_rate=0.08, n_estimators=500,
                    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.05, reg_lambda=3.0, is_unbalance=True,
                    verbose=-1, random_state=42, n_jobs=-1,
                )
            else:
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
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        wr = (preds == y_te).mean() * 100
        week_date = datetime.fromtimestamp(window_start, tz=timezone.utc).strftime('%Y-%m-%d')

        all_preds.extend(preds)
        all_labels_wf.extend(y_te)
        all_confs.extend(proba)
        all_test_X.append(X_te)
        all_test_ts.extend(all_ts_arr[test_mask])
        weekly_results.append((week_date, wr, len(y_te), y_te.mean() * 100))
        best_model = model
        window_start += STEP_SEC

    all_preds = np.array(all_preds)
    all_labels_wf = np.array(all_labels_wf)
    all_confs = np.array(all_confs)
    all_test_X = np.vstack(all_test_X) if all_test_X else np.empty((0, len(fnames)))
    all_test_ts = np.array(all_test_ts)

    print(f"\n{'Week':>12} {'WR':>7} {'Trades':>7} {'Base%':>7}")
    print("-" * 40)
    for date, wr, n, base in weekly_results:
        marker = "OK" if wr > 55 else "WARN" if wr > 52 else "BAD"
        print(f"{date:>12} {wr:>6.1f}% {n:>7} {base:>6.1f}% {marker}")

    overall_wr = (all_preds == all_labels_wf).mean() * 100 if len(all_preds) > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"OVERALL: {overall_wr:.1f}% WR on {len(all_labels_wf)} trades")
    if len(all_labels_wf) > 0:
        print(f"Base rate: {all_labels_wf.mean() * 100:.1f}% UP")
    print(f"Edge: {overall_wr - 50:.1f}pp")

    if len(all_confs) > 0:
        print(f"\nBy confidence:")
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
            mask = np.maximum(all_confs, 1 - all_confs) >= thresh
            if mask.sum() > 0:
                bwr = (all_preds[mask] == all_labels_wf[mask]).mean() * 100
                n = mask.sum()
                per_day = n / max(1, len(weekly_results) * 7)
                pnl = (bwr / 100 * 50 - (1 - bwr / 100) * 50) * per_day
                print(f"  >={thresh:.0%}: {bwr:.1f}% WR | {n} trades ({per_day:.0f}/day) | ${pnl:.0f}/day")

    if USE_LGBM and best_model is not None:
        print(f"\nTop 20 features:")
        imp = best_model.feature_importances_
        for i in np.argsort(imp)[::-1][:20]:
            print(f"  {fnames[i]:30s} {imp[i]:>6.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: FLIP RULE MINING — find clusters where model is reliably
    # wrong, then flip those predictions (wrong UP -> bet DOWN)
    # ═══════════════════════════════════════════════════════════════════════
    if len(all_preds) > 500 and len(fnames) > 0:
        print(f"\n{'=' * 60}")
        print("PHASE 3: Flip Rule Mining (wrong clusters -> flip direction)")
        print('=' * 60)

        is_correct = (all_preds == all_labels_wf).astype(int)
        is_wrong = 1 - is_correct
        n_total = len(is_correct)
        base_wrong_rate = is_wrong.mean()
        base_wr = 1 - base_wrong_rate
        print(f"Total trades: {n_total} | Model WR: {base_wr:.1%} | Wrong rate: {base_wrong_rate:.1%}")

        # ── A. Exhaustive single-feature threshold scan ───────────────────
        # For each feature, try many thresholds and find ranges where
        # wrong rate > 50% (= flipping would be profitable)
        print(f"\n--- A. Single-Feature Flip Zones (wrong > 50%) ---")
        print(f"{'Feature':<28s} {'Condition':<25s} {'Trades':>7} {'Wrong%':>7} {'If Flip':>7}")
        print("-" * 80)

        flip_candidates = []  # (wrong_rate, n, fname, op, thresh, fi)
        MIN_FLIP_TRADES = max(500, n_total // 100)  # need decent sample

        for fi, fname in enumerate(fnames):
            col = all_test_X[:, fi]
            if np.std(col) < 1e-8:
                continue
            # Try percentile thresholds: 10, 20, ..., 90
            for pct in range(5, 96, 5):
                thresh = np.percentile(col, pct)
                for op, mask in [('<=', col <= thresh), ('>', col > thresh)]:
                    n_match = mask.sum()
                    if n_match < MIN_FLIP_TRADES or n_match > n_total - MIN_FLIP_TRADES:
                        continue
                    wrong_in_mask = is_wrong[mask].mean()
                    if wrong_in_mask > 0.50:  # flipping would help
                        flip_candidates.append((wrong_in_mask, n_match, fname, op, thresh, fi))

        # Deduplicate: keep best threshold per (feature, direction)
        seen = {}
        for wr, n, fname, op, thresh, fi in flip_candidates:
            key = (fname, op)
            if key not in seen or wr > seen[key][0]:
                seen[key] = (wr, n, fname, op, thresh, fi)
        flip_candidates = sorted(seen.values(), reverse=True)

        for wr, n, fname, op, thresh, fi in flip_candidates[:20]:
            flip_wr = wr  # if we flip, wrong becomes right
            print(f"{fname:<28s} {op + ' ' + f'{thresh:.3f}':<25s} {n:>7} {wr:>6.1%} {flip_wr:>6.1%}")

        # ── B. Two-feature combo flip zones ───────────────────────────────
        print(f"\n--- B. Two-Feature Combo Flip Zones (wrong > 50%) ---")
        print(f"{'Condition':<55s} {'Trades':>7} {'Wrong%':>7} {'If Flip':>7}")
        print("-" * 80)

        # Use top 15 most important features for combo search
        if USE_LGBM and best_model is not None:
            imp = best_model.feature_importances_
            top_fi_list = np.argsort(imp)[::-1][:15].tolist()
        else:
            top_fi_list = list(range(min(15, len(fnames))))

        combo_flips = []  # (wrong_rate, n, condition_str, mask)
        MIN_COMBO_TRADES = max(300, n_total // 150)

        for i_idx in range(len(top_fi_list)):
            for j_idx in range(i_idx + 1, len(top_fi_list)):
                fi_a, fi_b = top_fi_list[i_idx], top_fi_list[j_idx]
                col_a, col_b = all_test_X[:, fi_a], all_test_X[:, fi_b]
                # Try median and tercile splits
                for pa in [33, 50, 67]:
                    thresh_a = np.percentile(col_a, pa)
                    for pb in [33, 50, 67]:
                        thresh_b = np.percentile(col_b, pb)
                        for opa, mask_a in [('<=', col_a <= thresh_a), ('>', col_a > thresh_a)]:
                            for opb, mask_b in [('<=', col_b <= thresh_b), ('>', col_b > thresh_b)]:
                                combo_mask = mask_a & mask_b
                                n_match = combo_mask.sum()
                                if n_match < MIN_COMBO_TRADES:
                                    continue
                                wrong_r = is_wrong[combo_mask].mean()
                                if wrong_r > 0.50:
                                    cond = f"{fnames[fi_a]} {opa} {thresh_a:.3f} & {fnames[fi_b]} {opb} {thresh_b:.3f}"
                                    combo_flips.append((wrong_r, n_match, cond, combo_mask))

        # Deduplicate overlapping combos: keep highest wrong rate per coverage band
        combo_flips.sort(key=lambda x: (x[0], x[1]), reverse=True)
        shown_combos = []
        for wr, n, cond, mask in combo_flips:
            # Skip if >80% overlap with an already-shown combo
            overlap = False
            for _, _, _, prev_mask in shown_combos:
                if (mask & prev_mask).sum() / max(mask.sum(), 1) > 0.8:
                    overlap = True
                    break
            if not overlap:
                shown_combos.append((wr, n, cond, mask))
            if len(shown_combos) >= 15:
                break

        for wr, n, cond, mask in shown_combos:
            print(f"{cond:<55s} {n:>7} {wr:>6.1%} {wr:>6.1%}")

        # ── C. Decision tree flip rules (deeper, for multi-feature) ───────
        print(f"\n--- C. Decision Tree Flip Clusters (depth=4) ---")
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(200, n_total // 80),
                                    class_weight='balanced', random_state=42)
        dt.fit(all_test_X, is_wrong)

        leaf_ids = dt.apply(all_test_X)
        unique_leaves = np.unique(leaf_ids)
        tree_ = dt.tree_

        def _trace_rules(tree_, sample_x, feature_names):
            path = dt.decision_path(sample_x.reshape(1, -1)).toarray()[0]
            nodes = np.where(path)[0]
            rules = []
            for i in range(len(nodes) - 1):
                node = nodes[i]
                feat = feature_names[tree_.feature[node]]
                thresh = tree_.threshold[node]
                if nodes[i + 1] == tree_.children_left[node]:
                    rules.append((feat, '<=', thresh))
                else:
                    rules.append((feat, '>', thresh))
            return rules

        print(f"{'Leaf':>6} {'Trades':>7} {'Wrong%':>8} {'Action':>8}  Rules")
        print("-" * 90)

        flip_leaves = {}  # leaf_id -> (wrong_rate, n, rules)
        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            n_leaf = mask.sum()
            wr_leaf = is_wrong[mask].mean()
            sample_idx = np.where(mask)[0][0]
            rules = _trace_rules(tree_, all_test_X[sample_idx], fnames)
            rule_str = " & ".join(f"{f} {o} {t:.3f}" for f, o, t in rules)

            if wr_leaf > 0.50:
                action = "FLIP"
                flip_leaves[leaf] = (wr_leaf, n_leaf, rules)
            elif wr_leaf < 0.42:
                action = "KEEP+"
            else:
                action = "keep"
            print(f"{leaf:>6} {n_leaf:>7} {wr_leaf:>7.1%} {action:>8}  {rule_str}")

        # ── D. Simulate flipping ─────────────────────────────────────────
        print(f"\n--- D. Flip Simulation ---")

        # Collect all flip rules from sections A, B, C
        # Use decision tree leaves as primary (they're non-overlapping)
        flipped_preds = all_preds.copy()
        flip_mask_total = np.zeros(n_total, dtype=bool)

        # Apply tree-based flips (non-overlapping by construction)
        for leaf, (wr_leaf, n_leaf, rules) in flip_leaves.items():
            lmask = leaf_ids == leaf
            flipped_preds[lmask] = 1 - flipped_preds[lmask]
            flip_mask_total |= lmask

        n_flipped = flip_mask_total.sum()
        new_correct = (flipped_preds == all_labels_wf).astype(int)
        new_wr = new_correct.mean() * 100

        # Per-segment stats
        kept_mask = ~flip_mask_total
        if kept_mask.sum() > 0:
            kept_wr = (all_preds[kept_mask] == all_labels_wf[kept_mask]).mean() * 100
        else:
            kept_wr = 0
        if flip_mask_total.sum() > 0:
            flip_wr_before = (all_preds[flip_mask_total] == all_labels_wf[flip_mask_total]).mean() * 100
            flip_wr_after = (flipped_preds[flip_mask_total] == all_labels_wf[flip_mask_total]).mean() * 100
        else:
            flip_wr_before = flip_wr_after = 0

        print(f"  Trades flipped:    {n_flipped:>7} ({n_flipped/n_total:.1%} of all)")
        print(f"  Kept trades WR:    {kept_wr:.1f}% ({kept_mask.sum()} trades)")
        print(f"  Flipped zone WR:   {flip_wr_before:.1f}% -> {flip_wr_after:.1f}% (after flip)")
        print(f"  COMBINED WR:       {base_wr*100:.1f}% -> {new_wr:.1f}%")

        # ── E. Walk-forward validation of flip rules ──────────────────────
        # Split timeline in half: discover rules on first half, test on second
        print(f"\n--- E. Walk-Forward Flip Validation ---")
        print("  (Rules discovered on first half, tested on second half)")

        half = n_total // 2
        first_X, first_wrong = all_test_X[:half], is_wrong[:half]
        second_X, second_preds = all_test_X[half:], all_preds[half:]
        second_labels = all_labels_wf[half:]
        second_wrong = is_wrong[half:]

        # Discover rules on first half
        dt_wf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(200, half // 80),
                                       class_weight='balanced', random_state=42)
        dt_wf.fit(first_X, first_wrong)

        first_leaves = dt_wf.apply(first_X)
        wf_flip_leaves = set()
        for leaf in np.unique(first_leaves):
            lmask = first_leaves == leaf
            if lmask.sum() >= 200 and first_wrong[lmask].mean() > 0.50:
                wf_flip_leaves.add(leaf)

        # Apply discovered rules to second half
        second_leaves = dt_wf.apply(second_X)
        wf_flip_mask = np.isin(second_leaves, list(wf_flip_leaves))
        wf_flipped = second_preds.copy()
        wf_flipped[wf_flip_mask] = 1 - wf_flipped[wf_flip_mask]

        wr_before_wf = (second_preds == second_labels).mean() * 100
        wr_after_wf = (wf_flipped == second_labels).mean() * 100
        n_flipped_wf = wf_flip_mask.sum()

        # Per-leaf OOS performance
        print(f"\n  {'Leaf':>6} {'1st Half':>10} {'2nd Half':>10} {'Trades(2H)':>11}  Status")
        print("  " + "-" * 55)
        for leaf in sorted(wf_flip_leaves):
            first_m = first_leaves == leaf
            second_m = second_leaves == leaf
            if first_m.sum() > 0 and second_m.sum() > 0:
                wr1 = first_wrong[first_m].mean()
                wr2 = second_wrong[second_m].mean()
                status = "HOLDS" if wr2 > 0.50 else "FAILS" if wr2 < 0.47 else "weak"
                print(f"  {leaf:>6} {wr1:>9.1%} {wr2:>9.1%} {second_m.sum():>11}  {status}")

        print(f"\n  2nd half before flip: {wr_before_wf:.1f}% WR")
        print(f"  2nd half after flip:  {wr_after_wf:.1f}% WR  ({n_flipped_wf} trades flipped)")
        print(f"  Improvement:          {wr_after_wf - wr_before_wf:+.1f}pp")

        # ── F. Summary of actionable flip rules ──────────────────────────
        # Only keep rules that hold in walk-forward validation
        print(f"\n--- F. Validated Flip Rules ---")
        validated_rules = []
        for leaf in sorted(wf_flip_leaves):
            first_m = first_leaves == leaf
            second_m = second_leaves == leaf
            if second_m.sum() >= 100:
                wr2 = second_wrong[second_m].mean()
                if wr2 > 0.50:
                    rules = _trace_rules(dt_wf.tree_, first_X[np.where(first_m)[0][0]], fnames)
                    rule_str = " & ".join(f"{f} {o} {t:.3f}" for f, o, t in rules)
                    n2 = second_m.sum()
                    validated_rules.append((wr2, n2, rule_str, rules))

        if validated_rules:
            validated_rules.sort(reverse=True)
            for wr2, n2, rule_str, _ in validated_rules:
                print(f"  FLIP when: {rule_str}")
                print(f"    -> {wr2:.1%} wrong (= {wr2:.1%} correct if flipped) on {n2} OOS trades")
        else:
            print("  No flip rules survived walk-forward validation.")

    print(f"\n{'=' * 60}")
    print("SUMMARY (zero look-ahead, incremental processing)")
    print('=' * 60)
    print(f"  Pure V24 rules:          {rule_wr:.1f}% WR ({rule_total} trades)")
    print(f"  V24 + LightGBM hybrid:   {overall_wr:.1f}% WR ({len(all_labels_wf)} trades)")
    print(f"  Baseline (old LightGBM): 51.4% WR")

    # ─── Save model ──────────────────────────────────────────────────────
    if best_model is not None:
        # Model path: asset_tf_model.pkl for non-default, legacy name for btcusdt/m5
        if ASSET == 'btcusdt' and TARGET_TF == 'm5':
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v24_fractal_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{ASSET}_{TARGET_TF}_model.pkl')
        if os.path.exists(model_path):
            os.replace(model_path, model_path + '.bak')

        print(f"\nTraining final model on last {TRAIN_DAYS + VAL_DAYS} days...")
        final_start = last_data_ts - (TRAIN_DAYS + VAL_DAYS) * DAY_SEC
        final_mask = all_ts_arr >= final_start
        X_final, y_final = all_X[final_mask], all_y[final_mask]

        if USE_LGBM:
            if TARGET_TF == 'm15':
                final_model = lgb.LGBMClassifier(
                    objective='binary', metric='binary_logloss',
                    max_depth=-1, num_leaves=20, learning_rate=0.08, n_estimators=500,
                    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.05, reg_lambda=3.0, is_unbalance=True,
                    verbose=-1, random_state=42, n_jobs=-1,
                )
            else:
                final_model = lgb.LGBMClassifier(
                    objective='binary', metric='binary_logloss',
                    max_depth=2, learning_rate=0.08, n_estimators=500,
                    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.01, reg_lambda=2.0, is_unbalance=True,
                    verbose=-1, random_state=42, n_jobs=-1,
                )
            val_size = min(len(y_final) // 8, 2016)
            if val_size >= 10:
                final_model.fit(X_final[:-val_size], y_final[:-val_size],
                                eval_set=[(X_final[-val_size:], y_final[-val_size:])],
                                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                                           lgb.log_evaluation(period=0)])
            else:
                final_model.fit(X_final, y_final)
        else:
            final_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            final_model.fit(X_final, y_final)

        model_data = {
            'model': final_model,
            'feature_names': fnames,
            'config': {
                'type': 'v24_fractal_hybrid',
                'model': 'lgbm_d6' if USE_LGBM else 'gbm_d3_n100',
                'n_features': len(fnames),
                'samples': len(y_final),
                'train_days': TRAIN_DAYS + VAL_DAYS,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'auto_retrained': False,
                'walk_forward_wr': round(overall_wr, 2),
                'pure_rules_wr': round(rule_wr, 2),
                'incremental': True,
            }
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved to {model_path} ({len(y_final)} samples, WF WR: {overall_wr:.1f}%)")



if __name__ == "__main__":
    main()
