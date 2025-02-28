import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import ccxt
from numba import njit
from scipy.stats import norm, t

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"

# =============================================================================
# Dashboard Title & Welcome Message
# =============================================================================
st.title("CNO Dashboard")
st.write(f"Welcome, {st.session_state['username']}!")

# =============================================================================
# GLOBAL VARIABLES & HELPER FUNCTIONS
# =============================================================================
lookback_options = {
    "1 Day": 1440,
    "3 Days": 4320,
    "1 Week": 10080,
    "2 Weeks": 20160,
    "1 Month": 43200
}
global_lookback_label = st.sidebar.selectbox(
    "Select Global Lookback Period",
    list(lookback_options.keys()),
    key="global_lookback_label"
)
global_lookback_minutes = lookback_options[global_lookback_label]
timeframe = st.sidebar.selectbox(
    "Select Timeframe", ["1m", "5m", "15m", "1h"],
    key="timeframe_widget"
)
bvc_model = st.sidebar.selectbox(
    "Select BVC Model", ["Hawkes", "ACD", "ACI"],
    key="bvc_model"
)

@njit(cache=True)
def ema(arr_in: np.ndarray, window: int, alpha: float = 0) -> np.ndarray:
    # If alpha is not provided, use default formula: 3/(window+1)
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    n = arr_in.size
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = (arr_in[i] * alpha) + (ewma[i - 1] * (1 - alpha))
    return ewma

def fetch_data(symbol, timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_ohlcv = []
    since = cutoff_ts
    max_limit = 1440  # max candles per request for Kraken
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_ts = ohlcv[-1][0]
        if last_ts <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_ts + 1
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    now = pd.to_datetime(now_ms, unit="ms")
    cutoff = now - pd.Timedelta(minutes=lookback_minutes)
    df = df[df["stamp"] >= cutoff]
    return df

@st.cache_data
def fetch_trades_kraken(symbol="BTC/USD", lookback_minutes=1440, limit=43200):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    since = cutoff_ts
    all_trades = []
    while True:
        trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        if not trades:
            break
        all_trades += trades
        since = trades[-1]['timestamp'] + 1
        if trades[-1]['timestamp'] >= now_ms or len(trades) < limit:
            break
    all_trades = [t for t in all_trades if t['timestamp'] >= cutoff_ts]
    if not all_trades:
        raise ValueError(f"No trades returned from Kraken in last {lookback_minutes} minutes.")
    df_tr = pd.DataFrame(all_trades)
    df_tr['time'] = df_tr['timestamp'] / 1000.0
    df_tr['vol'] = df_tr['amount']
    df_tr = df_tr[['time', 'price', 'vol']].copy()
    df_tr.sort_values('time', inplace=True)
    df_tr.reset_index(drop=True, inplace=True)
    return df_tr

# =============================================================================
# BVC MODEL CLASSES
# =============================================================================
class HawkesBVC:
    def __init__(self, window=20, kappa=0.1, dof=0.25):
        self._window = window
        self._kappa = kappa
        self._dof = dof
        self.metrics = None

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy()
        df['logret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        rolling_std = df['logret'].rolling(self._window).std().fillna(0)
        alpha_exp = np.exp(-self._kappa)

        def label_func(r, sigma):
            if sigma > 0:
                val = t.cdf(r / sigma, df=self._dof)
                return 2*val - 1
            return 0

        df['label'] = [label_func(r, s) for r, s in zip(df['logret'], rolling_std)]
        df['weighted_volume'] = df['volume'] * df['label']

        bvc_list = []
        current_bvc = 0.0
        for wv in df['weighted_volume']:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.abs(bvc).max() != 0:
            bvc /= np.abs(bvc).max()
        bvc *= scale

        self.metrics = pd.DataFrame({
            'stamp': df['stamp'],
            'bvc': bvc
        })
        return self.metrics

class ACDBVC:
    def __init__(self, kappa=0.1):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4) -> pd.DataFrame:
        df_tr = df_tr.dropna(subset=['time', 'price', 'vol']).copy()
        df_tr['duration'] = df_tr['time'].diff().shift(-1)
        df_tr.dropna(subset=['duration'], inplace=True)
        df_tr = df_tr[df_tr['duration'] > 0]
        if len(df_tr) < 10:
            st.warning(f"ACD: Insufficient data, found only {len(df_tr)} trades.")

        mean_d = df_tr['duration'].mean()
        std_d = df_tr['duration'].std()
        if std_d < 1e-12:
            # fallback to avoid dividing by zero
            std_d = 1e-6

        df_tr['standardized_residual'] = (df_tr['duration'] - mean_d) / std_d
        df_tr['price_change'] = np.log(df_tr['price'] / df_tr['price'].shift(1)).fillna(0)
        df_tr['label'] = -df_tr['standardized_residual'] * df_tr['price_change']
        df_tr['weighted_volume'] = df_tr['vol'] * df_tr['label']

        alpha_exp = np.exp(-self._kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df_tr['weighted_volume']:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.abs(bvc).max() != 0:
            bvc /= np.abs(bvc).max()
        bvc *= scale

        # If it's all zero, inject a tiny random offset
        if np.allclose(bvc, 0.0):
            st.warning("ACD BVC is entirely zero—injecting small random offset for visualization.")
            bvc += np.random.normal(0, 0.01, size=len(bvc))

        self.metrics = pd.DataFrame({
            'stamp': pd.to_datetime(df_tr['time'], unit='s'),
            'bvc': bvc
        })
        return self.metrics

class ACIBVC:
    def __init__(self, kappa=0.1):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4) -> pd.DataFrame:
        df_tr = df_tr.dropna(subset=['time', 'price', 'vol']).copy()
        times = df_tr['time'].values
        intensities = self.estimate_intensity(times, self._kappa)
        df_tr = df_tr.iloc[:len(intensities)]
        df_tr['intensity'] = intensities
        df_tr['price_change'] = np.log(df_tr['price'] / df_tr['price'].shift(1)).fillna(0)
        df_tr['label'] = df_tr['intensity'] * df_tr['price_change']
        df_tr['weighted_volume'] = df_tr['vol'] * df_tr['label']

        alpha_exp = np.exp(-self._kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df_tr['weighted_volume']:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.abs(bvc).max() != 0:
            bvc /= np.abs(bvc).max()
        bvc *= scale

        # If it's all zero, inject a tiny random offset
        if np.allclose(bvc, 0.0):
            st.warning("ACI BVC is entirely zero—injecting small random offset for visualization.")
            bvc += np.random.normal(0, 0.01, size=len(bvc))

        self.metrics = pd.DataFrame({
            'stamp': pd.to_datetime(df_tr['time'], unit='s'),
            'bvc': bvc
        })
        return self.metrics

    def estimate_intensity(self, times: np.ndarray, beta: float) -> np.ndarray:
        intensities = [0.0]
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-beta*dt) + 1)
        return np.array(intensities)

# =============================================================================
# MAIN DASHBOARD LOGIC
# =============================================================================
st.header("Section 1: Momentum, Skewness & BVC Analysis")

symbol_bsi1 = st.sidebar.text_input("Enter Ticker Symbol (Sec 1)",
                                    value="BTC/USD",
                                    key="symbol_bsi1")
st.write(f"Fetching data for: **{symbol_bsi1}** with a global lookback of "
         f"**{global_lookback_minutes}** minutes and timeframe **{timeframe}**.")

# 1) Fetch Candle Data
try:
    prices_bsi = fetch_data(symbol=symbol_bsi1,
                            timeframe=timeframe,
                            lookback_minutes=global_lookback_minutes)
    st.write("Data range:", prices_bsi["stamp"].min(), "to", prices_bsi["stamp"].max())
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close', 'volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)

# Compute scaled price & EMA
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close'] / prices_bsi['close'].iloc[0]) * 1e4
prices_bsi['ScaledPrice_EMA'] = ema(prices_bsi['ScaledPrice'].values, window=10)

# Compute VWAP
prices_bsi['cum_vol'] = prices_bsi['volume'].cumsum()
prices_bsi['cum_pv'] = (prices_bsi['close'] * prices_bsi['volume']).cumsum()
prices_bsi['vwap'] = prices_bsi['cum_pv'] / prices_bsi['cum_vol']
if prices_bsi['vwap'].iloc[0] == 0 or not np.isfinite(prices_bsi['vwap'].iloc[0]):
    st.warning("VWAP initial value is zero or invalid. Using ScaledPrice as fallback for VWAP plotting.")
    prices_bsi['vwap_transformed'] = prices_bsi['ScaledPrice']
else:
    prices_bsi['vwap_transformed'] = np.log(prices_bsi['vwap'] / prices_bsi['vwap'].iloc[0]) * 1e4

# 2) Evaluate BVC
if bvc_model == "Hawkes":
    model = HawkesBVC(window=20, kappa=0.1)
    bvc_metrics = model.eval(prices_bsi.reset_index())
else:
    # ACD or ACI => fetch trades
    try:
        df_trades = fetch_trades_kraken(symbol=symbol_bsi1,
                                        lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trade data: {e}")
        st.stop()
    if bvc_model == "ACD":
        model = ACDBVC(kappa=0.1)
    else:
        model = ACIBVC(kappa=0.1)
    bvc_metrics = model.eval(df_trades)

# 3) Merge the BVC results into our main df
df_merged = prices_bsi.reset_index().merge(bvc_metrics, on='stamp', how='left')
df_merged.sort_values('stamp', inplace=True)
df_merged['bvc'] = df_merged['bvc'].fillna(method='ffill').fillna(0)

global_min = df_merged['ScaledPrice'].min()
global_max = df_merged['ScaledPrice'].max()

# -----------------------------------------------------------------------------
# PRICE PLOT with BVC-based coloring (LineCollection approach)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

# Convert timestamps to numeric
x_vals = mdates.date2num(df_merged['stamp'].values)
y_vals = df_merged['ScaledPrice'].values
bvc_vals = df_merged['bvc'].values

# Create segments: each pair of adjacent points forms a line segment
points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a LineCollection, with color mapped to BVC
# 'bwr' means negative BVC is bluish, positive is reddish
lc = LineCollection(segments, cmap='bwr', norm=plt.Normalize(bvc_vals.min(), bvc_vals.max()))
lc.set_array(bvc_vals)
lc.set_linewidth(1.5)

# Add the colored line to the axes
ax.add_collection(lc)

# Plot the EMA and VWAP
ax.plot(df_merged['stamp'], df_merged['ScaledPrice_EMA'],
        color='black', linewidth=1, label="EMA(10)")
ax.plot(df_merged['stamp'], df_merged['vwap_transformed'],
        color='gray', linewidth=1, label="VWAP")

ax.set_xlim(x_vals.min(), x_vals.max())
ax.set_ylim(global_min - 50, global_max + 50)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax.set_xlabel("Time")
ax.set_ylabel("ScaledPrice")
ax.set_title("Price with EMA & VWAP (Segmented by BVC)")

plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
ax.legend(fontsize=8)

st.pyplot(fig)

# -----------------------------------------------------------------------------
# BVC Plot
# -----------------------------------------------------------------------------
fig_bvc, ax_bvc = plt.subplots(figsize=(10, 3), dpi=120)
ax_bvc.plot(df_merged['stamp'], df_merged['bvc'], color="blue", linewidth=1,
            label=f"BVC ({bvc_model})")
ax_bvc.set_xlabel("Time")
ax_bvc.set_ylabel("BVC")
ax_bvc.legend(fontsize=8)
ax_bvc.set_title("BVC Over Time")
ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax_bvc.get_yticklabels(), fontsize=7)

st.pyplot(fig_bvc)
