import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        if ohlcv[-1][0] <= cutoff_ts:
            break
        since = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[df["stamp"] >= pd.to_datetime(cutoff_ts, unit="ms")]

@st.cache_data
def fetch_trades_kraken(symbol="BTC/USD", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_trades = []
    since = cutoff_ts
    while True:
        trades = exchange.fetch_trades(symbol, since=since)
        if not trades:
            break
        all_trades += trades
        since = trades[-1]['timestamp'] + 1
        if trades[-1]['timestamp'] >= now_ms:
            break
    df_tr = pd.DataFrame([t for t in all_trades if t['timestamp'] >= cutoff_ts])
    df_tr['time'] = df_tr['timestamp'] / 1000.0
    return df_tr[['time', 'price', 'amount']].rename(columns={'amount': 'vol'})

# =============================================================================
# BVC MODELS
# =============================================================================
class HawkesBVC:
    def __init__(self, window=20, kappa=0.1):
        self.window = window
        self.kappa = kappa

    def eval(self, df):
        prices = df['close'].values
        r = np.log(prices[1:]/prices[:-1])
        sigma = pd.Series(r).rolling(self.window).std().ffill().values
        labels = np.array([2 * t.cdf(ri/(s if s>0 else 1e-10), df=0.25).clip(0,1) - 1 
                   for ri, s in zip(r, sigma)])
        bvc = np.zeros(len(df))
        current = 0
        alpha = np.exp(-self.kappa)
        for i in range(1, len(df)):
            current = current * alpha + df['volume'].iloc[i] * labels[i-1]
            bvc[i] = current
        return pd.DataFrame({'stamp': df['stamp'], 'bvc': bvc})

class ACDBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def eval(self, trades):
        trades = trades.sort_values('time')
        durations = trades['time'].diff().shift(-1).fillna(1)
        residuals = (durations - durations.mean()) / durations.std()
        price_changes = np.log(trades['price']).diff().fillna(0)
        labels = -residuals * price_changes
        bvc = np.zeros(len(trades))
        current = 0
        for i in range(len(trades)):
            current = current * np.exp(-self.kappa) + trades['vol'].iloc[i] * labels.iloc[i]
            bvc[i] = current
        return pd.DataFrame({'stamp': pd.to_datetime(trades['time'], unit='s'), 'bvc': bvc})

class ACIBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def eval(self, trades):
        times = trades['time'].values
        intensities = [0.0]
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-self.kappa * dt) + 1)
        price_changes = np.log(trades['price']).diff().fillna(0)
        labels = np.array(intensities) * price_changes
        bvc = np.zeros(len(trades))
        current = 0
        for i in range(len(trades)):
            current = current * np.exp(-self.kappa) + trades['vol'].iloc[i] * labels.iloc[i]
            bvc[i] = current
        return pd.DataFrame({'stamp': pd.to_datetime(trades['time'], unit='s'), 'bvc': bvc})

# =============================================================================
# MAIN DASHBOARD LOGIC
# =============================================================================
symbol = st.sidebar.text_input("Symbol", "BTC/USD")
try:
    ohlc_data = fetch_data(symbol, timeframe, global_lookback_minutes)
    ohlc_data['ScaledPrice'] = np.log(ohlc_data['close']/ohlc_data['close'].iloc[0]) * 1e4
    ohlc_data['ScaledPrice_EMA'] = ema(ohlc_data['ScaledPrice'].values, 10)
except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()

# BVC Calculation
if bvc_model == "Hawkes":
    model = HawkesBVC()
    bvc_data = model.eval(ohlc_data)
else:
    try:
        trades = fetch_trades_kraken(symbol, global_lookback_minutes)
        if bvc_model == "ACD":
            model = ACDBVC()
        else:
            model = ACIBVC()
        bvc_data = model.eval(trades)
        # Critical resampling step for trade-based models
        bvc_data = bvc_data.set_index('stamp').resample(timeframe).last().ffill().reset_index()
    except Exception as e:
        st.error(f"Trade processing error: {e}")
        st.stop()

# Merge and clean data
merged = ohlc_data.merge(bvc_data, on='stamp', how='left')
merged['bvc'] = merged['bvc'].ffill().fillna(0)

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))
norm = plt.Normalize(merged['bvc'].min(), merged['bvc'].max())
for i in range(len(merged)-1):
    ax.plot(merged['stamp'].iloc[i:i+2], merged['ScaledPrice'].iloc[i:i+2],
            color=plt.cm.coolwarm(norm(merged['bvc'].iloc[i])),
            linewidth=1.5)
ax.plot(merged['stamp'], merged['ScaledPrice_EMA'], 'grey', alpha=0.8, label='EMA')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
st.pyplot(fig)

# BVC Subplot
fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.plot(merged['stamp'], merged['bvc'], 'purple', linewidth=0.8)
ax2.fill_between(merged['stamp'], merged['bvc'], alpha=0.2)
st.pyplot(fig2)
