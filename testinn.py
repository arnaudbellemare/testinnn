import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ccxt
from numba import njit
from scipy.stats import t as studentt, norm
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from typing import Optional

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
    "Select Global Lookback Period", list(lookback_options.keys()),
    key="global_lookback_label"
)
global_lookback_minutes = lookback_options[global_lookback_label]

timeframe = st.sidebar.selectbox(
    "Select Timeframe", ["1m", "5m", "15m", "1h"],
    key="timeframe_widget"
)

@njit(cache=True)
def ema(arr_in: NDArray, window: int, alpha: Optional[float] = 0) -> NDArray:
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
    max_limit = 1440
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        if last_timestamp <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_timestamp + 1
    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    now = pd.to_datetime(now_ms, unit="ms")
    cutoff = now - pd.Timedelta(minutes=lookback_minutes)
    df = df[df["stamp"] >= cutoff]
    return df

@st.cache_data
def fetch_trades_kraken(symbol="BTC/USD", lookback_minutes=1440, limit=43200):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes*60*1000
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
        raise ValueError(f"No trades returned from Kraken in the last {lookback_minutes} minutes.")
    df_tr = pd.DataFrame(all_trades)
    df_tr['time'] = df_tr['timestamp']/1000.0
    df_tr['vol'] = df_tr['amount']
    df_tr = df_tr[['time','price','vol']].copy()
    df_tr.sort_values('time', inplace=True)
    df_tr.reset_index(drop=True, inplace=True)
    return df_tr

# =============================================================================
# BVC CLASSES
# =============================================================================
class HawkesBVC:
    def __init__(self, window: int, kappa: float, dof=0.25):
        self._window = window
        self._kappa = kappa
        self._dof = dof
        self.metrics = None

    def eval(self, df: pd.DataFrame, scale=1e4):
        times = df['stamp']
        prices = df['close']
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)
        volume = df['volume']
        sigma = r.rolling(self._window).std().fillna(0.0)
        alpha_exp = np.exp(-self._kappa)
        labels = []
        for i in range(len(r)):
            labels.append(self._label(r.iloc[i], sigma.iloc[i]))
        labels = np.array(labels)

        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc*alpha_exp + volume.iloc[i]*labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc /= np.max(np.abs(bvc))
            bvc *= scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': bvc})
        return self.metrics

    def _label(self, r: float, sigma: float):
        if sigma > 0.0:
            cum = studentt.cdf(r/sigma, df=self._dof)
            return 2*cum - 1.0
        else:
            return 0.0

class ACDBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4):
        df_tr = df_tr.dropna(subset=['time','price','vol']).copy()
        df_tr['duration'] = df_tr['time'].diff().shift(-1)
        df_tr = df_tr.dropna(subset=['duration'])
        df_tr = df_tr[df_tr['duration']>0]
        if len(df_tr) < 10:
            raise ValueError("Insufficient trade data for ACD model.")
        mean_duration = df_tr['duration'].mean()
        std_duration  = df_tr['duration'].std() or 1e-10
        df_tr['standardized_residual'] = (df_tr['duration'] - mean_duration)/std_duration
        df_tr['price_change'] = np.log(df_tr['price']/df_tr['price'].shift(1)).fillna(0)
        df_tr['label'] = -df_tr['standardized_residual']*df_tr['price_change']

        alpha_exp = np.exp(-self._kappa)
        bvc_list  = []
        current_bvc = 0.0
        df_tr['weighted_volume'] = df_tr['vol']*df_tr['label']
        for wv in df_tr['weighted_volume'].values:
            current_bvc = current_bvc*alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc)) != 0:
            bvc /= np.max(np.abs(bvc))
            bvc *= scale
        df_tr['stamp'] = pd.to_datetime(df_tr['time'], unit='s')
        self.metrics = pd.DataFrame({'stamp': df_tr['stamp'],'bvc': bvc})
        return self.metrics

class ACIBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4):
        df_tr = df_tr.dropna(subset=['time','price','vol']).copy()
        times = pd.to_datetime(df_tr['time'], unit='s')
        times_numeric = times.astype(np.int64)//10**9
        intensities = self.estimate_intensity(times_numeric, self._kappa)
        df_tr = df_tr.iloc[:len(intensities)]
        df_tr['intensity'] = intensities
        df_tr['price_change'] = np.log(df_tr['price']/df_tr['price'].shift(1)).fillna(0)
        df_tr['label'] = df_tr['intensity']*df_tr['price_change']

        alpha_exp = np.exp(-self._kappa)
        bvc_list  = []
        current_bvc = 0.0
        df_tr['weighted_volume'] = df_tr['vol']*df_tr['label']
        for wv in df_tr['weighted_volume'].values:
            current_bvc = current_bvc*alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc)) != 0:
            bvc /= np.max(np.abs(bvc))
            bvc *= scale
        df_tr['stamp'] = pd.to_datetime(df_tr['time'], unit='s')
        self.metrics = pd.DataFrame({'stamp': df_tr['stamp'],'bvc': bvc})
        return self.metrics

    def estimate_intensity(self, times: np.ndarray, beta: float)->np.ndarray:
        intensities = [0.0]
        for i in range(1,len(times)):
            delta_t = times[i]-times[i-1]
            intensities.append(intensities[-1]*np.exp(-beta*delta_t) + 1)
        return np.array(intensities)


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.header("Universal 2-Color BVC")

symbol_bsi1 = st.sidebar.text_input("Enter Ticker (Sec 1)", value="BTC/USD")
st.write(f"Fetching data for: **{symbol_bsi1}** using lookback = {global_lookback_minutes} minutes, timeframe = {timeframe}.")

try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe=timeframe, lookback_minutes=global_lookback_minutes)
    st.write("Data range:", prices_bsi["stamp"].min(), "to", prices_bsi["stamp"].max())
except Exception as e:
    st.error(f"Error fetching candle data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close','volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)

# Scaled Price
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close']/prices_bsi['close'].iloc[0]) * 1e4

# BVC Model Choice
bvc_choice = st.sidebar.selectbox("Select BVC Model", ["Hawkes", "ACD", "ACI"])

# Evaluate the selected BVC
if bvc_choice=="Hawkes":
    bvc_model = HawkesBVC(window=20, kappa=0.1)
    bvc_metrics = bvc_model.eval(prices_bsi.reset_index())
elif bvc_choice=="ACD":
    try:
        trades_df = fetch_trades_kraken(symbol=symbol_bsi1, lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trades for ACD: {e}")
        st.stop()
    acd_model = ACDBVC(kappa=0.1)
    bvc_metrics = acd_model.eval(trades_df)
elif bvc_choice=="ACI":
    try:
        trades_df = fetch_trades_kraken(symbol=symbol_bsi1, lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trades for ACI: {e}")
        st.stop()
    aci_model = ACIBVC(kappa=0.1)
    bvc_metrics = aci_model.eval(trades_df)
else:
    st.stop()

# Merge BVC with candle DataFrame
df_plot = prices_bsi.reset_index().merge(bvc_metrics, on='stamp', how='left')
df_plot.sort_values('stamp', inplace=True)

# 1) Take the BVC array
bvc_vals = df_plot['bvc'].values
if len(bvc_vals)==0 or np.all(np.isnan(bvc_vals)):
    st.warning("No BVC data to plot.")
    st.stop()

# 2) Find the absolute max for normalizing in [-1..+1]
abs_max = np.nanmax(np.abs(bvc_vals))
if abs_max==0:
    abs_max=1e-9

# 3) Create a scaled column
df_plot['bvc_scaled'] = df_plot['bvc'] / abs_max  # => in [-1..+1], ignoring NaNs

# Plot the line with 2-colors
fig, ax = plt.subplots(figsize=(10,4), dpi=120)
for i in range(len(df_plot)-1):
    xvals = df_plot['stamp'].iloc[i:i+2]
    yvals = df_plot['ScaledPrice'].iloc[i:i+2]
    val = df_plot['bvc_scaled'].iloc[i]
    if pd.isna(val):
        # If we have missing BVC, we can skip or plot in gray
        color = 'gray'
    else:
        if val>=0:
            # Range + => Blues from 0..1
            color = plt.cm.Blues(val) # val in [0..1]
        else:
            # Range - => Reds from 0..1
            color = plt.cm.Reds(-val) # -val in [0..1]
    ax.plot(xvals, yvals, color=color, linewidth=2)

ax.set_title(f"{bvc_choice} BVC Colored by ± Range [-1..+1]", fontsize=12)
ax.set_xlabel("Time")
ax.set_ylabel("ScaledPrice")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Also show the raw BVC
fig2, ax2 = plt.subplots(figsize=(10,3), dpi=120)
ax2.plot(df_plot['stamp'], df_plot['bvc'], color='blue')
ax2.set_title(f"{bvc_choice} BVC (Raw Values)", fontsize=12)
ax2.set_xlabel("Time")
ax2.set_ylabel("BVC")
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig2)
