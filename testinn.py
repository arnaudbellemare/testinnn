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
# HELPER FUNCTIONS
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
    "Select Timeframe",
    ["1m", "5m", "15m", "1h"],
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
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
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
        labels = np.array([
            self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))
        ])
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc*alpha_exp + volume.iloc[i]*labels[i]
            bvc[i] = current_bvc

        # Same as your original: scale to ±some max if non-zero
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc/np.max(np.abs(bvc))*scale

        self.metrics = pd.DataFrame({'stamp': times, 'bvc': bvc})
        return self.metrics

    def _label(self, r: float, sigma: float):
        if sigma>0.0:
            cum = studentt.cdf(r/sigma, df=self._dof)
            return 2*cum-1.0
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
        if len(df_tr)<10:
            raise ValueError("Insufficient trade data for ACD model.")
        mean_duration = df_tr['duration'].mean()
        std_duration  = df_tr['duration'].std() or 1e-10
        df_tr['standardized_residual'] = (df_tr['duration']-mean_duration)/std_duration
        df_tr['price_change'] = np.log(df_tr['price']/df_tr['price'].shift(1)).fillna(0)
        # negative sign to intensify negativity for longer durations
        df_tr['label'] = -df_tr['standardized_residual']*df_tr['price_change']
        
        alpha_exp = np.exp(-self._kappa)
        bvc_list = []
        current_bvc= 0.0
        df_tr['weighted_volume'] = df_tr['vol']*df_tr['label']
        for wv in df_tr['weighted_volume'].values:
            current_bvc = current_bvc*alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc))!=0:
            bvc = bvc/np.max(np.abs(bvc))*scale
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
        bvc_list = []
        current_bvc = 0.0
        df_tr['weighted_volume'] = df_tr['vol']*df_tr['label']
        for wv in df_tr['weighted_volume'].values:
            current_bvc = current_bvc*alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc))!=0:
            bvc = bvc/np.max(np.abs(bvc))*scale
        df_tr['stamp'] = pd.to_datetime(df_tr['time'], unit='s')
        self.metrics = pd.DataFrame({'stamp': df_tr['stamp'],'bvc': bvc})
        return self.metrics

    def estimate_intensity(self, times: np.ndarray, beta: float)->np.ndarray:
        intensities = [0.0]
        for i in range(1, len(times)):
            delta_t = times[i]-times[i-1]
            intensities.append(intensities[-1]*np.exp(-beta*delta_t) + 1)
        return np.array(intensities)

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.header("BVC Analysis with Hawkes, ACD, or ACI (Same Color Logic)")

symbol_bsi1 = st.sidebar.text_input("Enter Ticker Symbol", value="BTC/USD")
st.write(f"Fetching data for: {symbol_bsi1} with {global_lookback_minutes} minutes lookback, {timeframe} timeframe.")

try:
    prices_bsi = fetch_data(symbol_bsi1, timeframe, global_lookback_minutes)
    st.write(
        "Data range:",
        prices_bsi["stamp"].min(),
        "to",
        prices_bsi["stamp"].max()
    )
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close','volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close']/prices_bsi['close'].iloc[0])*1e4

bvc_choice = st.sidebar.selectbox("Select BVC Model", ["Hawkes","ACD","ACI"])

if bvc_choice=="Hawkes":
    hawkes = HawkesBVC(window=20, kappa=0.1)
    bvc_df = hawkes.eval(prices_bsi.reset_index())
elif bvc_choice=="ACD":
    try:
        df_trades = fetch_trades_kraken(symbol_bsi1, lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trades for ACD: {e}")
        st.stop()
    acd = ACDBVC(kappa=0.1)
    bvc_df = acd.eval(df_trades)
elif bvc_choice=="ACI":
    try:
        df_trades = fetch_trades_kraken(symbol_bsi1, lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trades for ACI: {e}")
        st.stop()
    aci = ACIBVC(kappa=0.1)
    bvc_df = aci.eval(df_trades)
else:
    st.stop()

df_plot = prices_bsi.reset_index().merge(bvc_df, on='stamp', how='left')
df_plot.sort_values('stamp', inplace=True)

# Same EXACT color logic as your Hawkes snippet:
fig, ax = plt.subplots(figsize=(10,4), dpi=120)

# 1) find min/max of BVC
bvc_min = df_plot['bvc'].min()
bvc_max = df_plot['bvc'].max()
norm_bvc = plt.Normalize(bvc_min, bvc_max)

for i in range(len(df_plot)-1):
    xvals = df_plot['stamp'].iloc[i:i+2]
    yvals = df_plot['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_plot['bvc'].iloc[i]

    if pd.isna(bvc_val):
        color = 'gray'
    else:
        if bvc_val>=0:
            color = plt.cm.Blues(norm_bvc(bvc_val))
        else:
            color = plt.cm.Reds(norm_bvc(bvc_val))
    ax.plot(xvals, yvals, color=color, linewidth=1)

ax.set_title(f"{bvc_choice} BVC with the same color logic as Hawkes", fontsize=10)
ax.set_xlabel("Time")
ax.set_ylabel("ScaledPrice")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Also plot the BVC in a separate figure
fig_bvc, ax_bvc = plt.subplots(figsize=(10,3), dpi=120)
ax_bvc.plot(df_plot['stamp'], df_plot['bvc'], color='blue', linewidth=1)
ax_bvc.set_title(f"{bvc_choice} BVC - Raw Values", fontsize=10)
ax_bvc.set_xlabel("Time")
ax_bvc.set_ylabel("BVC")
ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
st.pyplot(fig_bvc)
