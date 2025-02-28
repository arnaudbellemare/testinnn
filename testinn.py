import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ccxt
from numba import njit
from scipy.stats import norm, t
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme

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
global_lookback_label = st.sidebar.selectbox("Select Global Lookback Period", list(lookback_options.keys()), key="global_lookback_label")
global_lookback_minutes = lookback_options[global_lookback_label]
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h"], key="timeframe_widget")
# New analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Hawkes BVC", "ADC", "ACI"], key="analysis_type")

@njit(cache=True)
def ema(arr_in: np.ndarray, window: int, alpha: float = 0) -> np.ndarray:
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    n = arr_in.size
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = (arr_in[i] * alpha) + (ewma[i - 1] * (1 - alpha))
    return ewma

@njit(cache=True)
def bbw(klines: np.ndarray, length: int, multiplier: float) -> float:
    closes = klines[:, 4]
    dev = multiplier * np.std(closes[-length:])
    return 2 * dev

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
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    now = pd.to_datetime(now_ms, unit="ms")
    cutoff = now - pd.Timedelta(minutes=lookback_minutes)
    return df[df["stamp"] >= cutoff]

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
    all_trades = [trade for trade in all_trades if trade['timestamp'] >= cutoff_ts]
    if not all_trades:
        raise ValueError(f"No trades returned from Kraken in the last {lookback_minutes} minutes.")
    df_tr = pd.DataFrame(all_trades)
    # Keep time as seconds (float); convert later
    df_tr['time'] = df_tr['timestamp'] / 1000.0
    df_tr['vol'] = df_tr['amount']
    df_tr = df_tr[['time', 'price', 'vol']].copy()
    df_tr.sort_values('time', inplace=True)
    df_tr.reset_index(drop=True, inplace=True)
    return df_tr

def fraction_buy(prices):
    dp = np.diff(np.log(prices))
    if len(dp) < 2:
        return np.zeros_like(dp)
    mean_ = np.nanmean(dp)
    std_ = np.nanstd(dp)
    if std_ == 0:
        return np.zeros_like(dp)
    z = (dp - mean_) / std_
    return norm.cdf(z)

def vol_bin(volumes, w):
    group = np.zeros_like(volumes, dtype=int)
    cur_g = 0
    csum = 0
    for i in range(len(volumes)):
        csum += volumes[i]
        group[i] = cur_g
        if csum >= w:
            cur_g += 1
            csum = 0
    return group

# -----------------------------------------------------------------------------
# NEW FUNCTIONS FOR ADC and ACI Analysis
# -----------------------------------------------------------------------------
@njit(cache=True)
def directional_change(prices: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    n = len(prices)
    adc = np.zeros(n, dtype=np.float64)
    if n < 2:
        return adc
    ref_price = prices[0]
    direction = 0  # 0: not set, 1: up, -1: down
    cumulative_change = 0.0
    for i in range(1, n):
        pct_change = (prices[i] - ref_price) / ref_price
        if abs(pct_change) >= threshold:
            if direction == 0 or (direction == 1 and pct_change < -threshold) or (direction == -1 and pct_change > threshold):
                direction = 1 if pct_change > 0 else -1
                ref_price = prices[i]
                cumulative_change = direction * threshold
            else:
                cumulative_change += pct_change - (direction * threshold)
        adc[i] = cumulative_change
    return adc

@njit(cache=True)
def accumulated_candle_index(klines: np.ndarray, lookback: int = 20) -> np.ndarray:
    n = len(klines)
    aci = np.zeros(n, dtype=np.float64)
    if n < 2:
        return aci
    for i in range(1, n):
        open_price = klines[i, 1]
        high_price = klines[i, 2]
        low_price = klines[i, 3]
        close_price = klines[i, 4]
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        if close_price > open_price:
            candle_score = body_size + 0.5 * upper_shadow - 0.3 * lower_shadow
        else:
            candle_score = -body_size - 0.5 * lower_shadow + 0.3 * upper_shadow
        if i >= lookback:
            recent_highs = klines[i-lookback:i, 2]
            recent_lows = klines[i-lookback:i, 3]
            recent_range = np.mean(recent_highs - recent_lows)
            if recent_range > 0:
                candle_score = candle_score / recent_range
        if i > 1:
            aci[i] = 0.8 * aci[i-1] + candle_score
        else:
            aci[i] = candle_score
    return aci

# -----------------------------------------------------------------------------
# NEW CLASSES FOR ADC and ACI Analysis
# -----------------------------------------------------------------------------
class DirectionalChangeAnalysis:
    def __init__(self, window: int = 20, threshold: float = 0.005, kappa: float = 0.1):
        self._window = window
        self._threshold = threshold
        self._kappa = kappa
        self.metrics = None
        
    def eval(self, df: pd.DataFrame, scale=1e4):
        times = df['stamp']
        prices = df['close'].values
        adc_values = directional_change(prices, self._threshold)
        if self._kappa > 0:
            alpha_exp = np.exp(-self._kappa)
            smoothed_adc = np.zeros_like(adc_values)
            current_adc = 0.0
            for i in range(len(adc_values)):
                current_adc = current_adc * alpha_exp + adc_values[i]
                smoothed_adc[i] = current_adc
            adc_values = smoothed_adc
        if np.max(np.abs(adc_values)) != 0:
            adc_values = adc_values / np.max(np.abs(adc_values)) * scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': adc_values})
        return self.metrics

class AccumulatedCandleAnalysis:
    def __init__(self, window: int = 20, kappa: float = 0.1):
        self._window = window
        self._kappa = kappa
        self.metrics = None
        
    def eval(self, df: pd.DataFrame, scale=1e4):
        times = df['stamp']
        klines = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        aci_values = accumulated_candle_index(klines, self._window)
        if self._kappa > 0:
            alpha_exp = np.exp(-self._kappa)
            smoothed_aci = np.zeros_like(aci_values)
            current_aci = 0.0
            for i in range(len(aci_values)):
                current_aci = current_aci * alpha_exp + aci_values[i]
                smoothed_aci[i] = current_aci
            aci_values = smoothed_aci
        if np.max(np.abs(aci_values)) != 0:
            aci_values = aci_values / np.max(np.abs(aci_values)) * scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': aci_values})
        return self.metrics

# =============================================================================
# MAIN DASHBOARD LOGIC
# =============================================================================
st.header("Section 1: Momentum, Skewness & BVC Analysis")
symbol_bsi1 = st.sidebar.text_input("Enter Ticker Symbol (Sec 1)", value="BTC/USD", key="symbol_bsi1")
st.write(f"Fetching data for: **{symbol_bsi1}** with a global lookback of **{global_lookback_minutes}** minutes and timeframe **{timeframe}**.")

try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe=timeframe, lookback_minutes=global_lookback_minutes)
    st.write("Data range:", prices_bsi["stamp"].min(), "to", prices_bsi["stamp"].max())
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close', 'volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)
prices_bsi['ScaledPrice'] = np.log(prices_bsi['close'] / prices_bsi['close'].iloc[0]) * 1e4
prices_bsi['ScaledPrice_EMA'] = ema(prices_bsi['ScaledPrice'].values, window=10)
prices_bsi['cum_vol'] = prices_bsi['volume'].cumsum()
prices_bsi['cum_pv'] = (prices_bsi['close'] * prices_bsi['volume']).cumsum()
prices_bsi['vwap'] = prices_bsi['cum_pv'] / prices_bsi['cum_vol']
if prices_bsi['vwap'].iloc[0] == 0 or not np.isfinite(prices_bsi['vwap'].iloc[0]):
    st.warning("VWAP initial value is zero or invalid. Using ScaledPrice as fallback for VWAP plotting.")
    prices_bsi['vwap_transformed'] = prices_bsi['ScaledPrice']
else:
    prices_bsi['vwap_transformed'] = np.log(prices_bsi['vwap'] / prices_bsi['vwap'].iloc[0]) * 1e4
if 'buyvolume' not in prices_bsi.columns or 'sellvolume' not in prices_bsi.columns:
    prices_bsi['buyvolume'] = prices_bsi['volume'] * 0.5
    prices_bsi['sellvolume'] = prices_bsi['volume'] - prices_bsi['buyvolume']

# =============================================================================
# BVC MODEL EVALUATION & MERGE
# =============================================================================
if analysis_type == "Hawkes BVC":
    st.write("### Hawkes BVC Analysis")
    model = HawkesBVC(window=20, kappa=0.1)
    bvc_metrics = model.eval(prices_bsi.reset_index(), scale=1e4)
    indicator_title = "BVC"
elif analysis_type == "ADC":
    st.write("### Average Directional Change (ADC) Analysis")
    threshold_param = st.slider("Directional Change Threshold (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1) / 100
    adc_analysis = DirectionalChangeAnalysis(window=20, threshold=threshold_param, kappa=0.1)
    bvc_metrics = adc_analysis.eval(prices_bsi.reset_index(), scale=1e4)
    indicator_title = "ADC"
else:  # ACI
    st.write("### Accumulated Candle Index (ACI) Analysis")
    aci_analysis = AccumulatedCandleAnalysis(window=20, kappa=0.1)
    bvc_metrics = aci_analysis.eval(prices_bsi.reset_index(), scale=1e4)
    indicator_title = "ACI"

df_merged = prices_bsi.reset_index().merge(bvc_metrics, on='stamp', how='left')
df_merged.sort_values('stamp', inplace=True)
df_merged['bvc'] = df_merged['bvc'].fillna(method='ffill').fillna(0)

# Debug: Display BVC value summary
st.write("BVC Summary:", df_merged['bvc'].describe())

# Fixed normalization range for gradient coloring: -1 to 1
norm_bvc = plt.Normalize(-1, 1)

# =============================================================================
# PLOTTING SECTION - Price Chart Colored by Normalized BVC
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
for i in range(len(df_merged) - 1):
    xvals = df_merged['stamp'].iloc[i:i+2]
    yvals = df_merged['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_merged['bvc'].iloc[i]
    color = plt.cm.bwr(norm_bvc(bvc_val))
    ax.plot(xvals, yvals, color=color, linewidth=1.2)
ax.plot(df_merged['stamp'], df_merged['ScaledPrice_EMA'], color='black',
        linewidth=1, label="EMA(10)")
ax.plot(df_merged['stamp'], df_merged['vwap_transformed'], color='gray',
        linewidth=1, label="VWAP")
ax.set_xlabel("Time", fontsize=8)
ax.set_ylabel("ScaledPrice", fontsize=8)
ax.set_title(f"Price with EMA & VWAP (Colored by {indicator_title})", fontsize=10)
ax.legend(fontsize=7)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
ax.set_ylim(df_merged['ScaledPrice'].min() - 50, df_merged['ScaledPrice'].max() + 50)
plt.tight_layout()
st.pyplot(fig)

# =============================================================================
# BVC PLOT
# =============================================================================
fig_bvc, ax_bvc = plt.subplots(figsize=(10, 3), dpi=120)
ax_bvc.plot(bvc_metrics['stamp'], bvc_metrics['bvc'], color="blue", linewidth=1,
            label=f"{indicator_title}")
ax_bvc.set_xlabel("Time", fontsize=8)
ax_bvc.set_ylabel(indicator_title, fontsize=8)
ax_bvc.legend(fontsize=7)
ax_bvc.set_title(f"{indicator_title} Over Time", fontsize=10)
ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax_bvc.get_yticklabels(), fontsize=7)
st.pyplot(fig_bvc)
