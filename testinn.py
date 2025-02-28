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
    # If alpha not provided, use default formula: 3/(window+1)
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
    all_trades = [trade for trade in all_trades if trade['timestamp'] >= cutoff_ts]
    if not all_trades:
        raise ValueError(f"No trades returned from Kraken in the last {lookback_minutes} minutes.")
    df_tr = pd.DataFrame(all_trades)
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

# =============================================================================
# CLASSES FOR ANALYSIS
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
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        # Use .values for indexing to ensure compatibility with numpy arrays
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.values[i] * labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': bvc})
        return self.metrics

    def _label(self, r: float, sigma: float):
        if sigma > 0.0:
            val = t.cdf(r / sigma, df=self._dof)
            return 2 * val - 1.0
        else:
            return 0.0

class ACDBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4) -> pd.DataFrame:
        try:
            df_tr = df_tr.dropna(subset=['time', 'price', 'vol']).copy()
            df_tr['duration'] = df_tr['time'].diff().shift(-1)
            df_tr = df_tr.dropna(subset=['duration'])
            df_tr = df_tr[df_tr['duration'] > 0]
            if len(df_tr) < 10:
                raise ValueError("Insufficient trade data for custom ACD model.")
            
            mean_duration = df_tr['duration'].mean()
            std_duration = df_tr['duration'].std() or 1e-10
            df_tr['standardized_residual'] = (df_tr['duration'] - mean_duration) / std_duration
            df_tr['price_change'] = np.log(df_tr['price'] / df_tr['price'].shift(1)).fillna(0)
            df_tr['label'] = -df_tr['standardized_residual'] * df_tr['price_change']
            df_tr['weighted_volume'] = df_tr['vol'] * df_tr['label']
            alpha_exp = np.exp(-self._kappa)
            bvc_list = []
            current_bvc = 0.0
            # Use a standard loop over the weighted_volume values
            for wv in df_tr['weighted_volume'].values:
                current_bvc = current_bvc * alpha_exp + wv
                bvc_list.append(current_bvc)
            bvc = np.array(bvc_list)
            if np.max(np.abs(bvc)) != 0:
                bvc = bvc / np.max(np.abs(bvc)) * scale
            self.metrics = pd.DataFrame({
                'stamp': pd.to_datetime(df_tr['time'], unit='s'),
                'bvc': bvc
            })
            return self.metrics
        except Exception as e:
            st.error(f"Error in ACDBVC model: {e}")
            return pd.DataFrame()

class ACIBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame, scale=1e4) -> pd.DataFrame:
        try:
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
            for wv in df_tr['weighted_volume'].values:
                current_bvc = current_bvc * alpha_exp + wv
                bvc_list.append(current_bvc)
            bvc = np.array(bvc_list)
            if np.max(np.abs(bvc)) != 0:
                bvc = bvc / np.max(np.abs(bvc)) * scale
            self.metrics = pd.DataFrame({
                'stamp': pd.to_datetime(df_tr['time'], unit='s'),
                'bvc': bvc
            })
            return self.metrics
        except Exception as e:
            st.error(f"Error in ACIBVC model: {e}")
            return pd.DataFrame()

    def estimate_intensity(self, times: np.ndarray, beta: float) -> np.ndarray:
        intensities = [0.0]
        for i in range(1, len(times)):
            delta_t = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-beta * delta_t) + 1)
        return np.array(intensities)

class TradeClassification:
    def __init__(self, df_tr):
        self.df_tr = df_tr
    def classify(self, method='bvc', freq=0, window=60, window_type='time'):
        if method != 'bvc':
            raise ValueError("Only 'bvc' method is implemented.")
        if window_type == 'time':
            self.df_tr['group'] = (self.df_tr['time'] // window).astype(int)
        elif window_type == 'vol':
            self.df_tr['group'] = vol_bin(self.df_tr['vol'].values.astype(int), window)
        else:
            raise ValueError("window_type must be 'time' or 'vol'.")
        grouped = self.df_tr.groupby('group')
        group_keys = sorted(grouped.groups.keys())
        last_prices = []
        volumes = []
        for g in group_keys:
            chunk = grouped.get_group(g)
            last_price = chunk['price'].iloc[-1]
            total_vol = chunk['vol'].sum()
            last_prices.append(last_price)
            volumes.append(total_vol)
        last_prices = np.array(last_prices, dtype=float)
        volumes = np.array(volumes, dtype=float)
        f_b = np.zeros_like(last_prices, dtype=float)
        if len(last_prices) > 1:
            f_b[1:] = fraction_buy(last_prices)
        df_out = pd.DataFrame({'f_b': f_b, 'vol': volumes}, index=group_keys)
        df_out['buy_vol'] = df_out['f_b'] * df_out['vol']
        self.df_tr['Initiator'] = 0
        return df_out

# =============================================================================
# MAIN DASHBOARD LOGIC
# =============================================================================
st.header("Section 1: Momentum, Skewness & BVC Analysis")

symbol_bsi1 = st.sidebar.text_input(
    "Enter Ticker Symbol (Sec 1)", value="BTC/USD", key="symbol_bsi1"
)
st.write(f"Fetching data for: **{symbol_bsi1}** with a global lookback of **{global_lookback_minutes}** minutes and timeframe **{timeframe}**.")

# Fetch OHLC data
try:
    prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe=timeframe, lookback_minutes=global_lookback_minutes)
    st.write("Data range:", prices_bsi["stamp"].min(), "to", prices_bsi["stamp"].max())
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

prices_bsi.dropna(subset=['close', 'volume'], inplace=True)
prices_bsi['stamp'] = pd.to_datetime(prices_bsi['stamp'])
prices_bsi.set_index('stamp', inplace=True)

# Compute scaled price and EMA
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

if 'buyvolume' not in prices_bsi.columns or 'sellvolume' not in prices_bsi.columns:
    prices_bsi['buyvolume'] = prices_bsi['volume'] * 0.5
    prices_bsi['sellvolume'] = prices_bsi['volume'] - prices_bsi['buyvolume']

# Prepare skewness analysis DataFrame
st.write("### Skewness Analysis")
df_skew = prices_bsi.copy()
df_skew['hlc3'] = (df_skew['high'] + df_skew['low'] + df_skew['close']) / 3.0
SkewLength = 14
alpha_val = 2.0 / (1.0 + SkewLength)
df_skew['TrueRange'] = (np.abs(df_skew['hlc3'] - df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0]))
                         / df_skew['hlc3'].shift(1, fill_value=df_skew['hlc3'].iloc[0]))
dev_max_series = []
dev_min_series = []
dev_max_prev, dev_min_prev = 1.618, 1.618
for i in range(len(df_skew)):
    if i == 0:
        dev_max_series.append(dev_max_prev)
        dev_min_series.append(dev_min_prev)
    else:
        current_tr = df_skew['TrueRange'].iloc[i]
        prior_hlc3 = df_skew['hlc3'].iloc[i - 1]
        current_hlc3 = df_skew['hlc3'].iloc[i]
        if current_hlc3 > prior_hlc3:
            dev_max_prev = alpha_val * current_tr + (1 - alpha_val) * dev_max_prev
        else:
            dev_max_prev = alpha_val * 0 + (1 - alpha_val) * dev_max_prev
        if current_hlc3 < prior_hlc3:
            dev_min_prev = alpha_val * current_tr + (1 - alpha_val) * dev_min_prev
        else:
            dev_min_prev = alpha_val * 0 + (1 - alpha_val) * dev_min_prev
        dev_max_series.append(dev_max_prev)
        dev_min_series.append(dev_min_prev)
df_skew['deviation_max'] = dev_max_series
df_skew['deviation_min'] = dev_min_series
df_skew['normalized_skew'] = (df_skew['deviation_max'] / df_skew['deviation_min'] - 1) * 3
df_skew['normalized_z'] = (df_skew['normalized_skew'] + 3) / 6
df_skew['normalized_z'] = df_skew['normalized_z'].ffill().bfill()
df_skew['ScaledPrice'] = np.log(df_skew['close'] / df_skew['close'].iloc[0]) * 1e4
ema_window = 10
df_skew['ScaledPrice_EMA'] = ema(df_skew['ScaledPrice'].values, ema_window)

# =============================================================================
# BVC MODEL EVALUATION
# =============================================================================
if bvc_model == "Hawkes":
    model = HawkesBVC(window=20, kappa=0.1)
    bvc_metrics = model.eval(prices_bsi.reset_index())
elif bvc_model in ["ACD", "ACI"]:
    try:
        trades_df = fetch_trades_kraken(symbol=symbol_bsi1, lookback_minutes=global_lookback_minutes)
    except Exception as e:
        st.error(f"Error fetching trade data: {e}")
        st.stop()
    if bvc_model == "ACD":
        model = ACDBVC(kappa=0.1)
    else:
        model = ACIBVC(kappa=0.1)
    bvc_metrics = model.eval(trades_df)
else:
    st.error("Invalid BVC model selection.")
    st.stop()

# Merge BVC metrics with price data
df_merged = prices_bsi.reset_index().merge(bvc_metrics, on='stamp', how='left')
df_merged.sort_values('stamp', inplace=True)
df_merged['bvc'] = df_merged['bvc'].fillna(method='ffill').fillna(0)

global_min = df_merged['ScaledPrice'].min()
global_max = df_merged['ScaledPrice'].max()

# Force a symmetric color scale around zero for clear distinction
bvc_min = df_merged['bvc'].min()
bvc_max = df_merged['bvc'].max()
abs_max = max(abs(bvc_min), abs(bvc_max))

# =============================================================================
# PLOTTING SECTION - Price Chart Colored by Normalized BVC
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
norm_bvc = plt.Normalize(-abs_max, abs_max)
for i in range(len(df_merged) - 1):
    xvals = df_merged['stamp'].iloc[i:i+2]
    yvals = df_merged['ScaledPrice'].iloc[i:i+2]
    bvc_val = df_merged['bvc'].iloc[i]
    cmap_bvc = plt.cm.get_cmap('bwr')  # Blue-White-Red colormap
    color = cmap_bvc(norm_bvc(bvc_val))
    ax.plot(xvals, yvals, color=color, linewidth=1.2)
ax.plot(df_merged['stamp'], df_merged['ScaledPrice_EMA'], color='black',
        linewidth=1, label="EMA(10)")
ax.plot(df_merged['stamp'], df_merged['vwap_transformed'], color='gray',
        linewidth=1, label="VWAP")
ax.set_xlabel("Time", fontsize=8)
ax.set_ylabel("ScaledPrice", fontsize=8)
ax.set_title(f"Price with EMA & VWAP (Colored by {bvc_model} BVC)", fontsize=10)
ax.legend(fontsize=7)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
ax.set_ylim(global_min - 50, global_max + 50)
plt.tight_layout()
st.pyplot(fig)

# =============================================================================
# BVC PLOT
# =============================================================================
fig_bvc, ax_bvc = plt.subplots(figsize=(10, 3), dpi=120)
ax_bvc.plot(bvc_metrics['stamp'], bvc_metrics['bvc'], color="blue", linewidth=1,
            label=f"BVC ({bvc_model})")
ax_bvc.set_xlabel("Time", fontsize=8)
ax_bvc.set_ylabel("BVC", fontsize=8)
ax_bvc.legend(fontsize=7)
ax_bvc.set_title("BVC Over Time", fontsize=10)
ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right', fontsize=7)
plt.setp(ax_bvc.get_yticklabels(), fontsize=7)
st.pyplot(fig_bvc)
