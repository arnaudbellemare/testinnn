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
bvc_model = st.sidebar.selectbox("Select BVC Model", ["Hawkes", "ACD", "ACI"], key="bvc_model")

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
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.iloc[i] * labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': bvc})
        return self.metrics

    def _label(self, r: float, sigma: float):
        if sigma > 0.0:
            cum = t.cdf(r / sigma, df=self._dof)
            return 2 * cum - 1.0
        else:
            return 0.0

    def plot(self):
        return (
            ggplot(self.metrics, aes(x='stamp', y='bvc'))
            + geom_line(color='blue', size=0.5)
            + labs(title="B/S Imbalance", x="Time", y="BVC")
            + theme_minimal()
            + theme(figure_size=(11, 5))
        )

# ACDBVC: Custom simple ACD-like model
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
            bvc = np.zeros(len(df_tr), dtype=float)
            current_bvc = 0.0
            for i in range(len(df_tr)):
                current_bvc = current_bvc * np.exp(-self._kappa) + df_tr['weighted_volume'].iloc[i]
                bvc[i] = current_bvc
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

# ACIBVC: Custom simple ACI-like model
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
            bvc = np.zeros(len(df_tr), dtype=float)
            current_bvc = 0.0
            for i in range(len(df_tr)):
                current_bvc = current_bvc * np.exp(-self._kappa) + df_tr['weighted_volume'].iloc[i]
                bvc[i] = current_bvc
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
            intensity = intensities[-1] * np.exp(-beta * delta_t) + 1
            intensities.append(intensity)
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
# SECTION 1: Momentum, Skewness & BVC Analysis
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

if 'buyvolume' not in prices_bsi.columns or 
