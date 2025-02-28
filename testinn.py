import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plotnine import *
from scipy import stats
import ccxt
from typing import Optional

# Streamlit page configuration
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide")

# Helper Functions
def fetch_data(symbol: str, timeframe: str, lookback_minutes: int) -> Optional[pd.DataFrame]:
    try:
        exchange = ccxt.kraken()
        now_ms = exchange.milliseconds()
        since = now_ms - lookback_minutes * 60 * 1000
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=720)  # Kraken max limit per call
        if not ohlcv:
            raise ValueError("No OHLCV data returned from Kraken.")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['stamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[df['stamp'] >= pd.Timestamp(now_ms - lookback_minutes * 60 * 1000, unit='ms')]
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return None

def fetch_trades_kraken(symbol: str, lookback_minutes: int) -> Optional[pd.DataFrame]:
    try:
        exchange = ccxt.kraken()
        now_ms = exchange.milliseconds()
        since = now_ms - lookback_minutes * 60 * 1000
        all_trades = []
        while True:
            trades = exchange.fetch_trades(symbol, since=since, limit=1000)  # Kraken max limit
            if not trades:
                break
            all_trades.extend(trades)
            since = trades[-1]['timestamp'] + 1
            if since >= now_ms:
                break
        if not all_trades:
            raise ValueError("No trade data returned from Kraken.")
        df_tr = pd.DataFrame(all_trades)
        df_tr['time'] = df_tr['timestamp'] / 1000.0
        df_tr['price'] = df_tr['price']
        df_tr['vol'] = df_tr['amount']
        df_tr = df_tr[['time', 'price', 'vol']].sort_values('time').reset_index(drop=True)
        df_tr = df_tr[df_tr['time'] >= (now_ms / 1000.0 - lookback_minutes * 60)]
        return df_tr
    except Exception as e:
        st.error(f"Error fetching trade data: {e}")
        return None

def ema(data: pd.Series, alpha: float) -> pd.Series:
    return data.ewm(alpha=alpha, adjust=False).mean()

class ScaledPrice:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaled = self._scale_prices()

    def _scale_prices(self):
        min_price = self.df['close'].min()
        max_price = self.df['close'].max()
        if max_price == min_price:
            return pd.Series(np.zeros(len(self.df)), index=self.df.index)
        return (self.df['close'] - min_price) / (max_price - min_price)

# HawkesBVC Class
class HawkesBVC:
    def __init__(self, window: int, kappa: float):
        self._window = window
        self._kappa = kappa
        self.metrics = None

    def eval(self, df: pd.DataFrame):
        df = df.copy()
        df['logret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        rolling_std = df['logret'].rolling(self._window, min_periods=1).std().fillna(0)
        standardized = df['logret'] / rolling_std.replace(0, np.nan).fillna(1e-10)
        df['label'] = stats.t.cdf(standardized, df=5)
        df['weighted_volume'] = df['volume'] * df['label']
        bvc = [0] * len(df)
        for i in range(len(df)):
            if i == 0:
                bvc[i] = df['weighted_volume'].iloc[i]
            else:
                bvc[i] = bvc[i-1] * np.exp(-self._kappa) + df['weighted_volume'].iloc[i]
        self.metrics = pd.DataFrame({'stamp': df['stamp'], 'bvc': bvc})
        return self.metrics

# ACDBVC Class with a custom simple ACD-like estimation
class ACDBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame) -> pd.DataFrame:
        try:
            df_tr = df_tr.dropna(subset=['time', 'price', 'vol']).copy()
            df_tr['duration'] = df_tr['time'].diff().shift(-1)
            df_tr = df_tr.dropna(subset=['duration'])
            df_tr = df_tr[df_tr['duration'] > 0]
            if len(df_tr) < 10:
                raise ValueError("Insufficient trade data for custom ACD model.")
            
            # Custom simple ACD-like model: standardize durations
            mean_duration = df_tr['duration'].mean()
            std_duration = df_tr['duration'].std()
            if std_duration == 0:
                std_duration = 1e-10
            df_tr['standardized_residual'] = (df_tr['duration'] - mean_duration) / std_duration

            df_tr['price_change'] = np.log(df_tr['price'] / df_tr['price'].shift(1)).fillna(0)
            df_tr['label'] = -df_tr['standardized_residual'] * df_tr['price_change']
            df_tr['weighted_volume'] = df_tr['vol'] * df_tr['label']
            bvc = [0] * len(df_tr)
            for i in range(len(df_tr)):
                if i == 0:
                    bvc[i] = df_tr['weighted_volume'].iloc[i]
                else:
                    bvc[i] = bvc[i-1] * np.exp(-self._kappa) + df_tr['weighted_volume'].iloc[i]
            self.metrics = pd.DataFrame({
                'stamp': pd.to_datetime(df_tr['time'], unit='s'),
                'bvc_acd': bvc
            })
            return self.metrics
        except Exception as e:
            st.error(f"Error fitting custom ACD model: {e}")
            return pd.DataFrame()

# ACIBVC Class
def estimate_intensity(times: np.ndarray, beta: float) -> np.ndarray:
    intensities = [0]
    for i in range(1, len(times)):
        delta_t = times[i] - times[i-1]
        intensity = intensities[-1] * np.exp(-beta * delta_t) + 1
        intensities.append(intensity)
    return np.array(intensities)

class ACIBVC:
    def __init__(self, kappa: float):
        self._kappa = kappa
        self.metrics = None

    def eval(self, df_tr: pd.DataFrame) -> pd.DataFrame:
        df_tr = df_tr.dropna(subset=['time', 'price', 'vol']).copy()
        times = df_tr['time'].values
        intensities = estimate_intensity(times, self._kappa)
        df_tr = df_tr.iloc[:len(intensities)]
        df_tr['intensity'] = intensities
        df_tr['price_change'] = np.log(df_tr['price'] / df_tr['price'].shift(1)).fillna(0)
        df_tr['label'] = df_tr['intensity'] * df_tr['price_change']
        df_tr['weighted_volume'] = df_tr['vol'] * df_tr['label']
        bvc = [0] * len(df_tr)
        for i in range(len(df_tr)):
            if i == 0:
                bvc[i] = df_tr['weighted_volume'].iloc[i]
            else:
                bvc[i] = bvc[i-1] * np.exp(-self._kappa) + df_tr['weighted_volume'].iloc[i]
        self.metrics = pd.DataFrame({
            'stamp': pd.to_datetime(df_tr['time'], unit='s'),
            'bvc_aci': bvc
        })
        return self.metrics

# Streamlit Application
def main():
    st.title("Financial Analysis Dashboard")

    # Sidebar settings
    st.sidebar.header("Global Settings")
    global_lookback_minutes = st.sidebar.number_input(
        "Global Lookback (minutes)", min_value=60, max_value=10080, value=1440, step=60
    )
    timeframe = st.sidebar.selectbox(
        "Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=1
    )

    # Section 1: Momentum, Skewness & BVC Analysis
    st.header("Section 1: Momentum, Skewness & BVC Analysis")

    symbol_bsi1 = st.sidebar.text_input("Enter Ticker Symbol (e.g., BTC/USD)", value="BTC/USD", key="symbol_bsi1")
    bvc_method = st.sidebar.selectbox("Select BVC Method", ["Hawkes", "ACD", "ACI"], key="bvc_method")
    st.write(f"Fetching data for: **{symbol_bsi1}** with a lookback of **{global_lookback_minutes}** minutes and timeframe **{timeframe}**.")

    # Fetch OHLCV data
    with st.spinner("Fetching OHLCV data..."):
        prices_bsi = fetch_data(symbol=symbol_bsi1, timeframe=timeframe, lookback_minutes=global_lookback_minutes)
    if prices_bsi is None or prices_bsi.empty:
        st.stop()

    # Compute additional metrics
    scaled_price = ScaledPrice(prices_bsi)
    prices_bsi['ScaledPrice'] = scaled_price.scaled
    prices_bsi['EMA'] = ema(prices_bsi['close'], alpha=0.05)
    prices_bsi['VWAP'] = (prices_bsi['close'] * prices_bsi['volume']).cumsum() / prices_bsi['volume'].cumsum()

    # Price Plot
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(prices_bsi['stamp'], prices_bsi['close'], linewidth=0.8, label="Price", color="black")
    ax.plot(prices_bsi['stamp'], prices_bsi['EMA'], linewidth=0.8, label="EMA", color="orange")
    ax.plot(prices_bsi['stamp'], prices_bsi['VWAP'], linewidth=0.8, label="VWAP", color="purple")
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Price", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title(f"Price with EMA and VWAP for {symbol_bsi1}", fontsize=10)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    st.pyplot(fig)

    # Compute and Plot BVC
    if bvc_method == "Hawkes":
        hawkes_bvc = HawkesBVC(window=20, kappa=0.1)
        bvc_metrics = hawkes_bvc.eval(prices_bsi)
        bvc_column = 'bvc'
        label = "BVC (Hawkes)"
    else:
        with st.spinner("Fetching trade data..."):
            df_tr = fetch_trades_kraken(symbol=symbol_bsi1, lookback_minutes=global_lookback_minutes)
        if df_tr is None or df_tr.empty:
            st.warning("No trade data available for BVC calculation.")
            st.stop()
        if bvc_method == "ACD":
            bvc = ACDBVC(kappa=0.1)
            bvc_metrics = bvc.eval(df_tr)
            bvc_column = 'bvc_acd'
            label = "BVC (ACD)"
        elif bvc_method == "ACI":
            bvc = ACIBVC(kappa=0.1)
            bvc_metrics = bvc.eval(df_tr)
            bvc_column = 'bvc_aci'
            label = "BVC (ACI)"

    if bvc_metrics.empty:
        st.warning(f"No {label} metrics available to plot.")
    else:
        # BVC Plot
        fig_bvc, ax_bvc = plt.subplots(figsize=(10, 3), dpi=120)
        ax_bvc.plot(bvc_metrics['stamp'], bvc_metrics[bvc_column], color="blue", linewidth=0.8, label=label)
        ax_bvc.set_xlabel("Time", fontsize=8)
        ax_bvc.set_ylabel("BVC", fontsize=8)
        ax_bvc.legend(fontsize=7)
        ax_bvc.set_title(f"{label}", fontsize=10)
        ax_bvc.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_bvc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax_bvc.get_xticklabels(), rotation=30, ha='right', fontsize=7)
        plt.setp(ax_bvc.get_yticklabels(), fontsize=7)
        st.pyplot(fig_bvc)

if __name__ == "__main__":
    main()

