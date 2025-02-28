import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ccxt
from scipy.stats import t

# =============================================================================
# CORE FIX: PROPER TIME ALIGNMENT
# =============================================================================
def calculate_bvc(model_type, ohlc_df, trades_df, timeframe):
    # Common processing for trade-based models
    if model_type in ["ACD", "ACI"]:
        # Calculate micro-BVC at trade level
        trades_df = trades_df.sort_values('time')
        
        if model_type == "ACD":
            # ACD-specific calculations
            durations = trades_df['time'].diff().dt.total_seconds().shift(-1)
            residuals = (durations - durations.mean()) / durations.std()
            price_changes = np.log(trades_df['price']).diff()
            trades_df['bvc'] = -residuals * price_changes * trades_df['amount']
        else:  # ACI
            times = trades_df['time'].values.astype(np.int64) // 10**9
            intensities = [0.0]
            for i in range(1, len(times)):
                dt = times[i] - times[i-1]
                intensities.append(intensities[-1] * np.exp(-0.1 * dt) + 1)
            price_changes = np.log(trades_df['price']).diff()
            trades_df['bvc'] = np.array(intensities) * price_changes * trades_df['amount']

        # Critical alignment step
        trades_df['time'] = trades_df['time'].dt.floor(timeframe)
        bvc_df = trades_df.groupby('time')['bvc'].sum().reset_index()
        bvc_df['bvc'] = bvc_df['bvc'].cumsum()
        
    else:  # Hawkes
        returns = np.log(ohlc_df["close"]).diff()
        sigma = returns.rolling(20, min_periods=1).std()
        labels = 2 * t.cdf(returns/sigma, df=3).clip(0, 1) - 1
        bvc_df = pd.DataFrame({
            'time': ohlc_df['time'],
            'bvc': (labels * ohlc_df['volume']).cumsum()
        })

    # Normalize across all models
    bvc_df['bvc'] = (bvc_df['bvc'] - bvc_df['bvc'].mean()) / bvc_df['bvc'].std()
    return bvc_df

# =============================================================================
# DATA FETCHING & VISUALIZATION
# =============================================================================
@st.cache_data
def get_data(symbol, timeframe, lookback_minutes):
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=exchange.milliseconds()-lookback_minutes*60*1000)
    ohlc_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ohlc_df["time"] = pd.to_datetime(ohlc_df["timestamp"], unit="ms").dt.floor(timeframe)
    
    trades = exchange.fetch_trades(symbol, since=exchange.milliseconds()-lookback_minutes*60*1000)
    trades_df = pd.DataFrame(trades)[['timestamp', 'price', 'amount']]
    trades_df["time"] = pd.to_datetime(trades_df["timestamp"], unit="ms")
    trades_df["price"] = trades_df["price"].astype(float)
    
    return ohlc_df, trades_df

def plot_colored_price(ohlc_df, bvc_df):
    merged = ohlc_df.merge(bvc_df, on='time', how='left').ffill()
    norm = plt.Normalize(merged["bvc"].min(), merged["bvc"].max())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(len(merged)-1):
        ax.plot(
            merged["time"].iloc[i:i+2],
            merged["close"].iloc[i:i+2],
            color=plt.cm.coolwarm(norm(merged["bvc"].iloc[i])),
            linewidth=2,
            solid_capstyle='round'
        )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =============================================================================
# STREAMLIT APP
# =============================================================================
st.title("BVC Color Alignment Fix")
model_type = st.sidebar.selectbox("Model", ["Hawkes", "ACD", "ACI"])
symbol = st.sidebar.selectbox("Symbol", ["BTC/USD", "ETH/USD"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
lookback = st.sidebar.selectbox("Lookback", ["1h", "4h", "12h", "24h"])

lookback_map = {"1h": 60, "4h": 240, "12h": 720, "24h": 1440}
ohlc_data, trades_data = get_data(symbol, timeframe, lookback_map[lookback])

bvc_data = calculate_bvc(model_type, ohlc_data, trades_data, timeframe)
plot_colored_price(ohlc_data, bvc_data)
