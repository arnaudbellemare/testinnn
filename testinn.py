import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ccxt
from scipy.stats import t

# =============================================================================
# SETUP
# =============================================================================
st.set_page_config(layout="wide")
st.title("BVC Model Dashboard")

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.header("Parameters")
symbol = st.sidebar.selectbox("Symbol", ["BTC/USD", "ETH/USD"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
lookback = st.sidebar.selectbox("Lookback", ["4h", "12h", "24h", "3d"])
model_type = st.sidebar.radio("Model", ["Hawkes", "ACD", "ACI"])

# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=60)
def get_data(symbol, timeframe, lookback):
    # Convert lookback to minutes
    lookback_map = {"4h": 240, "12h": 720, "24h": 1440, "3d": 4320}
    exchange = ccxt.kraken()
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(
        symbol, 
        timeframe, 
        since=exchange.milliseconds() - lookback_map[lookback]*60*1000
    )
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Fetch trades
    trades = exchange.fetch_trades(
        symbol, 
        since=exchange.milliseconds() - lookback_map[lookback]*60*1000
    )
    trades_df = pd.DataFrame(trades)[['timestamp', 'price', 'amount']]
    trades_df["time"] = pd.to_datetime(trades_df["timestamp"], unit="ms")
    trades_df["price"] = trades_df["price"].astype(float)
    
    return df, trades_df

# =============================================================================
# BVC MODELS
# =============================================================================
class BVCModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.params = {
            "hawkes_kappa": 0.15,
            "acd_kappa": 0.1,
            "aci_kappa": 0.1
        }
        
    def process(self, ohlc_df, trades_df):
        if self.model_type == "Hawkes":
            return self._hawkes(ohlc_df)
        elif self.model_type == "ACD":
            return self._acd(trades_df)
        else:
            return self._aci(trades_df)
    
    def _hawkes(self, df):
        df = df.copy()
        returns = np.log(df["close"]).diff()
        sigma = returns.rolling(20, min_periods=1).std()
        labels = 2 * t.cdf(returns/sigma, df=3) - 1
        
        bvc = np.zeros(len(df))
        alpha = np.exp(-self.params["hawkes_kappa"])
        current = 0
        
        for i in range(1, len(df)):
            current = current * alpha + df["volume"].iloc[i] * labels.iloc[i]
            bvc[i] = current
            
        return pd.DataFrame({
            "time": df["time"],
            "bvc": bvc
        })
    
    def _acd(self, trades):
        df = trades.copy()
        df["duration"] = df["time"].diff().dt.total_seconds().shift(-1)
        df["residual"] = (df["duration"] - df["duration"].mean()) / df["duration"].std()
        df["price_change"] = np.log(df["price"]).diff()
        df["label"] = -df["residual"] * df["price_change"]
        
        bvc = np.zeros(len(df))
        alpha = np.exp(-self.params["acd_kappa"])
        current = 0
        
        for i in range(len(df)):
            current = current * alpha + df["amount"].iloc[i] * df["label"].iloc[i]
            bvc[i] = current
            
        return self._align_to_ohlc(df, bvc)
    
    def _aci(self, trades):
        df = trades.copy()
        times = df["time"].values.astype(np.int64) // 10**9  # Convert to seconds
        intensities = [0.0]
        
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-self.params["aci_kappa"] * dt) + 1)
            
        df["price_change"] = np.log(df["price"]).diff()
        df["label"] = np.array(intensities) * df["price_change"]
        
        bvc = np.zeros(len(df))
        alpha = np.exp(-self.params["aci_kappa"])
        current = 0
        
        for i in range(len(df)):
            current = current * alpha + df["amount"].iloc[i] * df["label"].iloc[i]
            bvc[i] = current
            
        return self._align_to_ohlc(df, bvc)
    
    def _align_to_ohlc(self, trades_df, bvc_values):
        # Critical alignment step
        aligned = pd.DataFrame({
            "time": trades_df["time"],
            "bvc": bvc_values
        })
        aligned = aligned.set_index("time").resample(timeframe).last().ffill().reset_index()
        return aligned

# =============================================================================
# MAIN DISPLAY
# =============================================================================
ohlc_data, trades_data = get_data(symbol, timeframe, lookback)

# Initialize and process model
model = BVCModel(model_type)
bvc_results = model.process(ohlc_data, trades_data)

# Merge data
merged = ohlc_data.merge(bvc_results, on="time", how="left").ffill()

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

# Price plot with BVC coloring
norm = plt.Normalize(merged["bvc"].min(), merged["bvc"].max())
cmap = plt.cm.coolwarm

for i in range(len(merged)-1):
    ax1.plot(
        merged["time"].iloc[i:i+2],
        merged["close"].iloc[i:i+2],
        color=cmap(norm(merged["bvc"].iloc[i])),
        linewidth=2
    )

# BVC plot
ax2.fill_between(merged["time"], merged["bvc"], alpha=0.4, color='purple')
ax2.plot(merged["time"], merged["bvc"], color='darkviolet', linewidth=0.8)

# Formatting
ax1.set_title(f"{symbol} Price Colored by {model_type} BVC", fontsize=14)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax2.set_title("BVC Value", fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()

st.pyplot(fig)
