# Add this to the GLOBAL VARIABLES & HELPER FUNCTIONS section
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Hawkes BVC", "ADC", "ACI"], key="analysis_type")

# Add these functions to the helper functions section
@njit(cache=True)
def directional_change(prices: NDArray, threshold: float = 0.005) -> NDArray:
    """
    Calculate Average Directional Change (ADC) indicator.
    
    Parameters:
    ----------
    prices : NDArray
        Array of price values.
    threshold : float
        Minimum percentage change to consider a directional change.
        
    Returns:
    -------
    NDArray : Array of ADC values.
    """
    n = len(prices)
    adc = np.zeros(n, dtype=np.float64)
    
    if n < 2:
        return adc
    
    # Initial reference price
    ref_price = prices[0]
    direction = 0  # 0 = no direction yet, 1 = uptrend, -1 = downtrend
    cumulative_change = 0.0
    
    for i in range(1, n):
        # Calculate percentage change from reference price
        pct_change = (prices[i] - ref_price) / ref_price
        
        # Determine if direction changed
        if abs(pct_change) >= threshold:
            if (direction == 1 and pct_change < -threshold) or (direction == -1 and pct_change > threshold) or direction == 0:
                # Direction changed or initialized
                direction = 1 if pct_change > 0 else -1
                ref_price = prices[i]
                cumulative_change = direction * threshold  # Start with the threshold value
            else:
                # Same direction, accumulate the change
                cumulative_change += pct_change - (direction * threshold)
        
        # Store ADC value
        adc[i] = cumulative_change
    
    return adc

def accumulated_candle_index(klines: NDArray, lookback: int = 20) -> NDArray:
    """
    Calculate Accumulated Candle Index (ACI) based on candle patterns.
    
    Parameters:
    ----------
    klines : NDArray
        Array of OHLCV candle data.
    lookback : int
        Number of candles to consider for normalization.
        
    Returns:
    -------
    NDArray : Array of ACI values.
    """
    n = len(klines)
    aci = np.zeros(n, dtype=np.float64)
    
    if n < 2:
        return aci
    
    for i in range(1, n):
        # Calculate candle body and shadows
        open_price = klines[i, 1]
        high_price = klines[i, 2]
        low_price = klines[i, 3]
        close_price = klines[i, 4]
        
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # Calculate candle score based on bullish/bearish properties
        if close_price > open_price:  # Bullish candle
            candle_score = body_size + 0.5 * upper_shadow - 0.3 * lower_shadow
        else:  # Bearish candle
            candle_score = -body_size - 0.5 * lower_shadow + 0.3 * upper_shadow
        
        # Normalize score using recent volatility
        if i >= lookback:
            recent_highs = klines[i-lookback:i, 2]
            recent_lows = klines[i-lookback:i, 3]
            recent_range = np.mean(recent_highs - recent_lows)
            
            if recent_range > 0:
                candle_score = candle_score / recent_range
        
        # Accumulate the score (with decay)
        if i > 1:
            aci[i] = 0.8 * aci[i-1] + candle_score
        else:
            aci[i] = candle_score
    
    return aci

# Add this class for ADC analysis
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
        
        # Apply exponential smoothing if kappa is provided
        if self._kappa > 0:
            alpha_exp = np.exp(-self._kappa)
            smoothed_adc = np.zeros_like(adc_values)
            current_adc = 0.0
            
            for i in range(len(adc_values)):
                current_adc = current_adc * alpha_exp + adc_values[i]
                smoothed_adc[i] = current_adc
            
            adc_values = smoothed_adc
        
        # Scale ADC values 
        if np.max(np.abs(adc_values)) != 0:
            adc_values = adc_values / np.max(np.abs(adc_values)) * scale
            
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': adc_values})
        return self.metrics

# Add this class for ACI analysis
class AccumulatedCandleAnalysis:
    def __init__(self, window: int = 20, kappa: float = 0.1):
        self._window = window
        self._kappa = kappa
        self.metrics = None
        
    def eval(self, df: pd.DataFrame, scale=1e4):
        times = df['stamp']
        
        # Convert DataFrame to numpy array for OHLCV
        klines = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        
        # Calculate ACI
        aci_values = accumulated_candle_index(klines, self._window)
        
        # Apply exponential smoothing if kappa is provided
        if self._kappa > 0:
            alpha_exp = np.exp(-self._kappa)
            smoothed_aci = np.zeros_like(aci_values)
            current_aci = 0.0
            
            for i in range(len(aci_values)):
                current_aci = current_aci * alpha_exp + aci_values[i]
                smoothed_aci[i] = current_aci
            
            aci_values = smoothed_aci
        
        # Scale ACI values
        if np.max(np.abs(aci_values)) != 0:
            aci_values = aci_values / np.max(np.abs(aci_values)) * scale
            
        self.metrics = pd.DataFrame({'stamp': times, 'bvc': aci_values})
        return self.metrics

# Replace the HawkesBVC evaluation code in SECTION 1 with this:
st.write("### Analysis Options")
window_param = st.slider("Window Length", min_value=5, max_value=50, value=20, step=1)
kappa_param = st.slider("Kappa (Decay Factor)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

if analysis_type == "Hawkes BVC":
    st.write("### Hawkes BVC Analysis")
    hawkes_bvc = HawkesBVC(window=window_param, kappa=kappa_param)
    bvc_metrics = hawkes_bvc.eval(prices_bsi.reset_index())
    indicator_title = "BVC"
elif analysis_type == "ADC":
    st.write("### Average Directional Change Analysis")
    threshold_param = st.slider("Directional Change Threshold (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1) / 100
    adc_analysis = DirectionalChangeAnalysis(window=window_param, threshold=threshold_param, kappa=kappa_param)
    bvc_metrics = adc_analysis.eval(prices_bsi.reset_index())
    indicator_title = "ADC"
else:  # ACI
    st.write("### Accumulated Candle Index Analysis")
    aci_analysis = AccumulatedCandleAnalysis(window=window_param, kappa=kappa_param)
    bvc_metrics = aci_analysis.eval(prices_bsi.reset_index())
    indicator_title = "ACI"

df_skew = df_skew.merge(bvc_metrics, on='stamp', how='left')

# Then update the plot titles to use the correct indicator name:
ax_bvc.set_ylabel(indicator_title, fontsize=8)
ax_bvc.set_title(indicator_title, fontsize=10)
