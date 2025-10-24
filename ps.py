import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analyzer — تحليل الأسهم (MVP)")

# Sidebar inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("رمز السهم (مثال: AAPL أو TSLA)", value="AAPL")
today = datetime.utcnow().date()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("تاريخ البدء", default_start)
end_date = st.sidebar.date_input("تاريخ الانتهاء", today)

ma_short = st.sidebar.selectbox("متوسط متحرك قصير المدى (أيام)", [5, 10, 20, 50], index=2)
ma_long = st.sidebar.selectbox("متوسط متحرك طويل المدى (أيام)", [50, 100, 200], index=0)
show_rsi = st.sidebar.checkbox("عرض RSI", value=True)
rsi_period = st.sidebar.number_input("فترة RSI", min_value=5, max_value=30, value=14)

show_macd = st.sidebar.checkbox("عرض MACD", value=True)
show_bbands = st.sidebar.checkbox("عرض Bollinger Bands", value=True)

if start_date >= end_date:
    st.sidebar.error("تاريخ البدء يجب أن يكون قبل تاريخ الانتهاء.")

# Make sure the cache key changes when these parameters change by passing them into the function
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end, ma_short: int, ma_long: int, rsi_period: int):
    # Ensure ints
    ma_short = int(ma_short)
    ma_long = int(ma_long)
    rsi_period = int(rsi_period)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df

    # Drop NA rows from the downloaded data
    df = df.dropna(how='any').copy()

    # Ensure Close is a 1-D numeric Series
    # If df['Close'] somehow becomes 2D (e.g., DataFrame slice), .squeeze() ensures 1D
    df['Close'] = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
    df = df.dropna(subset=['Close']).copy()

    # Moving averages
    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()

    # Compute indicators inside try/except to avoid crashing the whole app on library errors
    try:
        # RSI: pass a 1D Series and integer window
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_period).rsi()

        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
    except Exception as e:
        # Log the exception to Streamlit (visible in logs) and continue without those indicators
        st.warning(f"حدث خطأ عند حساب بعض المؤشرات الفنية: {e}")
        # Ensure those columns exist so downstream code doesn't KeyError
        for col in ['RSI','MACD','MACD_signal','MACD_diff','BB_high','BB_low','BB_mid']:
            if col not in df.columns:
                df[col] = np.nan

    # Simple MA crossover signal
    df['Signal'] = np.where(df['MA_short'] > df['MA_long'], 1, 0)
    df['Signal_change'] = df['Signal'].diff().fillna(0)

    # Strategy returns (approximate)
    df['Daily_ret'] = df['Close'].pct_change().fillna(0)
    df['Strategy_ret'] = df['Daily_ret'] * df['Signal'].shift(1).fillna(0)
    df['Strategy_cum'] = (1 + df['Strategy_ret']).cumprod()
    df['Buy_and_hold_cum'] = (1 + df['Daily_ret']).cumprod()

    return df

with st.spinner(f"تحميل بيانات {ticker}..."):
    # pass ma_short, ma_long, rsi_period explicitly so cache invalidates when they change
    df = load_data(ticker, start_date, end_date, ma_short, ma_long, rsi_period)

if df.empty:
    st.error("لم أجد بيانات لهذا الرمز أو أن النطاق الزمني لا يحتوي على بيانات.")
else:
    st.subheader(f"البيانات: {ticker} ({start_date} → {end_date})")
    col1, col2 = st.columns([3,1])

    # Build figure: price (+BB, MAs), optional RSI & MACD
    rows = 1
    if show_rsi: rows += 1
    if show_macd: rows += 1

    row_heights = []
    if show_rsi and show_macd:
        row_heights = [0.5, 0.25, 0.25]
    elif show_rsi or show_macd:
        row_heights = [0.7, 0.3]
    else:
        row_heights = [1.0]

    fig = make_subplots(rows=len(row_heights), cols=1, shared_xaxes=True, row_heights=row_heights)

    # Price Candles
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candles'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_short'], line=dict(color='blue', width=1.5), name=f"MA {ma_short}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_long'], line=dict(color='orange', width=1.5), name=f"MA {ma_long}"), row=1, col=1)

    if show_bbands and 'BB_high' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], line=dict(color='gray', width=1), name='BB High', opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], line=dict(color='gray', width=1), name='BB Low', opacity=0.5), row=1, col=1)

    current_row = 2
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1

    if show_macd and 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='green', width=1), name='MACD'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='red', width=1), name='MACD Signal'), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Key metrics and backtest summary
    with col2:
        last_close = df['Close'].iloc[-1]
        st.metric("آخر سعر", f"${last_close:.2f}")
        total_ret = df['Buy_and_hold_cum'].iloc[-1] - 1
        strat_ret = df['Strategy_cum'].iloc[-1] - 1
        period_days = (df.index[-1] - df.index[0]).days if isinstance(df.index[0], pd.Timestamp) else len(df)
        years = max(period_days/252, 1e-6)
        cagr_bh = (df['Buy_and_hold_cum'].iloc[-1]) ** (1/years) - 1
        cagr_strat = (df['Strategy_cum'].iloc[-1]) ** (1/years) - 1

        def max_drawdown(cum_returns):
            peak = cum_returns.cummax()
            dd = (cum_returns - peak) / peak
            return dd.min()

        mdd_bh = max_drawdown(df['Buy_and_hold_cum'])
        mdd_strat = max_drawdown(df['Strategy_cum'])

        st.markdown("### ملخص الأداء")
        st.write(f"Buy & Hold إجمالي العائد: {(total_ret*100):.2f}% — CAGR: {(cagr_bh*100):.2f}% — MaxDD: {(mdd_bh*100):.2f}%")
        st.write(f"الاستراتيجية (MA Crossover) إجمالي العائد: {(strat_ret*100):.2f}% — CAGR: {(cagr_strat*100):.2f}% — MaxDD: {(mdd_strat*100):.2f}%")
        st.write("إشارات تداول (تغيرات MA):")
        st.write(df[df['Signal_change'] != 0][['Signal_change']].tail(10))

        st.write("آخر بيانات الحجم والسعر")
        st.write(df[['Open','High','Low','Close','Volume']].tail(5))

    # Data download
    csv = df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("تنزيل البيانات (CSV)", data=csv, file_name=f"{ticker}_{start_date}_{end_date}.csv", mime="text/csv")

    # Show raw
    with st.expander("عرض الجدول الكامل"):
        st.dataframe(df)
