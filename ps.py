import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Analyzer — بسيط وواضح", layout="wide")
st.title("Stock Analyzer — بسيط وواضح مع توقع وسجل ثقة")

# -------- Sidebar (إعدادات بسيطة) ----------
st.sidebar.header("إعدادات")
ticker = st.sidebar.text_input("رمز السهم", value="AAPL")
today = datetime.utcnow().date()
start_date = st.sidebar.date_input("تاريخ البدء", value=today - timedelta(days=365))
end_date = st.sidebar.date_input("تاريخ الانتهاء", value=today)
ma_window = st.sidebar.number_input("فترة المتوسط المتحرك (يوم)", min_value=2, max_value=200, value=20)
predict_horizon = st.sidebar.number_input("عدد الأيام للتوقع", min_value=1, max_value=30, value=5)
lags = st.sidebar.number_input("عدد القيم السابقة (lags) لنموذج التوقع", min_value=1, max_value=60, value=10)

if start_date >= end_date:
    st.sidebar.error("تأكد أن تاريخ البدء قبل تاريخ الانتهاء.")

# -------- Helper: تحميل وتحضير البيانات ----------
@st.cache_data
def load_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    df = df.dropna(how='any').copy()

    # Ensure Close is 1D numeric
    df['Close'] = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
    df = df.dropna(subset=['Close']).copy()

    # Simple moving average
    df[f"MA_{ma_window}"] = df['Close'].rolling(window=ma_window).mean()
    return df

# -------- Simple forecasting function (transparent) ----------
def train_and_forecast(close_series: pd.Series, n_lags: int, horizon: int):
    """
    Build lag features X from close_series to predict next value y.
    Train LinearRegression and forecast 'horizon' days iteratively.
    Return predictions list, prediction_intervals, confidence_percent, model_metrics.
    """
    prices = close_series.values
    n = len(prices)
    if n < n_lags + 5:
        return {
            "error": "البيانات قصيرة جداً لبناء نموذج التوقع. زِد المدى الزمني أو خفّض عدد الـlags."
        }

    # Build dataset
    X, y = [], []
    for i in range(n_lags, n):
        X.append(prices[i - n_lags:i])
        y.append(prices[i])
    X = np.array(X)
    y = np.array(y)

    # Train/test split (آخر 20% كاختبار)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on test
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mean_price = np.mean(y_test) if len(y_test) > 0 else np.mean(y_train)
    rel_rmse = rmse / mean_price if mean_price != 0 else 1.0
    # Confidence: simple interpretable score: higher when relative RMSE small.
    confidence = max(0.0, 1.0 - rel_rmse) * 100  # as percent, clipped at 0

    # Residual std to produce simple prediction intervals
    residuals = y_test - y_pred_test if len(y_test) > 0 else y_train - model.predict(X_train)
    resid_std = np.std(residuals) if len(residuals) > 0 else 0.0
    z95 = 1.96

    # Forecast next 'horizon' days iteratively using last available window
    last_window = prices[-n_lags:].tolist()
    preds = []
    intervals = []
    for _ in range(horizon):
        x_in = np.array(last_window[-n_lags:]).reshape(1, -1)
        p = model.predict(x_in)[0]
        preds.append(float(p))
        lower = p - z95 * resid_std
        upper = p + z95 * resid_std
        intervals.append((float(lower), float(upper)))
        # append predicted price to window for next-step forecast (naive iter)
        last_window.append(p)

    metrics = {"rmse": float(rmse), "rel_rmse": float(rel_rmse), "confidence_pct": float(confidence)}
    return {"predictions": preds, "intervals": intervals, "metrics": metrics, "model": model}

# -------- Main UI and execution ----------
with st.spinner("تحميل البيانات والمعالجة..."):
    df = load_data(ticker, start_date, end_date)

if df.empty:
    st.error("لم أجد بيانات لهذا الرمز أو النطاق الزمني فارغ.")
else:
    st.subheader(f"{ticker} — الملخص السريع")
    col1, col2 = st.columns([2,1])

    # Left: Charts
    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Candles'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"MA_{ma_window}"], line=dict(color='blue'), name=f"MA {ma_window}"), row=1, col=1)

        # Forecast area placeholder (we will append predicted points if available)
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Right: Key info + prediction
    with col2:
        last_price = df['Close'].iloc[-1]
        first_price = df['Close'].iloc[0]
        change_pct = (last_price / first_price - 1) * 100 if first_price != 0 else 0.0
        st.metric("آخر سعر", f"${last_price:.2f}", f"{change_pct:.2f}% خلال المدى")

        st.markdown("### توقع السعر")
        result = train_and_forecast(df['Close'], n_lags=lags, horizon=predict_horizon)
        if "error" in result:
            st.warning(result["error"])
        else:
            preds = result["predictions"]
            intervals = result["intervals"]
            metrics = result["metrics"]

            # show numeric
            for i, p in enumerate(preds, start=1):
                low, high = intervals[i-1]
                st.write(f"اليوم +{i}: {p:.2f} (مدى ثقة 95%: {low:.2f} — {high:.2f})")

            st.markdown(f"**مؤشر الثقة للتوقع**: {metrics['confidence_pct']:.1f}%")
            st.write(f"RMSE (اختبار): {metrics['rmse']:.4f} — نسبة RMSE إلى السعر المتوسط: {metrics['rel_rmse']:.3f}")

            # Append forecast to chart (rebuild chart including preds)
            future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=len(preds))
            fig2 = make_subplots(rows=1, cols=1)
            fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
            fig2.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='Prediction', line=dict(color='red')))
            # add intervals as filled area
            lowers = [iv[0] for iv in intervals]
            uppers = [iv[1] for iv in intervals]
            fig2.add_trace(go.Scatter(x=list(future_dates)+list(future_dates[::-1]),
                                      y=list(uppers)+list(lowers[::-1]),
                                      fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name='95% interval'))
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### بيانات أخيرة")
    st.dataframe(df[['Open','High','Low','Close','Volume', f"MA_{ma_window}"]].tail(10))

    st.markdown("### ملاحظات سريعة عن نموذج التوقع")
    st.write("- النموذج بسيط وشفاف: Linear Regression يستخدم آخر N قيم (lags) للتنبؤ باليوم التالي، ثم يُستخدم التنبؤ تسلسليًا للـhorizon.")
    st.write("- مؤشر الثقة مُبني من خطأ النموذج على مجموعة الاختبار: Confidence % = max(0, 1 - (RMSE / متوسط السعر)) × 100.")
    st.write("- مجال الثقة (interval) مبني على انحراف البقايا المفروض مبدئيًا طبيعيًا (تقريب).")
    st.write("- القيود: لا يأخذ بعين الاعتبار فروق التنفيذ/العمولات/الانزلاق، ولا يعتمد على بيانات سوقية/أخبار أو حجم تداول بشكل مُفصّل.")

    # CSV download
    csv = df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("تنزيل البيانات (CSV)", data=csv, file_name=f"{ticker}_{start_date}_{end_date}.csv", mime="text/csv")
