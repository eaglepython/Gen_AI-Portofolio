import streamlit as st
import requests
import base64
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Stock Forecasting Demo", layout="centered")
st.title("Stock Price Forecasting Demo")

st.markdown("""
Enter a stock symbol and number of days to forecast. View predicted prices and trends.
""")

api_url = "http://localhost:8000/forecast"

symbol = st.text_input("Stock Symbol", "AAPL")
days = st.number_input("Days to Forecast", min_value=1, max_value=30, value=7)

if st.button("Get Forecast"):
    try:
        response = requests.post(api_url, json={"symbol": symbol, "days": days})
        if response.status_code == 200:
            data = response.json()
            st.success(f"Forecast for {symbol}:")
            forecast = data.get("forecast", [])
            st.line_chart(forecast)
            # Export forecast data button
            forecast_text = '\n'.join(str(x) for x in forecast)
            b64_txt = base64.b64encode(forecast_text.encode()).decode()
            href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="forecast_{symbol}.txt">Download Forecast Data</a>'
            st.markdown(href_txt, unsafe_allow_html=True)
            # Export chart as PNG
            fig, ax = plt.subplots()
            ax.plot(forecast, marker='o')
            ax.set_title(f'Forecast for {symbol}')
            ax.set_xlabel('Day')
            ax.set_ylabel('Price')
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="forecast_chart_{symbol}.png">Download Chart as PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error(f"API error: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")

st.info("Make sure the FastAPI app is running at http://localhost:8000 before using this dashboard.")
