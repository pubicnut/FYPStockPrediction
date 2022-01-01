import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date
import time
import requests
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly
import cufflinks as cf

#title of proj
st.title('Stock Prediction')

#enter stock symbol
st.sidebar.subheader('Select ')
user_input = st.sidebar.text_input("ENTER STOCK SYMBOL")
#select date range if you want, or you can just see current time stock
st.sidebar.subheader('Select date range')
start_date = st.sidebar.date_input("Start date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

#current year button
current_year = st.sidebar.button("Current year")
if current_year:
    start_date = date(date.today().year, 1, 1)
    end_date = date(date.today().year, 12, 31)

#retrieve particular stock data
df = yf.download(user_input, start = start_date, end = end_date, progress = False )

#plot graph for "Close" column
# df["Close"].plot(figsize=(16,6))
# chart_data = pd.DataFrame(df["Close"])
# st.line_chart(chart_data)

#bollinger bands, looks like better chart but not very sure
# st.header('**Graph**')
# qf=cf.QuantFig(df,title= user_input,legend='top',name='GS')
# qf.add_bollinger_bands()
# fig = qf.iplot(asFigure=True)
# st.plotly_chart(fig)

import plotly.graph_objs as go

#Interval required 1 minute
# data = yf.download(tickers=user_input, period='5d', interval='1m')
data = yf.download(tickers=user_input, start = start_date, end = end_date)

#declare figure
fig = go.Figure()

#Candlestick
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))

# Add titles
fig.update_layout(
    title= user_input + ' Share Price',
    yaxis_title='Stock Price (USD per Shares)',
    width =700, height=700 )

# X-Axes #2h\1d\2d\1wk\1mt\3mt\6mt\1y\2y\5y\
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1D", step="day", stepmode="todate"),
            dict(count=7, label="1 week", step="day", stepmode="todate"),
            dict(count=1, label="1 month", step="month", stepmode="todate"),
            dict(count=3, label="3 month", step="month", stepmode="todate"),
            dict(count=6, label="6 month", step="month", stepmode="todate"),
            dict(count=1, label="1 Year", step="year", stepmode="todate"),
            dict(count=5, label="5 Year", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)


#Show
st.plotly_chart(fig)
