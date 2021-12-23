import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
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
user_input = st.text_input("ENTER STOCK SYMBOL")

#select date range if you want, or you can just see current time stock
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

#retrieve particular stock data
df = yf.download(user_input, start = start_date, end = end_date, progress = False )
st.text(df.head(6))

#plot graph for "Close" column
# df["Close"].plot(figsize=(16,6))
# chart_data = pd.DataFrame(df["Close"])
# st.line_chart(chart_data)

#bollinger bands, looks like better chart but not very sure
st.header('**Graph**')
qf=cf.QuantFig(df,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)
