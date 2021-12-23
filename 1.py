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
user_input = st.text_input("ENTER STOCK SYMBOL")

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
st.text(df.head(6))

#plot graph for "Close" column
# df["Close"].plot(figsize=(16,6))
# chart_data = pd.DataFrame(df["Close"])
# st.line_chart(chart_data)

#bollinger bands, looks like better chart but not very sure
st.header('**Graph**')
qf=cf.QuantFig(df,title= user_input,legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

tickerData = yf.Ticker(user_input)
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)
string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)
