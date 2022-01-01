import streamlit as st #
import pandas as pd #
import yfinance as yf #
import datetime
from datetime import date
from datetime import timedelta
import time
import requests
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly
import cufflinks as cf
#ML
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#title of proj
st.title('Stock Prediction')

#enter stock symbol
st.sidebar.subheader('Select ')
user_input = st.sidebar.text_input("ENTER STOCK SYMBOL")
#select date range if you want, or you can just see current time stock
st.sidebar.subheader('Select date range')
start_date = st.sidebar.date_input("Start date", datetime.date(2021, 1, 1)) #remember to change to 2015-1-1
end_date = st.sidebar.date_input("End date", datetime.date(2021, 12, 22))#remember to change to datetime.date.today()

#current year button
current_year = st.sidebar.button("Current year")
if current_year:
    start_date = date(date.today().year, 1, 1)
    end_date = date(date.today().year, 12, 31)

#retrieve particular stock data
df = yf.download(user_input, start = start_date, end = end_date, progress = False )

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


def prediction_test():
    chart_data = pd.DataFrame(
         df["Close"])

    #count the number of entries
    entry_count = len(df['Close'])
    training_set = df['Close']
    training_set = pd.DataFrame(training_set)
    df.isna().any()
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []

    for i in range(60, entry_count):  #60 because take data from day 1 to day 60, then making predicition on 61st day. #
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #Training model
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = "adam", loss = "mean_squared_error")
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)

    price_data = df["Close"]
    test_set = pd.DataFrame(price_data)

    dataset_total = pd.concat((df['Close'], price_data), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(price_data) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 307):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    st.text(X_test)
    st.text(X_test.shape)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[-1], 1))
    stock_dates = df.index
    real_stock_price = df.iloc[:,3:4]
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # st.text(predicted_stock_price)
    predicted_stock_price = pd.DataFrame(predicted_stock_price,columns = ['Predicted Stock Price'])

    global var_real_stock_price
    global var_predicted_stock_price
    global var_stock_dates
    var_real_stock_price = real_stock_price
    var_predicted_stock_price = predicted_stock_price
    var_stock_dates = stock_dates
# if 'predicted_stock_price' not in st.session.state:
#     st.session.state['predicted_stock_price'] = ''

if st.button("run prediction"):
    prediction_test()
    pred_date = var_real_stock_price.index[len(var_real_stock_price)-1]
    pred_date += timedelta(days=1)
# pred_date_2 = pred_date
# pred_date_2 +=timedelta(days=1)
    new_index = (list(var_real_stock_price.index))
    new_index.append(pred_date)
# new_index.append(pred_date_2)
    new_index = pd.Index(new_index)
# print(new_index)
    predicted_stock_price_2 = var_predicted_stock_price.set_index(new_index)
    # predicted_stock_price_2.head(250)
    df = pd.concat([predicted_stock_price_2,var_real_stock_price ], axis=1)
    df.tail()
    chart = df
    st.line_chart(chart)
