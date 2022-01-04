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

m = st.markdown("""
<style>
div.stButton > button:first-child { 
    position:relative;left:30%;
}

</style>""", unsafe_allow_html=True)

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
current_year = st.sidebar.button("Current year", key = 'sidebar')
if current_year:
    start_date = date(date.today().year, 1, 1)
    end_date = date(date.today().year, 12, 31)

#retrieve particular stock data
df = yf.download(user_input, start = start_date, end = end_date, progress = False )
# st.text("df length {}, start: {}, end: {}".format(str(len(df)), start_date, end_date))

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
    X_train, y_train = np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32)
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
    price_data.fillna(value=0, inplace=True)

    dataset_total = pd.concat((df['Close'], price_data), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(price_data) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, len(df) + 62):
        X_test.append(np.array(inputs[i-60:i, 0]).tolist())
    # X_test_pred = np.asarray(X_test).astype(np.float32)
    X_test_len = max(map(len, X_test))
    X_test_pred = np.array([xi+[0.0]*(X_test_len-len(xi)) for xi in X_test]).astype(np.float32)

    # X_test = np.reshape(X_test_pred, (X_test_pred.shape[0], X_test_pred.shape[1]))
    X_test = X_test_pred
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



if st.button("run prediction"):
    prediction_test()
    data_predicted_stock_price = var_predicted_stock_price["Predicted Stock Price"].to_numpy()

    date_index = (list(var_real_stock_price.index))
    for i in range(0, len(data_predicted_stock_price) - len(date_index)):
        pred_date = date_index[len(date_index)-1]
        pred_date += timedelta(days=1)
        date_index.append(pred_date)

    data_real_stock_price = var_real_stock_price["Close"].to_numpy().tolist()
    for i in range(0, len(data_predicted_stock_price) - len(data_real_stock_price)):
        data_real_stock_price.append(np.nan)

    df = pd.DataFrame(
        {
            "Predicted Stock Price": data_predicted_stock_price,
            "Real Stock Price": data_real_stock_price,
        },
        index=date_index
    )
    chart = df
    st.line_chart(chart)
