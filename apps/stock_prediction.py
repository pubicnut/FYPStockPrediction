import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st #
import pandas as pd #
import yfinance as yf #
import datetime
from datetime import date
from datetime import datetime
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
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

def prediction_test():

    # df = yf.download(user_input_gbl, start = start_date_gbl, end = datetime.date.today(), progress = False )
    df = yf.download(user_input_gbl, start = start_date_gbl, end = date(2021,12,24), progress = False )
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
    model.fit(X_train, y_train, epochs = 1, batch_size = 32)

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

def app():
    global user_input_gbl
    global start_date_gbl
    st.title('View Stock Prediction')
    col1, col2, = st.columns(2)
    with col1:
        user_input = st.text_input("ENTER STOCK SYMBOL")
        user_input_gbl = user_input
        df = yf.download(user_input)
        date_index = str(df.index[0])
        dt_obj = datetime.strptime(date_index, "%Y-%m-%d %H:%M:%S")
        start_date_gbl = dt_obj
            #select date range if you want, or you can just see current time stock

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
