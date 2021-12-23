import streamlit as st

import pandas as pd
import yfinance as yf
import datetime
import time
import requests
import io

import math
import matplotlib.pyplot as plt
import keras
import numpy as np
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

st.title('Stock Prediction')

user_input = st.text_input("ENTER STOCK SYMBOL")

df = yf.download(user_input, start = '2016-11-09', end = '2021-11-09', progress = False )
st.text(df.head(6))
