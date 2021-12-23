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


st.title('Stock Prediction')

user_input = st.text_input("ENTER STOCK SYMBOL")

df = yf.download(user_input, start = '2016-11-09', end = '2021-11-09', progress = False )
st.text(df.head(6))
