from ml_code import Model
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# this need to be the same as when trained
INTERVAL = 30
LAYER2 = 128
LAYER3 = 64
LAYER4 = 1
learning_rate = 0.0001

TRAIN_TIMES = 100

START_DATE = "2020-01-01"
time_interval = "1d"
TRAIN_YEARS = 5

model = Model(INTERVAL, LAYER2, LAYER3, LAYER4, learning_rate)
model.load()

# NASDAQ: ^IXIC
# 500: SPY
# nvidia: NVDA
# microsoft: MSFT
# apple: AAPL



stock = yf.download(
    "AAPL",
    start=START_DATE,
    interval=time_interval,
    auto_adjust=True,
    multi_level_index=False
)
prices = stock["Close"].dropna().to_numpy()
windows = np.lib.stride_tricks.sliding_window_view(prices, INTERVAL+2)

def time_to_num(c):
    if c == "1d":
        return 252
    if c == "1wk":
        return 36

trains = int(TRAIN_YEARS * time_to_num(time_interval))

for t in range(TRAIN_TIMES):
    for i in range(trains):
        correct = 100 * (windows[i][INTERVAL + 1] - windows[i][INTERVAL]) / windows[i][INTERVAL]

        for j in range(1, INTERVAL + 1):
            model.a1[j - 1] = 100 * (windows[i][j] - windows[i][j - 1]) / windows[i][j - 1]

        guess = model.forward()
        model.train(correct)
    print(f"Train {t+1} done")

model.save()