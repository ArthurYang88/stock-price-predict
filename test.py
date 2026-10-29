from ml_code import Model
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# this need to be the same as when trained
INTERVAL = 30
LAYER2 = 128
LAYER3 = 64
LAYER4 = 1

# NASDAQ: ^IXIC
# 500: SPY
# nvidia: NVDA
# microsoft: MSFT
# apple: AAPL

STOCK = "MSFT"
START_DATE = "2020-01-01"
time_interval = "1d"
TRAIN_YEARS = 5

model = Model(INTERVAL, LAYER2, LAYER3, LAYER4)
model.load()


stock = yf.download(
    STOCK,
    start=START_DATE,
    interval=time_interval,
    auto_adjust=True,
    multi_level_index=False
)
prices = stock["Close"].dropna().to_numpy()
windows = np.lib.stride_tricks.sliding_window_view(prices, INTERVAL+2)

guess_lst = []
correct_lst = []

def time_to_num(c):
    if c == "1d":
        return 252
    if c == "1wk":
        return 36

trains = int(TRAIN_YEARS * time_to_num(time_interval))

for i in range(trains):
    correct = 100 * (windows[i][INTERVAL+1]-windows[i][INTERVAL])/windows[i][INTERVAL]
    correct_lst.append(correct)

    for j in range(1, INTERVAL+1):
        model.a1[j-1] = 100 * (windows[i][j]-windows[i][j-1])/windows[i][j-1]

    guess = model.forward()
    guess_lst.append(guess)


# plot graphs
interval = 30
kern = np.ones(interval)/interval
guess_avg = np.convolve(guess_lst, kern, mode="valid")
correct_avg = np.convolve(correct_lst, kern, mode="valid")
x_axis = range(len(guess_avg))

plt.plot(x_axis, guess_avg, label="Prediction")
plt.plot(x_axis, correct_avg, label="Correct")
plt.autoscale()

plt.legend()
plt.show()