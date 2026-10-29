from ml_code import Model
import yfinance as yf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# this need to be the same as when trained
#################
# don't touch
#################
INTERVAL = 30
LAYER2 = 128
LAYER3 = 64
LAYER4 = 1
learning_rate = 0.0001

################################
# settings
################################
TRAIN_TIMES_TOTAL = 100
TRAIN_TIMES_EACH = 1

START_DATE = "2021-01-01"
time_interval = "1d"
TRAIN_YEARS = 5

stocks_lst = [
    "MSFT",
    "NVDA",
    "GOOGL",
    "NFLX",
    "AMZN",
    "HD",
    "PG",
    "KO",
    "JNJ",
    "MRK",
    "JPM",
    "BAC",
    "CAT",
    "UNP",
    "XOM",
    "CVX",
    "NUE",
    "FCX",
    "SO",
    "DUK",
    "PLD",
    "O"
]

def time_to_num(c):
    if c == "1d":
        return 252
    if c == "1wk":
        return 36

model = Model(INTERVAL, LAYER2, LAYER3, LAYER4, learning_rate)
model.load()

price_dict = {}
try:
    with open("train_prices.pkl", "rb") as f:
        price_dict = pickle.load(f)
except EOFError:
    with open("train_prices.pkl", "wb") as f:
        for stk in stocks_lst:
            stock = yf.download(
                stk,
                start=START_DATE,
                interval=time_interval,
                auto_adjust=True,
                multi_level_index=False
            )
            prices = stock["Close"].dropna().to_numpy()
            price_dict[stk] = prices
        pickle.dump(price_dict, f)

for t0 in range(TRAIN_TIMES_TOTAL):
    for stk in stocks_lst:
        prices = price_dict[stk]
        windows = np.lib.stride_tricks.sliding_window_view(prices, INTERVAL + 2)

        trains = int(TRAIN_YEARS * time_to_num(time_interval))

        for t in range(TRAIN_TIMES_EACH):
            for i in range(trains):
                correct = 100 * (windows[i][INTERVAL + 1] - windows[i][INTERVAL]) / windows[i][INTERVAL]

                for j in range(1, INTERVAL + 1):
                    model.a1[j - 1] = 100 * (windows[i][j] - windows[i][j - 1]) / windows[i][j - 1]

                guess = model.forward()
                model.train(correct)
            # print(f"Train {t + 1} on {stk} done")
    print(f"Big train round {t0+1} done")

model.save()