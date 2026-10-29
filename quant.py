from ml_code import Model
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pickle

# this need to be the same as when trained
INTERVAL = 30
LAYER2 = 128
LAYER3 = 64
LAYER4 = 1

test_lst = [
    "MSFT",
    "NVDA",
    "GOOGL",
    "NFLX",
    "KO",
    "JPM"
]
START_DATE = "2025-01-01"
time_interval = "1d"
TRAIN_YEARS = 1

# test investing
P_no_short = 10000
P_short = 10000
P_hold = 10000

P_hold_lst = [P_hold/len(test_lst)] * len(test_lst)
P_short_lst = [P_hold/len(test_lst)] * len(test_lst)
P_no_short_lst = [P_hold/len(test_lst)] * len(test_lst)

hold_hist = [P_hold]
no_short_hist = [P_no_short]
short_hist = [P_short]

# strategy: principal split into all the stocks evenly
# if price tmr is higher: sell
# if price tmr is lower: buy
# ignoring everything

def time_to_num(c):
    if c == "1d":
        return 252
    if c == "1wk":
        return 36

model = Model(INTERVAL, LAYER2, LAYER3, LAYER4)
model.load()

price_dict = {}
try:
    with open("test_prices.pkl", "rb") as f:
        price_dict = pickle.load(f)
except EOFError:
    with open("test_prices.pkl", "wb") as f:
        for stk in test_lst:
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

trains = int(TRAIN_YEARS * time_to_num(time_interval))
wds = dict.fromkeys(test_lst)

for k, stk in enumerate(test_lst):
    prices = []
    try:
        prices = price_dict[stk]
    except KeyError:
        stock = yf.download(
            stk,
            start=START_DATE,
            interval=time_interval,
            auto_adjust=True,
            multi_level_index=False
        )
        prices = stock["Close"].dropna().to_numpy()
    wd = np.lib.stride_tricks.sliding_window_view(prices, INTERVAL+2)
    wds[stk] = wd

for i in range(trains):

    for k, stk in enumerate(test_lst):
        windows = wds[stk]
        correct = 100 * (windows[i][INTERVAL + 1] - windows[i][INTERVAL]) / windows[i][INTERVAL]

        for j in range(1, INTERVAL + 1):
            model.a1[j - 1] = 100 * (windows[i][j] - windows[i][j - 1]) / windows[i][j - 1]

        guess = model.forward()

        old_hold = P_hold_lst[k]
        old_short = P_short_lst[k]
        old_no_short = P_no_short_lst[k]

        P_hold_lst[k] *= 1 + correct / 100
        if guess > 0:
            P_no_short_lst[k] *= 1 + correct / 100
            P_short_lst[k] *= 1 + correct / 100
        else:
            P_short_lst[k] *= 1 - correct / 100

        P_hold += -(old_hold - P_hold_lst[k])
        P_short += -(old_short - P_short_lst[k])
        P_no_short += -(old_no_short - P_no_short_lst[k])

    hold_hist.append(P_hold)
    short_hist.append(P_short)
    no_short_hist.append(P_no_short)

x_axis = range(trains+1)

plt.title("Investing")
plt.plot(x_axis, hold_hist, label="Hold")
plt.plot(x_axis, short_hist, label="Short")
plt.plot(x_axis, no_short_hist, label="No short")
plt.autoscale()
plt.legend()

print(f"Hold: {P_hold}")
print(f"Short: {P_short}")
print(f"No short: {P_no_short}")

plt.show()