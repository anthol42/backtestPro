import numpy as np
from StockAPI.fetch import Stock
import matplotlib.pyplot as plt
def rollingLinReg(data: np.ndarray, period: int = 50, forecast: int = 0):
    ma = np.full((len(data), ), dtype=np.float32, fill_value=np.nan)
    sumX = 0     # Equivalent to 'u'
    sumY = 0     # Equivalent to 'c'
    sumXY = 0    # Equivalent to 'd'
    o = period * (period + 1) / 2
    h = o * (2 * period + 1) / 3
    l = o**2
    for i in range(len(ma)):
        current_value = data[i]
        if not np.isnan(current_value):
            sumX += period * current_value - sumY
            sumY += current_value
            sumXY += current_value ** 2
            if i >= period - 1:    # We can start calculating indicator's value
                if i > period - 1:  # We need to remove older values
                    removed_value = data[i-period]
                    sumY -= removed_value
                    sumXY -= removed_value ** 2

                # Now calculate incremental linear regression
                slope = (period * sumX - o * sumY) / (period*h - l)
                intercept = (sumY - slope * o) / period
                ma[i] = intercept + slope * (period + forecast)
    return ma

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[0:n] = np.nan
    return ret[1:] / n

if __name__ == "__main__":
    stock = Stock("TQQQ")
    stock.add_chart('5d', '5y')
    close = stock.chart["Close"].to_numpy()
    ma = rollingLinReg(close, 10, forecast=-2)
    # ma2 = rollingLinReg(close, 10, forecast=0)
    ma2 = moving_average(ma[9:], 3)
    ma2 = np.concatenate((np.full(10, fill_value=np.nan), ma2), dtype=np.float32)
    plt.plot(close, color='g')
    plt.plot(ma, color='b')
    plt.plot(ma2, color='r')
    plt.show()