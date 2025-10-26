import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_3d = np.load("generated_data.npy")
    data = data_3d.reshape(-1, data_3d.shape[2]) / 10000
    best_ask = data[:, 0]
    best_bid = data[:, 2]
    x = np.arange(0, len(best_ask))
    plt.plot(x, best_ask, label="Best Ask Price")
    plt.plot(x, best_bid, label="Best Bid Price")
    plt.show()
