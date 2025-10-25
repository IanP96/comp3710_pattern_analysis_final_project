"""
Data loading and preprocessing
"""

"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

data.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
(3) load_data: download or generate data
(4): batch_generator: mini-batch generator
"""

import logging
from pathlib import Path

# from os.path import dirname, abspath
import numpy as np

from utils import DATA_DIR, ORDERBOOK_DATA_FILENAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """

    original_data = np.loadtxt(
        Path("data", ORDERBOOK_DATA_FILENAME), delimiter=",", skiprows=0
    )

    # If the data is in reverse chronological data (the LOBSTER data isn't), flip the data to make
    # chronological data
    # original_data = original_data[::-1]

    # Normalise the data
    original_data = MinMaxScaler(original_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(original_data) - seq_len):
        data_slice = original_data[i : i + seq_len]
        temp_data.append(data_slice)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def load_data(opt):
    # Data loading
    # if opt.data_name in ["stock", "energy"]:
    #     ori_data = real_data_loading(opt.data_name, opt.seq_len)  # list: 3661; [24,6]
    # elif opt.data_name == "sine":
    #     # Set number of samples and its dimensions
    #     no, dim = 10000, 5
    #     ori_data = sine_data_generation(no, opt.seq_len, dim)
    original_data = real_data_loading(opt.seq_len)  # list: 3661; [24,6]
    logger.info("Stock dataset has been loaded.")

    return original_data


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb
