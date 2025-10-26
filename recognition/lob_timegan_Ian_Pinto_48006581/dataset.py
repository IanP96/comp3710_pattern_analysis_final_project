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
from argparse import Namespace

# from os.path import dirname, abspath
import numpy as np
from numpy.typing import NDArray

from utils import DATA_DIR, ORDERBOOK_DATA_FILENAME

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def min_max_scaler(data: NDArray[np.float32]) -> NDArray[np.float32]:
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


def load_data(opt: Namespace) -> NDArray[np.float32]:
    """
    Load and preprocess stock data

    Args:
        opt (Options): command-line options. opt.seq_len should be the sequence length for slicing

    Returns:
        preprocessed data of shape (num_batches, seq_len, num_features)
    """
    # todo update docstring

    logger.info("Loading and preprocessing stock dataset...")

    # Data loading
    seq_len = opt.seq_len

    original_data = np.loadtxt(
        Path("data", ORDERBOOK_DATA_FILENAME), delimiter=",", skiprows=0, dtype=np.int64
    )

    # If the data is in reverse chronological data (the LOBSTER data isn't), flip the data to make
    # chronological data
    # original_data = original_data[::-1]

    # From the README about the LOBSTER data:
    # ---
    # Unoccupied Price Levels:
    # When the selected number of levels exceeds the number of levels
    # available the empty order book positions are filled with dummy
    # information to guarantee a symmetric output. The extra bid
    # and/or ask prices are set to -9999999999 and 9999999999,
    # respectively. The Corresponding volumes are set to 0.
    # ---
    # So remove any rows where this applies
    filtered_data = np.array([row for row in original_data if 0 not in row])

    filtered_data_float = filtered_data.astype("float32")

    # Normalise the data
    filtered_data_float = min_max_scaler(filtered_data_float)

    # Get dimensions
    n_samples = filtered_data_float.shape[0]
    n_features = filtered_data_float.shape[1]
    n_batches = n_samples - seq_len + 1

    # Cut data by sequence length
    sliced_data = np.empty((n_batches, seq_len, n_features), dtype=np.float32)
    for i in range(0, len(filtered_data_float) - seq_len + 1):
        data_slice = filtered_data_float[i : i + seq_len]
        sliced_data[i] = data_slice

    # Mix the datasets (to make it similar to i.i.d)
    np.random.shuffle(sliced_data)

    logger.info("Stock dataset has been loaded and preprocessed.")
    return sliced_data


def batch_generator(
    data: NDArray[np.float32], time: NDArray[np.int32], batch_size: int
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
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

    X_mb = np.array([data[i] for i in train_idx], dtype=np.float32)
    T_mb = np.array([time[i] for i in train_idx], dtype=np.int32)

    return X_mb, T_mb
