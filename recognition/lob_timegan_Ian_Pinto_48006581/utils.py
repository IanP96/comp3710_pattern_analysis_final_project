"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) random_generator: random vector generator
(4) NormMinMax: return data info
"""

import logging

import numpy as np
from numpy.typing import NDArray
import torch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DATA_DIR = "data"
ORDERBOOK_DATA_FILENAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

# 
NUM_TRAINING_ITERATIONS = 3

if NUM_TRAINING_ITERATIONS < 1_000:
    logger.warning(
        "Number of training iterations is set to a low value of %s for testing purposes.",
        NUM_TRAINING_ITERATIONS,
    )

# Device configuration
# source: https://edstem.org/au/courses/26755/discussion/2844412?answer=6302304
TORCH_DEVICE_NAME: str
if torch.cuda.is_available():
    logger.info("CUDA is available")
    TORCH_DEVICE_NAME = "cuda"
elif torch.backends.mps.is_available():
    logger.info("CUDA not found. Using MPS")
    TORCH_DEVICE_NAME = "mps"
else:
    logger.warning("CUDA and MPS not found. Using CPU")
    TORCH_DEVICE_NAME = "cpu"
TORCH_DEVICE = torch.device(TORCH_DEVICE_NAME)


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def extract_time(data: NDArray) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    # Get lengths of each sequence (num of timesteps)
    time = np.array([d.shape[0] for d in data])

    # Get the maximum sequence length
    max_seq_len = time.max()

    return time, max_seq_len


def random_generator(
    batch_size: int,
    z_dim: int,
    T_mb,
    max_seq_len,
    mean: float | None = None,
    std: float | None = None,
) -> NDArray[np.float64]:
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        noise_shape = (T_mb[i], z_dim)
        if mean is None and std is None:
            temp_Z = np.random.uniform(0.0, 1, noise_shape)
        else:
            assert mean is not None and std is not None
            # todo try using normal distribution as well
            # using formula for st dev of uniform distribution
            interval_size = std * (12**0.5)
            temp_Z = np.random.uniform(
                mean - interval_size / 2, mean + interval_size / 2, noise_shape
            )
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    Z_mb_np = np.array(Z_mb)
    return Z_mb_np


def norm_min_max(
    data: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Min-Max Normaliser.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalised data
      - min_val: minimum values (for renormalisation)
      - max_val: maximum values (for renormalisation)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val  # [3661, 24, 6]

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val
