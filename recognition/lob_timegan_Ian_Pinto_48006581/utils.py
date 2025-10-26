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
import matplotlib.pyplot as plt

from constants import MPR_RANGE, SPREAD_RANGE

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    seq_len: int,
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
        noise_shape = (seq_len, z_dim)
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
        Z_mb.append(temp_Z)
    Z_mb_np = np.array(Z_mb)
    return Z_mb_np


def norm_min_max(
    data: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Min-Max Normaliser. Normalises based on feature-wise min and max values.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalised data
      - min_val: minimum values (for renormalisation)
      - max_val: maximum values (for renormalisation)
    """
    # todo delete old code once verified
    min_val_old = np.min(np.min(data, axis=0), axis=0)
    min_val = np.min(data, axis=(0, 1))
    assert np.all(min_val == min_val_old)
    data = data - min_val  # [3661, 24, 6]

    max_val_old = np.max(np.max(data, axis=0), axis=0)
    max_val = np.max(data, axis=(0, 1))
    assert np.all(max_val == max_val_old)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def kl_metric(
    original_data: NDArray, generated_data: NDArray, metric_type: str, show_plot: bool = False
) -> float:

    # best ask is feature 0, best bid is feature 2

    assert (
        len(original_data.shape) == len(generated_data.shape) == 2
    ), "data should be 2D"
    assert metric_type in {"spread", "mpr"}

    real_and_generated = []
    bins = None
    for data in [original_data, generated_data]:
        source_data: NDArray
        bin_range: tuple[float, float]
        if metric_type == "mpr":
            mid = 0.5 * (data[:, 2] + data[:, 0])
            source_data = np.log(mid[1:]) - np.log(mid[:-1])
            bin_range = MPR_RANGE
        else:
            source_data = data[:, 0] - data[:, 2]  # spread
            bin_range = SPREAD_RANGE
        assert len(source_data.shape) == 1
        hist_values, bins = np.histogram(
            source_data, bins=100, density=True, range=bin_range
        )
        real_and_generated.append(hist_values)
    assert bins is not None
    dx = bins[1] - bins[0]
    real = real_and_generated[0]
    generated = real_and_generated[1]
    mask = (real > 0) & (generated > 0)
    real = real[mask]
    generated = generated[mask]
    bins = bins[:-1][mask]
    if show_plot:
        plt.plot(bins, real, label="real")
        plt.plot(bins, generated, label="generated")
        plt.title(f"KL Divergence {metric_type} histograms")
        plt.legend()
        plt.show()
    kl_divergence = np.sum(real * np.log(real / generated)).item() * dx
    assert isinstance(kl_divergence, float)
    assert kl_divergence > -1e-6, "KL Divergence should be non-negative"
    return kl_divergence
