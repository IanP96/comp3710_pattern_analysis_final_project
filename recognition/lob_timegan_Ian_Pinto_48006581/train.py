"""
Code for training, validating and saving model.
The model should be imported from “modules.py” and the data loader should be imported from
“dataset.py”.
Make sure to plot the losses and metrics during training.
"""

"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""


import os

from options import Options
from dataset import load_data
from modules import TimeGAN

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    """Training"""

    # Arguments
    command_line_options = Options().parse()

    # Load data
    original_data = load_data(command_line_options)

    # Load model
    model = TimeGAN(command_line_options, original_data)

    # Train model
    model.train()


if __name__ == "__main__":
    train()
