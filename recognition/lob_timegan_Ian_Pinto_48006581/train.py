"""
Code for training, validating and saving model.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Original code author: Zhiwei Zhang (bitzzw@gmail.com)
Modified by: Ian Pinto
"""

from options import Options
from dataset import load_data
from modules import TimeGAN

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    """Training"""

    # Arguments
    command_line_options = Options().parse()

    # Load data
    original_data, validate_data, test_data = load_data(command_line_options)

    # Load model
    model = TimeGAN(command_line_options, original_data, validate_data, test_data)

    # Train model
    model.train_and_save()


if __name__ == "__main__":
    train()
