"""
Example usage of your trained model. Print out any results and/or provide visualisations where
applicable
"""

from options import Options
from dataset import load_data
from modules import TimeGAN


def main():
    # Arguments
    command_line_options = Options().parse()

    # Load data
    original_data, validate_data, test_data = load_data(command_line_options)

    # Load model
    model = TimeGAN(command_line_options, original_data, validate_data, test_data, load_weights=True)

    # Inference
    model.run_inference()


if __name__ == "__main__":
    main()
