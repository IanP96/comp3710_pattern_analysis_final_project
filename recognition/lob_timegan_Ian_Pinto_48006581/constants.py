"""
Constants used throughout the project
"""

OUTPUT_DIR = "output"

# Relative to OUTPUT_DIR
WEIGHTS_DIR = "weights"

# How many training iterations to use (max) on Rangpur
RANGPUR_NUM_TRAINING_ITERATIONS = 30_000
RANGPUR_VALIDATE_INTERVAL = 300

# How many training iterations to use (max) running locally on laptop
LOCAL_NUM_TRAINING_ITERATIONS = 3
LOCAL_VALIDATE_INTERVAL = 1

TRAIN_TEST_VALIDATE = (0.6, 0.2, 0.2)
assert (
    abs(sum(TRAIN_TEST_VALIDATE) - 1) < 1e-6
), "TRAIN_TEST_VALIDATE ratios must sum to 1.0"

DATA_DIR = "data"
ORDERBOOK_DATA_FILENAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
