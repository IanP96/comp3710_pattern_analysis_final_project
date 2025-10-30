import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

from dataset import load_data
from constants import NUM_LEVELS
from options import Options

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_ssim(path_img1: Path | str, path_img2: Path | str) -> float:
    """Calculate SSIM between two images.

    Args:
      - path_img1: path to first image
      - path_img2: path to second image

    Returns:
      - ssim_value: calculated SSIM value
    """
    img1 = img_as_float(plt.imread(path_img1))
    img2 = img_as_float(plt.imread(path_img2))
    print(img1.shape)
    ssim_value = ssim(img1, img2, multichannel=True, channel_axis=2, data_range=1)
    return ssim_value


def plot_heatmap(
    data: NDArray,
    plot_title: str | None = None,
    plot_save_path: Path | str | None = None,
) -> None:

    volume_data = data[:, 1:40:2]
    max_volume = np.max(volume_data)
    # min_volume = np.min(volume_data)
    logger.debug("max_volume=%s", max_volume)

    price_data = data[:, 0:40:2]
    max_price = np.max(price_data)
    min_price = np.min(price_data)
    logger.debug("max_price=%s", max_price)

    # red for ask, blue for bid
    x = np.empty((len(data) * NUM_LEVELS * 2,), dtype=np.float32)
    y = np.empty_like(x)
    colour = np.empty((len(data) * NUM_LEVELS * 2, 4), dtype=np.float32)  # RGBA
    i = 0
    BID = 0
    ASK = 1
    for time_point, row in enumerate(data):
        for level in range(NUM_LEVELS):
            for order_type in range(2):  # 0 for ask, 1 for bid
                row = data[time_point]
                feature_num = level * 4 + order_type * 2
                price = row[feature_num]
                volume = row[feature_num + 1]
                x[i] = time_point
                y[i] = price
                colour[i] = np.array(
                    [
                        0.99 if order_type == ASK else 0.01,
                        0.01,
                        0.99 if order_type == BID else 0.01,
                        volume / max_volume,
                    ]
                )
                if not 0 <= volume / max_volume <= 1:
                    raise ValueError(str(volume / max_volume))
                i += 1

    assert i == len(x) == len(y) == len(colour)

    plt.ylim(min_price, max_price)
    plt.xlabel("Time point")
    plt.ylabel("Price")
    plt.scatter(x, y, c=colour)
    if plot_title is not None:
        plt.title(plot_title)
    if plot_save_path is not None:
        plt.savefig(plot_save_path)
    plt.show()


if __name__ == "__main__":

    # heatmap and SSIM on test data
    command_line_options = Options().parse()
    _, _, test_data = load_data(command_line_options)
    cutoff = int(len(test_data) / 100)
    plot_heatmap(
        test_data[:cutoff], "Heatmap of test LOBSTER data", "test_heatmap.png"
    )
    plot_heatmap(
        test_data[:cutoff] + np.random.rand(cutoff, NUM_LEVELS * 4) * 100,
        "Heatmap of test LOBSTER data plus noise",
        "test_heatmap_noise.png",
    )
    test_data_ssim = get_ssim(Path("test_heatmap.png"), Path("test_heatmap_noise.png"))
    print(f"{test_data_ssim = }")
