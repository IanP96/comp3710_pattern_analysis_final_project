# Limit-order book TimeGAN project

Student name: Ian Pinto

Student ID: 48006581

For COMP3710 at The University of Queensland

# Project brief

Train a generative time-series model such as TimeGAN to generate synthetic sequences of limit order
book (LOB) events using the [LOBSTER dataset](https://lobsterdata.com/info/DataSamples.php) (Use AMZN level 10 data). Evaluate on a held-out test
split using the following metrics:

- **Distribution similarity**: KL divergence ≤ 0.1 between the generated and real spread and
  midprice return distributions.
- **Visual similarity**: SSIM > 0.6 between heatmaps of generated vs real LOB depth snapshots.

In your report, include model architecture and parameter count, training strategy (full vs variants such
as adversarial-only or supervised-only losses), GPU type, VRAM, epochs, and total training time. Also,
include 3–5 representative heatmap visualizations of generated vs real order books with a short error
analysis paragraph discussing where the synthetic LOBs succeed and fail. [Hard Difficulty]

*Note: You are likely to need some ’heavy’ GPU resources and the use of torch.run for distributed training. You may
also need to do your own reading on how to order books, volatility clustering, and how stock markets work.*