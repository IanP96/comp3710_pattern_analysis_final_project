# Links

## Info

- [Article - Modeling and Generating Time-Series Data using
  TimeGAN](https://towardsdatascience.com/modeling-and-generating-time-series-data-using-timegan-29c00804f54d/)
  - read
- [Research article - Modeling and Generating Time-Series Data using TimeGAN](https://www.researchgate.net/publication/373686527_Generative_AI_for_End-to-End_Limit_Order_Book_Modelling_A_Token-Level_Autoregressive_Generative_Model_of_Message_Flow_Using_a_Deep_State_Space_Network)
  - Not read
- [Generating Synthetic Market Data](https://jonathankinlay.com/2022/07/generating-synthetic-market-data/)
  - Read
  - Is TimeGAN the best choice?
  - "For stock data, there are some very basic tests that should first be performed to ensure the
    consistency of the synthetic output. In particular, in each row of the window, the High should
    exceed the Open, Low and Close prices, with the Low price falling below the Open, High and Close
    prices."
- [A New Approach to Generating Synthetic Market Data](https://jonathankinlay.com/2022/07/a-new-approach-to-generating-synthetic-market-data/)
  - Follows on from previous article
  - Read (kinda)
- [synthetic-data-for-finance TimeGAN Github repo](https://github.com/stefan-jansen/synthetic-data-for-finance?tab=readme-ov-file#code-example-timegan-adversarial-training-for-synthetic-financial-data)

## Code

- [QuantGAN](https://github.com/PakAndrey/QuantGANforRisk)
- [ydata TimeGAN GitHub](https://github.com/ydataai/ydata-synthetic/)
  - Seems to be the best option
- [ydata PyPi](https://pypi.org/project/ydata-synthetic/)
- [TimeGAN on stock data (jupyter notebook)](https://colab.research.google.com/github/ydataai/ydata-synthetic/blob/master/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb#scrollTo=FGzo4LZqjOWA)
- [ydata time-series data documentation](https://docs.synthetic.ydata.ai/latest/synthetic_data/time_series/timegan_example/)
- [TimeGAN Pytorch GitHub](https://github.com/zwzhang123/TimeGAN-pytorch/tree/main)

# Thoughts

- Is the task to generate order books or order messages?
- Use KL divergence as an evaluation metric
  - but it is non-differentiable in this case so don't use as a cost function
- Should ensure consistency of data i.e. asks higher than bids etc
  - Idea - what if I set the discriminator to always give an output of 0 (fake) for data that
    violated the ask > bid property?

# Model

- 40 features (10 levels of LOBSTER data. Each level gives bid price, bid volume, ask price, ask
  volume)
- Slices are shuffled before training. Good for learning statistical patterns, not so much for
  long-term trends
- Slice size of 24 could maybe be longer if I wanted the model to learn long-term patterns better
- 