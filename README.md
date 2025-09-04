# timesc

This package offers tools to perform NLP-inspired self-consistency for time series forecasting. It applies to any forecasting model.

## Setup

## Usage

Notebook `tirex.ipynb` provides example of usage of quantile based forecasting trajectory sampling, both on synthetic time series datasets and real datasets. Self consistency is then applied to these sampled trajectories, by taking their median.

To sample next step from quantile-based forecast, a pseudo distribution has to be rtetrieved from the quantiles. `metalog.py` provides a function that outputs inverse cdf based on quantiles. `quantile_sampling.py` provides a function that samples trajectories.

![example of sampled trajectories](figs/samples.png)