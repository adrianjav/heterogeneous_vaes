# Heterogeneous VAEs

_Beware: This repository is under construction 🛠️_

Pytorch implementation of different VAE models to model heterogeneous data.
Here, we call heterogeneous data those for which we assume that each feature is of a different type, and therefore each feature is assumed to have a different likelihood. 
Heterogeneous data is also known as _mixed-type_ data and _tabular_ data.

## Usage

This repository is not meant to be a library which you can install and use as it is, but rather as a ML project code which you can freely fork and modify to fit your particular needs.

## Dependencies

We are working on providing a conda requirements file. For the moment, there is a Dockerfile which you can build and use, or simply look at the project dependencies from there.

## Example

You can find information about all the available arguments via `python main.py --help`. For example, you can train the Wine dataset on a heterogeneous VAE with default arguments using:

```{bash}
python main.py -model=vae -dataset=datasets/Wine -seed=2 -miss-perc=20 -miss-suffix=1
```

## Models

This repository contains implementations of the following models, adapted for heterogeneous likelihoods (if you use them in your work, _make sure to cite the original authors_):
- _Autoencoding Variational Bayes_ (VAE): https://arxiv.org/abs/1312.6114
- _Importance Weighted Autoencoder_ (IWAE): http://arxiv.org/abs/1509.00519
- _Doubly Reparametrized Gradient Estimators for Monte Carlo objectives_ (DReG): http://arxiv.org/abs/1810.04152
- _Handling Incomplete Heterogeneous Data using VAEs_ (HI-VAE): http://arxiv.org/abs/1807.03653

## Likelihoods

The code supports the following likelihoods at the moment: 
- Gaussian, for real-valued features.
- Log-normal, for positive-valued real features.
- Bernoulli, for binary features.
- Categorical, for categorical features.
- Poisson, for positive-valued integer (count) features.

## Datasets

We provide with this code some example datasets taken from [UCI](https://archive.ics.uci.edu/ml/datasets.php) and [R package datasets](https://vincentarelbundock.github.io/Rdatasets/datasets.html).
You can use any dataset as long as the format is the same.

## Contributing

The code can be further simplified and polished, and we still have some legacy code. Pull requests and issues are more than welcome, as long as it contributes to making the code clean, simple, general, and elegant.
