import torch
import matplotlib.pyplot as plt

import numpy as np


def plot_distribution(data, dist, **kwargs):
    """
    Plots the values according to whether the distribution is discrete or continuous (1-dimensional)
    """
    if dist.real_dist.is_discrete:
        if not isinstance(data, torch.Tensor):
            weights = np.ones_like(data[0]) / float(len(data[0]))
            weights = [weights] * len(data)
        else:
            weights = np.ones_like(data) / float(len(data))

        bins = sorted(list(set(np.unique(data[1])).union(set(np.unique(data[0])))))
        plt.hist([d.tolist() for d in data], alpha=0.5, weights=weights, bins=bins, **kwargs)  # bins=data[0].unique(), **kwargs)
    else:
        if 'color' in kwargs.keys():
            colors = kwargs.pop('color')
        else:
            colors = [None] * len(data)

        for d, color in zip(data, colors):
            plt.hist(d, bins=100, alpha=0.5, density=True, color=color, **kwargs)


def plot_together(all_data, prob_model, title, path, dims=None, legend=None, **kwargs):
    colors = ['r', 'b', 'g']
    if dims is None:
        dims = range(len(prob_model.gathered))

    for dim in dims:
        if str(prob_model.gathered[dim][1]) != 'lognormal':
            plot_distribution([d[..., dim].numpy() for d in all_data], prob_model.gathered[dim][1],
                              color=colors[:len(all_data)], **kwargs)
        else:
            plot_distribution([torch.log1p(d[..., dim]).numpy() for d in all_data], prob_model.gathered[dim][1],
                              color=colors[:len(all_data)], **kwargs)

        plt.suptitle(title)
        if legend:
            plt.legend(legend)

        plt.savefig(f'{path}_{dim}' if len(dims) > 1 else path)
        plt.close()
