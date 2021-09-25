from functools import wraps, partial, reduce

import torch
import torch.optim

import src.distributions as my_dists
from .probabilistc_model import ProbabilisticModel


def normalize_per_dimension(prob_model, dims, func, name):
    if dims == 'all':
        dims = range(len(prob_model))
    elif dims == 'continuous':
        dims = [idxs for idxs, dist in prob_model.gathered if dist.real_dist.is_continuous]
        if len(dims) > 0:
            dims = reduce(list.__add__, dims)

    @wraps(func)
    def normalize_per_dimension_(x, mask=None):
        if len(dims) > 0:
            print('method:', name)

        for i in dims:
            dist_i = prob_model[i] if isinstance(prob_model, ProbabilisticModel) else prob_model[i][1]

            if dist_i.is_discrete:
                continue

            data = x[:, i] if (mask is None or mask[:, i].all()) else torch.masked_select(x[:, i], mask[:, i])
            data = dist_i >> data

            if isinstance(dist_i, my_dists.LogNormal):  # special case
                data = data.log()

            weight = func(data)
            dist_i.weight *= weight

            print(f'normalizing [dim={i}] [weight={dist_i.weight.item()}]')

        if len(dims) > 0:
            print('')

        return x
    return normalize_per_dimension_


normalize = partial(normalize_per_dimension, name='normalization', func=lambda x: 1/torch.max(torch.abs(x)).item())
standardize = partial(normalize_per_dimension, name='standardization', func=lambda x: 1/torch.std(x).item())
