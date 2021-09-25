from __future__ import annotations

from typing import List
from functools import reduce

import torch
from .distributions import get_distribution_by_name, Base


def _get_distributions(dists_names) -> List[Base]:
    dists = []

    for i, name in enumerate(dists_names):
        if 'categorical' in name or 'ordinal' in name:
            pos = name.find('(')
            num_probs = int(name[pos + 1 : name.find(')')])
            name = name[:pos]
        else:
            num_probs = 1

        if num_probs == 1:
            dist_i = get_distribution_by_name(name)()
        else:
            dist_i = get_distribution_by_name(name)(num_probs)

        dists += [dist_i]

    return dists


class ProbabilisticModel(object):
    def __init__(self, dists_names):
        self.dists = _get_distributions(dists_names)
        self.indexes = reduce(list.__add__, [[[i, j] for j in range(d.num_dists)] for i, d in enumerate(self.dists)])

    def to(self, device):
        for d in self:
            d._weight = d._weight.to(device)
        return self

    @property
    def weights(self):
        return [d.weight for d in self]

    @weights.setter
    def weights(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().tolist()

        for w, d in zip(values, self):
            d.weight = w

    def scale_data(self, x):
        new_x = []
        for i, d in enumerate(self):
            new_x.append(d >> x[:, i])
        return torch.stack(new_x, dim=-1)

    def __rshift__(self, data):
        return self.scale_data(data)

    def params_from_data(self, x, mask):
        params = []
        for i, d in enumerate(self):
            pos = self.gathered_index(i)
            data = x[..., i] if mask is None or mask[..., pos].all() else torch.masked_select(x[..., i], mask[..., pos])
            params += d.params_from_data(data)
        return params

    def preprocess_data(self, x, mask=None):
        new_x = []
        for i, dist_i in enumerate(self.dists):
            new_x += dist_i.preprocess_data(x[:, i], mask)

        for i in range(len(self.dists), x.size(1)):
            new_x += [x[:, i]]

        return torch.stack(new_x, 1)

    def gathered_index(self, index):
        return self.indexes[index][0]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item) -> Base:
        if isinstance(item, int):
            return self.__getitem__(self.indexes[item])

        return self.dists[item[0]][item[1]]

    @property
    def gathered(self):
        class GatherProbabilisticModel(object):
            def __init__(self, model):
                self.model = model

            def __len__(self):
                return len(self.model.dists)

            def __getitem__(self, item):
                offset = sum([d.num_dists for d in self.model.dists[: item]])
                idxs = range(offset, offset + self.model.dists[item].num_dists)

                return idxs, self.model.dists[item]

            @property
            def weights(self):
                return [d.weight for [_, d] in self]

            @weights.setter
            def weights(self, values):
                if isinstance(values, torch.Tensor):
                    values = values.detach().tolist()

                for w, [_, d] in zip(values, self):
                    d.weight = w

            def __iter__(self):
                offset = 0
                for i, d in enumerate(self.model.dists):
                    yield list(range(offset, offset + d.num_dists)), d
                    offset += d.num_dists

            def get_param_names(self):
                names = []
                for i, dist_i in enumerate(self.model.dists):
                    if dist_i.num_dists > 1 or dist_i.size_params[0] > 1:
                        param_name = dist_i.real_parameters[0]
                        num_classes = dist_i.size_params[0] if dist_i.num_dists == 1 else dist_i.num_dists
                        names += [f'{dist_i}_{param_name}{j}_dim{i}' for j in range(num_classes)]
                    else:
                        names += [f'{dist_i}_{v}_dim{i}' for v in dist_i.real_parameters]

                return names

            def scale_data(self, x):
                new_x = []
                for i, [_, d] in enumerate(self):
                    new_x.append(d >> x[:, i])
                return torch.stack(new_x, dim=-1)

            def __rshift__(self, data):
                return self.scale_data(data)

        return GatherProbabilisticModel(self)

