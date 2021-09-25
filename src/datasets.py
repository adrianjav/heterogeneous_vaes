import os.path
from collections import namedtuple
from typing import Optional, Tuple, Sequence, Callable

import torch
import torch.distributions
from torch.distributions import constraints
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import numpy as np

import pytorch_lightning as pl


# Dictionary with the datasets allowed and their options.
DatasetOptions = namedtuple('DatasetOptions', ['with_header', 'has_nan_mask', 'create_mask_on_fly'])

allowed_datasets = {
    'defaultCredit': DatasetOptions(False, False, False),
    'Adult': DatasetOptions(False, True, False),
    'Breast': DatasetOptions(False, True, False),
    'spam': DatasetOptions(False, False, False),
    'Wine': DatasetOptions(False, False, False),
    'letter': DatasetOptions(False, False, False),
    'diamonds': DatasetOptions(True, False, True),
    'bank': DatasetOptions(True, False, True),
    'movies': DatasetOptions(True, True, True),
    'HI': DatasetOptions(True, False, True),
    'rwm5yr': DatasetOptions(True, False, True),
    'labour': DatasetOptions(True, False, True)
}


def read_data_file(file_path: str, categoricals: Sequence[int], prob_model, with_header: bool) -> torch.Tensor:
    r"""
    Reads the csv located in `file_path` and it makes sures that the data fits the support constraints from
    `prob_model`. If `with_header` is True, then the csv is expected to have a header and an initial index column.
    """

    if with_header:
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.read_csv(file_path, ',', header=None)

    for i in categoricals:
        df.iloc[:, i] -= df.iloc[:, i].min()

    assert df.shape[1] == len(prob_model)
    for i in range(df.shape[1]):
        # replace nans with a value
        df.iloc[:, i] = df.iloc[:, i].fillna(df.iloc[:, i].max())

        if prob_model.gathered[i][1].real_dist.dist.support == constraints.positive:
            df.iloc[:, i] = df.iloc[:, i].astype('float32').clip(lower=1e-20)  # ensure that is positive

        if str(prob_model.gathered[i][1]) == 'poisson':
            if df.iloc[:, i].min() != 0.:
                df.iloc[:, i] -= 1

        if str(prob_model.gathered[i][1]) == 'normal':
            df.iloc[:, i] -= df.iloc[:, i].mean()

    data = torch.tensor(df.to_numpy(float), dtype=torch.float32)
    return data


def read_mask_file(file_path: str, n_rows: int, n_cols: int) -> torch.Tensor:
    df = pd.read_csv(file_path, ',', names=['row', 'col'], header=None)
    df -= 1
    df = df.groupby('row')['col'].apply(list)

    mask = []
    for i in range(n_rows):
        row = torch.tensor(df[i] if i in df.index else []).long()

        mask_i = torch.ones(n_cols)
        mask_i.index_fill_(0, row, 0.)

        mask += [mask_i]

    return torch.stack(mask, dim=0)


class InductiveDataModule(pl.LightningDataModule):
    r"""
    DataModule for supervised learning. It performs the usual training/validation/test splits. Like in the transductive
    case, we return masks for the three sets in case that the user wants to evaluate imputation tasks as well.
    """

    train_data: TensorDataset
    val_data: TensorDataset
    test_data: TensorDataset

    def __init__(self, dir_path: str, miss_perc: int, mask_seed: int, categoricals: Sequence[int], prob_model,
                 batch_size: int, preprocess_fn: Sequence[Callable] = (), splits: Tuple[int, int, int] = (70, 10, 20)):
        r"""

        :param dir_path: Path to the dataset directory.
        :param prob_model: Heterogeneous likelihood class.
        :param missing_percentage: Percentage of missing data (at random) per column.
        :param mask_seed: Random seed for the generator (if mask is created on the fly), or file suffix (otherwise).
        :param batch_size: Batch size for the data loader.
        :param splits: Percentages for the train/val/test splits, respectively.
        """
        super().__init__()

        if dir_path[-1] == '/':
            dir_path = dir_path[:-1]

        self.dir_path = dir_path
        self.dataset_name = dir_path[dir_path.rindex('/') + 1:]
        self.miss_perc = miss_perc
        self.mask_seed = mask_seed
        self.categoricals = categoricals
        self.prob_model = prob_model
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size
        self.splits = splits

        assert self.dataset_name in allowed_datasets.keys()
        self.options = allowed_datasets[self.dataset_name]

    def prepare_data(self):
        assert os.path.exists(self.dir_path) and os.path.isdir(self.dir_path)

    def setup(self, stage: Optional[str] = None) -> None:
        data = read_data_file(f'{self.dir_path}/data.csv', self.categoricals, self.prob_model, self.options.with_header)
        size, n_features = data.size()

        if self.options.has_nan_mask:
            if self.options.with_header:
                nan_mask = pd.read_csv(f'{self.dir_path}/MissingTrue.csv', index_col=0)
                nan_mask = torch.from_numpy(nan_mask.values.astype('float32'))
            else:
                nan_mask = f'{self.dir_path}/MissingTrue.csv'
                nan_mask = read_mask_file(nan_mask, size, n_features)
        else:
            nan_mask = torch.ones_like(data, dtype=torch.float32)

        if self.options.create_mask_on_fly:
            np.random.seed(self.mask_seed)
            to_be_masked = int(size*(self.miss_perc/100))
            mask = torch.ones([size, n_features], dtype=torch.float32)
            for col in range(n_features):  # for each column select miss_perc% of rows randomly
                row_indices = np.random.choice(size, to_be_masked, replace=False)
                mask[:, col].index_fill_(0, torch.tensor(row_indices), 0.)
        else:
            mask = f'{self.dir_path}/Missing{self.miss_perc}_{self.mask_seed}.csv'
            mask = read_mask_file(mask, size, n_features)
        mask = (mask.long() + nan_mask.long()) == 2

        dataset = TensorDataset(data, mask, nan_mask)

        split_train = int(size * self.splits[0] / 100.)
        split_val = int(size * self.splits[1] / 100.)
        splits = (split_train, split_val, size - split_train - split_val)
        assert all([x > 0 for x in splits])

        generator = torch.Generator().manual_seed(self.mask_seed)
        self.train_data, self.val_data, self.test_data = random_split(dataset, splits, generator=generator)
        dataset[:][1].index_fill_(0, torch.tensor(self.train_data.indices), True)

        for fn in self.preprocess_fn:
            fn(self.train_data[:][0], self.train_data[:][1])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
