import sys
import argparse
import subprocess
import datetime

import torch

import src.feature_scaling as scaling
from src.datasets import InductiveDataModule
from src.probabilistc_model import ProbabilisticModel
from src.miscelanea import test_mie_ll

from src.models import VAE, IWAE, DREG, HIVAE

import pytorch_lightning as pl


def validate(args) -> None:
    args.timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    if args.dataset[-1] == '/':
        args.dataset = args.dataset[:-1]

    args.dataset = args.root + '/' + args.dataset

    # Read types of the dataset
    arguments = ['./read_types.sh', f'{args.dataset}/data_types.csv']

    proc = subprocess.Popen(arguments, stdout=subprocess.PIPE)
    out = eval(proc.communicate()[0].decode('ascii'))

    args.probabilistic_model = out['probabilistic model']
    args.categoricals = out['categoricals']


def print_data_info(prob_model, data):
    print()
    print('#' * 20)
    print('Original data')

    x = data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print()
    print(f'weights = {[x.item() for x in prob_model.weights]}')
    print()

    print('Scaled data')

    x = prob_model >> data
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, i].min()} {x[:, i].max()}')

    print('#' * 20)
    print()


def main(hparams):
    validate(hparams)
    pl.seed_everything(hparams.seed)

    if hparams.to_file:
        sys.stdout = open(f'{hparams.path}/test_{hparams.test_on}_stdout.txt', 'w')
        sys.stderr = open(f'{hparams.path}/test_{hparams.test_on}_stderr.txt', 'w')

    prob_model = ProbabilisticModel(hparams.probabilistic_model)
    print('Likelihoods:', [str(d) for d in prob_model])
    print('Dataset:', hparams.dataset)

    preprocess_fn = [scaling.standardize(prob_model, 'continuous')]
    dm = InductiveDataModule(hparams.dataset, hparams.miss_perc, hparams.miss_suffix, hparams.categoricals, prob_model,
                                 hparams.batch_size, preprocess_fn)

    dm.prepare_data()
    dm.setup(stage='test')

    test_loader = dm.test_dataloader()

    # Evaluate
    prob_model = prob_model.to('cpu')

    print('Loading and evaluating best model.')
    model = {
        'vae': VAE, 'iwae': IWAE, 'dreg': DREG, 'hivae': HIVAE
    }[hparams.model].load_from_checkpoint(f'{hparams.path}/{hparams.test_on}.ckpt', prob_model=prob_model)

    test_mie_ll(model, prob_model, test_loader.dataset, hparams.device)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    # Configuration
    parser = argparse.ArgumentParser('')

    # General
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-root', type=str, default='.', help='Output folder (default: \'%(default)s)\'')
    parser.add_argument('-to-file', action='store_true', help='Redirect output to \'test_best_stdout.txt\'')
    parser.add_argument('-test-on', type=str, default='best', choices=['best', 'last'])

    parser.add_argument('-model', type=str, required=True, choices=['vae', 'iwae', 'hivae', 'dreg'])

    # Tracking
    parser.add_argument('-tensorboard', action='store_true', help='Activates tensorboard logs.')

    # Dataset
    group = parser.add_argument_group('dataset')
    group.add_argument('-batch-size', type=int, default=1024, help='Batch size (%(default)s)')
    group.add_argument('-dataset', type=str, required=True, help='Dataset to use (path to folder)')
    group.add_argument('-miss-perc', type=int, required=True, help='Missing percentage')
    group.add_argument('-miss-suffix', type=int, required=True, help='Suffix of the missing percentage file')

    parser.add_argument('-path', type=str, required=True, help='Path to the experiment folder')

    args = parser.parse_args()
    main(args)

    sys.exit(0)
