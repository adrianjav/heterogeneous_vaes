import yaml
import os
import sys
import argparse
import subprocess
import datetime

import torch

import src.plotting as plt
import src.feature_scaling as scaling
from src.datasets import InductiveDataModule
from src.probabilistc_model import ProbabilisticModel
from src.miscelanea import test_mie_ll

from src.models import VAE, IWAE, DREG, HIVAE

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks


def validate(args) -> None:
    args.timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    if args.dataset[-1] == '/':
        args.dataset = args.dataset[:-1]

    assert args.model != 'iwae' or args.samples > 0

    dataset = args.dataset
    if dataset[-1] == '/':
        dataset = dataset[:-1]

    args.dataset = args.root + '/' + args.dataset

    args.root = f'{args.root}/results/{args.model}/{dataset}/' \
                f'dropout_{args.dropout}/' \
                f'Missing{args.miss_perc}_{args.miss_suffix}'

    # Read types of the dataset
    arguments = ['./read_types.sh', f'{args.dataset}/data_types.csv']

    proc = subprocess.Popen(arguments, stdout=subprocess.PIPE)
    out = eval(proc.communicate()[0].decode('ascii'))

    args.probabilistic_model = out['probabilistic model']
    args.categoricals = out['categoricals']

    if args.max_epochs is None:
        args.max_epochs = {
            'Wine': 2000, 'letter': 400, 'spam': 2000,
            'Adult': 400, 'defaultCredit': 400, 'Breast': 3000,
            'labour': 400, 'HI': 400, 'diamonds': 400, 'CPS1988': 400,
            'rwm5yr': 400, 'movies': 400, 'bank': 400
        }[args.dataset[args.dataset.rindex('/')+1:]]


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


@torch.no_grad()
def test(model, prob_model, loader, device):
    model.eval()
    mask_bc = loader.dataset[:][1].to(device)
    generated_data = model([loader.dataset[:][0].to(device), mask_bc, None], mode=False).cpu()

    data = loader.dataset[:][0]
    plt.plot_together([data, generated_data], prob_model, title='', legend=['original', 'generated'],
                      path=f'{args.root}/marginal')


def main(hparams):
    validate(hparams)
    pl.seed_everything(hparams.seed)
    os.makedirs(hparams.root, exist_ok=True)

    if hparams.to_file:
        sys.stdout = open(f'{hparams.root}/stdout.txt', 'w')
        sys.stderr = open(f'{hparams.root}/stderr.txt', 'w')

    prob_model = ProbabilisticModel(hparams.probabilistic_model)
    print('Likelihoods:', [str(d) for d in prob_model])

    if hparams.latent_size is None:
        if hparams.latent_perc is not None:
            hparams.latent_size = max(1, int(len(prob_model.gathered) * (hparams.latent_perc / 100) + 0.5))
        else:
            hparams.latent_size = max(1, int(len(prob_model.gathered) * 0.75 + 0.5))

    if not hasattr(hparams, 'size_s') or hparams.size_s is None:
        hparams.size_s = hparams.latent_size

    if not hasattr(hparams, 'size_z') or hparams.size_z is None:
        hparams.size_z = hparams.latent_size

    if not hasattr(hparams, 'size_y') or hparams.size_y is None:
        hparams.size_y = hparams.hidden_size

    print('Dataset:', hparams.dataset)

    preprocess_fn = [scaling.standardize(prob_model, 'continuous')]
    dm = InductiveDataModule(hparams.dataset, hparams.miss_perc, hparams.miss_suffix, hparams.categoricals, prob_model,
                                 hparams.batch_size, preprocess_fn)

    dm.prepare_data()
    dm.setup(stage='fit')

    train_loader = dm.train_dataloader()
    test_loader = dm.val_dataloader()
    print_data_info(prob_model, train_loader.dataset[:][0])

    with open(f'{hparams.root}/args.yml', 'w') as outfile:
        yaml.dump(hparams, outfile)

    # Crete model and trainer
    model = {
        'vae': VAE, 'iwae': IWAE, 'dreg': DREG, 'hivae': HIVAE
    }[hparams.model](prob_model, hparams)

    tb_logger = None
    if hparams.tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(f'{hparams.root}/tb_logs')

    timer = pl_callbacks.Timer()
    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=hparams.root, filename='best',
                                                       monitor='validation/re', save_last=True)

    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs, logger=tb_logger, default_root_dir=hparams.root,
        callbacks=[timer, checkpoint_callback]
    )

    # Train
    trainer.fit(model, dm)

    seconds = timer.time_elapsed('train')
    print(f'Training finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')

    # Evaluate
    prob_model = prob_model.to('cpu')

    print('Loading and evaluating best model.')
    model = type(model).load_from_checkpoint(trainer.checkpoint_callback.best_model_path, prob_model=prob_model)
    test(model, prob_model, test_loader, hparams.device)

    test_mie_ll(model, prob_model, train_loader.dataset, hparams.device, title='Train', missing=False)
    test_mie_ll(model, prob_model, test_loader.dataset, hparams.device, missing=True)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    # Configuration
    parser = argparse.ArgumentParser('')

    # General
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('-root', type=str, default='.', help='Output folder (default: \'%(default)s)\'')
    parser.add_argument('-to-file', action='store_true', help='Redirect output to \'stdout.txt\'')

    parser.add_argument('-model', type=str, required=True, choices=['vae', 'iwae', 'hivae', 'dreg'])

    # Tracking
    parser.add_argument('-tensorboard', action='store_true', help='Activates tensorboard logs.')

    # Dataset
    group = parser.add_argument_group('dataset')
    group.add_argument('-batch-size', type=int, default=1024, help='Batch size (%(default)s)')
    group.add_argument('-dataset', type=str, required=True, help='Dataset to use (path to folder)')
    group.add_argument('-miss-perc', type=int, required=True, help='Missing percentage')
    group.add_argument('-miss-suffix', type=int, required=True, help='Suffix of the missing percentage file')

    # Training
    group = parser.add_argument_group('training')
    group.add_argument('-learning-rate', type=float, default=0.001, help='Learning rate')
    group.add_argument('-decay', type=float, default=1., help='Learning rate\'s exponential decay rate.')  # 0.999999
    group.add_argument('-max-epochs', type=int, default=None, help='Number of epochs.')

    # VAE
    group = parser.add_argument_group('vae/iwae')
    group.add_argument('-latent-size', type=int, default=None)
    group.add_argument('-latent-perc', type=int, default=None)
    group.add_argument('-dropout', type=float, default=0.1, help='Dropout percentage on the input layer')
    group.add_argument('-hidden-size', type=int, default=200, help='Size of the hidden layers')

    # IWAE
    group = parser.add_argument_group('iwae')
    group.add_argument('-use-dreg', action='store_true', help='Whether to use the doubly rep. estimator')
    group.add_argument('-samples', type=int, default=None, help='Number of importance samples')

    # HI-VAE
    group = parser.add_argument_group('hivae')
    group.add_argument('-size-z', type=int, default=None)
    group.add_argument('-size-s', type=int, default=None)
    group.add_argument('-size-y', type=int, default=None)

    args = parser.parse_args()
    main(args)

    sys.exit(0)
