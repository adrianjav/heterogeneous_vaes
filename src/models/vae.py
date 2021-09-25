from functools import partial

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs

import pytorch_lightning as pl


def init_weights(m, gain=1.):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)


class Encoder(nn.Module):
    def __init__(self, prob_model, latent_size, hidden_size, dropout):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout), nn.BatchNorm1d(len(prob_model)),
            nn.Linear(len(prob_model), hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        self.z_loc = nn.Linear(hidden_size, latent_size)
        self.z_log_scale = nn.Linear(hidden_size, latent_size)

        self.encoder.apply(partial(init_weights, gain=nn.init.calculate_gain('tanh')))
        self.z_loc.apply(init_weights)
        self.z_log_scale.apply(init_weights)

    def q_z(self, loc, logscale):
        scale = softplus(logscale)
        return dists.Normal(loc, scale)

    def forward(self, x):
        h = self.encoder(x)

        loc = self.z_loc(h)  # constraints.real
        log_scale = self.z_log_scale(h)  # constraints.real

        return loc, log_scale


class VAEHead(nn.Module):
    def __init__(self, dist, hidden_size):
        super().__init__()
        self.dist = dist

        self.head = nn.Linear(hidden_size, sum(self.dist.size_params))
        self.head.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    def unpack_params(self, theta):
        noise = 1e-15

        params = []
        pos = 0
        for i in range(self.dist.num_params):
            value = theta[..., pos: pos + self.dist.size_params[i]]
            value = value.squeeze(-1)

            if isinstance(self.dist.arg_constraints[i], constraints.greater_than):
                lower_bound = self.dist.arg_constraints[i].lower_bound
                value = lower_bound + noise + softplus(value)

            elif isinstance(self.dist.arg_constraints[i], constraints.less_than):
                upper_bound = self.dist.arg_constraints[i].upper_bound
                value = upper_bound - noise - softplus(value)

            elif self.dist.arg_constraints[i] == constraints.simplex:
                value = logits_to_probs(value)

            params += [value]
            pos += self.dist.size_params[i]

        return torch.stack(params, dim=0)

    def forward(self, y):
        raw_params = self.head(y)
        params = self.unpack_params(raw_params)
        return params


class VAE(pl.LightningModule):
    def __init__(self, prob_model, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.prob_model = prob_model
        self.samples = 1

        hparams = self.hparams

        # Prior
        self.prior_z_loc = torch.zeros(hparams.latent_size)
        self.prior_z_scale = torch.ones(hparams.latent_size)

        # Encoder
        self.encoder = Encoder(prob_model, hparams.latent_size, hparams.hidden_size, hparams.dropout)

        # Decoder
        self.decoder_shared = nn.Sequential(
            nn.Linear(hparams.latent_size, hparams.hidden_size), nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size), nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size), nn.ReLU(),
        )
        self.decoder_shared.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))
        self.heads = nn.ModuleList([VAEHead(dist, hparams.hidden_size) for dist in prob_model])

    @property
    def prior_z(self):
        return dists.Normal(self.prior_z_loc, self.prior_z_scale)

    def _run_step(self, x, mask):
        z_params = self.encoder(x if mask is None else x * mask.float())
        z = self.encoder.q_z(*z_params).rsample([self.samples])

        y = self.decoder_shared(z)
        x_params = [head(y) for head in self.heads]

        x_scaled = self.prob_model >> x
        x_scaled = x_scaled.unsqueeze(dim=0).tile((self.samples, 1, 1))
        mask = mask.unsqueeze(dim=0).tile((self.samples, 1, 1))

        # samples x batch_size x D
        log_px_z = [self.log_likelihood(x_scaled, mask, i, params_i) for i, params_i in enumerate(x_params)]

        log_pz = self.prior_z.log_prob(z).sum(dim=-1)  # samples x batch_size
        log_qz_x = self.encoder.q_z(*z_params).log_prob(z).sum(dim=-1)  # samples x batch_size
        kl_z = log_qz_x - log_pz

        return log_px_z, kl_z

    def _step(self, batch, batch_idx):
        x, mask, _ = batch
        log_px_z, kl_z = self._run_step(x, mask)

        elbo = sum(log_px_z) - kl_z
        loss = -elbo.squeeze(dim=0).sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(x, mask) * mask).sum(dim=0) / mask.sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl'] = kl_z.squeeze(dim=0).mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'training/{k}': v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'validation/{k}': v for k, v in logs.items()})
        return loss

    def _infer_step(self, x, mask, mode):
        z_params = self.encoder(x if mask is None else x * mask.float())
        if mode:
            z = z_params[0]  # Mode of a Normal distribution
        else:
            z = self.encoder.q_z(*z_params).sample()

        y = self.decoder_shared(z)
        x_params = [head(y) for head in self.heads]

        return x_params

    def _impute_step(self, x, mask, mode):
        x_params = self._infer_step(x, mask, mode=mode)

        new_x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = torch.cat([x_params[i] for i in idxs], dim=0)
            new_x_i = dist_i.impute(dist_i << params).float().flatten()
            new_x.append(new_x_i)

        return torch.stack(new_x, dim=-1), x_params

    def forward(self, batch, mode=True):
        x, mask, _ = batch
        return self._impute_step(x, mask, mode=mode)[0]

    # Measures
    def log_likelihood(self, x, mask, i, params_i):
        log_prob_i = self.prob_model[i].log_prob(x[..., i], params_i)
        if mask is not None:
            log_prob_i = log_prob_i * mask[..., i].float()
        return log_prob_i

    def _log_likelihood(self, x, x_params):
        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            params = torch.cat([x_params[i] for i in idxs], dim=0)
            log_prob_i = dist_i.real_log_prob(x[..., i], dist_i << params)
            log_prob.append(log_prob_i)

        return torch.stack(log_prob, dim=-1).squeeze(dim=0)  # batch_size x num_dimensions

    def log_likelihood_real(self, x, mask):
        x_params = self._infer_step(x, mask, mode=True)
        return self._log_likelihood(x, x_params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.parameters(), 'lr': self.hparams.learning_rate},
        ])

        if self.hparams.decay == 1.:
            return optimizer

        # We cannot set different schedulers if we want to avoid manual optimization
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.decay)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'  # Alternatively: "step"
            },
        }