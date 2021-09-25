from functools import partial

import torch
import torch.nn as nn
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs

import pytorch_lightning as pl

from src.distributions import GumbelDistribution
from src.miscelanea import to_one_hot


def init_weights(m, gain=1.):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.05)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class Encoder(nn.Module):
    def __init__(self, prob_model, size_s, size_z):
        super().__init__()
        input_size = sum([d.size for d in prob_model])

        # Encoder
        self.encoder_s = nn.Linear(input_size, size_s)

        self.encoder_z = nn.Identity()  # Just in case we want to increase this part
        self.q_z_loc = nn.Linear(input_size + size_s, size_z)
        self.q_z_log_scale = nn.Linear(input_size + size_s, size_z)

        self.encoder_z.apply(partial(init_weights))
        self.q_z_loc.apply(init_weights)
        self.q_z_log_scale.apply(init_weights)

        self.temperature = 1.

    def q_z(self, loc, log_scale):
        scale = torch.exp(log_scale)
        return dists.Normal(loc, scale)

    def q_s(self, logits):
        return GumbelDistribution(logits=logits, temperature=self.temperature)

    def forward(self, x, mode=False):
        s_logits = self.encoder_s(x)
        if mode:
            s_samples = to_one_hot(torch.argmax(s_logits, dim=-1), s_logits.size(-1)).float()
        else:
            s_samples = self.q_s(s_logits).rsample() if self.training else self.q_s(s_logits).sample()

        x_and_s = torch.cat((x, s_samples), dim=-1)  # batch_size x (input_size + latent_s_size)

        h = self.encoder_z(x_and_s)
        z_loc = self.q_z_loc(h)
        z_log_scale = self.q_z_log_scale(h)
        # z_log_scale = torch.clamp(z_log_scale, -7.5, 7.5)

        return s_samples, [s_logits, z_loc, z_log_scale]


class HIVAEHead(nn.Module):
    def __init__(self, dist, size_s, size_z, size_y):
        super().__init__()
        self.dist = dist

        # Generates its own y from z
        self.net_y = nn.Linear(size_z, size_y)

        # First parameter generated with y and s
        self.head_y_and_s = nn.Linear(size_y + size_s, self.dist.size_params[0], bias=False)

        # Next parameters (if any) generated only with s
        self.head_s = None
        if len(self.dist.size_params) > 1:
            self.head_s = nn.Linear(size_s, sum(self.dist.size_params[1:]), bias=False)
            self.head_s.apply(partial(init_weights))

        self.net_y.apply(partial(init_weights))
        self.head_y_and_s.apply(partial(init_weights))

    def unpack_params(self, theta, first_parameter):
        noise = 1e-15

        params = []
        pos = 0
        for i in ([0] if first_parameter else range(1, self.dist.num_params)):
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

            elif self.dist.size > 1:
                value[..., 0] = value[..., 0] * 0.

            params += [value]
            pos += self.dist.size_params[i]

        return torch.stack(params, dim=0)

    def forward(self, z, s):
        y = self.net_y(z)
        y_and_s = torch.cat((y, s), dim=-1)  # batch_size x (hidden_size + latent_s_size)

        raw_params = self.head_y_and_s(y_and_s)  # First parameter
        params = self.unpack_params(raw_params, first_parameter=True)

        if self.head_s is not None:  # Other parameters (if any)
            raw_params = self.head_s(s)
            params_s = self.unpack_params(raw_params, first_parameter=False)
            params = torch.cat((params, params_s), dim=0)

        return params


class HIVAE(pl.LightningModule):
    def __init__(self, prob_model, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.prob_model = prob_model
        self.samples = 1

        hparams = self.hparams

        # Parameters for the normalization layers
        self.mean_data = [0. for _ in range(len(prob_model))]
        self.std_data = [1. for _ in range(len(prob_model))]

        # Priors
        self.prior_s_pi = torch.ones(hparams.size_s) / hparams.size_s
        self.p_z_loc = nn.Linear(hparams.size_s, hparams.size_z)
        self.p_z_loc.apply(partial(init_weights))

        # Encoder
        self.encoder = Encoder(prob_model, hparams.size_s, hparams.size_z)

        # Decoder
        self.decoder_shared = nn.Identity()  # In case we want to increase this part
        self.decoder_shared.apply(partial(init_weights))

        self.heads = nn.ModuleList([
            HIVAEHead(dist, hparams.size_s, hparams.size_z, hparams.size_y) for dist in prob_model
        ])

    def prior_z(self, loc):
        return dists.Normal(loc, 1.)

    @property
    def prior_s(self):
        return dists.OneHotCategorical(probs=self.prior_s_pi, validate_args=False)

    def normalize_data(self, x, mask, epsilon=1e-6):
        assert len(self.prob_model) == x.size(-1)

        new_x = []
        for i, d in enumerate(self.prob_model):
            x_i = torch.masked_select(x[..., i], mask[..., i].bool()) if mask is not None else x[..., i]
            new_x_i = torch.unsqueeze(x[..., i], 1)

            if str(d) == 'normal':
                self.mean_data[i] = x_i.mean()
                self.std_data[i] = x_i.std()
                self.std_data[i] = torch.clamp(self.std_data[i], 1e-6, 1e20)

                new_x_i = (new_x_i - self.mean_data[i]) / (self.std_data[i] + epsilon)
            elif str(d) == 'lognormal':
                x_i = torch.log1p(x_i)
                self.mean_data[i] = x_i.mean()
                self.std_data[i] = x_i.std()
                self.std_data[i] = torch.clamp(self.std_data[i], 1e-10, 1e20)

                new_x_i = (torch.log1p(new_x_i) - self.mean_data[i]) / (self.std_data[i] + epsilon)
            elif str(d) == 'poisson':
                new_x_i = torch.log1p(new_x_i)  # x[..., i] can have 0 values (just as a poisson distribution)

            elif 'categorical' in str(d) or 'bernoulli' in str(d):
                new_x_i = to_one_hot(torch.squeeze(new_x_i, 1), d.size)


            new_x.append(new_x_i)

        # new_x = torch.stack(new_x, dim=-1)
        new_x = torch.cat(new_x, 1)

        def broadcast_mask(mask, prob_model):
            if all([d.size == 1 for d in prob_model]):
                return mask

            new_mask = []
            for i, d in enumerate(self.prob_model):
                new_mask.append(mask[:, i].unsqueeze(-1).expand(-1, d.size))

            return torch.cat(new_mask, dim=-1)

        mask = broadcast_mask(mask, self.prob_model)

        if mask is not None:
            new_x = new_x * mask
        return new_x

    def denormalize_params(self, etas):
        new_etas = []
        for i, d in enumerate(self.prob_model):
            etas_i = etas[i]

            if str(d) == 'normal':
                mean_data, std_data = self.mean_data[i], self.std_data[i]
                std_data = torch.clamp(std_data, min=1e-3)

                mean, std = d.to_params(etas_i)
                mean = mean * std_data + mean_data
                std = torch.clamp(std, min=1e-3, max=1e20)
                std = std * std_data

                etas_i = d.to_naturals([mean, std])
                etas_i = torch.stack(etas_i, dim=0)
            elif str(d) == 'lognormal':
                mean_data, std_data = self.mean_data[i], self.std_data[i]
                # std_data = torch.clamp(std_data, min=1e-10)

                mean, std = d.to_params(etas_i)
                mean = mean * std_data + mean_data
                # std = torch.clamp(std, min=1e-6) #, max=1)
                std = std * std_data

                etas_i = d.to_naturals([mean, std])
                etas_i = torch.stack(etas_i, dim=0)

            new_etas.append(etas_i)
        return new_etas

    def _run_step(self, x, mask):
        # Normalization layer
        new_x = self.normalize_data(x, mask)

        # Sampling s and obtaining z and s parameters
        s_samples, params = self.encoder(new_x)
        s_logits, z_loc, z_log_scale = params

        # Sampling z
        z = self.encoder.q_z(z_loc, z_log_scale).rsample()

        # Obtaining the parameters of x
        y_shared = self.decoder_shared(z)
        x_params = [head(y_shared, s_samples) for head in self.heads]
        x_params = self.denormalize_params(x_params)  # Denormalizing parameters

        # Compute all the log-likelihoods

        # batch_size x D
        log_px_z = [self.log_likelihood(x, mask, i, params_i) for i, params_i in enumerate(x_params)]

        pz_loc = self.p_z_loc(s_samples)
        log_pz = self.prior_z(pz_loc).log_prob(z).sum(dim=-1)  # batch_size
        log_qz_x = self.encoder.q_z(z_loc, z_log_scale).log_prob(z).sum(dim=-1)  # batch_size
        kl_z = log_qz_x - log_pz

        # batch_size
        log_ps = self.prior_s.log_prob(s_samples)
        log_qs_x = dists.OneHotCategorical(logits=s_logits, validate_args=False).log_prob(s_samples)
        kl_s = log_qs_x - log_ps

        return log_px_z, kl_z, kl_s

    def _step(self, batch, batch_idx):
        x, mask, _ = batch
        log_px_z, kl_z, kl_s = self._run_step(x, mask)

        elbo = sum(log_px_z) - kl_z - kl_s
        loss = -elbo.sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(x, mask) * mask).sum(dim=0) / mask.sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl_z'] = kl_z.mean(dim=0)
            logs['kl_s'] = kl_s.mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

            if self.training:
                logs['temperature'] = self.encoder.temperature

        return loss, logs

    def training_step(self, batch, batch_idx):
        self.encoder.temperature = max(1e-3, 1. - 0.01 * self.trainer.current_epoch)
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'training/{k}': v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'validation/{k}': v for k, v in logs.items()})
        return loss

    def _infer_step(self, x, mask, mode):
        new_x = self.normalize_data(x, mask)
        s_samples, params = self.encoder(new_x, mode=mode)
        s_logits, z_loc, z_log_scale = params

        if mode:
            z = z_loc  # Mode of a Normal distribution
        else:
            z = self.encoder.q_z(z_loc, z_log_scale).sample()

        y_shared = self.decoder_shared(z)
        x_params = [head(y_shared, s_samples) for head in self.heads]
        x_params = self.denormalize_params(x_params)  # Denormalizing parameters

        return x_params

    def _impute_step(self, x, mask, mode):
        x_params = self._infer_step(x, mask, mode=mode)

        new_x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = torch.cat([x_params[i] for i in idxs], dim=0)
            new_x_i = dist_i.impute(params).float().flatten()
            if str(dist_i) == 'lognormal':
                # new_x_i = torch.where(new_x_i > 20, new_x_i, new_x_i.expm1().log())
                new_x_i = torch.clamp(new_x_i, 1e-20, 1e20)
            new_x.append(new_x_i)

        return torch.stack(new_x, dim=-1), x_params

    def forward(self, batch, mode=True):
        x, mask, _ = batch
        return self._impute_step(x, mask, mode=mode)[0]

    # Measures
    def log_likelihood(self, x, mask, i, params_i):
        x_i = x[..., i]

        log_prob_i = self.prob_model[i].log_prob(x_i, params_i)
        if mask is not None:
            log_prob_i = log_prob_i * mask[..., i].float()
        return log_prob_i

    def _log_likelihood(self, x, x_params):
        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            x_i = x[..., i]

            params = torch.cat([x_params[i] for i in idxs], dim=0)
            log_prob_i = dist_i.real_log_prob(x_i, params)
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
