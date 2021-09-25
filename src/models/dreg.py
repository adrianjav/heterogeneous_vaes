import torch

from .vae import VAE


class DREG(VAE):
    def __init__(self, prob_model, hparams):
        super().__init__(prob_model, hparams)
        hparams = self.hparams
        self.samples = hparams.samples

    def _run_step(self, x, mask):
        z_params = self.encoder(x if mask is None else x * mask.float())
        z = self.encoder.q_z(*z_params).rsample([self.samples])

        x_scaled = self.prob_model >> x
        x_scaled = x_scaled.unsqueeze(dim=0).tile((self.samples, 1, 1))
        mask = mask.unsqueeze(dim=0).tile((self.samples, 1, 1))

        # Decoder
        y = self.decoder_shared(z.detach())
        x_params = [head(y) for head in self.heads]
        log_px_z_decoder = [self.log_likelihood(x_scaled, mask, i, params_i) for i, params_i in enumerate(x_params)]

        # Encoder
        if self.training:
            self.decoder_shared.requires_grad_(False)
            self.heads.requires_grad_(False)

        y = self.decoder_shared(z)
        x_params = [head(y) for head in self.heads]
        log_px_z_encoder = [self.log_likelihood(x_scaled, mask, i, params_i) for i, params_i in enumerate(x_params)]

        if self.training:
            self.decoder_shared.requires_grad_(True)
            self.heads.requires_grad_(True)

        z_params = [param_i.detach() for param_i in z_params]
        log_qz_x = self.encoder.q_z(*z_params).log_prob(z).sum(dim=-1)  # samples x batch_size

        log_pz = self.prior_z.log_prob(z).sum(dim=-1)  # samples x batch_size
        kl_z = log_qz_x - log_pz

        return log_px_z_decoder, log_px_z_encoder, kl_z

    def _step(self, batch, batch_idx):
        x, mask, _ = batch
        log_px_z_decoder, log_px_z_encoder, kl_z = self._run_step(x, mask)

        lw = sum(log_px_z_encoder) - kl_z  # samples x batch_size
        with torch.no_grad():
            w_tilde = torch.exp(lw - torch.logsumexp(lw, dim=0, keepdim=True))

        loss_encoder = torch.sum(w_tilde**2 * lw, dim=0)
        loss_decoder = torch.sum(w_tilde * sum(log_px_z_decoder), dim=0)
        loss = 0.5 * (loss_decoder + loss_encoder)
        loss = -loss.sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(x, mask) * mask).sum(dim=0) / mask.sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl'] = kl_z.mean(dim=0).mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

        return loss, logs

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