import torch

from .vae import VAE


class IWAE(VAE):
    def __init__(self, prob_model, hparams):
        super().__init__(prob_model, hparams)
        hparams = self.hparams
        self.samples = hparams.samples

    def _step(self, batch, batch_idx):
        x, mask, _ = batch
        log_px_z, kl_z = self._run_step(x, mask)

        lw = sum(log_px_z) - kl_z  # samples x batch_size

        loss = torch.logsumexp(lw, dim=0)
        loss = loss - torch.ones_like(loss).mul(self.samples).log()
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
