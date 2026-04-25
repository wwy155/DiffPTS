"""D3U-style diffusion wrapper: same noise schedule as DiffPTS, D3U MLP as epsilon predictor."""

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.nn.tmdm_diffusion_utils import make_beta_schedule
from src.models.DiffPTS import compute_gx_term, compute_hat_alpha, compute_tilde_alpha
from src.d3u.denoise_models.MLP import MLP


class D3UDiffusion(nn.Module):
    """
    Diffusion tensors match ``DiffPTS``; denoiser is D3U ``MLP(y_t, t, enc_out)``.
    Patch ``enc_out`` (from ``cond_pred_model``) must be set via ``set_enc_out`` before ``forward``.
    """

    def __init__(self, configs, device):
        super().__init__()
        self.args = configs
        self.device = device
        self.num_timesteps = configs.timesteps

        betas = make_beta_schedule(
            schedule=configs.beta_schedule,
            num_timesteps=configs.timesteps,
            start=configs.beta_start,
            end=configs.beta_end,
        )
        self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(self.betas)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(self.device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.betas_bar = 1 - self.alphas_cumprod
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas)
        self.alphas_tilde = self.alphas_cumprod_sum
        self.alphas_hat = compute_hat_alpha(alphas).to(self.device)
        self.betas_tilde = self.alphas_tilde - self.alphas_hat
        self.gx_term = compute_gx_term(alphas).to(self.device)
        assert (torch.tensor(self.betas_tilde) >= 0).all()
        assert ((self.betas_bar - self.betas_tilde) >= 0).all()

        self.betas_tilde_m_1 = torch.cat([torch.ones(1, device=self.device), self.betas_tilde[:-1]], dim=0)
        self.betas_bar_m_1 = torch.cat([torch.ones(1, device=self.device), self.betas_bar[:-1]], dim=0)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if configs.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_sum_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod_sum[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coeff_2 = torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.tau = None

        patch_len = getattr(configs, "d3u_patch_len", 16)
        stride = getattr(configs, "d3u_patch_stride", 8)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        ml_cfg = SimpleNamespace(diffusion=SimpleNamespace(timesteps=configs.timesteps))
        self.diffussion_model = MLP(ml_cfg, configs, patch_num=patch_num)
        self._enc_out = None

    def set_enc_out(self, enc_out):
        """[B, n_vars, patch_num, d_model_c] from patch conditional model."""
        self._enc_out = enc_out

    def forward(self, x, x_mark, y_t, y_0_hat, gx, t):
        del x, x_mark, y_0_hat, gx
        if self._enc_out is None:
            raise RuntimeError("D3UDiffusion.forward: call set_enc_out(enc_out) first (patch encoder output).")
        dec_out = self.diffussion_model(y_t, t, self._enc_out)
        return dec_out, torch.zeros(1, device=y_t.device, dtype=y_t.dtype)
