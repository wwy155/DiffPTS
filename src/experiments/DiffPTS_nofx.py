"""
DiffPTS ablation: **no fx training**

Trains:
- g(x) variance/scale estimator (cond_pred_model_g)
- diffusion model (model)

Freezes:
- f(x) conditional predictor (cond_pred_model)

Additionally:
- fx is fixed to 0 (not learned / not predicted)
"""
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import wandb
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.DiffPTS import DiffPTS
import src.layer.mu_backbone as ns_Transformer
import src.layer.mu_linearbackbone as LinearBackbone
import argparse
import src.layer.g_backbone as G
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type

from torch_timeseries.utils.early_stop import EarlyStopping
from src.layer.diffpts_utils import q_sample, p_sample_loop, cal_sigma12, cal_sigma_tilde, cal_forward_noise
import yaml
import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing
from torch_timeseries.nn.embedding import freq_map


from src.nn.iTransformerBackbone import iTransformerEnc
from src.nn.PatchTSTBackbone import PatchTSTEnc

import torch
from dataclasses import dataclass, field

from src.experiments.DiffPTS import DiffPTSForecast as _BaseDiffPTSForecast
from torch_timeseries.utils.parse_type import parse_type

@dataclass
class DiffPTSForecast(_BaseDiffPTSForecast):
    """
    Same CLI/Fire surface as `src/experiments/DiffPTS.py`, but with fx frozen.
    """
    model_type: str = "DiffPTS_nofx"

    def _init_model(self):
        super()._init_model()
        for p in self.cond_pred_model.parameters():
            p.requires_grad_(False)

    def _init_optimizer(self):
        # Only optimize diffusion + gx
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{"params": self.model.parameters()}, {"params": self.cond_pred_model_g.parameters()}],
            lr=self.lr,
        )

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Same as base, but fx is fixed to 0.
        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        fx = torch.zeros_like(batch_y, device=self.device)
        gx = self.cond_pred_model_g(batch_x)

        log_p_yT_given_X = 0.5 * (torch.log(gx) + (fx - batch_y).square() / gx)
        nll_loss = log_p_yT_given_X.mean()

        e = torch.randn_like(batch_y).to(self.device)
        from src.layer.diffpts_utils import cal_forward_noise, q_sample

        forward_noise = cal_forward_noise(self.model.betas_bar, gx, t)
        noise = e * torch.sqrt(forward_noise)

        y_t_batch = q_sample(
            batch_y,
            fx,
            self.model.alphas_bar_sqrt,
            self.model.one_minus_alphas_bar_sqrt,
            t,
            noise=noise,
        )

        output, _ = self.model(batch_x, batch_x_mark, y_t_batch, fx, gx, t)
        kl_loss = (e - output).square().mean()
        return kl_loss + nll_loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, plot=True):
        # Same as base, but fx is fixed to 0.
        b = batch_x.shape[0]
        minisample = self.diffusion_config.testing.minisample

        y_0_hat_batch = torch.zeros(
            (batch_x.size(0), self.pred_len, self.dataset.num_features),
            device=self.device,
            dtype=batch_x.dtype,
        )
        gx = self.cond_pred_model_g(batch_x)

        preds = []
        from src.layer.diffpts_utils import p_sample_loop

        f_dim = -1 if self.args.features == "MS" else 0

        for _ in range(self.diffusion_config.testing.n_z_samples // minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile

            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gen_y_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                    y_tile_seq = p_sample_loop(
                        self.model,
                        x_tile,
                        x_mark_tile,
                        y_0_hat_tile,
                        gx_tile,
                        y_T_mean_tile,
                        self.model.num_timesteps,
                        self.model.alphas,
                        self.model.one_minus_alphas_bar_sqrt,
                        self.model.alphas_cumprod,
                        self.model.alphas_cumprod_sum,
                        self.model.alphas_cumprod_prev,
                        self.model.alphas_cumprod_sum_prev,
                        self.model.betas_tilde,
                        self.model.betas_bar,
                        self.model.betas_tilde_m_1,
                        self.model.betas_bar_m_1,
                    )
                gen_y = y_tile_seq[self.model.num_timesteps].reshape(
                    b, minisample, self.model.args.pred_len, self.model.args.c_out
                ).detach().cpu()
                gen_y_box.append(gen_y)
            outputs = torch.concat(gen_y_box, dim=1)
            outputs = outputs[:, :, -self.pred_len :, f_dim:]  # B, S, O, N
            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds, dim=1)
        batch_y = batch_y[:, -self.pred_len :, f_dim:].to(self.device)
        outs = preds.permute(0, 2, 3, 1)  # B, O, N, S
        return outs, batch_y


if __name__ == "__main__":
    import fire

    fire.Fire(DiffPTSForecast)

