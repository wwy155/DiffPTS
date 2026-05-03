"""
DiffPTS ablation: **no denoising / no diffusion training**

Trains:
- f(x) conditional predictor (cond_pred_model)
- g(x) variance/scale estimator (cond_pred_model_g)

Freezes:
- diffusion model (model)

Validation sampling:
- directly sample: fx + gx * randn_like
  (i.e. no p_sample_loop denoising)
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
    Same CLI/Fire surface as `src/experiments/DiffPTS.py`, but without diffusion training
    and with direct Gaussian sampling at validation time.
    """
    model_type: str = "DiffPTS_nodenoise"

    def _init_model(self):
        super()._init_model()
        # self.model_type='DiffPTS_nodenoise'
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _init_optimizer(self):
        # Only optimize fx + gx
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{"params": self.cond_pred_model.parameters()}, {"params": self.cond_pred_model_g.parameters()}],
            lr=self.lr,
        )

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Optimize only the Gaussian NLL from fx/gx; no diffusion KL term.
        batch_y_mark_input = torch.concat(
            [batch_x_mark[:, -self.label_len :, :], batch_y_mark], dim=1
        )

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features], device=self.device
        )
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        fx, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = self.cond_pred_model_g(batch_x)

        # Same NLL term as the base experiment.
        log_p_yT_given_X = 0.5 * (
            torch.log(gx) + (fx - batch_y).square() / gx
        )
        nll_loss = log_p_yT_given_X.mean()
        return nll_loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, plot=True):
        # Produce samples directly from fx/gx without denoising.
        batch_y_mark_input = torch.concat(
            [batch_x_mark[:, -self.label_len :, :], batch_y_mark], dim=1
        )

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features], device=self.device
        )
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        fx, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = self.cond_pred_model_g(batch_x)

        f_dim = -1 if self.args.features == "MS" else 0
        fx = fx[:, -self.pred_len :, f_dim:]
        gx = gx[:, -self.pred_len :, f_dim:]
        true = batch_y[:, -self.pred_len :, f_dim:].to(self.device)

        s = int(self.diffusion_config.testing.n_z_samples)
        # Requested form: fx + gx * rand_like (using standard normal noise).
        eps = torch.randn(fx.shape[0], fx.shape[1], fx.shape[2], s, device=self.device)
        outs = fx.unsqueeze(-1) + gx.unsqueeze(-1) * eps  # B, O, N, S
        return outs, true


if __name__ == "__main__":
    import fire

    fire.Fire(DiffPTSForecast)

