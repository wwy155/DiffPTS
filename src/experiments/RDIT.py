from dataclasses import dataclass
import os
import time
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from torch.optim import *

from torch_timeseries.nn.embedding import freq_map
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible

from src.experiments.prob_forecast import ProbForecastExp
from src.layer.diffpts_utils import q_sample, p_sample_loop, cal_forward_noise
from src.models.DiffPTS import DiffPTS
from src.rdit.model.TimeFilter import Model as TimeFilter


def dict2namespace(config):
    import argparse

    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


EPS = 1e-8


class RDITEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model["model"].state_dict(), os.path.join(self.path, "model.pth"))
        torch.save(model["cond_pred_model"].state_dict(), os.path.join(self.path, "cond_pred_model.pth"))
        self.val_loss_min = val_loss


@dataclass
class RDITParameters:
    num_samples: int = 101
    beta_start: float = 0.0001
    beta_end: float = 0.01
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 1024
    diffusion_steps: int = 20
    moving_avg: int = 25
    factor: int = 3
    distil: bool = True
    dropout: float = 0.05
    activation: str = "gelu"
    CART_input_x_embed_dim: int = 32
    p_hidden_layers: int = 2

    # RDIT / TimeFilter patch config
    # IMPORTANT: TimeFilter requires seq_len divisible by patch_len.
    # For default windows=168, patch_len=12 makes seq_len/patch_len an integer.
    patch_len: int = 12
    alpha: float = 0.1
    top_p: float = 0.5
    pos: bool = True

    # Residual diffusion config
    sigma_mode: str = "res"  # x_std | ones | res (sqrt EMA of (y-f(x))^2 per feature)


@dataclass
class RDITForecast(ProbForecastExp, RDITParameters):
    model_type: str = "RDIT"

    def _get_mask(self):
        dtype = torch.float32
        L = self.windows * self.dataset.num_features // self.patch_len
        N = self.windows // self.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = (
                (torch.arange(L) >= k // N * N)
                & (torch.arange(L) < k // N * N + N)
                & (torch.arange(L) != k)
            ).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        return torch.stack(masks, dim=0)

    def _init_model(self):
        self.label_len = self.windows // 2
        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": "M",
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "enc_in": self.dataset.num_features,
            "dec_in": self.dataset.num_features,
            "c_out": self.dataset.num_features,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "moving_avg": self.moving_avg,
            "timesteps": self.diffusion_steps,
            "factor": self.factor,
            "distil": self.distil,
            "beta_schedule": "linear",
            "embed": "timeF",
            "dropout": self.dropout,
            "activation": self.activation,
            "output_attention": False,
            "do_predict": True,
            "freq": self.dataset.freq,
            "CART_input_x_embed_dim": self.CART_input_x_embed_dim,
            "p_hidden_layers": self.p_hidden_layers,
            "diffusion_config_dir": "./configs/nsdiff.yml",
            "t_in": freq_map[self.dataset.freq],
            # TimeFilter args
            "patch_len": self.patch_len,
            "alpha": self.alpha,
            "top_p": self.top_p,
            "pos": self.pos,
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        self.model = DiffPTS(self.args, self.device).to(self.device)
        self.cond_pred_model = TimeFilter(self.args).float().to(self.device)
        self.masks = self._get_mask()
        # EMA of per-feature mean squared training residual (y - f(x))^2; used when sigma_mode == "res"
        self._res_var_ema: Optional[torch.Tensor] = None

    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{"params": self.model.parameters()}, {"params": self.cond_pred_model.parameters()}],
            lr=self.lr,
        )

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "model.pth")
        self.best_cond_checkpoint_filepath = os.path.join(self.run_save_dir, "cond_pred_model.pth")
        self.early_stopper = RDITEarlyStopping(self.patience, verbose=True, path=self.run_save_dir)

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
            "res_var_ema": self._res_var_ema,
        }
        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")

    def _resume_run(self, seed):
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        self.early_stopper.set_state(check_point["early_stopping"])
        if "res_var_ema" in check_point:
            rve = check_point["res_var_ema"]
            self._res_var_ema = None if rve is None else rve.to(self.device)

    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))
        self.cond_pred_model.load_state_dict(torch.load(self.best_cond_checkpoint_filepath, map_location=self.device))

    def _sigma(self, batch_x):
        b, n = batch_x.size(0), self.dataset.num_features
        if self.sigma_mode == "ones":
            return torch.ones(b, 1, n, device=self.device)
        if self.sigma_mode == "res":
            ve = getattr(self, "_res_var_ema", None)
            if ve is None:
                return torch.ones(b, 1, n, device=self.device)
            s = torch.sqrt(ve.to(self.device) + EPS)
            return s.expand(b, -1, -1)
        # x_std: per-sample std from history x
        return batch_x.std(dim=1, keepdim=True, unbiased=False) + EPS

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for batch_x, batch_y, _, origin_y, batch_x_mark, batch_y_mark in self.train_loader:
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_mark = batch_x_mark.to(self.device).float()
                batch_y_mark = batch_y_mark.to(self.device).float()

                loss = self._process_train_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss.backward()
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len :, :], batch_y_mark], dim=1)
        dec_inp_pred = torch.zeros([batch_x.size(0), self.pred_len, self.dataset.num_features]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        fx, _ = self.cond_pred_model(batch_x, self.masks, is_training=True)
        fx = fx[:, -self.pred_len :, :]

        # additional point loss to train f(x) toward y
        fx_loss = (fx - batch_y).square().mean()

        if self.sigma_mode == "res":
            with torch.no_grad():
                r = batch_y - fx.detach()
                vb = r.pow(2).mean(dim=(0, 1), keepdim=True)
                if self._res_var_ema is None:
                    self._res_var_ema = vb.clone()
                else:
                    m = 0.05 #self.res_sigma_ema_momentum #  
                    self._res_var_ema.mul_(1 - m).add_(vb, alpha=m)

        sigma = self._sigma(batch_x)  # [B,1,N]
        res0 = (batch_y - fx) / sigma

        gx = torch.ones_like(batch_y).to(self.device)
        e = torch.randn_like(res0).to(self.device)
        forward_noise = cal_forward_noise(self.model.betas_bar, gx, t)
        noise = e * torch.sqrt(forward_noise)

        res_t = q_sample(res0, torch.zeros_like(res0), self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt, t, noise=noise)
        output, _ = self.model(batch_x, batch_x_mark, res_t, fx, gx, t)
        diff_loss = ((e - output)).square().mean()
        loss = diff_loss +  fx_loss
        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, plot=True):
        # Same sampling structure as DiffPTS, but on residuals and with sigma scaling.
        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        minisample = self.diffusion_config.testing.minisample

        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            current_t = self.diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(
                b,
                minisample,
                (config.pred_len),
                config.c_out,
            ).cpu()
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat(
                    [gen_y_by_batch_list[current_t], gen_y], dim=0
                ).detach().cpu()
            return gen_y

        # point estimate f(x)
        fx, _ = self.cond_pred_model(batch_x, self.masks, is_training=False)
        fx = fx[:, -self.pred_len :, :]
        sigma = self._sigma(batch_x)  # [B,1,N]

        gx = torch.ones_like(batch_y).to(self.device)

        preds = []
        for _ in range(self.diffusion_config.testing.n_z_samples // minisample):
            repeat_n = int(minisample)

            fx_tile = fx.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)
            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)
            gx_tile = gx.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)
            sigma_tile = sigma.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)

            # residual diffusion prior mean is 0
            y_T_mean_tile = torch.zeros_like(fx_tile)

            gen_y_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                    y_tile_seq = p_sample_loop(
                        self.model,
                        x_tile,
                        x_mark_tile,
                        torch.zeros_like(fx_tile).to(self.device),
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
                gen_y = store_gen_y_at_step_t(
                    config=self.model.args,
                    config_diff=self.diffusion_config,
                    idx=self.model.num_timesteps,
                    y_tile_seq=y_tile_seq,
                )
                gen_y_box.append(gen_y.detach().cpu())

            outputs = torch.concat(gen_y_box, dim=1)
            f_dim = -1 if self.args.features == "MS" else 0
            # residual samples
            outputs = outputs[:, :, -self.pred_len :, f_dim:]
            # residual -> y
            outputs = outputs * sigma_tile[:, None, :, f_dim:].detach().cpu() + fx_tile[:, None, :, f_dim:].detach().cpu()

            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds, dim=1)
        batch_y = batch_y[:, -self.pred_len :, :].to(self.device)
        outs = preds.permute(0, 2, 3, 1)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
            self.pred_len,
            self.dataset.num_features,
            self.diffusion_config.testing.n_z_samples,
        )
        return outs, batch_y

    def run(self, seed=42) -> Dict[str, float]:
        if self._use_wandb() and not self._init_wandb(self.project, seed):
            return {}

        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                break

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(self.current_epoch + 1, time.time() - epoch_start_time)
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            _ = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result["crps"], model={"model": self.model, "cond_pred_model": self.cond_pred_model})
            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({"training_loss": np.mean(train_losses)}, step=self.current_epoch)
                wandb.log({f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items():
                wandb.run.summary[f"best_test_{k}"] = v
            wandb.finish()
        return best_test_result


if __name__ == "__main__":
    import fire

    fire.Fire(RDITForecast)

