import math

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 2:
            front = x[0:1, :].repeat((self.kernel_size - 1) // 2, 1)
            end = x[-1:, :].repeat((self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=0)
            x = self.avg(x.permute(1, 0))
            x = x.permute(1, 0)
        else:
            raise ValueError(
                f"Unsupported data shape: {x.shape}, expected [bsz, seq_len, n_vars] or [series_len, n_vars]."
            )
        return x


class raw_series_decomp(nn.Module):
    """Series decomposition block used in data loading stage."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res.numpy(), moving_mean.numpy()


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):
    """Model seasonality using inverse DFT."""

    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t, device=x.device)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)

        # repeat f to [b, f, d] then gather with same indices, then [b, f, 1, d]
        f_rep = f[None, :, None].expand(x_freq.size(0), -1, x_freq.size(2))
        f_sel = f_rep[index_tuple].unsqueeze(2)
        return self.extrapolate(x_freq, f_sel, t)

    def extrapolate(self, x_freq, f, t_0):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = torch.arange(t_0, dtype=torch.float, device=x_freq.device)[None, None, :, None]

        amp = (x_freq.abs() * 2.0 / t_0).unsqueeze(2)
        phase = x_freq.angle().unsqueeze(2)
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return x_time.sum(dim=1)

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        top_k = max(1, min(top_k, length))
        _, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)

        mesh_a = torch.arange(x_freq.size(0), device=x_freq.device)[:, None].expand(-1, x_freq.size(2))
        mesh_b = torch.arange(x_freq.size(2), device=x_freq.device)[None, :].expand(x_freq.size(0), -1)
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple

