# Vendored from D3U: https://github.com/zhangzhenBrave/D3U/blob/main/model9_NS_transformer/denoise_models/MLP.py
# Lin0 input dim uses patch_num * d_model_c (dynamic) instead of a hardcoded patch count.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        t = t.long()
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(t.size(0), -1, self.num_out) * out
        return out


class MLP(nn.Module):
    """D3U denoise MLP: predicts noise from (y_t, t) and patch encoder output enc_out."""

    def __init__(self, config, MTS_args, patch_num: int):
        super().__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = MTS_args.enc_in * 2
        self.lin0 = nn.Linear(patch_num * MTS_args.d_model_c, MTS_args.pred_len)
        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, MTS_args.enc_in)

    def forward(self, y_t, t, enc_out):
        # enc_out: [batch, nvar, patch_num, d_model_c]
        batch, nvar, patch_num, d_model_c = enc_out.shape
        enc_flat = enc_out.reshape(batch, nvar, patch_num * d_model_c)
        enc_proj = self.lin0(enc_flat).permute(0, 2, 1)
        eps_pred = torch.cat((y_t, enc_proj), dim=-1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred = self.lin4(eps_pred)
        return eps_pred
