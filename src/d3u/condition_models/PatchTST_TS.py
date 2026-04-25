import torch
from torch import nn

from src.d3u.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.d3u.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.d3u.layers.Embed import PatchEmbedding
from src.d3u.layers.Decompose import series_decomp, FourierLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor_c,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model_c,
                        configs.n_heads_c,
                    ),
                    configs.d_model_c,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers_c)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model_c), Transpose(1, 2)
            ),
        )

    def forward(self, x_enc, attn_mask=None, tau=None, delta=None):
        enc_out, attns = self.encoder(x_enc, attn_mask, tau, delta)
        return enc_out, attns


class Model(nn.Module):
    """
    Patch-based conditional predictor (D3U style).
    Returns:
      - dec_out: [B, pred_len, n_vars]
      - dummy: tensor([0.0]) (for compatibility with D3U training code)
      - enc_out: [B, n_vars, patch_num, d_model_c]
    """

    def __init__(self, configs, patch_len=None, stride=None):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        if patch_len is None:
            patch_len = int(getattr(configs, "d3u_patch_len", 16))
        if stride is None:
            stride = int(getattr(configs, "d3u_patch_stride", 8))
        self.patch_len = patch_len
        self.stride = stride
        padding = stride
        self.decomposition = getattr(configs, "decomposition", False)

        self.patch_embedding = PatchEmbedding(
            configs.d_model_c,
            patch_len,
            stride,
            padding,
            configs.padding_patch,
            configs.dropout,
        )

        self.decomp = series_decomp(kernel_size=configs.kernel_size) if self.decomposition else None
        self.seasonal = (
            FourierLayer(d_model=configs.d_model_c, factor=configs.fourier_factor)
            if self.decomposition
            else None
        )

        self.encoder = _Encoder(configs)

        self.head_nf = configs.d_model_c * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)  # [B, n_vars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*n_vars, patch_num, d_model_c]

        enc_out, _ = self.encoder(enc_out)  # [B*n_vars, patch_num, d_model_c]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model_c, patch_num]
        condition_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, patch_num, d_model_c]

        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, condition_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.decomposition:
            res, trend = self.decomp(x_enc)
            seasonal = self.seasonal(res)
            x_enc = trend + seasonal
        dec_out, enc_out = self.forecast(x_enc)
        return dec_out, torch.tensor([0.0], device=dec_out.device), enc_out

