import torch
import torch.nn as nn
from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding




class Model(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        
        self.proj = nn.Sequential(
            nn.Linear(self.seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, configs.pred_len),
        )
        # self.t_proj = nn.Sequential(
        #     nn.Linear(configs.t_in, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, configs.c_out),
        # )
        
        self.use_revin = configs.revin

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: B, T, N
        # x_mark_dec: B, O, TE


        if self.use_revin:
            # Normalization
            mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x N
            x_enc = x_enc - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x N
            x_enc = x_enc / std_enc

        x_enc = x_enc.permute(0, 2, 1)
        e_x = self.proj(x_enc).permute(0, 2, 1)
        # e_t = self.t_proj(x_mark_dec)
        
        dec_out = e_x #+ e_t[:, -self.pred_len:, :] # B, O, N

        if self.use_revin:
            dec_out = dec_out * std_enc + mean_enc


        return dec_out[:, -self.pred_len:, :], None
