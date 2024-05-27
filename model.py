import numpy as np
import torch
from torch import nn
from hparams import d_model, num_heads, d_mlp, act_type, num_layers, d_vocab, n_ctx

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))

    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

global_embedder = nn.Embedding(d_vocab, d_model)
pos_embedder = PosEmbed(n_ctx, d_model)
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                     nhead=num_heads,
                                     dim_feedforward=d_mlp,
                                     activation=act_type,
                                     batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
unembedder = nn.Linear(d_model, d_vocab)

model = nn.Sequential(global_embedder, pos_embedder, encoder, unembedder)
