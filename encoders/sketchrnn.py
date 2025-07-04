import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import einops

class SketchRNNEmbedding(Module):
    """
    ## Encoder module

    This consists of a bidirectional LSTM
    """

    def __init__(self, enc_hidden_size: int = 256, is_global=True):
        """
        :param enc_hidden_size:
        :param is_global: True: 返回全局特征， False: 返回局部特征
        """
        super().__init__()
        self.is_global = is_global
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)

        if self.is_global:
            self.mu_head = nn.Linear(2 * enc_hidden_size, 2 * enc_hidden_size)
        else:
            self.mu_head = nn.Conv1d(2 * enc_hidden_size, 2 * enc_hidden_size, 1)

    def forward(self, inputs: torch.Tensor, state=None):
        """
        :param inputs: [bs, len, emb]
        :param mask: [bs, len]
        :param state:
        :return:
        """
        inputs = inputs.transpose(0, 1)

        # -> output: [n_pnt, bs, channel]
        output, (hidden, cell) = self.lstm(inputs.float(), state)

        if self.is_global:
            hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')  # fb: forward backward
        else:
            hidden = output.permute(1, 2, 0)

        emb = self.mu_head(hidden)
        return emb

if __name__ == "__main__":
    sketch_rnn_encoder = SketchRNNEmbedding()
    sketch_data_test = np.loadtxt(r"E:\Dataset\Sketchy\sketches_s5\airplane\n02691156_58-2.txt", delimiter=',')
    print(sketch_data_test.shape)
    print(sketch_rnn_encoder(torch.randn((32, 312, 5))).size())