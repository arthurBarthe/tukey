import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear
from torch.nn import ReLU, BatchNorm1d


class NN(Sequential):
    def __init__(self, structure: list, n_out: int, batch_norm: bool = False):
        subblocks = []
        self.n_out = n_out
        self.batch_norm = batch_norm
        for i in range(len(structure) - 1):
            subblocks.extend(self._make_sublock(structure[i], structure[i + 1]))
        subblocks.extend(self._make_final_layer(structure[-1]))
        super().__init__(*subblocks)

    def _make_sublock(self, n_in: int, n_out: int):
        if not self.batch_norm:
            return [Linear(n_in, n_out), ReLU()]
        return [Linear(n_in, n_out), ReLU(), BatchNorm1d(n_out)]

    def _make_final_layer(self, n_in : int):
        return [Linear(n_in, self.n_out), ]


class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden = self.initHidden()

        self.i2h = NN((input_size + hidden_size, 256, 256), hidden_size)
        self.i2o = NN((input_size + hidden_size, 256, 256), output_size)

    def forward(self, input):
        combined = torch.cat((input, self.hidden[-1]), 1)
        self.hidden.append(self.i2h(combined))
        output = self.i2o(combined)
        return output

    def run_over_sequence(self, sequence):
        batch_size, seq_len = sequence.shape[:2]
        output = torch.zeros((batch_size, seq_len))
        for i in range(seq_len):
            output[:, i] = self.forward(sequence[:, i: i + 1])
        return output

    def initHidden(self):
        return [torch.zeros(1, self.hidden_size), ]