import torch
from torch import nn

from module.estimator.memory import Experience  # latency


def weighted_loss(output, target):
    # squared error
    loss = (output - target)**2
    # weighted loss
    loss = torch.ones_like(target) / target * loss
    # calculate mean
    loss = torch.mean(loss)
    return loss


class Predictor(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                num_layers=1)
        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)

        self.logits_epsilon0 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)
        self.logits_epsilon1 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)
        self.logits_epsilon2 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)
        self.logits_epsilon3 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)
        self.logits_epsilon4 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)
        self.logits_epsilon5 = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)

    def forward(self, x, hidden=None):
        out, (hidden, cell) = self.rnn_cell(x, hidden)

        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        out = out.view(-1)

        oute0 = self.logits_epsilon0(hidden)
        oute0 = torch.sigmoid(oute0) * 2
        oute0 = oute0.view(-1)

        oute1 = self.logits_epsilon1(hidden)
        oute1 = torch.sigmoid(oute1) * 2
        oute1 = oute1.view(-1)

        oute2 = self.logits_epsilon2(hidden)
        oute2 = torch.sigmoid(oute2) * 2
        oute2 = oute2.view(-1)

        oute3 = self.logits_epsilon3(hidden)
        oute3 = torch.sigmoid(oute3) * 2
        oute3 = oute3.view(-1)

        oute4 = self.logits_epsilon4(hidden)
        oute4 = torch.sigmoid(oute4) * 2
        oute4 = oute4.view(-1)

        oute5 = self.logits_epsilon5(hidden)
        oute5 = torch.sigmoid(oute5) * 2
        oute5 = oute5.view(-1)
        return out,oute0,oute1,oute2,oute3,oute4,oute5
