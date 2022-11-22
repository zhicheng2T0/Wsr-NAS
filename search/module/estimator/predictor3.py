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

    def __init__(self, input_size, hidden_size, epsilon_num):
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                num_layers=1)
        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=1+epsilon_num)

    def forward(self, x, hidden=None):

        out, (hidden, cell) = self.rnn_cell(x, hidden)

        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        return out

# if __name__=='__main__':
#     temp_input=torch.rand(5,28,8)
