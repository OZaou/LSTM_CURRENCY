
import torch
import torch.nn as nn

class ForexModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False):
        super(ForexModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, 1)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0), self.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
