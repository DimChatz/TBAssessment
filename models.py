import torch.nn as nn
import torch


#################
### MLSTMFCNN ###
#################

class LSTMConvBlock(nn.Module):         
    def __init__(self, ni, no, ks):
        super(LSTMConvBlock, self).__init__() 
        self.conv = nn.Conv1d(ni, no, ks, padding='same')
        self.bn = nn.BatchNorm1d(no, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeExciteBlock(nn.Module):
    def __init__(self, ni, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(nn.Linear(ni, ni // reduction, bias=False), nn.ReLU(),  nn.Linear(ni // reduction, ni, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__() 
        self.dim = dim
    def forward(self, x): 
        return torch.cat(x, dim=self.dim)


class Reshape(nn.Module):
    def __init__(self, *shape): 
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__() 
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))


class MLSTMFCN(nn.Module):
    def __init__(self, numClasses):
        super(MLSTMFCN, self).__init__()
        # LSTM
        self.LSTM = nn.LSTM(input_size=22, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.LSTMdropout = nn.Dropout(0.8)

        # FCN
        self.convblock1 = LSTMConvBlock(22, 128, 7)
        self.se1 = SqueezeExciteBlock(128, 16)
        self.convblock2 = LSTMConvBlock(128, 256, 5)
        self.se2 = SqueezeExciteBlock(256, 16)
        self.convblock3 = LSTMConvBlock(256, 128, 3)
        self.gap = GAP1d(1)

        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(100 + 92, numClasses)

    def forward(self, x):
        # RNN
        LSTMInput = torch.permute(x, (0, 2, 1))  # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, (_, _) = self.LSTM(LSTMInput)
        y = self.LSTMdropout(output[:, -1, :])  # Using the output at the last time step

        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x).squeeze(-1)  # Remove the last dimension after GAP

        # Concat
        combined = self.concat((y, x))
        combined = self.fc_dropout(combined)
        combined = self.fc(combined)
        return combined