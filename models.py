import torch.nn as nn
import torch

#################
### MLSTMFCNN ###
#################

class LSTMConvBlock(nn.Module):
    """Convolutional block that applies a 1D convolution followed by batch normalization and ReLU activation.

    Args:
        ni (int): Number of input channels.
        no (int): Number of output channels.
        ks (int): Kernel size for the convolution.
    """
    def __init__(self, ni, no, ks):
        super(LSTMConvBlock, self).__init__()
        self.conv = nn.Conv1d(ni, no, ks, padding='same')
        self.bn = nn.BatchNorm1d(no, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after applying convolution, batch normalization, and ReLU.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GAP1d(nn.Module):
    """Global Adaptive Pooling 1D module.

    Applies adaptive average pooling over 1D data and then flattens the output.

    Args:
        output_size (int, optional): Target output size. Defaults to 1.
    """
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    
    def forward(self, x):
        """Forward pass through the GAP1d module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Flattened tensor after adaptive average pooling.
        """
        return self.flatten(self.gap(x))


class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration.

    Args:
        ni (int): Number of input channels.
        reduction (int, optional): Reduction factor for the intermediate layer. Defaults to 16.
    """
    def __init__(self, ni, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ni, ni // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(ni // reduction, ni, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through the squeeze-and-excitation block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after channel recalibration.
        """
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)


class Concat(nn.Module):
    """Module to concatenate a list of tensors along a specified dimension.

    Args:
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.
    """
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        """Forward pass to concatenate tensors.

        Args:
            x (list of torch.Tensor): List of tensors to concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.cat(x, dim=self.dim)


class Reshape(nn.Module):
    """Reshape module that reshapes an input tensor to a specified shape.

    Args:
        *shape: The target shape. If empty, flattens the input tensor except for the batch dimension.
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """Forward pass to reshape the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        if not self.shape:
            return x.reshape(x.shape[0], -1)
        elif self.shape == (-1,):
            return x.reshape(-1)
        else:
            return x.reshape(x.shape[0], *self.shape)


class MLSTMFCNRegression(nn.Module):
    """Multi-LSTM Fully Convolutional Network for regression tasks.

    This model combines a bidirectional LSTM with a fully convolutional network (FCN) to extract features
    from sequential data and performs regression on the combined features.

    Args:
        input_channels (int, optional): Number of input channels for the FCN. Defaults to 66.
        lstm_input_size (int, optional): Input size for the LSTM. Defaults to 66.
        lstm_hidden_size (int, optional): Hidden size for the LSTM. Defaults to 64.
        output_size (int, optional): Number of regression targets. Defaults to 2.
    """
    def __init__(self, input_channels=66, lstm_input_size=66, lstm_hidden_size=64, output_size=2):
        super(MLSTMFCNRegression, self).__init__()
        # LSTM branch
        self.LSTM = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.LSTMdropout = nn.Dropout(0.8)

        # Fully Convolutional Network (FCN) branch
        self.convblock1 = LSTMConvBlock(input_channels, 128, 7)
        self.se1 = SqueezeExciteBlock(128, 16)
        self.convblock2 = LSTMConvBlock(128, 256, 5)
        self.se2 = SqueezeExciteBlock(256, 16)
        self.convblock3 = LSTMConvBlock(256, 128, 3)
        self.gap = GAP1d(1)

        # Common layers for feature concatenation and output
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(128 + 2 * lstm_hidden_size, output_size)  # 128 from FCN, 128 from bidirectional LSTM

    def forward(self, x):
        """Forward pass through the MLSTMFCNRegression model.

        Processes input through both the LSTM branch and the FCN branch, concatenates their outputs,
        applies dropout, and then passes the result through a fully connected layer to produce the final output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_size).
        """
        # LSTM branch
        lstm_input = x.permute(0, 2, 1)  # Change shape to (batch, sequence_length, channels)
        output, _ = self.LSTM(lstm_input)
        y = self.LSTMdropout(output[:, -1, :])  # Use the last time step

        # FCN branch
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x).squeeze(-1)

        # Concatenate features from LSTM and FCN branches
        combined = self.concat((y, x))
        combined = self.fc_dropout(combined)
        out = self.fc(combined)

        return out
