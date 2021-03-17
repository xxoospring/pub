import torch
import torch.nn.functional as F
from torch import nn


class LinearNorm(nn.Module):
    """ apply linear layer with xavier weights init. out = x*w^T + bias
    Args
        :param in_dim: x[...,-1] dimension
        :param out_dim: out[..., -1] dimension
        :param bias: add bias or not
        :param w_init_gain: gain factor generate method
    Shape:
        Input: [N, *, in_dim]
        Output: [N, *, out_dim]
    Example:
        inputs = torch.rand([3, 16, 17, 64])
        linear = LinearNorm(64, 1024)
        out = linear(inputs) # torch.Size([3, 16, 17, 1024])
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):

        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, t_x):
        return self.linear_layer(t_x)


class Conv1dNorm(nn.Module):
    """ apply conv1d with xavier weights init
    Args:
        :param in_channels: input channels -> [B, chs, seq_len]
        :param out_channels: output channels
        :param kernel_size: odd only
        :param stride: stride of the convolution
        :param padding: kernel // 2
        :param dilation: dilation conv
        :param bias: add bias or not
        :param w_init_gain: gain factor generate method
    Shape:
        Input: [N, n_ch_in, seq_len_in]
        Output: [N, n_ch_out, seq_len_out]
    Example:
        inputs = torch.rand([3, 128, 13])
        conv1d = Conv1dNorm(128, 512, 5, padding=5//2)
        out = conv1d(inputs) # torch.Size([3, 512, 13])
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv1dNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def mish(t_x):
    return t_x * torch.tanh(F.softplus(t_x))


class PreNet(nn.Module):
    """ Prenet always apply in feature transformation before feed into model.
    Args:
        :param in_dim: input feature last dimension
        :param sizes: layer list: [layer1_dim, layer2_dim,....]
        :param activate_func_cb: callback function of activate function
    Shape:
        Input: [N, *, c_in], c_in = in_dim
        Output: [N, *, c_out], c_out = sizes[-1]
    Example:
        inputs = torch.rand([3, 13, 64])
        prenet = PreNet(64, [128, 128, 512], 0.3, mish)
        out = prenet(inputs) # torch.Size([3, 13, 512])
    """
    def __init__(self, in_dim, sizes, drop_prob=0.3, activate_func_cb=mish):
        super(PreNet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

        self.drop = nn.Dropout(drop_prob)
        self.ac = activate_func_cb

    def forward(self, t_x):
        for linear in self.layers:
            t_x = self.drop(self.ac(linear(t_x)))
        return t_x


class LocalTemporalExtraction(nn.Module):
    """ apply a set of conv1d to extract local temporal relationship feature
    Args:
        :param in_dim: input channels dim
        :param feat_chs: list, channels in each layer
        :param kernel_size: kernel size
        :param drop_prob: dropout prob
        :param activate_func_cb: activate callback function
    Shape:
        Input: [N, ch_in, seq_len]
        Output: [N, ch_out, seq_len]
    Example:
        inputs = torch.rand([3, 64, 13])
        lte = LocalTemporalExtraction(64, [128, 512, 1024], 5, 0.1, mish)
        out = lte(inputs) # torch.Size([3, 1024, 13])
    """
    def __init__(self, in_dim, feat_chs=[512]*3, kernel_size=5, drop_prob=0.1, activate_func_cb=mish):
        super(LocalTemporalExtraction, self).__init__()

        conv1ds = []
        in_chs = [in_dim]+feat_chs
        out_chs = in_chs[1:]
        for in_ch, out_ch in zip(in_chs, out_chs):
            conv = nn.Sequential(
                Conv1dNorm(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch)
            )
            conv1ds.append(conv)
        self.convolutions = nn.ModuleList(conv1ds)
        self.drop = nn.Dropout(drop_prob)
        self.ac = activate_func_cb

    def forward(self, t_x):
        for layer in self.convolutions:
            t_x = self.drop(self.ac(layer(t_x)))
        return t_x
