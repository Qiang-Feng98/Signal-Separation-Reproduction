"""
@File     : mccastasnet.py
@Project  : 41-PGSNet
@Time     : 2025/10/17 20:40
@Author   : FQQ # (Feng Qiang)
@License  : (C)Copyright 2023-2027, UESTC
@Note     :
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
"""
Modified from YaokaiZhang/MC-ConvTasNet: https://github.com/YaokaiZhang/MC-ConvTasNet
Noticed: input and output lengths to be equal
"""

# MC-Conv-TasNet
class MCTasNet(nn.Module):
    def __init__(self, input_dim, out_dim, enc_dim, features, sr, win, layer, stack, kernel, causal=False):
        """
        :param input_dim: dimension of the input feature
        :param out_dim: dimension of the output feature
        :param enc_dim: dimension of the encoder feature
        :param features: dimension of the feature vector
        :param sr: sampling rate
        :param win: window size
        :param layer: number of each tac block layers
        :param stack: number of stack layers
        :param kernel: kernel size
        :param causal: causal or not
        """
        super(MCTasNet, self).__init__()

        # hyper parameters
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.features = features

        self.window = int(sr * win / 1000)
        self.stride = self.window // 2
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.causal = causal

        # input encoder
        self.encoder = nn.Conv1d(self.input_dim, self.enc_dim, self.window, bias=False, stride=self.stride)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 30, (2, self.window), bias=False, stride=self.stride),
            nn.ReLU(),
        )

        # TCN separator
        tcn_input_dim = enc_dim + (30 * self.input_dim)
        self.TCN = TCN(tcn_input_dim, enc_dim * self.out_dim, self.features, self.features * 4,
                       layer, stack, kernel, causal=causal)
        # self.TCN = TCN(self.enc_dim+180, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
        #                       self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.window, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, C, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        # Ensure input has self.in_ch channels by repeating if needed
        # Expand single-channel input to multi-channel input
        # torch.Size([2, 1, 32000]) ==> torch.Size([2, 6, 32000])
        if input.size(1) == 1 and self.input_dim > 1:
            input = input.repeat(1, self.input_dim, 1)

        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.window - (self.stride + nsample % self.window) % self.window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, self.input_dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, self.input_dim, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):

        # print(input.shape)
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = self.encoder(output)  # B, E, L

        # 2d spatial conv
        s_in = output.unsqueeze(1)
        pair1 = torch.cat((s_in[:, :, 0, :].unsqueeze(2), s_in[:, :, 1, :].unsqueeze(2)), dim=2)
        pair2 = torch.cat((s_in[:, :, 1, :].unsqueeze(2), s_in[:, :, 2, :].unsqueeze(2)), dim=2)
        pair3 = torch.cat((s_in[:, :, 2, :].unsqueeze(2), s_in[:, :, 3, :].unsqueeze(2)), dim=2)
        pair4 = torch.cat((s_in[:, :, 3, :].unsqueeze(2), s_in[:, :, 4, :].unsqueeze(2)), dim=2)
        pair5 = torch.cat((s_in[:, :, 4, :].unsqueeze(2), s_in[:, :, 5, :].unsqueeze(2)), dim=2)
        pair6 = torch.cat((s_in[:, :, 5, :].unsqueeze(2), s_in[:, :, 6, :].unsqueeze(2)), dim=2)
        pair7 = torch.cat((s_in[:, :, 6, :].unsqueeze(2), s_in[:, :, 7, :].unsqueeze(2)), dim=2)
        pair8 = torch.cat((s_in[:, :, 7, :].unsqueeze(2), s_in[:, :, 0, :].unsqueeze(2)), dim=2)

        spa_out = torch.cat(
            (
                self.conv(pair1),
                self.conv(pair2),
                self.conv(pair3),
                self.conv(pair4),
                self.conv(pair5),
                self.conv(pair6),
                self.conv(pair7),
                self.conv(pair8),
            ),
            dim=1
        )

        B, C, N, L = spa_out.shape
        spa_out = spa_out.view(B, C * N, L)

        _spa_spec = torch.cat((enc_output, spa_out), dim=1)

        spa_spec = self.TCN(_spa_spec)
        spa_spec = spa_spec.view(batch_size, self.out_dim, self.enc_dim, -1)  # B, E+C*N, L

        # generate masks
        masks = torch.sigmoid(spa_spec).view(batch_size, self.out_dim, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.out_dim, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.out_dim, -1)  # B, C, T

        return output


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        """
        :param dimension: input dimension
        :param eps: eps parameter
        :param trainable: whether to train or not
        """
        super().__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            # 用 register_buffer 替代 Variable
            self.register_buffer('gain', torch.ones(1, dimension, 1))
            self.register_buffer('bias', torch.zeros(1, dimension, 1))

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # 累积统计（在 C 维度上求和）
        cum_sum = torch.cumsum(x.sum(dim=1, keepdim=True), dim=2)  # (B, 1, T)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(dim=1, keepdim=True), dim=2)  # (B, 1, T)

        # 累积计数 [C, 2C, ..., TC]
        entry_cnt = torch.arange(1, T + 1, device=x.device, dtype=x.dtype) * C  # (T,)
        entry_cnt = entry_cnt.view(1, 1, -1)  # (1, 1, T)

        # 计算累积均值和方差
        cum_mean = cum_sum / entry_cnt  # (B, 1, T)
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # (B, 1, T)
        cum_std = (cum_var + self.eps).sqrt()  # (B, 1, T)

        # 归一化并应用增益和偏置（自动广播）
        x = (x - cum_mean) / cum_std  # (B, C, T)
        return x * self.gain + self.bias  # 广播 (1, C, 1) 到 (B, C, T)


class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        """
        :param input_channel: input channel
        :param hidden_channel: hidden channel
        :param kernel: kernel size
        :param padding: padding size
        :param dilation: dilation size
        :param skip: skip connection
        :param causal: causal
        """
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True):
        """
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param BN_dim: BN dimension
        :param hidden_dim: hidden dimension
        :param layer: layer number
        :param stack: stack number
        :param kernel: kernel size
        :param skip: skip connection
        :param causal: causal
        :param dilated: dilated
        """
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip, causal=causal))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )

        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)

        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output


if __name__ == '__main__':
    pass
