"""
@File     : afrcnn.py
@Project  : 41-PGSNet
@Time     : 2025/11/12 14:18
@Author   : FQQ # (Feng Qiang)
@License  : (C)Copyright 2023-2027, UESTC
@Note     :
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
"""
Modified from JusperLee/AFRCNN-For-Speech-Separation: https://github.com/JusperLee/AFRCNN-For-Speech-Separation
"""


class AFRCNN(nn.Module):
    def __init__(self, input_dim, out_dim, features=128, enc_kernel_size=21, num_blocks=16, upsampling_depth=4, enc_dim=512):
        """
        :param input_dim: input dimension
        :param out_dim: output dimension
        :param features: number of bottleneck channels
        :param enc_kernel_size: kernel size
        :param num_blocks: number of blocks
        :param upsampling_depth: upsampling depth
        :param enc_dim: encoder dimension
        """
        super(AFRCNN, self).__init__()

        # Number of sources to produce
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.features = features # bottleneck channel
        self.enc_dim = enc_dim  # encoder channel
        self.enc_kernel_size = enc_kernel_size
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
            self.enc_kernel_size // 2,
            2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=self.input_dim, out_channels=self.enc_dim,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_dim)
        self.bottleneck = nn.Conv1d(in_channels=self.enc_dim, out_channels=self.features, kernel_size=1)

        # Separation module
        self.sm = Recurrent(self.features, self.enc_dim, self.upsampling_depth, self.num_blocks)  #  channel of separation module == channel of encoder

        mask_conv = nn.Conv1d(self.features, self.out_dim * self.enc_dim, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_dim * self.out_dim,
            out_channels=self.out_dim,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    # Forward pass
    def forward(self, input_wav):

        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)
        x = self.mask_net(x)
        x = x.view(x.shape[0], self.out_dim, self.enc_dim, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)

        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = self.remove_trailing_zeros(estimated_waveforms, input_wav)

        return estimated_waveforms

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32).to(x.device)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Blocks(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=5,
                                               stride=2,
                                               groups=in_channels, d=1))
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j-i == 1:
                    fuse_layer.append(None)
                elif i-j == 1:
                    fuse_layer.append(DilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=2,
                                                      groups=in_channels, d=1))
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth-1:
                self.concat_layer.append(ConvNormAct(
                    in_channels*2, in_channels, 1, 1))
            else:
                self.concat_layer.append(ConvNormAct(
                    in_channels*3, in_channels, 1, 1))

        self.last_layer = nn.Sequential(
            ConvNormAct(in_channels*upsampling_depth, in_channels, 1, 1)
        )
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat((self.fuse_layers[i][0](output[i-1]) if i-1 >= 0 else torch.Tensor().to(output1.device),
                           output[i],
                           F.interpolate(output[i+1], size=wav_length, mode='nearest') if i+1 < self.depth else torch.Tensor().to(output1.device)), dim=1)
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(
                x_fuse[i], size=wav_length, mode='nearest')

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual
        #return expanded

class Recurrent(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 _iter=4):
        super().__init__()
        self.blocks = Blocks(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        #self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.blocks(x)
            else:
                #m = self.attention(mixture, x)
                x = self.blocks(self.concat_block(mixture+x))
        return x


if __name__ == '__main__':
    model = AFRCNN(8, 8, features=128, enc_dim=512,
                   enc_kernel_size=21,
                   num_blocks=8,
                   upsampling_depth=5,
                   )

    x = torch.rand(2, 8, 5000)
    estimated_sources = model(x)
    print(estimated_sources.shape)
