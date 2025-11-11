import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
Modified from f90/Wave-U-Net: https://github.com/f90/Wave-U-Net-Pytorch
Modified: input and output lengths to be equal (Using padding at both ends and center cropping)
"""


class WaveUNet(nn.Module):
    def __init__(self, input_dim, output_dim, features, kernel_size, strides, conv_type, res, depth=1):
        """
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param features: number of hidden layers features
        :param kernel_size: kernel size
        :param strides: strides
        :param conv_type: convolution type
        :param res: residual type
        :param depth: depth of the network
        """
        super(WaveUNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.levels = len(features)
        self.features = features
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_type = conv_type
        self.res = res
        self.depth = depth

        # Only odd filter kernels allowed
        assert (self.kernel_size % 2 == 1)

        # Create a model with downsample and upsample
        self.downsampling_blocks = nn.ModuleList()  # Create a list of downsampling modules in the model structure
        for i in range(self.levels - 1):
            in_ch = self.input_dim if i == 0 else self.features[i]
            self.downsampling_blocks.append(
                DownsamplingBlock(in_ch, self.features[i], self.features[i + 1], self.kernel_size, self.strides, self.depth, self.conv_type, self.res))

        self.upsampling_blocks = nn.ModuleList()  # Create a list of upsampling modules in the model structure
        for i in range(0, self.levels - 1):
            self.upsampling_blocks.append(
                UpsamplingBlock(
                    self.features[-1 - i],
                    self.features[-2 - i],
                    self.features[-2 - i],
                    self.kernel_size, self.strides, self.depth, self.conv_type, self.res))

        # Create residual connection
        self.bottlenecks = nn.ModuleList([
            ConvLayer(self.features[-1], self.features[-1], self.kernel_size, 1, self.conv_type) for _ in range(self.depth)])

        # Output conv
        self.output_conv = nn.Conv1d(self.features[0], self.output_dim, 1)

    def forward(self, x):
        input = x
        curr_input_size = x.shape[-1]

        # total downsample times
        valid_length = compute_valid_length(curr_input_size, self.strides, self.levels-1)

        if valid_length != curr_input_size:
            total_pad = valid_length - curr_input_size
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
            x = F.pad(x, (left_pad, right_pad), mode='reflect')
        else:
            x = x

        # calculate output
        shortcuts = []
        out = x
        # Encoder
        for block in self.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)
        # Bottleneck
        for conv in self.bottlenecks:
            out = conv(out)
        # Decoder
        for idx, block in enumerate(self.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])
        # Final output conv
        out = self.output_conv(out)

        # Ensure output length matches input
        out = centre_crop(out, input)

        return out


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert stride > 1

        self.pre_shortcut_convs = nn.ModuleList([
            ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)
        ] + [
            ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)
        ])

        self.post_shortcut_convs = nn.ModuleList([
            ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)
        ] + [
            ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)
        ])

        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride)  # fixed lowpass
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # Prepare shortcut (same length as input)
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # Process for downsampling
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        out = self.downconv(out)
        return out, shortcut


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert stride > 1

        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([
            ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)
        ] + [
            ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)
        ])

        self.post_shortcut_convs = nn.ModuleList([
            ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)
        ] + [
            ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)
        ])

    def forward(self, x, shortcut):
        upsampled = self.upconv(x)
        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Enforce natural alignment — no runtime fix!
        if upsampled.shape[-1] != shortcut.shape[-1]:
            raise RuntimeError(
                f"Length mismatch in UpsamplingBlock: "
                f"upsampled={upsampled.shape[-1]}, shortcut={shortcut.shape[-1]}"
            )

        combined = torch.cat([upsampled, shortcut], dim=1)
        for conv in self.post_shortcut_convs:
            combined = conv(combined)
        return combined


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        NORM_CHANNELS = 8  # for GroupNorm

        if transpose:
            padding = kernel_size - 1
            self.filter = nn.ConvTranspose1d(
                n_inputs, n_outputs, kernel_size, stride, padding=padding
            )
        else:
            padding = kernel_size // 2
            self.filter = nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride, padding=padding
            )

        if conv_type == "gn":
            assert n_outputs % NORM_CHANNELS == 0, f"n_outputs={n_outputs} not divisible by {NORM_CHANNELS}"
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        else:
            raise NotImplementedError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x):
        out = self.filter(x)

        # Only adjust length for transposed convolutions
        if self.transpose:
            target_length = (x.shape[-1] - 1) * self.stride + 1
            current_length = out.shape[-1]
            diff = current_length - target_length
            if diff > 0:
                crop_left = diff // 2
                crop_right = diff - crop_left
                out = out[:, :, crop_left:-crop_right]
            elif diff < 0:
                pad_left = (-diff) // 2
                pad_right = (-diff) - pad_left
                out = F.pad(out, (pad_left, pad_right), mode='reflect')

        # Activation
        if self.norm is not None:
            out = F.relu(self.norm(out))
        else:
            out = F.leaky_relu(out, negative_slope=0.01)
        return out


class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=False):
        """
        Creates a resampling layer for time series data (using 1D convolution) - (N, C, W) input format
        :param channels: Number of features C at each time-step
        :param kernel_size: Width of sinc-based lowpass-filter (>= 15 recommended for good filtering performance)
        :param stride: Resampling factor (integer)
        :param transpose: False for down-, true for upsampling
        :param padding: Either "reflect" to pad or "valid" to not pad
        :param trainable: Optionally activate this to train the lowpass-filter, starting from the sinc initialisation
        """
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert (kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)
        assert (padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        input_size = x.shape[2]  # (B, C, T)

        if not self.transpose:
            # Down-sampling path
            assert input_size % self.stride == 1, \
                f"[Resample1d] Input length {input_size} must satisfy L ≡ 1 (mod {self.stride}) " \
                f"for downsampling with stride={self.stride}. Please use compute_valid_length()."

            if self.padding == "valid":
                out = x
            else:
                num_pad = (self.kernel_size - 1) // 2
                out = F.pad(x, (num_pad, num_pad), mode=self.padding)

            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            return out

        else:
            # Up-sampling path (unchanged from your original logic)
            if self.padding != "valid":
                num_pad = (self.kernel_size - 1) // 2
                out = F.pad(x, (num_pad, num_pad), mode=self.padding)
            else:
                out = x

            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps -= self.kernel_size - 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert diff_steps % 2 == 0, f"Upsampling output asymmetry: diff={diff_steps}"
                out = out[:, :, diff_steps // 2: -diff_steps // 2]

            return out


def build_sinc_filter(kernel_size, cutoff):
    assert kernel_size % 2 == 1
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M // 2:
            filter[i] = 2 * np.pi * cutoff
        else:
            sinc = np.sin(2 * np.pi * cutoff * (i - M // 2)) / (i - M // 2)
            blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / M) + 0.08 * np.cos(4 * np.pi * i / M)
            filter[i] = sinc * blackman
    filter /= np.sum(filter)
    return filter


def centre_crop(x, target):
    """
    Center-crop 3D tensor along last axis to match target length.
    :param x: Input tensor of shape (B, C, L)
    :param target: Target tensor (only shape[-1] used)
    :return: Cropped tensor
    """
    if x is None or target is None:
        return x
    x_len, t_len = x.shape[-1], target.shape[-1]
    if x_len == t_len:
        return x
    assert x_len >= t_len, f"x_len={x_len} < t_len={t_len}"
    diff = x_len - t_len
    crop_left = diff // 2
    crop_right = diff - crop_left
    return x[:, :, crop_left:-crop_right].contiguous()


def compute_valid_length(length, stride, num_levels):
    if num_levels <= 0:
        return length
    modulus = stride ** num_levels
    # We want L ≡ 1 (mod modulus)  => (L - 1) % modulus == 0
    offset = (length - 1) % modulus
    if offset == 0:
        return length
    else:
        return length + (modulus - offset)


if __name__ == '__main__':
    def test_waveunet():
        model = WaveUNet(
            input_dim=8,
            output_dim=8,
            features=[16, 32, 64, 192],
            kernel_size=15,
            strides=4,
            conv_type="gn",
            res="fixed",  # highly recommended
            depth=1
        )
        model.eval()
        print("Testing various input lengths...")
        with torch.no_grad():
            for L in [1000, 1023, 1024, 1025, 4096, 8000, 16385]:
                x = torch.randn(1, 8, L)
                y = model(x)
                assert y.shape[-1] == L, f"Length mismatch: {y.shape[-1]} vs {L}"
                print(f"✅ Input length {L} → Output length {y.shape[-1]}")

    test_waveunet()
