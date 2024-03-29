import mlx.nn as nn
import mlx.core as mx
from typing import Union


class ConvLSTMCell(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 1,
            dilation: Union[int, tuple] = 1,
            bias: bool = True,
    ):
        super(ConvLSTMCell, self).__init__()

        # creating one conv for all matrix generation
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels * 4, kernel_size, stride, padding, dilation,
                              bias)

    def __call__(self, x_t, prev):
        h_t, c_t = prev

        # concatenating input and hidden by channel dimension
        x = mx.concatenate([x_t, h_t], axis=-1)

        # getting weights for input, forget, cell, and hidden gates respectively through Conv2d layer
        xh_i, xh_f, xh_c, xh_o = mx.split(self.conv(x), 4, axis=-1)

        # forget gate
        f = mx.sigmoid(xh_f)

        # update cell state
        c = c_t * f

        # input gate
        i = mx.sigmoid(xh_i)

        # new candidate
        c_candidate = mx.tanh(xh_c)

        # update cell state
        c = c + i * c_candidate

        # output gate
        o = mx.sigmoid(xh_o)

        # update hidden state
        h = o * mx.tanh(c)

        # returns a tuple of the hidden state and cell state
        return h, c


r"""A Convolutional LSTM recurrent layer.

    The input has shape ``NLHWC`` or ``LHWC`` where:
    * ``N`` is the optional batch dimension
    * ``L`` is the sequence length
    * ``H`` is the input's spatial height dimension
    * ``W`` is the input's spatial weight dimension
    * ``C`` is the input's channel dimension
    Concretely, for each element of the sequence, this layer computes:
    .. math::
        \begin{align*}
        i_t &= \sigma (W_{xi} \ast X_t + W_{hi} \ast H_{t-1} + W_{ci} \odot C_{t-1} + b_{i}) \\
        f_t &= \sigma (W_{xf} \odot X_t + W_{hf} \ast H_{t-1} + W_{cf} \odot C_{t-1} + b_{f}) \\
        C_t &= f_t \odot C_{t-1} + i_t \odot tanh(W_{xc} \ast X_{t} + W_{hc} * H_{t-1} + b_{c} \\
        o_t &= \sigma (W_{xo} * X_{t} + W_{ho} \ast H_{t-1} + W_{co} \odot C_{t} + b_{o} \\
        H_t &= o_{t} \dot tanh(C_{t})
        \end{align*}
    The hidden state :math:`H` and cell state :math:`C` have shape ``NHWO``
    or ``HWO``, depending on whether the input is batched or not.
    The cell returns two arrays, the hidden state and the cell state at
    each time step, both of shape ``NLHWO`` or ``LHWO``.
    Args:
        in_channels (int): The number of input channels, ``C``.
        out_channels (int): The number of output channels,  ``O``.
        kernel_size (int): The size of the convolution filters. Default: ``5``.
        bias (bool): Whether to use biases or not. Default: ``True``.
    """


class ConvLSTM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 5,
            bias: bool = True,
    ):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        assert self.kernel_size % 2 == 1, (f"Kernel size should be odd for same-dimensional spatial padding. Got: "
                                           f"{self.kernel_size}")

        padding = kernel_size // 2
        self.cell = ConvLSTMCell(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=1, bias=bias)

    def __call__(self, x):
        b, t, h, w, c = x.shape
        assert c == self.in_channels, f"Channel dimension should be {self.in_channels}. Got: {c}"

        # initializing tensor for initial hidden and cell states
        h_t = mx.zeros([b, h, w, self.out_channels])
        c_t = mx.zeros([b, h, w, self.out_channels])

        # initializing time-first array for unrolling over time-steps
        hidden_states = mx.zeros([t, b, h, w, self.out_channels])

        # unroll the cell over time
        for i in range(t):
            h_t, c_t = self.cell(x[:, i, :, :, :], (h_t, c_t))
            hidden_states[i] = h_t

        # reshaping back to batch-first
        hidden_states = hidden_states.reshape([b, t, h, w, self.out_channels])

        return hidden_states