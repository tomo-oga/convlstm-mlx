# An Implementation for ConvLSTM in Apple's Array Framework `mlx`

A Convolutional LSTM recurrent layer. 


### `_conv_lstm_cell` 
This `Module` computes the hidden and cell state for a time-step, expressed as:

$` i_t = \sigma (W_{xi} \ast X_t + W_{hi} \ast H_{t-1} + W_{ci} \odot C_{t-1} + b_i) `$ \
$` f_t = \sigma (W_{xf} \odot X_t \ast H_{t-1} + W_{cf} \odot C_{t-1} + b_f) `$\
$` C_t = f_t \odot X_t + i_t \odot tanh(W_{xc} \ast X_{t} + W_{hc} \ast H_{t-1} + b_c) `$ \
$` o_t = \sigma(W_{xo} \ast X_t +  W_{ho} \ast H_{t-1} + W_{co} \odot C_t + b_o `$ \
$` H_t = o_t \odot tanh(C_t) `$ \

Where $`\sigma`$ and $`\odot`$ represent the hyperbolic sigmoid function and Hadamard product respectively.

The expected input for this layer has shape `NHWC` or  `HWC` where:

* `N` is the optional batch dimension
* `H` is the input's spatial height dimension
* `W` is the input's spatial width dimension
* `C` is the input's channel dimension

And returns a `Tuple` of the hidden state, $`H_t`$, and the cell state, $`C_t`$, each with shape `NHWO`. 

**Args:** 

`in_channels (int)`: The number of input channels, `C`.\
`out_channels (int)`: The number of output channels, `O`.\
`kernel_size (int)`: The size of the convolution filters, must be odd to keep spatial dimensions with padding. Default: `5`. \
`stride (Union[int, tuple]` : The stride of the convolution.
`padding (Union[int, tuple]` : Padding to add to the input for convolution.
`dilation (Union[int, tuple]` : Dilation of the convolution.
`bias` (bool): Whether the convolutional calculation should use biases or not. Default: `True.`

### `ConvLSTM`

Unrolls a `_conv_lstm_cell` sequentially over time-steps.

The expected input for this layer has shape `NLHWC` or  `LHWC` where:

* `N` is the optional batch dimension
* `L` is the length of the sequence
* `H` is the input's spatial height dimension
* `W` is the input's spatial width dimension
* `C` is the input's channel dimension

**Args:**
`in_channels (int)`: The number of input channels, `C`.\
`out_channels (int)`: The number of output channels, `O`.\
`kernel_size (int)`: The size of the convolution filters, must be odd to keep spatial dimensions with padding. Default: `5`. \
`bias` (bool): Whether the convolutional calculation should use biases or not. Default: `True.`

The following features are yet to be implemented from initial release:\
[] Bi-directionality - allows the conv-lstm to unroll both forwards and backwards across the sequence
[] Allow for stride customization
[] Allow for customizable padding along with modes 'same' and 'valid'




