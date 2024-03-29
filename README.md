# An Implementation for ConvLSTM in Apple's Array Framework `mlx`

A Convolutional LSTM recurrent layer. 


### `conv_lstm_cell` 
This `Module` computes the hidden and cell state for a time-step, expressed as:

$` i_t = \sigma (W_{xi} \ast X_t + W_{hi} \ast H_{t-1} + W_{ci} \odot C_{t-1} + b_i) `$ \
$` f_t = \sigma (W_{xf} \odot X_t \ast H_{t-1} + W_{cf} \odot C_{t-1} + b_f) `$\
$` c_t = f_t \odot X_t + tanh(W_{xc} \ast X_{t} + W_{hc} \odot H_{t-1} + b_c `$ \
The expected input for this layer has shape `NLHWC` or  `LHWC` where:

* `N` is the optional batch dimension
* `L` is the length of the sequence
* `H` is the input's spatial height dimension
* `W` is the input's spatial width dimension
* `C` is the input's channel dimension

Conretely, for each element of the sequence, this layer computes `conv_lstm_cell` recurrently, expressed as:


