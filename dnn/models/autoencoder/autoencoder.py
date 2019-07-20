from torch import nn


class _AutoencoderModule(nn.Module):
    """
    Represents either an encoder or a decoder module.

    This class should not be initialized directly. It servers as a
    base class for both :class:`.Encoder` and :class:`.Decoder`.

    Attributes
    ----------
    _conv : :class:`type`
        An initializer for the final convolutional layer the module
        uses. This will be need to be defined as a class attribute in a
        derived class.

    _final_activation : :class:`type`
        An initializer for the final activation function on the module.
        This will need to be defined as a class attribute in a derived
        class.

    _layers : :class:`torch.Sequential`
        The layers of the module.

    Methods
    -------
    :meth:`forward`

    """

    def __init__(self, channels, kernel_sizes, strides, paddings):
        """
        Initialize an :class:`._AutoencoderModule`.

        Parameters
        ----------
        channels : :class:`list` of :class:`int`
            The number of channels in each layer. This includes
            the input and output layer. As a result, this :class:`list`
            will be longer by 1 than `kernel_sizes`, `strides` or
            `paddings`.

        kernel_sizes : :class:`list` of :class:`int`
            The kernel size of each convolutional layer.

        strides : :class:`list` of :class:`int`
            The stride of each convolutional layer.

        paddings : :class:`list` of :class:`int`
            The padding of each convolutional layer.

        """

        super().__init__()
        num_conv_layers = len(kernel_sizes)
        last_conv_layer_index = num_conv_layers - 1

        layers = []
        for i in range(num_conv_layers):
            in_channels = channels[i]
            out_channels = channels[i+1]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]

            conv = self._conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            layers.append(conv)

            if i < last_conv_layer_index:
                batch_norm = nn.BatchNorm2d(num_features=out_channels)
                layers.append(batch_norm)
                activation = nn.ReLU(inplace=True)
            else:
                activation = self._final_activation

            layers.append(activation)

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class Encoder(_AutoencoderModule):
    """
    Represents an encoder in a :class:`.Autoencoder`.

    """

    _conv = nn.Conv2d
    _final_activation = nn.ReLU(inplace=True)


class Decoder(_AutoencoderModule):
    """
    Represents a decoder in a :class:`.Autoencoder`.

    """

    _conv = nn.ConvTranspose2d
    _final_activation = nn.Tanh()


class Autoencoder(nn.Module):
    """
    Represents an autoencoder network.

    Attributes
    ----------
    _encoder : :class:`.Encoder`
        The :class:`.Encoder` network.

    _decoder : :class:`.Decoder`
        The :class:`.Decoder` network.

    _autoencoder : :class:`torch.Sequential`
        The layers of the autoencoder network.

    Methds
    ------
    :meth:`forward`

    """

    def __init__(self, encoder, decoder):
        """
        Initialize a :class:`.Autoencoder`.

        Parameters
        ----------
        encoder : :class:`.Encoder`
            The :class:`.Encoder` network.

        decoder : :class:`.Decoder`
            The :class:`.Decoder` network.

        """

        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._autoencoder = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self._autoencoder(x)
