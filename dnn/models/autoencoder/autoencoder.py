from torch import nn


class Module(nn.Module):
    """
    Represents either an encoder or a decoder module.

    """

    def __init__(self, channels, kernel_sizes, strides, paddings):
        """
        Initializes a encoder or decoder module.

        Parameters
        ----------
        channels : :class:`list` of :class:`int`
            The number of channels in each layer. This includes
            the input and output layer. As a result this list will be
            longer by 1 than `kernel_sizes`, `strides` or
            `paddings`.

        kernel_sizes : :class:`list` of :class:`int`
            The kernel size of each convolutional layer.

        strides : :class:`list` of :class:`int`
            The stride of each convoluational layer.

        paddings : :class:`list` of :class:`int`
            The padding of each convoluational layer.

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

            conv = self.conv(
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
                activation = self.final_activation

            layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(Module):
    conv = nn.Conv2d
    final_activation = nn.ReLU(inplace=True)


class Decoder(Module):
    conv = nn.ConvTranspose2d
    final_activation = nn.Tanh()


class AutoEncoder(nn.Module):
    """

    """

    def __init__(self, encoder, decoder):
        """

        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self.autoencoder(x)
