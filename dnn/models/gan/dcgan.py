from torch import nn


class Generator(nn.Module):
    """
    The DCGAN generator.

    Attributes
    ----------
    layers : :class:`torch.Sequential`
        All the layers of the newtork.

    """

    def __init__(self,
                 channels,
                 kernel_sizes,
                 strides,
                 paddings):
        """
        Initializes the generator.

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

            conv = nn.ConvTranspose2d(
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

            if i < last_conv_layer_index:
                activation = nn.ReLU(inplace=True)
            else:
                activation = nn.Tanh()
            layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    """
    The DCGAN discriminator.

    Attributes
    ----------
    layers : :class:`torch.Sequential`
        All the layers of the newtork.

    """

    def __init__(self,
                 channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 lrelu_alpha):
        """
        Initializes the discriminator.

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

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
            layers.append(conv)

            if i < last_conv_layer_index:
                activation = nn.LeakyReLU(lrelu_alpha)
                layers.append(activation)

            if i != 0 and i < last_conv_layer_index:
                batch_norm = nn.BatchNorm2d(num_features=out_channels)
                layers.append(batch_norm)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
