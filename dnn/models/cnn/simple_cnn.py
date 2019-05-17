from torch import nn


class SimpleCNN(nn.Module):
    """
    A simple CNN.

    The network consists of a series of convolutional layers, each
    followed by a max pooling layer. It is then followed by a series
    of fully connected layers.

    Attributes
    ----------
    conv_layers : :class:`nn.Sequential`
        The convoluational layers of the network.

    fc_layers : :class:`nn.Sequential`
        The fully connected layers of the network.

    fc_input_size : :class:`int`
        The number of features to the first fully connected layer.

    """

    def __init__(self,
                 channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_paddings,
                 conv_dilations,
                 pool_kernel_sizes,
                 pool_strides,
                 pool_paddings,
                 pool_dilations,
                 fc_input_size,
                 fcs,
                 final_activation):
        """
        Initialize a :class:`SimpleCNN`.

        Parameters
        ----------
        channels : :class:`list` of :class:`int`
            The number of channels in each layer of the network,
            including the input and output layer. This means the
            length of this :class:`list` is larger by 1 than
            `conv_kernel_sizes`.

        conv_kernel_sizes : :class:`list` of :class:`int`
            The kernel size of each convolutional layer.

        conv_strides : :class:`list` of :class:`int`
            The stride of each convolutional layer.

        conv_paddings : :class:`list` of :class:`int`
            The padding of each convolutional layer.

        conv_dilations : :class:`list` of :class:`int`
            The dilation of each convolutional layer.

        pool_kernel_sizes : :class:`list` of :class:`int`
            The kernel size of each max pooling layer.

        pool_strides : :class:`list` of :class:`int`
            The stride of each max pooling layer.

        pool_paddings : :class:`list` of :class`int`
            The padding of each max pooling layer.

        pool_dilations : :class:`list` of :class:`int`
            The dilation of each max pooling layer.

        fc_input_size : :class:`int`
            The number of features to the first fully connected layer.

        fcs : :class:`list` of :class:`int`
            The number of neurons in each fully connected layer.

        final_activation : :class:`function`
            The activation function to be applied to the output layer.

        """

        assert (len(channels) - 1 == len(conv_kernel_sizes) and
                len(conv_kernel_sizes) == len(conv_strides) and
                len(conv_strides) == len(conv_paddings) and
                len(conv_paddings) == len(conv_dilations) and
                len(conv_dilations) == len(pool_kernel_sizes) and
                len(pool_kernel_sizes) == len(pool_strides) and
                len(pool_strides) == len(pool_paddings) and
                len(pool_paddings) == len(pool_dilations))

        super().__init__()

        conv_layers = []
        for i in range(len(conv_kernel_sizes)):
            conv = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=conv_kernel_sizes[i],
                stride=conv_strides[i],
                padding=conv_paddings[i],
                dilation=conv_dilations[i]
            )
            conv_layers.append(conv)
            conv_layers.append(nn.ReLU(inplace=True))

            pool = nn.MaxPool2d(
                kernel_size=pool_kernel_sizes[i],
                stride=pool_strides[i],
                padding=pool_paddings[i],
                dilation=pool_dilations[i]
            )
            conv_layers.append(pool)

        self.conv_layers = nn.Sequential(*conv_layers)

        fc_layers = []
        self.fc_input_size = in_features = fc_input_size
        for out_features in fcs:
            fc_layer = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            fc_layers.append(fc_layer)

            if i != len(fcs)-1:
                activation = nn.ReLU(inplace=True)
            else:
                activation = final_activation
            fc_layers.append(activation)

            in_features = out_features

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_size)
        return self.fc_layers(x)
