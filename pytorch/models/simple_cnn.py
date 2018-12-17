from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple CNN.

    The network consists of a series of convolutional layers, each
    followed by a max pooling layer. It is then followed by a series
    of fully connected layers.

    Attributes
    ----------
    convs : :class:`list` of :class:`torch.nn.Conv2d`
        The convolutional layers of the network.

    pools : :class:`list` of :class:`torch.nn.MaxPool2d`
        The pooling layers of the network.

    fcs : :class:`list` of :class:`torch.nn.Linear`
        The fully connect layers of the network.

    fc_input_size : :class:`int`
        The number of features to the first fully connected layer.

    final_activation : :class:`function`
        The activation function to be applied to the output layer.

    """

    def __init__(self,
                 conv_in_channels,
                 conv_out_channels,
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
        Initialize a SimpleCNN.

        Parameters
        ----------
        conv_in_channels : :class:`list` of :class:`int`
            The number of input channels to each convolutional layer.

        conv_out_channels : :class:`list` of :class:`int`
            The number of output channels each convolutional layer has.

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

        assert (len(conv_in_channels) == len(conv_out_channels) and
                len(conv_out_channels) == len(conv_kernel_sizes) and
                len(conv_kernel_sizes) == len(conv_strides) and
                len(conv_strides) == len(conv_paddings) and
                len(conv_paddings) == len(conv_dilations) and
                len(conv_dilations) == len(pool_kernel_sizes) and
                len(pool_kernel_sizes) == len(pool_strides) and
                len(pool_strides) == len(pool_paddings) and
                len(pool_paddings) == len(pool_dilations))

        super().__init__()
        self.fc_input_size = fc_input_size

        convs = []
        conv_params = zip(conv_in_channels,
                          conv_out_channels,
                          conv_kernel_sizes,
                          conv_strides,
                          conv_paddings,
                          conv_dilations)

        for params in conv_params:
            (in_channels,
             out_channels,
             kernel_size,
             stride,
             padding,
             dilation) = params
            layer = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
            convs.append(layer)
        self.convs = nn.ModuleList(convs)

        pools = []
        pool_params = zip(pool_kernel_sizes,
                          pool_strides,
                          pool_paddings,
                          pool_dilations)

        for p in pool_params:
            kernel_size, stride, padding, dilation = p
            layer = nn.MaxPool2d(kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation)
            pools.append(layer)
        self.pools = nn.ModuleList(pools)

        fc_layers = []
        in_features = fc_input_size
        for fc in fcs:
            layer = nn.Linear(in_features=in_features,
                              out_features=fc)
            fc_layers.append(layer)
            in_features = fc
        self.fcs = nn.ModuleList(fc_layers)
        self.final_activation = final_activation

    def forward(self, x):
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x))
            x = self.pools[i](x)

        x = x.view(-1, self.fc_input_size)

        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x = (F.relu(x) if i != len(self.fcs) - 1 else
                 self.final_activation(x))
        return x
