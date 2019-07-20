from torch import nn


class Generator(nn.Module):
    """
    A generator consisting of feed-forward layers.

    The networks uses leaky ReLU activations, except the last layer
    which is tanh.

    Attributes
    ----------
    _layers : :class:`torch.Sequential`
        All the layers of the network.

    """

    def __init__(self, fc_layers, lrelu_alpha):
        """
        Initialize a :class:`Generator`.

        Parameters
        ----------
        fc_layers : :class:`list` of :class:`int`
            The number of neurons in each layer, including the input
            and output layers.

        lrelu_alpha : :class:`float`
            The alpha value for the leaky ReLU activations units.

        """

        super().__init__()
        num_torch_layers = len(fc_layers) - 1
        last_torch_layer_index = num_torch_layers - 1

        layers = []
        for i in range(num_torch_layers):
            in_features = fc_layers[i]
            out_features = fc_layers[i+1]

            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            layers.append(layer)

            if i != last_torch_layer_index:
                activation = nn.LeakyReLU(lrelu_alpha)
            else:
                activation = nn.Tanh()
            layers.append(activation)

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class Discriminator(nn.Module):
    """
    A discriminator consisting of feed-forward layers.

    All layers use the leaky ReLU activation except the output layer,
    which does not use an activation.

    Attributes
    ----------
    _layers : :class:`torch.Sequential`
        All the layers of the network.

    """

    def __init__(self, fc_layers, lrelu_alpha):
        """
        Initialize a :class:`Discriminator`.

        Parameters
        ----------
        fc_layers : :class:`list` of :class:`int`
            The number of neurons in each layer, including the input
            and output layers.

        lrelu_alpha : :class:`float`
            The alpha value for the leaky ReLU activations units.

        """

        super().__init__()
        num_torch_layers = len(fc_layers) - 1
        last_torch_layer_index = num_torch_layers - 1

        layers = []
        for i in range(num_torch_layers):
            in_features = fc_layers[i]
            out_features = fc_layers[i+1]

            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            layers.append(layer)

            if i != last_torch_layer_index:
                activation = nn.LeakyReLU(lrelu_alpha)
                layers.append(activation)

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)
