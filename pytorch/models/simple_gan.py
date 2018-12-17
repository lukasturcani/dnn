from torch import nn


class Generator(nn.Module):
    def __init__(self, input_size, fc_layers, lrelu_alpha):
        super().__init__()

        layers = []
        activations = []
        in_features = input_size
        for i, out_features in enumerate(fc_layers):
            layer = nn.Linear(in_features=in_features,
                              out_features=out_features)
            layers.append(layer)
            in_features = out_features

            activation = (nn.LeakyReLU(lrelu_alpha) if
                          i != len(fc_layers) - 1 else
                          nn.Tanh())
            activations.append(activation)

        self.fc_layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.activations[i](x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, fc_layers, lrelu_alpha):
        super().__init__()

        layers = []
        activations = []
        in_features = input_size
        for i, out_features in enumerate(fc_layers):
            layer = nn.Linear(in_features=in_features,
                              out_features=out_features)
            layers.append(layer)
            in_features = out_features

            activation = (nn.LeakyReLU(lrelu_alpha) if
                          i != len(fc_layers) - 1 else
                          nn.LogSoftmax(dim=1))
            activations.append(activation)

        self.fc_layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.activations[i](x)

        return x
