from torch import nn


class Discriminator(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 lrelu_alpha):

        super().__init__()

        params = zip(input_channels,
                     output_channels,
                     kernel_sizes,
                     strides,
                     paddings)

        convs, batch_norms, activations = [], [], []
        for i, p in enumerate(params):
            in_channels, out_channels, kernel_size, stride, padding = p

            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             bias=False)
            convs.append(conv)

            if i != len(input_channels)-1:
                activation = nn.LeakyReLU(lrelu_alpha)
                activations.append(activation)

            if i != 0 and i != len(input_channels)-1:
                batch_norm = nn.BatchNorm2d(num_features=out_channels)
                batch_norms.append(batch_norm)

        self.convs = nn.ModuleList(convs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):

        for i in range(len(self.convs)):
            x = self.convs[i](x)

            if i != 0 and i < len(self.convs)-1:
                x = self.batch_norms[i-1](x)

            if i < len(self.activations):
                x = self.activations[i](x)

        return x


class Generator(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_sizes,
                 strides,
                 paddings):

        super().__init__()

        params = zip(input_channels,
                     output_channels,
                     kernel_sizes,
                     strides,
                     paddings)

        convs, batch_norms, activations = [], [], []
        for i, p in enumerate(params):
            in_channels, out_channels, kernel_size, stride, padding = p

            conv = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=False)
            convs.append(conv)

            if i < len(input_channels)-1:
                batch_norm = nn.BatchNorm2d(num_features=out_channels)
                batch_norms.append(batch_norm)

            activation = (nn.ReLU(inplace=True) if
                          i < len(input_channels)-1 else
                          nn.Tanh())
            activations.append(activation)

        self.convs = nn.ModuleList(convs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):

        for i in range(len(self.convs)):
            x = self.convs[i](x)

            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activations[i](x)

        return x
