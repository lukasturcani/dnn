from torch import nn


class Generator(nn.Module):
    def __init__(self,
                 channels,
                 kernel_sizes,
                 strides,
                 paddings):

        super().__init__()
        self.phase = 0
        self.image_channels = channels[-1]

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(kernel_sizes)):
            conv = nn.ConvTranspose2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=False
            )
            nn.init.kaiming_normal_(
                tensor=conv.weight,
                nonlinearity='relu'
            )
            self.convs.append(conv)

            batch_norm = nn.BatchNorm2d(num_features=channels[i+1])
            self.batch_norms.append(batch_norm)

            self.activations.append(nn.ReLU(inplace=True))

        self.final_activation = nn.Tanh()

    def grow(self):
        self.phase += 1
        self.final_conv = nn.Conv2d(
            in_channels=self.convs[self.phase-1].out_channels,
            out_channels=self.image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        for i in range(self.phase):
            x = self.convs[i](x)

            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activations[i](x)

        x = self.final_conv(x)
        x = self.final_activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self,
                 channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 final_kernel_size,
                 lrelu_alpha):

        super().__init__()
        self.phase = 0
        self.image_channels = channels[0]
        self.final_kernel_size = final_kernel_size

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            conv = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=False
            )
            nn.init.kaiming_normal_(
                tensor=conv.weight,
                a=lrelu_alpha
            )
            self.convs.append(conv)

            self.activations.append(nn.LeakyReLU(lrelu_alpha))

            if i != 0:
                batch_norm = nn.BatchNorm2d(num_features=channels[i+1])
                self.batch_norms.append(batch_norm)

    def grow(self):
        self.phase += 1
        self.final_conv = nn.Conv2d(
            in_channels=self.convs[self.phase-1].out_channels,
            out_channels=self.image_channels,
            kernel_size=self.final_kernel_size,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        for i in range(self.phase):
            x = self.convs[i](x)

            if i != 0:
                x = self.batch_norms[i-1](x)

            x = self.activations[i](x)

        return self.final_conv(x)


class GrowingGan:
    def __init__(self, args):
        self.generator = Generator(
            channels=args.g_channels,
            kernel_sizes=args.g_kernel_sizes,
            strides=args.g_strides,
            paddings=args.g_paddings
        )

        self.discriminator = Discriminator(
            channels=args.d_channels,
            kernel_sizes=args.d_kernel_sizes,
            strides=args.d_strides,
            paddings=args.d_paddings,
            final_kernel_size=args.d_final_kernel_size,
            lrelu_alpha=args.lrelu_alpha
        )

    def grow(self):
        self.generator.grow()
        self.discriminator.grow()
