from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self,
                 image_channels,
                 input_channels,
                 output_channels,
                 kernel_sizes,
                 strides,
                 paddings):

        super().__init__()
        self.phase = 0
        self.image_channels = image_channels

        params = zip(input_channels,
                     output_channels,
                     kernel_sizes,
                     strides,
                     paddings)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i, p in enumerate(params):
            in_channels, out_channels, kernel_size, stride, padding = p

            conv = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=False)
            self.convs.append(conv)

            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            self.batch_norms.append(batch_norm)

            self.activations.append(nn.ReLU(inplace=True))

        self.final_activation = nn.Tanh()

    def grow(self):
        self.fade_alpha = 0
        self.phase += 1
        if self.phase > 1:
            self.fade_final_conv = self.final_conv
        self.final_conv = nn.Conv2d(
                    in_channels=self.convs[self.phase-1].out_channels,
                    out_channels=self.image_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)

    def fade_forward(self, x):
        for i in range(self.phase-1):
            x = self.convs[i](x)

            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activations[i](x)

        x = self.fade_final_conv(x)
        x = self.final_activation(x)
        return F.interpolate(x, scale_factor=2)

    def forward(self, x):
        inp = x
        for i in range(self.phase):
            x = self.convs[i](x)

            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)

            x = self.activations[i](x)

        x = self.final_conv(x)
        x = self.final_activation(x)

        if self.fade_alpha < 1 and self.phase > 1:
            y = self.fade_forward(inp)
            return self.fade_alpha*x + (1-self.fade_alpha)*y
        else:
            return x


class Discriminator(nn.Module):
    def __init__(self,
                 image_channels,
                 input_channels,
                 output_channels,
                 kernel_sizes,
                 strides,
                 paddings,
                 final_kernel_size,
                 lrelu_alpha):

        super().__init__()
        self.phase = 0
        self.image_channels = image_channels
        self.final_kernel_size = final_kernel_size

        params = zip(input_channels,
                     output_channels,
                     kernel_sizes,
                     strides,
                     paddings)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i, p in enumerate(params):
            in_channels, out_channels, kernel_size, stride, padding = p

            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             bias=False)
            self.convs.append(conv)

            self.activations.append(nn.LeakyReLU(lrelu_alpha))

            if i != 0:
                batch_norm = nn.BatchNorm2d(num_features=out_channels)
                self.batch_norms.append(batch_norm)

    def grow(self):
        self.fade_alpha = 0
        self.phase += 1
        if self.phase > 1:
            self.fade_final_conv = self.final_conv
        self.final_conv = nn.Conv2d(
                    in_channels=self.convs[self.phase-1].out_channels,
                    out_channels=self.image_channels,
                    kernel_size=self.final_kernel_size,
                    stride=1,
                    padding=0,
                    bias=False)

    def fade_forward(self, x):
        x = F.interpolate(x, scale_factor=0.5)

        for i in range(self.phase-1):
            x = self.convs[i](x)

            if i != 0:
                x = self.batch_norms[i-1](x)

            x = self.activations[i](x)

        return self.fade_final_conv(x)

    def forward(self, x):
        inp = x
        for i in range(self.phase):
            x = self.convs[i](x)

            if i != 0:
                x = self.batch_norms[i-1](x)

            x = self.activations[i](x)

        x = self.final_conv(x)

        if self.fade_alpha < 1 and self.phase > 1:
            y = self.fade_forward(inp)
            return self.fade_alpha*x + (1-self.fade_alpha)*y
        else:
            return x


class GrowingGan:
    def __init__(self, args):
        self.generator = Generator(
                    image_channels=args.image_channels,
                    input_channels=args.g_input_channels,
                    output_channels=args.g_output_channels,
                    kernel_sizes=args.g_kernel_sizes,
                    strides=args.g_strides,
                    paddings=args.g_paddings)

        self.discriminator = Discriminator(
                    image_channels=args.image_channels,
                    input_channels=args.d_input_channels,
                    output_channels=args.d_output_channels,
                    kernel_sizes=args.d_kernel_sizes,
                    strides=args.d_strides,
                    paddings=args.d_paddings,
                    final_kernel_size=args.d_final_kernel_size,
                    lrelu_alpha=args.lrelu_alpha)

    def grow(self):
        self.generator.grow()
        self.discriminator.grow()
