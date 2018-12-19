from torch import nn


class GrowingGan:
    def __init__(self, args):
        self.phase = 0
        self.args = args
        self.generator = nn.Sequential()
        self.discriminator = nn.Sequential()

    def grow(self):
        self.grow_generator()
        self.grow_discriminator()
        self.phase += 1

    def grow_generator(self):

        # Change the last activation from tanh to leaky relu.
        if self.phase:
            removed = list(self.generator.children())[:-2]
            self.generator = nn.Sequential(*removed)

        conv = nn.ConvTranspose2d(
                in_channels=self.args.g_input_channels[self.phase],
                out_channels=self.args.g_output_channels[self.phase],
                kernel_size=self.args.g_kernel_sizes[self.phase],
                stride=self.args.g_strides[self.phase],
                padding=self.args.g_paddings[self.phase],
                bias=False)
        self.generator.add_module(f'conv_{self.phase}', conv)

        num_features = self.args.g_output_channels[self.phase]
        batch_norm = nn.BatchNorm2d(num_features=num_features)
        name = f'batch_norm_{self.phase}'
        self.generator.add_module(name, batch_norm)

        activation = nn.LeakyReLU(self.args.lrelu_alpha)
        name = f'activation_{self.phase}'
        self.generator.add_module(name, activation)

        conv = nn.Conv2d(
                    in_channels=self.args.g_output_channels[self.phase],
                    out_channels=self.args.image_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)
        self.generator.add_module('final_conv', conv)
        self.generator.add_module('final_activation', nn.Tanh())

    def grow_discriminator(self):
        # Remove last layer which performs classification.
        if self.phase:
            removed = list(self.discriminator.children())[:-1]
            self.discriminator = nn.Sequential(*removed)

        conv = nn.Conv2d(
                    in_channels=self.args.d_input_channels[self.phase],
                    out_channels=self.args.d_output_channels[self.phase],
                    kernel_size=self.args.d_kernel_sizes[self.phase],
                    stride=self.args.d_strides[self.phase],
                    padding=self.args.d_paddings[self.phase],
                    bias=False)
        self.discriminator.add_module(f'conv_{self.phase}', conv)

        if self.phase:
            num_features = self.args.d_output_channels[self.phase]
            batch_norm = nn.BatchNorm2d(num_features=num_features)
            name = f'batch_norm_{self.phase}'
            self.discriminator.add_module(name, batch_norm)

        activation = nn.LeakyReLU(self.args.lrelu_alpha)
        name = f'activation_{self.phase}'
        self.discriminator.add_module(name, activation)

        conv = nn.Conv2d(
                in_channels=self.args.d_output_channels[self.phase],
                out_channels=1,
                kernel_size=self.args.d_final_kernel_size,
                stride=1,
                padding=0,
                bias=False)
        self.discriminator.add_module('final_conv', conv)
