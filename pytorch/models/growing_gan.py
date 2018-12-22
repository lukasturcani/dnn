from torch import nn
import torch.nn.functional as F


def same_padding(size_out,
                 size_in,
                 kernel_size,
                 stride,
                 dilation):

    t1 = (size_out - 1) * stride
    t2 = (kernel_size - 1) * dilation
    padding = t1 - size_in + t2 + 1
    left = padding // 2
    right = padding - left
    return (left, right)


class Generator(nn.Module):
    def __init__(self,
                 init_img_size,
                 img_channels,
                 latent_space_channels,
                 lrelu_alpha):
        super().__init__()
        self.fading = False
        self.fade_alpha = 0
        self.img_size = init_img_size
        self.img_channels = img_channels
        self.in_channels = latent_space_channels
        self.lrelu_alpha = lrelu_alpha
        self.blocks = nn.ModuleList()

    def grow(self, block_params):
        block = []

        for out_channels, kernel_size in block_params:
            padding = same_padding(
                           size_out=self.img_size,
                           size_in=self.img_size,
                           kernel_size=kernel_size,
                           stride=1,
                           dilation=1)
            conv = nn.Conv2d(
                           in_channels=self.in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=padding)
            block.append(conv)
            block.append(nn.LeakyReLU(self.lrelu_alpha))
            self.in_channels = out_channels

        self.blocks.append(nn.Sequential(*block))

        if len(self.blocks) > 1:
            self.prev_to_rgb = self.to_rgb

        self.to_rgb = nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=self.img_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)

        self.img_size *= 2

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i != 0:
                x = F.interpolate(x, scale_factor=2)
            x = block(x)

            if self.fading and i == len(self.blocks)-2:
                prev_x = F.interpolate(x, scale_factor=2)
                prev_img = self.prev_to_rgb(prev_x)

        if self.fading:
            prev = (1-self.fade_alpha)*prev_img
            current = self.fade_alpha*self.to_rgb(x)
            return prev + current

        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self,
                 init_img_size,
                 img_channels,
                 lrelu_alpha):
        super().__init__()
        self.fading = False
        self.img_size = init_img_size
        self.img_channels = img_channels
        self.lrelu_alpha = lrelu_alpha
        self.blocks = nn.ModuleList()

    def grow(self, block_params):

        if len(self.blocks):
            self.prev_from_rgb = self.from_rgb

        conv1_out_channels, _ = block_params[0]
        self.from_rgb = nn.Conv2d(
                        in_channels=self.img_channels,
                        out_channels=conv1_out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)

        block = []

        in_channels = conv1_out_channels
        for out_channels, kernel_size in block_params:

            padding = same_padding(
                           size_out=self.img_size,
                           size_in=self.img_size,
                           kernel_size=kernel_size,
                           stride=1,
                           dilation=1)
            conv = nn.Conv2d(
                           in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=padding)
            block.append(conv)
            block.append(nn.LeakyReLU(self.lrelu_alpha))
            in_channels = out_channels

        self.blocks.append(nn.Sequential(*block))

        if len(self.blocks) == 1:
            self.final_conv = nn.Conv2d(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=self.img_size,
                                        stride=1,
                                        padding=0)
            self.final_activation = nn.LeakyReLU(self.lrelu_alpha)
            self.fc = nn.Linear(in_features=out_channels,
                                out_features=1)

        self.img_size *= 2

    def forward(self, x):
        if self.fading:
            prev_x = F.avg_pool2d(x, kernel_size=2)
            prev_x = self.prev_from_rgb(prev_x)*(1-self.fade_alpha)

        x = self.from_rgb(x)
        for i, block in enumerate(reversed(self.blocks)):
            x = block(x)

            if i < len(self.blocks)-1:
                x = F.avg_pool2d(x, kernel_size=2)

            if self.fading and i == 0:
                x = prev_x + self.fade_alpha*x

        x = self.final_conv(x)
        x = self.final_activation(x)
        batch_size = x.size()[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)
