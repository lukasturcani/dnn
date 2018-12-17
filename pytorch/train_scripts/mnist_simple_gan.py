import logging
import argparse
import os
import shutil
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dnn_models.pytorch.models.simple_gan import (Generator,
                                                  Discriminator)

logger = logging.getLogger(__name__)


def train(args,
          generator,
          discriminator,
          g_optimizer,
          d_optimizer,
          train_loader,
          epoch):

    generator.train()
    discriminator.train()

    real_label = 1.
    smooth_real_label = 1. - args.label_smoothing
    fake_label = 0.
    # criterion = nn.BCEWithLogitsLoss()

    for batch_id, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to('cuda')
        real_images = real_images.view(-1, 28*28)
        batch_size = real_images.size()[0]

        # Train the discriminator.
        d_optimizer.zero_grad()
        noise = torch.rand(batch_size, args.g_input_size)
        noise = noise.to('cuda')
        fake_images = generator(noise).view_as(real_images)

        real_output = discriminator(real_images).view(batch_size)
        fake_output = discriminator(fake_images).view(batch_size)

        real_target = torch.full((batch_size, ),
                                 smooth_real_label,
                                 device='cuda')
        # d_real_loss = criterion(real_output, real_target)
        d_real_loss = F.nll_loss

        fake_target = torch.full((batch_size, ),
                                 fake_label,
                                 device='cuda')
        d_fake_loss = criterion(fake_output, fake_target)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train the generator.
        # g_optimizer.zero_grad()
        # noise = torch.rand(batch_size, args.g_input_size)
        # noise = noise.to('cuda')
        #
        # fake_images = generator(noise).view_as(real_images)
        # fake_output = discriminator(fake_images).view(batch_size)
        #
        # fake_target = torch.full((batch_size, ),
        #                          real_label,
        #                          device='cuda')
        # g_loss = criterion(fake_output, fake_target)
        # g_loss.backward()
        # g_optimizer.step()
        #
        # if batch_id % args.log_interval == 0:
        #     msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
        #            'Discriminator Loss: {:.6f}\t'
        #            'Generator Loss: {:.6f}')
        #     msg = msg.format(
        #         epoch,
        #         batch_id * len(real_images),
        #         len(train_loader.dataset),
        #         100. * batch_id / len(train_loader),
        #         d_loss.item(),
        #         g_loss.item())
        #     logger.info(msg)


def save_images(images, epoch, img_dir):
    filename = os.path.join(img_dir, f'epoch_{epoch}.png')
    save_image(images, filename)


def test(args, generator, discriminator, test_loader, epoch):
    generator.eval()
    discriminator.eval()
    test_d_loss = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()

    smooth_real_label = 1. - args.label_smoothing
    fake_label = 0.

    with torch.no_grad():
        for real_images, _ in test_loader:
            real_images = real_images.to('cuda')
            real_images = real_images.view(-1, 28*28)
            batch_size = real_images.size()[0]

            real_output = discriminator(real_images).view(batch_size)

            noise = torch.rand(batch_size,
                               args.g_input_size)
            noise = noise.to('cuda')
            fake_images = generator(noise).view_as(real_images)
            fake_output = discriminator(fake_images).view(batch_size)

            real_target = torch.full((batch_size, ),
                                     smooth_real_label,
                                     device='cuda')
            d_real_loss = criterion(real_output, real_target)

            fake_target = torch.full((batch_size, ),
                                     fake_label,
                                     device='cuda')
            d_fake_loss = criterion(fake_output, fake_target)

            test_d_loss += d_real_loss + d_fake_loss

            real_pred = real_output.round()
            real_correct = real_pred.eq(torch.ones_like(real_pred))
            correct += real_correct.sum().item()
            print('real_correct', real_correct.sum().item())

            fake_pred = fake_output.round()
            fake_correct = fake_pred.eq(torch.zeros_like(fake_pred))
            correct += fake_correct.sum().item()
            print('fake_correct', fake_correct.sum().item())

    test_d_loss /= len(test_loader.dataset)
    msg = ('\nTest set: Average discriminator loss: {:.4f}, '
           'Accuracy: {}/{} ({:.0f}%)\n')
    msg = msg.format(test_d_loss,
                     correct,
                     2*len(test_loader.dataset),
                     100. * correct / (2*len(test_loader.dataset)))
    logger.info(msg)

    noise = torch.rand(20, args.g_input_size).to('cuda')
    g_images = generator(noise).view(20, 1, 28, 28)
    save_images(g_images, epoch, args.img_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--d_fc_layers',
                        default=[1024, 512, 256, 2],
                        type=int,
                        nargs='+')
    parser.add_argument('--g_input_size', default=100, type=int)
    parser.add_argument('--g_fc_layers',
                        default=[256, 512, 1024, 784],
                        nargs='+',
                        type=int)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--logging_level',
                        default=logging.DEBUG,
                        type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--database_root',
                        default='/home/lukas/databases')
    parser.add_argument('--img_dir', default='generated_images')
    args = parser.parse_args()

    if os.path.exists(args.img_dir):
        shutil.rmtree(args.img_dir)
    os.mkdir(args.img_dir)

    logging_fmt = ('%(asctime)s - %(levelname)s - '
                   '%(module)s - %(msg)s')
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(level=args.logging_level,
                        format=logging_fmt,
                        datefmt=date_fmt)
    torch.manual_seed(args.seed)

    generator = Generator(
                    input_size=args.g_input_size,
                    fc_layers=args.g_fc_layers,
                    lrelu_alpha=args.lrelu_alpha)
    generator.to('cuda')

    discriminator = Discriminator(
                        input_size=28*28,
                        fc_layers=args.d_fc_layers,
                        lrelu_alpha=args.lrelu_alpha)
    discriminator.to('cuda')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_mnist = datasets.MNIST(root=args.database_root,
                                 train=True,
                                 download=True,
                                 transform=transform)

    train_loader = DataLoader(
                        dataset=train_mnist,
                        batch_size=args.train_batch_size,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True)

    test_mnist = datasets.MNIST(root=args.database_root,
                                train=False,
                                download=True,
                                transform=transform)

    test_loader = DataLoader(
                    dataset=test_mnist,
                    batch_size=args.test_batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True)

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=args.learning_rate,
                             betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=args.learning_rate,
                             betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        train(args=args,
              generator=generator,
              discriminator=discriminator,
              g_optimizer=g_optimizer,
              d_optimizer=d_optimizer,
              train_loader=train_loader,
              epoch=epoch)

        test(args=args,
             generator=generator,
             discriminator=discriminator,
             test_loader=test_loader,
             epoch=epoch)


if __name__ == '__main__':
    main()
