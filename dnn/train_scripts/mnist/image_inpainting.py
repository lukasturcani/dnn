import argparse
import logging
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from os.path import join
import os
import shutil

from dnn.pytorch.models.autoencoder.autoencoder import (
    Encoder, Decoder, AutoEncoder
)

logger = logging.getLogger(__name__)


def mask(loader):
    batch_size = loader.batch_sampler.batch_size
    m = torch.zeros(batch_size, 1, 64, 64)
    m[:, :, :32, :] = 1
    return m.to('cuda')


def train(args, model, train_loader, optimizer, epoch):
    model.train()

    m = mask(train_loader)
    for batch_id, (data, _) in enumerate(train_loader):
        data = data.to('cuda')
        optimizer.zero_grad()
        output = model(data*m)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_id % args.log_interval == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            msg = msg.format(
                epoch,
                batch_id * len(data),
                len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                loss.item()
            )
            logger.info(msg)


def test(args, model, test_loader, epoch):
    model.eval()
    i = iter(test_loader)
    m = mask(test_loader)
    with torch.no_grad():
        data, _ = next(i)
        data = data.to('cuda')
        output = model(data*m)
        save_image(
            tensor=data,
            filename=join(
                args.output_dir,
                'images',
                f'epoch_{epoch}_original.png'
            ),
            nrow=10
        )
        save_image(
            tensor=data*m,
            filename=join(
                args.output_dir,
                'images',
                f'epoch_{epoch}_masked.png'
            ),
            nrow=10
        )
        save_image(
            tensor=output,
            filename=join(
                args.output_dir,
                'images',
                f'epoch_{epoch}_output.png'
            ),
            nrow=10
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--encoder_channels',
                        default=[1, 256, 256, 512, 1024],
                        nargs='+',
                        type=int)
    parser.add_argument('--encoder_kernel_sizes',
                        default=[4, 4, 4, 4],
                        nargs='+',
                        type=int)
    parser.add_argument('--encoder_strides',
                        default=[2, 2, 2, 2],
                        nargs='+',
                        type=int)
    parser.add_argument('--encoder_paddings',
                        default=[1, 1, 1, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--decoder_channels',
                        default=[1024, 512, 256, 256, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--decoder_kernel_sizes',
                        default=[4, 4, 4, 4],
                        nargs='+',
                        type=int)
    parser.add_argument('--decoder_strides',
                        default=[2, 2, 2, 2],
                        nargs='+',
                        type=int)
    parser.add_argument('--decoder_paddings',
                        default=[1, 1, 1, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--database_root',
                        default='/home/lukas/databases')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--logging_level',
                        default=logging.DEBUG,
                        type=int)

    args = parser.parse_args()

    logging_fmt = (
        '%(asctime)s - %(levelname)s - %(module)s - %(msg)s'
    )
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(
        level=args.logging_level,
        format=logging_fmt,
        datefmt=date_fmt
    )

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    os.mkdir(join(args.output_dir, 'images'))

    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_mnist = datasets.MNIST(
        root=args.database_root,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        dataset=train_mnist,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    test_mnist = datasets.MNIST(
        root=args.database_root,
        train=False,
        transform=transform
    )
    test_loader = DataLoader(
        dataset=test_mnist,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    encoder = Encoder(
        channels=args.encoder_channels,
        kernel_sizes=args.encoder_kernel_sizes,
        strides=args.encoder_strides,
        paddings=args.encoder_paddings
    )
    decoder = Decoder(
        channels=args.decoder_channels,
        kernel_sizes=args.decoder_kernel_sizes,
        strides=args.decoder_strides,
        paddings=args.decoder_paddings
    )
    autoencoder = AutoEncoder(encoder, decoder)
    autoencoder.to('cuda')
    logging.debug(autoencoder)

    optimizer = optim.Adam(
        params=autoencoder.parameters(),
        lr=args.learning_rate
    )

    for epoch in range(1, args.epochs+1):
        train(args, autoencoder, train_loader, optimizer, epoch)
        test(args, autoencoder, test_loader, epoch)


if __name__ == '__main__':
    main()
