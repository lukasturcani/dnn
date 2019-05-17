import logging
import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dnn.models.gan.dcgan import Generator, Discriminator
from dnn.train_scripts.utils.trainers import GANTrainer

logger = logging.getLogger(__name__)


def data_loaders(args):
    """
    Creates MNIST train and test data loaders.

    Parameters
    ----------
    args : :class:`Namespace`
        Command line arguments passed to the script, holding model
        hyperparameters.

    Returns
    -------
    :class:`tuple` of :class:`torch.DataLoader`
        The train and test DataLoaders, respectively.

    """

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
        pin_memory=True
    )

    test_mnist = datasets.MNIST(
        root=args.database_root,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        dataset=test_mnist,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    return train_loader, test_loader


def main():

    ###################################################################
    # Define command line parameters.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        default=42,
        type=int
    )
    parser.add_argument(
        '--train_batch_size',
        default=100,
        type=int
    )
    parser.add_argument(
        '--test_batch_size',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float
    )
    parser.add_argument(
        '--beta1',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--beta2',
        default=0.999,
        type=float
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int
    )
    parser.add_argument(
        '--lrelu_alpha',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--label_smoothing',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--g_noise_shape',
        default=[100, 1, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--g_channels',
        default=[100, 1024, 512, 256, 128, 1],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--g_kernel_sizes',
        default=[4, 4, 4, 4, 4],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--g_strides',
        default=[1, 2, 2, 2, 2],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--g_paddings',
        default=[0, 1, 1, 1, 1],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--d_channels',
        default=[1, 128, 256, 512, 1024, 1],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--d_kernel_sizes',
        default=[4, 4, 4, 4, 4],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--d_strides',
        default=[2, 2, 2, 2, 1],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--d_paddings',
        default=[1, 1, 1, 1, 0],
        type=int,
        nargs='+'
    )
    parser.add_argument(
        '--database_root',
        default='/home/lukas/databases'
    )
    parser.add_argument(
        '--img_dir',
        default='generated_images'
    )
    parser.add_argument(
        '--saved_img_scale',
        default=1.,
        type=float
    )
    parser.add_argument(
        '--log_interval',
        default=50,
        type=int
    )
    parser.add_argument(
        '--logging_level',
        default=logging.DEBUG,
        type=int
    )
    args = parser.parse_args()

    ###################################################################
    # Set up output directory.
    ###################################################################

    if os.path.exists(args.img_dir):
        shutil.rmtree(args.img_dir)
    os.mkdir(args.img_dir)

    ###################################################################
    # Set up logging.
    ###################################################################

    logging_fmt = '%(asctime)s - %(levelname)s - %(module)s - %(msg)s'
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(
        level=args.logging_level,
        format=logging_fmt,
        datefmt=date_fmt
    )

    ###################################################################
    # Set random seed.
    ###################################################################

    torch.manual_seed(args.seed)

    ###################################################################
    # Create the generator.
    ###################################################################

    generator = Generator(
        channels=args.g_channels,
        kernel_sizes=args.g_kernel_sizes,
        strides=args.g_strides,
        paddings=args.g_paddings
    )
    generator.to('cuda')
    logging.debug(generator)

    ###################################################################
    # Create the discriminator.
    ###################################################################

    discriminator = Discriminator(
        channels=args.d_channels,
        kernel_sizes=args.d_kernel_sizes,
        strides=args.d_strides,
        paddings=args.d_paddings,
        lrelu_alpha=args.lrelu_alpha
    )
    discriminator.to('cuda')
    logging.debug(discriminator)

    ###################################################################
    # Create the data loaders.
    ###################################################################

    train_loader, test_loader = data_loaders(args)

    ###################################################################
    # Create the trainer.
    ###################################################################

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        args=args,
        img_shape=[1, 64, 64]
    )

    ###################################################################
    # Train.
    ###################################################################

    for epoch in range(args.epochs):
        trainer.train(train_loader)
        trainer.eval(test_loader)


if __name__ == '__main__':
    main()
