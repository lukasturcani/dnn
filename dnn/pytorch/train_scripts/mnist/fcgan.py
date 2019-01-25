import logging
import argparse
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dnn.pytorch.models.gan.fcgan import Generator, Discriminator
from dnn.pytorch.train_scripts.utils.trainers import GANTrainer

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
        transforms.ToTensor(),
        lambda image: (image - 0.5) * 2,
        lambda image: image.view(28*28)
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

    return train_loader, test_loader


def main():

    ###################################################################
    # Create the command line arguments.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_batch_size', default=100, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--img_dir', default='generated_images')

    parser.add_argument('--g_noise_shape',
                        default=[100],
                        nargs='+',
                        type=int)

    parser.add_argument('--d_fc_layers',
                        default=[28*28, 1024, 512, 256, 1],
                        type=int,
                        nargs='+')

    parser.add_argument('--g_fc_layers',
                        default=[256, 512, 1024, 784],
                        nargs='+',
                        type=int)

    parser.add_argument('--logging_level',
                        default=logging.DEBUG,
                        type=int)
    parser.add_argument('--database_root',
                        default='/home/lukas/databases')

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

    logging_fmt = ('%(asctime)s - %(levelname)s - '
                   '%(module)s - %(msg)s')
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(level=args.logging_level,
                        format=logging_fmt,
                        datefmt=date_fmt)

    ###################################################################
    # Set random seed.
    ###################################################################

    torch.manual_seed(args.seed)

    ###################################################################
    # Create the generator.
    ###################################################################

    generator = Generator(
                    fc_layers=args.g_fc_layers,
                    lrelu_alpha=args.lrelu_alpha)
    generator.to('cuda')

    ###################################################################
    # Create the discriminator.
    ###################################################################

    discriminator = Discriminator(
                        fc_layers=args.d_fc_layers,
                        lrelu_alpha=args.lrelu_alpha)
    discriminator.to('cuda')

    ###################################################################
    # Create the data loaders.
    ###################################################################

    train_loader, test_loader = data_loaders(args)

    ###################################################################
    # Create the trainer.
    ###################################################################

    trainer = GANTrainer(generator=generator,
                         discriminator=discriminator,
                         args=args,
                         img_shape=[1, 28, 28])

    ###################################################################
    # Train.
    ###################################################################

    for epoch in range(args.epochs):
        trainer.train(train_loader)
        trainer.eval(test_loader)


if __name__ == '__main__':
    main()
