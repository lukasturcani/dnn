import argparse
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from os.path import join
import os
import shutil

from dnn.models.autoencoder.autoencoder import (
    Encoder, Decoder, AutoEncoder
)
from dnn.models.gan.dcgan import Discriminator


logger = logging.getLogger(__name__)


def mask(loader):
    batch_size = loader.batch_sampler.batch_size
    m = torch.zeros(batch_size, 1, 64, 64)
    m[:, :, :32, :] = 1
    return m.to('cuda')


class GANTrainer:
    """
    Takes care of running the training loop.

    Attributes
    ----------
    generator : :class:`torch.Module`
        The generator network.

    discriminator : :class:`torch.Module`
        The discriminator network.

    args : :class:`Namespace`
        A namespace holding various hyperparameters the trainer needs.

    criterion : :class:`torch.Module`
        The loss function.

    epochs : :class:`int`
        The total number of epochs the trainer has gone through.

    g_optimizer : :class:`torch.Optimizer`
        The generator's optimizer.

    d_optimizer : :class:`torch.Optimizer`
        The discriminator's optimizer.

    img_shape : :class:`list` of :class:`int`
        The shape of the images the generator outputs.
        ``[channels, height, width]``.

    """

    def __init__(self, generator, discriminator, args, img_shape):
        """
        Initializes the trainer.

        Parameters
        ----------
        generator : :class:`torch.Module`
            The generator network.

        discriminator : :class:`torch.Module`
            The discriminator network.

        args : :class:`Namespace`
            A namespace hodling various hyperparameters the trainer
            needs.

        img_shape : :class:`list` of :class:`int`
            The shape of the images the generator outputs.
            ``[channels, height, width]``.

        """

        self.generator = generator
        self.discriminator = discriminator
        self.args = args
        self.img_shape = img_shape
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = 0

        self.g_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )
        self.d_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )

    def d_train_step(self, batch_size, real_images):
        """
        Carries out a single training step on the discriminator.

        Parameters
        ----------
        batch_size : :class:`int`
            The batch size.

        real_images : :class:`torch.Tensor`
            A batch of real images used for this training step.

        Returns
        -------
        None : :class:`NoneType`

        """

        inpainted_images = self.generator(real_images*self.mask)

        # Reset accumulated discriminator gradient.
        self.d_optimizer.zero_grad()

        # Get logits.
        real_logits = self.discriminator(real_images)
        inpainted_logits = self.discriminator(inpainted_images)

        # Get target labels for real images.
        real_target = torch.ones(batch_size, device='cuda')
        real_target -= self.args.label_smoothing
        real_target = real_target.view_as(real_logits)

        # Calculate loss on real iamges.
        d_real_loss = self.criterion(real_logits, real_target)

        # Get target labels for inpainted images.
        inpainted_target = torch.zeros(batch_size, device='cuda')
        inpainted_target = inpainted_target.view_as(inpainted_logits)

        # Calculate loss on inpainted images.
        d_inpainted_loss = self.criterion(
            inpainted_logits,
            inpainted_target
        )

        # Calculate total loss.
        d_loss = d_real_loss + d_inpainted_loss
        self.d_loss = d_loss.item()

        # Backprop.
        d_loss.backward()
        self.d_optimizer.step()

    def g_train_step(self, batch_size, real_images):
        """
        Carries out a single training step on the generator.

        Parameters
        ----------
        batch_size : :class:`int`
            The batch size used in this training step.

        real_images : :class:`torch.Tensor`
            A batch of real images used for this training step.

        Returns
        -------
        None : :class:`NoneType`

        """

        # Reset accumulated generator gradient.
        self.g_optimizer.zero_grad()

        # Create inpainted images.
        inpainted_images = self.generator(real_images*self.mask)

        # Get logits.
        inpainted_logits = self.discriminator(inpainted_images)

        # Get target labels for inpainted images.
        inpainted_target = torch.ones(batch_size, device='cuda')
        inpainted_target = inpainted_target.view_as(inpainted_logits)

        # Calculate generator loss.
        g_loss = self.criterion(inpainted_logits, inpainted_target)
        g_loss += F.mse_loss(inpainted_images, real_images)
        self.g_loss = g_loss.item()

        # Backprop.
        g_loss.backward()
        self.g_optimizer.step()

    def train(self, train_loader):
        """
        Trains the generator and discriminator for an epochself.

        Parameters
        ----------
        train_loader : :class:`DataLoader`
            A loader which provides the training set.

        Returns
        -------
        None : :class:`NoneType`

        """

        self.epochs += 1
        self.generator.train()
        self.discriminator.train()

        self.mask = mask(train_loader)
        self.d_loss = self.g_loss = 0
        for batch_id, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size()[0]
            real_images = real_images.to('cuda')

            if batch_id % 2 == 0:
                self.d_train_step(batch_size, real_images)
            else:
                self.g_train_step(batch_size, real_images)

            if batch_id % self.args.log_interval == 0:
                msg = (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                    'Discriminator Loss: {:.6f}\t'
                    'Generator Loss: {:.6f}'
                )
                msg = msg.format(
                    self.epochs,
                    batch_id * len(real_images),
                    len(train_loader.dataset),
                    100. * batch_id / len(train_loader),
                    self.d_loss,
                    self.g_loss
                )
                logger.info(msg)

    def eval(self, test_loader):
        """
        Evaluates the GAN performance on a test set.

        Parameters
        ----------
        test_loader : :class:`test_loader`
            A loader which provides the test set.

        Returns
        -------
        None : :class:`NoneType`

        """

        self.generator.eval()
        self.discriminator.eval()

        # Total correct prediction counts.
        correct = inpainted_correct = real_correct = 0

        m = mask(test_loader)
        with torch.no_grad():
            for real_images, _ in test_loader:
                real_images = real_images.to('cuda')
                batch_size = real_images.size()[0]

                # Correct prediction counts in the batch.
                batch_inpainted_correct = batch_real_correct = 0

                # Generate inpainted images.
                inpainted_images = self.generator(real_images*m)

                # Get predictions on real images.
                real_logits = self.discriminator(real_images)
                real_predictions = torch.sigmoid(real_logits).round()

                # Get target labels for real images.
                real_target = torch.ones(batch_size, device='cuda')
                real_target = real_target.view_as(real_predictions)

                # Check number of correct predictions on real images.
                batch_real_correct = (
                    real_predictions.eq(real_target).sum()
                )
                real_correct += batch_real_correct.item()

                # Get predictions on inpainted images.
                inpainted_logits = self.discriminator(inpainted_images)
                inpainted_predictions = (
                    torch.sigmoid(inpainted_logits).round()
                )

                # Get target labels for inpainted images.
                inpainted_target = (
                    torch.zeros(batch_size, device='cuda')
                )
                inpainted_target = (
                    inpainted_target.view_as(inpainted_predictions)
                )

                # Check number of correct predictions on inpainted.
                batch_inpainted_correct = (
                    inpainted_predictions.eq(inpainted_target).sum()
                )
                inpainted_correct += batch_inpainted_correct.item()

                # Get total number of correct predictions.
                correct += real_correct + inpainted_correct

        # Log results.
        msg = (
            '\nTest set: Accuracy: {}/{} ({:.0f}%)'
            '\tInpainted correct: {}\tReal correct: {}\n'
        )
        msg = msg.format(
            correct,
            2*len(test_loader.dataset),
            100. * correct / (2*len(test_loader.dataset)),
            inpainted_correct,
            real_correct
        )
        logger.info(msg)

        # Save some inpainted images.
        real_images, _ = next(iter(test_loader))
        real_images = real_images.to('cuda')
        filename = os.path.join(
            self.args.output_dir,
            'images',
            f'epoch_{self.epochs}_original.jpg'
        )
        save_image(real_images, filename, nrow=10)

        masked_images = m*real_images
        filename = os.path.join(
            self.args.output_dir,
            'images',
            f'epoch_{self.epochs}_masked.jpg'
        )
        save_image(masked_images, filename, nrow=10)

        inpainted_images = self.generator(masked_images)
        filename = os.path.join(
            self.args.output_dir,
            'images',
            f'epoch_{self.epochs}_inpainted.jpg'
        )
        save_image(inpainted_images, filename, nrow=10)


def main():

    ###################################################################
    # Define command line parameters.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        default=4,
        type=int
    )
    parser.add_argument(
        '--train_batch_size',
        default=64,
        type=int
    )
    parser.add_argument(
        '--test_batch_size',
        default=200,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=0.002,
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
        '--encoder_channels',
        default=[1, 256, 256, 512, 1024],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--encoder_kernel_sizes',
        default=[4, 4, 4, 4],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--encoder_strides',
        default=[2, 2, 2, 2],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--encoder_paddings',
        default=[1, 1, 1, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--decoder_channels',
        default=[1024, 512, 256, 256, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--decoder_kernel_sizes',
        default=[4, 4, 4, 4],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--decoder_strides',
        default=[2, 2, 2, 2],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--decoder_paddings',
        default=[1, 1, 1, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--d_channels',
        default=[1, 128, 256, 512, 1024, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--d_kernel_sizes',
        default=[4, 4, 4, 4, 4],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--d_strides',
        default=[2, 2, 2, 2, 1],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--d_paddings',
        default=[1, 1, 1, 1, 0],
        nargs='+',
        type=int
    )
    parser.add_argument(
        '--d_lrelu_alpha',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--label_smoothing',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--database_root',
        default='/home/lukas/databases'
    )
    parser.add_argument(
        '--output_dir',
        default='output'
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
    # Set up output directory.
    ###################################################################

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    os.mkdir(join(args.output_dir, 'images'))
    os.mkdir(join(args.output_dir, 'models'))

    ###################################################################
    # Set random seed.
    ###################################################################

    torch.manual_seed(args.seed)

    ###################################################################
    # Set up data lodaers.
    ###################################################################

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        lambda image: (image - 0.5) * 2
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
        pin_memory=True,
        drop_last=True
    )

    ###################################################################
    # Create the generator.
    ###################################################################

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

    ###################################################################
    # Create the discriminator.
    ###################################################################

    discriminator = Discriminator(
        channels=args.d_channels,
        kernel_sizes=args.d_kernel_sizes,
        strides=args.d_strides,
        paddings=args.d_paddings,
        lrelu_alpha=args.d_lrelu_alpha
    )
    discriminator.to('cuda')

    ###################################################################
    # Create the trainer.
    ###################################################################

    trainer = GANTrainer(
        generator=autoencoder,
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
        filename = (
            join(args.output_dir, 'models', f'g_epoch_{epoch+1}')
        )
        torch.save(autoencoder, filename)
        filename = (
            join(args.output_dir, 'models', f'd_epoch_{epoch+1}')
        )
        torch.save(discriminator, filename)


if __name__ == '__main__':
    main()
