import logging
import argparse
import os
import shutil
import torch
from torch import optim, nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dnn.pytorch.models.gan.growing_gan import Discriminator, Generator

logger = logging.getLogger(__name__)


def mnist_loaders(args, img_size):
    """
    Creates new MNIST loaders of given `img_size`.

    Parameters
    ----------
    args : :class:`Namespace`
        The command line arguments provided to the script.

    img_size : :class:`int`
        The output image size.

    Returns
    -------
    :class:`tuple` of :class:`torch.DataLoader`
        Returns the train and test data loaders, respectively.

    """

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
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


def train(args,
          generator,
          discriminator,
          g_optimizer,
          d_optimizer,
          train_loader,
          epoch):

    generator.train()
    discriminator.train()
    criterion = nn.BCEWithLogitsLoss()

    for batch_id, (real_images, _) in enumerate(train_loader):
        real_images = (real_images - 0.5) / 0.5
        batch_size = real_images.size()[0]

        real_images = real_images.to('cuda')

        # Create fake images.
        noise = torch.randn(batch_size,
                            args.latent_space_channels,
                            args.init_img_size,
                            args.init_img_size,
                            device='cuda')
        fake_images = generator(noise)

        ###############################################################
        # Train the discriminator.
        ###############################################################

        d_optimizer.zero_grad()

        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images)

        real_target = torch.ones(batch_size, device='cuda')
        real_target -= args.label_smoothing
        real_target = real_target.view_as(real_logits)
        d_real_loss = criterion(real_logits, real_target)

        fake_target = torch.zeros(batch_size, device='cuda')
        fake_target = fake_target.view_as(fake_logits)
        d_fake_loss = criterion(fake_logits, fake_target)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        ###############################################################
        # Train the generator.
        ###############################################################

        g_optimizer.zero_grad()
        noise = torch.randn(batch_size,
                            args.latent_space_channels,
                            args.init_img_size,
                            args.init_img_size,
                            device='cuda')

        fake_images = generator(noise)
        fake_logits = discriminator(fake_images)

        fake_target = torch.ones(batch_size, device='cuda')
        fake_target = fake_target.view_as(fake_logits)
        g_loss = criterion(fake_logits, fake_target)

        g_loss.backward()
        g_optimizer.step()

        if batch_id % args.log_interval == 0:
            msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                   'Discriminator Loss: {:.6f}\t'
                   'Generator Loss: {:.6f}')
            msg = msg.format(
                epoch,
                batch_id * len(real_images),
                len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                d_loss.item(),
                g_loss.item())
            logger.info(msg)


def test(args, generator, discriminator, test_loader, epoch):
    generator.eval()
    discriminator.eval()
    correct = fake_correct = real_correct = 0

    with torch.no_grad():
        for real_images, _ in test_loader:
            batch_real_correct = batch_fake_correct = 0
            batch_size = real_images.size()[0]

            real_images = real_images.to('cuda')

            noise = torch.randn(batch_size,
                                args.latent_space_channels,
                                args.init_img_size,
                                args.init_img_size,
                                device='cuda')
            fake_images = generator(noise)

            real_logits = discriminator(real_images)
            real_predictions = torch.sigmoid(real_logits).round()
            real_target = torch.ones(batch_size, device='cuda')
            real_target = real_target.view_as(real_predictions)
            batch_real_correct = real_predictions.eq(real_target).sum()
            real_correct += batch_real_correct.item()

            fake_logits = discriminator(fake_images)
            fake_predictions = torch.sigmoid(fake_logits).round()
            fake_target = torch.zeros(batch_size, device='cuda')
            fake_target = fake_target.view_as(fake_predictions)
            batch_fake_correct = fake_predictions.eq(fake_target).sum()
            fake_correct += batch_fake_correct.item()

    correct = real_correct + fake_correct

    msg = ('Test set: Accuracy: {}/{} ({:.0f}%)'
           '\tFake correct: {}\tReal correct: {}\n')
    msg = msg.format(correct,
                     2*len(test_loader.dataset),
                     100. * correct / (2*len(test_loader.dataset)),
                     fake_correct,
                     real_correct)
    logger.info(msg)

    noise = torch.randn(20,
                        args.latent_space_channels,
                        args.init_img_size,
                        args.init_img_size,
                        device='cuda')
    g_images = generator(noise)
    save_image(g_images*0.5 + 0.5,
               os.path.join(args.output_dir,
                            'images',
                            f'epoch_{epoch}_generated.png'))


def main():

    ###################################################################
    # Define command line parameters.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--beta1', default=0.0, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--init_img_size', default=4, type=int)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0, type=float)
    parser.add_argument('--log_interval', default=150, type=int)

    parser.add_argument('--latent_space_channels',
                        default=128,
                        type=int)

    parser.add_argument('--logging_level',
                        default=logging.DEBUG,
                        type=int)

    parser.add_argument('--database_root',
                        default='/home/lukas/databases')

    args = parser.parse_args()

    ###################################################################
    # Set up the output directory.
    ###################################################################

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    os.mkdir(os.path.join(args.output_dir, 'images'))
    os.mkdir(os.path.join(args.output_dir, 'models'))

    ###################################################################
    # Set up logging.
    ###################################################################

    log_fmt = '%(asctime)s - %(levelname)s - %(module)s - %(msg)s'
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(level=args.logging_level,
                        format=log_fmt,
                        datefmt=date_fmt)

    ###################################################################
    # Set random seed.
    ###################################################################

    torch.manual_seed(args.seed)

    ###################################################################
    # Define network parameters.
    ###################################################################

    # The output channels and kernel size of each convolutional layer.
    # Each tuple represents a single convolutional layer.
    # Tuples in the same sublist belong to the same block.
    generator_blocks = [
        [(128, 3), (128, 3)],
        [(128, 3), (128, 3)],
        [(128, 3), (128, 3)],
        [(64, 3), (64, 3)],
        [(64, 3), (64, 3)]
    ]

    # The output channels and kernel size of each convoluational layer.
    # Each tuple represents a single convolutional layer.
    # Tuples in the same sublist belong to the same block.
    discriminator_blocks = [
        [(128, 3)],
        [(128, 3), (128, 3)],
        [(128, 3), (128, 3)],
        [(64, 3), (128, 3)],
        [(64, 3), (64, 3)]
    ]

    # The epochs on which a new block is added to the networks.
    grow_epochs = { 1, 3, 7, 11, 15 }
    # During each epoch, if a new convolutional block is being faded
    # in, the fade_alpha of the networks increases by this amount.
    epoch_fade = 0.5

    ###################################################################
    # Build the networks.
    ###################################################################

    generator = Generator(
                     init_img_size=args.init_img_size,
                     img_channels=1,
                     latent_space_channels=args.latent_space_channels,
                     lrelu_alpha=args.lrelu_alpha)

    discriminator = Discriminator(
                     init_img_size=args.init_img_size,
                     img_channels=1,
                     lrelu_alpha=args.lrelu_alpha)

    ###################################################################
    # Train.
    ###################################################################

    block = 0
    img_size = args.init_img_size
    for epoch in range(1, args.epochs+1):

    ###################################################################

        if epoch in grow_epochs:

            # On the first epoch there is only 1 block so there is
            # nothing to fade in.
            if epoch != 1:
                generator.fading = True
                discriminator.fading = True

                generator.fade_alpha = 0
                discriminator.fade_alpha = 0

            ###########################################################
            # Save the models.
            ###########################################################

            torch.save(
                generator,
                os.path.join(args.output_dir, 'models', f'g_{epoch}'))
            torch.save(
                discriminator,
                os.path.join(args.output_dir, 'models', f'd_{epoch}'))

            ###########################################################
            # Grow the networks.
            ###########################################################

            generator.grow(generator_blocks[block])
            discriminator.grow(discriminator_blocks[block])

            g_optimizer = optim.Adam(generator.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2))
            d_optimizer = optim.Adam(discriminator.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2))

            block += 1

            discriminator.to('cuda')
            generator.to('cuda')

            # Log the new generator architecture.
            logging.debug(generator)
            g_input_size = (args.latent_space_channels,
                            args.init_img_size,
                            args.init_img_size)
            summary(generator, g_input_size)

            # Log the new discrimonator architecture.
            logging.debug(discriminator)
            d_input_size = (1, img_size, img_size)
            summary(discriminator, d_input_size)

            ###########################################################
            # Create new dataset loaders of the correct image size.
            ###########################################################

            train_loader, test_loader = mnist_loaders(args, img_size)

            # Save reference dataset images at the new size.
            save_image(next(iter(test_loader))[0][:20],
                       os.path.join(args.output_dir,
                                    'images',
                                    f'epoch_{epoch}_real.png'))

            ###########################################################
            # Update the image size.
            ###########################################################

            img_size *= 2

    ###################################################################

        if generator.fading:
            generator.fade_alpha += epoch_fade
            discriminator.fade_alpha += epoch_fade

        if generator.fading >= 1:
            generator.fading = False
            discriminator.fading = False

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
