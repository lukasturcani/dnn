import os
import logging
import torch
from torch import nn, optim
from torchvision.utils import save_image
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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

        self.g_optimizer = optim.Adam(generator.parameters(),
                                      lr=args.learning_rate,
                                      betas=(args.beta1, args.beta2))

        self.d_optimizer = optim.Adam(discriminator.parameters(),
                                      lr=args.learning_rate,
                                      betas=(args.beta1, args.beta2))

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

        # Use generator to make fake images.
        noise = torch.randn(batch_size,
                            *self.args.g_noise_shape,
                            device='cuda')

        fake_images = self.generator(noise)

        # Reset accumulated discriminator gradient.
        self.d_optimizer.zero_grad()

        # Get logits.
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images)

        # Get target labels for real images.
        real_target = torch.ones(batch_size, device='cuda')
        real_target -= self.args.label_smoothing
        real_target = real_target.view_as(real_logits)

        # Calculate loss on real iamges.
        d_real_loss = self.criterion(real_logits, real_target)

        # Get target labels for fake images.
        fake_target = torch.zeros(batch_size, device='cuda')
        fake_target = fake_target.view_as(fake_logits)

        # Calculate loss on fake images.
        d_fake_loss = self.criterion(fake_logits, fake_target)

        # Calculate total loss.
        d_loss = d_real_loss + d_fake_loss
        self.d_loss = d_loss.item()

        # Backprop.
        d_loss.backward()
        self.d_optimizer.step()

    def g_train_step(self, batch_size):
        """
        Carries out a single training step on the generator.

        Parameters
        ----------
        batch_size : :class:`int`
            The batch size used in this training step.

        Returns
        -------
        None : :class:`NoneType`

        """

        # Reset accumulated generator gradient.
        self.g_optimizer.zero_grad()

        # Create fake images.
        noise = torch.randn(batch_size,
                            *self.args.g_noise_shape,
                            device='cuda')

        fake_images = self.generator(noise)

        # Get logits.
        fake_logits = self.discriminator(fake_images)

        # Get target labels for fake images.
        fake_target = torch.ones(batch_size, device='cuda')
        fake_target = fake_target.view_as(fake_logits)

        # Calculate generator loss.
        g_loss = self.criterion(fake_logits, fake_target)
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

        for batch_id, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size()[0]
            real_images = real_images.to('cuda')

            self.d_train_step(batch_size, real_images)
            self.g_train_step(batch_size)

            if batch_id % self.args.log_interval == 0:
                msg = ('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                       'Discriminator Loss: {:.6f}\t'
                       'Generator Loss: {:.6f}')
                msg = msg.format(
                                self.epochs,
                                batch_id * len(real_images),
                                len(train_loader.dataset),
                                100. * batch_id / len(train_loader),
                                self.d_loss,
                                self.g_loss)
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
        correct = fake_correct = real_correct = 0

        with torch.no_grad():
            for real_images, _ in test_loader:
                real_images = real_images.to('cuda')
                batch_size = real_images.size()[0]

                # Correct prediction counts in the batch.
                batch_fake_correct = batch_real_correct = 0

                # Generate fake images.
                noise = torch.randn(batch_size,
                                    *self.args.g_noise_shape,
                                    device='cuda')
                fake_images = self.generator(noise)

                # Get predictions on real images.
                real_logits = self.discriminator(real_images)
                real_predictions = torch.sigmoid(real_logits).round()

                # Get target labels for real images.
                real_target = torch.ones(batch_size, device='cuda')
                real_target = real_target.view_as(real_predictions)

                # Check number of correct predictions on real images.
                batch_real_correct = (real_predictions
                                      .eq(real_target)
                                      .sum())
                real_correct += batch_real_correct.item()

                # Get predictions on fake images.
                fake_logits = self.discriminator(fake_images)
                fake_predictions = torch.sigmoid(fake_logits).round()

                # Get target labels for fake images.
                fake_target = torch.zeros(batch_size, device='cuda')
                fake_target = fake_target.view_as(fake_predictions)

                # Check number of correct predictions on fake.
                batch_fake_correct = (fake_predictions
                                      .eq(fake_target)
                                      .sum())
                fake_correct += batch_fake_correct.item()

                # Get total number of correct predictions.
                correct += real_correct + fake_correct

        # Log results.
        msg = ('\nTest set: Accuracy: {}/{} ({:.0f}%)'
               '\tFake correct: {}\tReal correct: {}\n')
        msg = msg.format(correct,
                         2*len(test_loader.dataset),
                         100. * correct / (2*len(test_loader.dataset)),
                         fake_correct,
                         real_correct)
        logger.info(msg)

        # Save some generated images.
        noise = torch.randn(20,
                            *self.args.g_noise_shape,
                            device='cuda')
        g_images = self.generator(noise).view(20, *self.img_shape)
        g_images = F.interpolate(g_images, scale_factor=2.29)

        filename = os.path.join(self.args.img_dir,
                                f'epoch_{self.epochs}.jpg')
        save_image(g_images*0.5 + 0.5, filename, nrow=10)
