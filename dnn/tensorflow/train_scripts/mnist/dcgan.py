import tensorflow as tf
from tensorflow.keras.datasets import mnist
import argparse
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

from dnn.tensorflow.models.gan.dcgan import model_fn


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


def get_input_fn(batch_size, images, labels, params):
    """
    Creates an input function which feeds the network with data.

    Parameters
    ----------
    batch_size : :class:`int`
        The training batch size.

    images : :class:`numpy.array`
        An array of the images belonging to the training set.

    labels : :class:`numpy.array`
        An array of the labels belonging to the training set.

    params : :class:`Namespace`
        Holds the model hyperparameters as attributes.

    Returns
    -------
    :class:`function`
        The input function.

    """

    images_array = (images.reshape([-1, 28, 28, 1])/255 - 0.5) * 2
    labels_array = LabelBinarizer().fit_transform(labels)
    init_hook = InitHook()

    def input_fn():
        images_input = tf.placeholder(dtype=images_array.dtype,
                                      shape=images_array.shape)
        labels_input = tf.placeholder(dtype=labels_array.dtype,
                                      shape=labels_array.shape)

        inputs = (images_input, labels_input)
        dset = tf.data.Dataset.from_tensor_slices(inputs)
        dset = dset.shuffle(images.shape[0]).batch(batch_size,
                                                   drop_remainder=True)
        dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        next_images, next_labels = iterator.get_next()
        next_images = tf.image.resize_images(next_images, [64, 64])
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {images_input: images_array,
                               labels_input: labels_array}

        next_noise = tf.random_normal(shape=[batch_size, 1, 1, 100])
        next_gen_labels = tf.random_uniform(shape=[batch_size],
                                            minval=0,
                                            maxval=10,
                                            dtype=tf.int32)

        next_gen_labels = tf.one_hot(indices=next_gen_labels,
                                     depth=10,
                                     dtype=tf.float32)

        features = {
            'noise': next_noise,
            'gen_labels': next_gen_labels,
            'images': next_images
        }
        return features, next_labels

    return input_fn, init_hook


def sample_generator(params):
    """
    Uses the generator to create new images and saves them.

    Parameters
    ----------
    params : :class:`Namespace`
        The parameters passed to the script.

    Returns
    -------
    None : :class:`NoneType`

    """

    # Create a new input_fn which feeds the generator with data.
    # A custom one is defined here to avoid loading the MNIST dataset.
    # Because all we are doing is sampling the generator, we only
    # need a noise tensor really.

    def input_fn():
        noise = tf.random_normal(shape=[params.num_images, 1, 1, 100])

        # This is ignored if --labels is not used. When --labels is
        # used, the indices argument determines which digits get
        # sampled.
        indices = [i % 10 for i in range(params.num_images)]
        gen_labels = tf.one_hot(indices=indices,
                                depth=10,
                                dtype=tf.float32)
        return {
            'noise': noise,
            'gen_labels': gen_labels,
            # Create a dummy tensor which feeds noise where MNIST
            # images should normally go. As we are only sampling
            # the generator, this should never be used but must exist
            # because the graph defintion requires it.
            'images': tf.random_normal(
                                  shape=[params.num_images, 64, 64, 1])
        }

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       params=params)

    sample = estimator.predict(input_fn=input_fn,
                               predict_keys='g_images',
                               yield_single_examples=False)
    images = next(sample)['g_images'].reshape(-1, 1, 64, 64)
    images = torch.from_numpy(images*0.5 + 0.5)
    images = F.interpolate(images,
                           scale_factor=params.saved_img_scale)
    save_image(images, 'generator_sample.jpg', nrow=10)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_generator', action='store_true')
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--train_steps', default=1_000_000, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--saved_img_scale', default=1., type=float)

    parser.add_argument('--g_dropout_layers',
                        default=[],
                        type=int,
                        nargs='+')

    parser.add_argument('--g_depths',
                        default=[1024, 512, 256, 128, 1],
                        type=int,
                        nargs='+')

    parser.add_argument('--g_kernel_sizes',
                        default=[4, 4, 4, 4, 4],
                        type=int,
                        nargs='+')

    parser.add_argument('--g_strides',
                        default=[1, 2, 2, 2, 2],
                        type=int,
                        nargs='+')

    parser.add_argument('--g_padding',
                        default=['valid',
                                 'same',
                                 'same',
                                 'same',
                                 'same'],
                        nargs='+')

    parser.add_argument('--d_depths',
                        default=[128, 256, 512, 1024, 1],
                        type=int,
                        nargs='+')

    parser.add_argument('--d_kernel_sizes',
                        default=[4, 4, 4, 4, 4],
                        type=int,
                        nargs='+')

    parser.add_argument('--d_strides',
                        default=[2, 2, 2, 2, 1],
                        type=int,
                        nargs='+')

    parser.add_argument('--d_padding',
                        default=['same',
                                 'same',
                                 'same',
                                 'same',
                                 'valid'],
                        nargs='+')

    parser.add_argument('--num_images',
                        default=20,
                        type=int,
                        help=('The number of images to '
                              'save when using --sample_generator.'))

    return parser


def main():
    params = make_parser().parse_args()

    # Check if generation of images requested.
    if params.sample_generator:
        sample_generator(params)
        exit()

    ###################################################################
    # Create the RunConfig.
    ###################################################################

    config = tf.estimator.RunConfig(
                                model_dir=params.model_dir,
                                tf_random_seed=42,
                                save_summary_steps=None,
                                save_checkpoints_steps=None,
                                save_checkpoints_secs=None
    )

    ###################################################################
    # Create the input function.
    ###################################################################

    train_data, _ = mnist.load_data()
    train_images, train_labels = train_data
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.float32)

    input_fn, init_hook = get_input_fn(
                                batch_size=params.batch_size,
                                images=train_images,
                                labels=train_labels,
                                params=params
    )

    ###################################################################
    # Create the estimator.
    ###################################################################

    estimator = tf.estimator.Estimator(
                                model_fn=model_fn,
                                model_dir=params.model_dir,
                                config=config,
                                params=params
    )

    ###################################################################
    # Train.
    ###################################################################

    saver = tf.train.CheckpointSaverHook(
                                checkpoint_dir=params.model_dir,
                                save_steps=params.save_step
    )

    estimator.train(
                               input_fn=input_fn,
                               hooks=[init_hook, saver],
                               max_steps=params.train_steps
    )


if __name__ == '__main__':
    main()
