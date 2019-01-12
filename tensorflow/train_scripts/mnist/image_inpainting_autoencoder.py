import tensorflow as tf
import argparse
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import pickle
from skimage.transform import resize
from imageio import imwrite

from dnn.tensorflow.models.gan.autoencoder import model_fn


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

        d1, d2, d3 = img_shape = next_images.shape.as_list()[1:]
        mask = np.ones(img_shape)
        mask[d1//2:, :, :] = 0.
        mask = mask == 1.

        next_masked_images = next_images * mask

        features = {
            'images': next_images,
            'masked_images': next_masked_images
        }
        return features, next_labels

    return input_fn, init_hook


def write_image(filename, img):
    img = img*0.5 + 0.5
    img = np.uint8(img*255)
    imwrite(filename, img)


def write_masked_img(filename, img):
    img = img.reshape([28, 28, 1])
    img = resize(np.array(img), [64, 64])
    d1, d2, d3 = img.shape
    img[d1//2:, :, :] = 0
    imwrite(filename, img)


def write_original_img(filename, img):
    img = img.reshape([28, 28, 1])
    img = resize(img, [64, 64])
    imwrite(filename, img)


def sample_generator(params):

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       params=params)

    _, test_data = mnist.load_data()
    test_images, test_labels = test_data
    test_images = test_images.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    input_fn, init_hook = get_input_fn(
                                batch_size=1,
                                images=test_images,
                                labels=test_labels,
                                params=params
    )

    prediction = estimator.predict(input_fn,
                                   predict_keys='g_images',
                                   hooks=[init_hook])

    if not os.path.exists('pics'):
        os.mkdir('pics')
    for i in range(25):
        filename = f'pics/{i}_restored.jpg'

        write_image(filename, next(prediction)['g_images'])
        write_masked_img(f'pics/{i}_masked.jpg', test_images[i])
        write_original_img(f'pics/{i}_original.jpg', test_images[i])


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--d_pretrain_steps', default=3000, type=int)
    parser.add_argument('--g_pretrain_steps', default=3000, type=int)
    parser.add_argument('--both_train_steps',
                        default=1_000_000,
                        type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)
    parser.add_argument('--g_nconvs', default=4, type=int)
    parser.add_argument('--g_dropout_layers',
                        default=[5, 6],
                        type=int,
                        nargs='+')
    parser.add_argument('--g_depths',
                        default=[256, 256, 512, 1024,
                                 512, 256, 256, 1],
                        type=int,
                        nargs='+')
    parser.add_argument('--g_kernel_sizes',
                        default=[4, 4, 4, 4,
                                 4, 4, 4, 4],
                        type=int,
                        nargs='+')
    parser.add_argument('--g_strides',
                        default=[2, 2, 2, 2,
                                 2, 2, 2, 2],
                        type=int,
                        nargs='+')
    parser.add_argument('--g_padding',
                        default=['same', 'same', 'same', 'same',
                                 'same', 'same', 'same', 'same'],
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
    parser.add_argument('--sample_generator', action='store_true')

    return parser


def main():
    params = make_parser().parse_args()

    # Check if generation of images requested.
    if params.sample_generator:
        sample_generator(params)
        exit()

    ###################################################################
    # Save the hyperparameters.
    ###################################################################

    if not os.path.exists(params.model_dir):
        os.mkdir(params.model_dir)

    with open(f'{params.model_dir}/hyperparams.pkl', 'wb') as f:
        pickle.dump(params, f)

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
    # Pre-train discriminator.
    ###################################################################

    params.training_network = 'd'
    params.train_steps = params.d_pretrain_steps

    # Create the estimator.
    estimator = tf.estimator.Estimator(
                                model_fn=model_fn,
                                model_dir=params.model_dir,
                                config=config,
                                params=params
    )

    saver = tf.train.CheckpointSaverHook(
                                checkpoint_dir=params.model_dir,
                                save_steps=params.save_step
    )

    # Pre-train.
    estimator.train(
                               input_fn=input_fn,
                               hooks=[init_hook, saver],
                               max_steps=params.train_steps
    )

    ###################################################################
    # Pre-train generator.
    ###################################################################

    params.training_network = 'g'
    params.train_steps = params.g_pretrain_steps

    # Create the estimator.
    estimator = tf.estimator.Estimator(
                                model_fn=model_fn,
                                model_dir=params.model_dir,
                                config=config,
                                params=params
    )

    saver = tf.train.CheckpointSaverHook(
                                checkpoint_dir=params.model_dir,
                                save_steps=params.save_step
    )

    # Pre-train.
    estimator.train(
                               input_fn=input_fn,
                               hooks=[init_hook, saver],
                               max_steps=params.train_steps
    )

    ###################################################################
    # Train
    ###################################################################

    params.training_network = 'both'
    params.train_steps = params.both_train_steps

    # Create the estimator.
    estimator = tf.estimator.Estimator(
                                model_fn=model_fn,
                                model_dir=params.model_dir,
                                config=config,
                                params=params
    )

    saver = tf.train.CheckpointSaverHook(
                                checkpoint_dir=params.model_dir,
                                save_steps=params.save_step
    )

    # Pre-train.
    estimator.train(
                               input_fn=input_fn,
                               hooks=[init_hook, saver],
                               max_steps=params.train_steps
    )


if __name__ == '__main__':
    main()
