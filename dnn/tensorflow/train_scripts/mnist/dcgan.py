import tensorflow as tf
from tensorflow.keras.datasets import mnist
import argparse
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import pickle
from imageio import imwrite
from skimage.transform import resize

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


def write_image(filename, img):
    img = resize(img, [280, 280])
    img = (img*0.5 + 0.5)*255
    imwrite(filename, np.uint8(img))


def sample_generator(params):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       params=params)

    def input_fn():
        noise = tf.random_normal(shape=[1, 1, 1, 100])
        gen_labels = tf.one_hot(indices=[0],
                                depth=10,
                                dtype=tf.float32)
        return {
            'noise': noise,
            'gen_labels': gen_labels,
            'images': tf.random_normal(shape=[1, 64, 64, 1])
        }

    prediction = estimator.predict(input_fn=input_fn,
                                   predict_keys='g_images')

    if not os.path.exists('pics'):
        os.mkdir('pics')

    for i in range(25):
        write_image(f'pics/eg_{i}.jpg', next(prediction)['g_images'])


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_generator', action='store_true')
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--train_steps', default=1000000, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--lrelu_alpha', default=0.2, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--g_dropout_layers',
                        default=[2, 3],
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
