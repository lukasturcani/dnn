import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import argparse
from sklearn.preprocessing import LabelBinarizer

from dnn.tensorflow.models.cnn.simple_cnn import model_fn


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


def get_input_fn(batch_size, images, labels, train):
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

    train : :class:`bool`
        Toggles between getting the train and eval input functions.

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
        dset = dset.shuffle(len(images_array)).batch(batch_size)
        if train:
            dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        next_images, next_labels = iterator.get_next()
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {images_input: images_array,
                               labels_input: labels_array}
        return {'images': next_images}, next_labels

    return input_fn, init_hook


def main():

    ###################################################################
    # Set the hyperparameters.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--save_summary_steps', default=30, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--save_checkpoints_steps',
                        default=1000,
                        type=int)
    parser.add_argument('--save_checkpoints_secs', default=5, type=int)
    parser.add_argument('--training_batch_size', default=50, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--training_steps', default=50000, type=int)
    parser.add_argument('--convolution_sizes',
                        default=[5, 5],
                        nargs='+',
                        type=int)
    parser.add_argument('--convolution_depths',
                        default=[32, 64],
                        nargs='+',
                        type=int)
    parser.add_argument('--pool_size', default=2, type=int)
    parser.add_argument('--fc_layers',
                        default=[1024, 100],
                        nargs='+',
                        type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', action='store_true')

    params = parser.parse_args()
    params.output_size = 10

    ###################################################################
    # Create the RunConfig.
    ###################################################################

    run_config = tf.estimator.RunConfig(
                tf_random_seed=params.random_seed,
                save_summary_steps=params.save_summary_steps,
                save_checkpoints_steps=params.save_checkpoints_steps,
                keep_checkpoint_max=params.keep_checkpoint_max)

    ###################################################################
    # Create the input functions.
    ###################################################################

    train_data, test_data = mnist.load_data()

    train_images, train_labels = train_data
    train_images = train_images.astype(np.float32)
    train_labels = train_labels.astype(np.float32)

    test_images, test_labels = test_data
    test_images = test_images.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    train_input_fn, init_training_set = get_input_fn(
                            batch_size=params.training_batch_size,
                            images=train_images,
                            labels=train_labels,
                            train=True)
    eval_input_fn, init_test_set = get_input_fn(
                            batch_size=params.eval_batch_size,
                            images=test_images,
                            labels=test_labels,
                            train=False)

    ###################################################################
    # Create the estimator.
    ###################################################################

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       config=run_config,
                                       params=params)

    ###################################################################
    # Create the TrainSpec and EvalSpec.
    ###################################################################

    train_spec = tf.estimator.TrainSpec(
                                    input_fn=train_input_fn,
                                    max_steps=params.training_steps,
                                    hooks=[init_training_set])

    eval_spec = tf.estimator.EvalSpec(
                        input_fn=eval_input_fn,
                        steps=None,
                        hooks=[init_test_set],
                        start_delay_secs=params.save_checkpoints_secs,
                        throttle_secs=params.save_checkpoints_secs)

    ###################################################################
    # Train.
    ###################################################################

    estimator.train(input_fn=train_input_fn,
                    hooks=[init_training_set],
                    max_steps=1)
    estimator.evaluate(input_fn=eval_input_fn,
                       hooks=[init_test_set])
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)


if __name__ == '__main__':
    main()
