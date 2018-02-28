"""
Builds and trains vanilla CNN using tensorflow's high level API.

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


def get_input_fn(batch_size, images, labels, repeat):
    """
    Creates an input function which feeds the training set.

    Parameters
    ----------
    batch_size : :class:`int`
        The training batch size.

    images : :class:`numpy.array`
        An array of the images belonging to the training set.

    labels : :class:`numpy.array`
        An array of the labels belonging to the training set.

    repeat : :class:`bool`
        Toggles if the dataset should run in an endless loop.

    Returns
    -------
    :class:`function`
        The input function.

    """

    images_array = images.reshape([-1, 28, 28, 1])
    labels_array = LabelBinarizer().fit_transform(labels)
    init_hook = InitHook()

    def input_fn():
        images_ph = tf.placeholder(images_array.dtype, images_array.shape)
        labels_ph = tf.placeholder(labels_array.dtype, labels_array.shape)

        dset = tf.data.Dataset.from_tensor_slices((images_ph, labels_ph))
        dset = dset.shuffle(10000).batch(batch_size)
        if repeat:
            dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        next_images, next_labels = iterator.get_next()
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {images_ph: images_array,
                               labels_ph: labels_array}
        return {'images': next_images}, next_labels

    return input_fn, init_hook


def cnn(images, convolution_sizes, convolution_depths, pool_size, fc_layers):
    """
    Assembles a CNN model.

    Parameters
    ----------
    images : :class:`tf.Tensor`
        The tensor which feeds the images.

    convolution_sizes : :class:`list` of :class:`int`
        Holds the size of every convolutional layer. Must have equal
        length to `convolution_depths`.

    convolution_depths : :class:`list` of :class:`int`
        Holds the depth of every convolutional layer. Must have equal
        length to `convolution_sizes`.

    pool_size : :class:`int`
        The size of the max pooling layer.

    fc_layers : :class:`list` of :class:`int`
        Each number represents the number of neurons in a fully
        connected layer.

    Returns
    -------
    :class:`tf.Tensor`
        The logits tensor of the CNN.

    """

    prev_layer = images
    for i, (size, depth) in enumerate(zip(convolution_sizes,
                                          convolution_depths)):
        with tf.variable_scope(f'convolutional_layer_{i}'):
            conv = tf.layers.conv2d(inputs=prev_layer,
                                    filters=depth,
                                    kernel_size=size,
                                    padding='same',
                                    activation=tf.nn.relu,
                                    name='convolution')
            pool = tf.layers.max_pooling2d(inputs=conv,
                                           pool_size=pool_size,
                                           strides=2,
                                           name='pooling')
            prev_layer = pool

    pixels = np.prod(prev_layer.shape.as_list()[1:])
    prev_layer = tf.reshape(prev_layer, [-1, pixels])
    for i, size in enumerate(fc_layers):
        prev_layer = tf.layers.dense(inputs=prev_layer,
                                     units=size,
                                     activation=tf.nn.relu,
                                     name=f'fully_connected_{i}')

    return tf.layers.dense(inputs=prev_layer,
                           units=10,
                           name='logits')


def model_fn(features, labels, mode, params):
    """
    Defines the architecture.

    Parameters
    ----------
    features : :class:`dict`
        Maps the name of a feature to the :class:`tf.Tensor` which
        holds it.

    labels : :class:`tf.Tensor`
        Contains the labels passed to the model via the input
        functions.

    mode : :class:`str`
        The mode in which the estimator is to be executed.

    params : :class:`tf.contrib.training.HParams`
        Holds the hyperparameters for the model.

    Returns
    -------
    :class:`tf.EstimatorSpec`
        Defines the model.


    """

    logits = cnn(images=features['images'],
                 convolution_sizes=params.convolution_sizes,
                 convolution_depths=params.convolution_depths,
                 pool_size=params.pool_size,
                 fc_layers=params.fc_layers)
    predictions = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    trainer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    train_op = trainer.minimize(loss=loss,
                                global_step=tf.train.get_global_step())

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
                        labels=tf.argmax(labels, axis=1),
                        predictions=tf.argmax(predictions, axis=1))}

    return tf.estimator.EstimatorSpec(
                                    mode=mode,
                                    predictions=predictions,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops=eval_metric_ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--save_summary_steps', default=30, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--save_checkpoints_steps', default=1000, type=int)
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

    params = parser.parse_args()

    run_config = tf.estimator.RunConfig(
                tf_random_seed=params.random_seed,
                save_summary_steps=params.save_summary_steps,
                save_checkpoints_steps=params.save_checkpoints_steps,
                keep_checkpoint_max=params.keep_checkpoint_max)

    mnist = input_data.read_data_sets('mnist')
    train_input_fn, init_training_set = get_input_fn(
                            batch_size=params.training_batch_size,
                            images=mnist.train.images,
                            labels=mnist.train.labels,
                            repeat=True)
    eval_input_fn, init_test_set = get_input_fn(
                            batch_size=params.eval_batch_size,
                            images=mnist.test.images,
                            labels=mnist.test.labels,
                            repeat=False)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       config=run_config,
                                       params=params)

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
