
"""
Builds a vanilla CNN using tensorflow's high level API.

"""

import tensorflow as tf
import numpy as np


def cnn(images, mode, params):
    """
    Assembles a CNN model.

    Parameters
    ----------
    images : :class:`tf.Tensor`
        The tensor which feeds the images.

    mode : :class:`str`
        The mode of the CNN, i.e. training or eval. Use tensorflow
        defined values, for example: `tf.estimator.ModeKeys.TRAIN` or
        `tf.estimator.ModeKeys.EVAL`.

    params : :class:`Namespace`
        Holds the model hyperparameters as attributes.

    Returns
    -------
    :class:`tf.Tensor`
        The logits tensor of the CNN.

    """

    training = mode == tf.estimator.ModeKeys.TRAIN
    prev_layer = images
    for i, (size, depth) in enumerate(zip(params.convolution_sizes,
                                          params.convolution_depths)):
        with tf.variable_scope(f'convolutional_layer_{i}'):
            conv = tf.layers.conv2d(
                        inputs=prev_layer,
                        filters=depth,
                        kernel_size=size,
                        padding='same',
                        use_bias=False if params.batch_norm else True,
                        name='convolution')

            if params.batch_norm:
                conv = tf.layers.batch_normalization(
                                                inputs=conv,
                                                training=training)

            prev_layer = tf.layers.max_pooling2d(
                                           inputs=tf.nn.relu(conv),
                                           pool_size=params.pool_size,
                                           strides=2,
                                           name='pooling')

    pixels = np.prod(prev_layer.shape.as_list()[1:])
    prev_layer = tf.reshape(prev_layer, [-1, pixels])
    for i, size in enumerate(params.fc_layers):
        with tf.variable_scope(f'fully_connected_{i}'):
            prev_layer = tf.layers.dense(
                         inputs=prev_layer,
                         units=size,
                         use_bias=False if params.batch_norm else True)

            if params.batch_norm:
                prev_layer = tf.layers.batch_normalization(
                                                    inputs=prev_layer,
                                                    training=training)

            prev_layer = tf.nn.relu(prev_layer)

            if params.dropout:
                prev_layer = tf.layers.dropout(inputs=prev_layer,
                                               rate=params.dropout,
                                               training=training)

    return tf.layers.dense(inputs=prev_layer,
                           units=params.output_size,
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

    logits = cnn(images=features['images'], mode=mode, params=params)
    predictions = tf.nn.softmax(logits=logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    trainer = tf.train.AdamOptimizer(
                            learning_rate=params.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = trainer.minimize(
                               loss=loss,
                               global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
                            labels=tf.argmax(labels, 1),
                            predictions=tf.argmax(predictions, 1))
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
