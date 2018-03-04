import tensorflow as tf
import argparse
import pickle
from sklearn.preprocessing import LabelBinarizer
import os
import numpy as np


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


def get_input_fn(batch_size, dataset_path, repeat):
    """

    """

    with open(dataset_path, 'rb') as f:
        cifar = pickle.load(f, encoding='bytes')
        feature_array = cifar[b'data']
        label_array = cifar[b'coarse_labels']

    feature_array = (feature_array.reshape((-1, 3, 32, 32))
                                  .transpose((0, 2, 3, 1)))
    label_array = LabelBinarizer().fit_transform(label_array)
    init_hook = InitHook()

    def input_fn():
        feature_ph = tf.placeholder(tf.float32,
                                    feature_array.shape,
                                    'images')
        label_ph = tf.placeholder(tf.float32,
                                  label_array.shape,
                                  'labels')
        dset = tf.data.Dataset.from_tensor_slices((feature_ph, label_ph))
        dset = dset.shuffle(10000).batch(batch_size)
        if repeat:
            dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {feature_ph: feature_array,
                               label_ph: label_array}
        next_images, next_labels = iterator.get_next()
        return {'images': next_images}, next_labels

    return input_fn, init_hook


def cnn(images, mode, params):
    """

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
                prev_layer = tf.layers.drouput(inputs=prev_layer,
                                               rate=params.dropout,
                                               training=training)

    return tf.layers.dense(inputs=prev_layer,
                           units=20,
                           name='logits')


def model_fn(features, labels, mode, params):
    """

    """

    logits = cnn(features['images'], mode, params)
    predictions = tf.nn.softmax(logits=logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    trainer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = trainer.minimize(
                               loss=loss,
                               global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
                            labels=tf.argmax(labels, 1),
                            predictions=tf.argmax(predictions, 1))}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar100_path',
                        default='/home/lukas/datasets/cifar-100-python')
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--train_steps', default=1000000, type=int)
    parser.add_argument('--save_summary_steps', default=1000, type=int)
    parser.add_argument('--save_checkpoints_secs', default=60, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--convolution_sizes',
                        default=[5, 5],
                        type=int,
                        nargs='+')
    parser.add_argument('--convolution_depths',
                        default=[32, 64],
                        type=int,
                        nargs='+')
    parser.add_argument('--pool_size', default=2, type=int)
    parser.add_argument('--fc_layers',
                        default=[1000, 100],
                        type=int,
                        nargs='+')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=False, type=bool)

    params = parser.parse_args()

    config = tf.estimator.RunConfig(
                    model_dir=params.model_dir,
                    tf_random_seed=42,
                    save_summary_steps=params.save_summary_steps,
                    save_checkpoints_secs=params.save_checkpoints_secs)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       config=config,
                                       params=params)

    train_input_fn, train_init_hook = get_input_fn(
            batch_size=params.train_batch_size,
            dataset_path=os.path.join(params.cifar100_path, 'train'),
            repeat=True)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.train_steps,
                                        hooks=[train_init_hook])

    eval_input_fn, eval_init_hook = get_input_fn(
        batch_size=params.eval_batch_size,
        dataset_path=os.path.join(params.cifar100_path, 'test'),
        repeat=False)
    eval_spec = tf.estimator.EvalSpec(
                  input_fn=eval_input_fn,
                  steps=None,
                  hooks=[eval_init_hook],
                  start_delay_secs=params.save_checkpoints_secs,
                  throttle_secs=params.save_checkpoints_secs)

    estimator.train(input_fn=train_input_fn,
                    hooks=[train_init_hook],
                    max_steps=1)
    estimator.evaluate(input_fn=eval_input_fn,
                       hooks=[eval_init_hook])

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)


if __name__ == '__main__':
    main()
