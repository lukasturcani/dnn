import tensorflow as tf
import argparse
import pickle
from sklearn.preprocessing import LabelBinarizer
import os

from dnn.tensorflow.models.cnn.simple_cnn import model_fn


class InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(self.init, self.feed_dict)


def get_input_fn(batch_size, dataset_path, train):
    """
    Creates an input function which feeds the training set.

    Parameters
    ----------
    batch_size : :class:`int`
        The training batch size.

    database_path : :class:`str`
        The path to the CIFAR-100 folder.

    train : :class:`bool`
        Toggles between getting the train and eval input functions.

    Returns
    -------
    :class:`function`
        The input function.

    """

    with open(dataset_path, 'rb') as f:
        cifar = pickle.load(f, encoding='bytes')
        images_array = cifar[b'data']
        labels_array = cifar[b'coarse_labels']

    images_array = (images_array.reshape((-1, 3, 32, 32))
                                .transpose((0, 2, 3, 1)))
    labels_array = LabelBinarizer().fit_transform(labels_array)
    init_hook = InitHook()

    def input_fn():
        images_input = tf.placeholder(dtype=tf.float32,
                                      shape=images_array.shape,
                                      name='images')
        labels_input = tf.placeholder(dtype=tf.float32,
                                      shape=labels_array.shape,
                                      name='labels')

        inputs = (images_input, labels_input)
        dset = tf.data.Dataset.from_tensor_slices(inputs)
        dset = dset.shuffle(len(images_array)).batch(batch_size)
        if train:
            dset = dset.repeat()

        iterator = dset.make_initializable_iterator()
        init_hook.init = iterator.initializer
        init_hook.feed_dict = {images_input: images_array,
                               labels_input: labels_array}
        next_images, next_labels = iterator.get_next()
        return {'images': next_images}, next_labels

    return input_fn, init_hook


def main():

    ###################################################################
    # Set the hyperparameters.
    ###################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
                    '--cifar100_path',
                    default='/home/lukas/datasets/cifar-100-python')
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--model_dir', default='output')
    parser.add_argument('--train_steps', default=1000000, type=int)
    parser.add_argument('--save_summary_steps', default=1000, type=int)
    parser.add_argument('--save_checkpoints_secs',
                        default=60,
                        type=int)
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
    parser.add_argument('--batch_norm', action='store_true')

    params = parser.parse_args()
    params.output_size = 20

    ###################################################################
    # Create the RunConfig.
    ###################################################################

    config = tf.estimator.RunConfig(
                    model_dir=params.model_dir,
                    tf_random_seed=42,
                    save_summary_steps=params.save_summary_steps,
                    save_checkpoints_secs=params.save_checkpoints_secs)

    ###################################################################
    # Create the input functions.
    ###################################################################

    train_input_fn, train_init_hook = get_input_fn(
            batch_size=params.train_batch_size,
            dataset_path=os.path.join(params.cifar100_path, 'train'),
            train=True)

    eval_input_fn, eval_init_hook = get_input_fn(
        batch_size=params.eval_batch_size,
        dataset_path=os.path.join(params.cifar100_path, 'test'),
        train=False)

    ###################################################################
    # Create the estimator.
    ###################################################################

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.model_dir,
                                       config=config,
                                       params=params)

    ###################################################################
    # Create the TrainSpec and EvalSpec.
    ###################################################################

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.train_steps,
                                        hooks=[train_init_hook])

    eval_spec = tf.estimator.EvalSpec(
                  input_fn=eval_input_fn,
                  steps=None,
                  hooks=[eval_init_hook],
                  start_delay_secs=params.save_checkpoints_secs,
                  throttle_secs=params.save_checkpoints_secs)

    ###################################################################
    # Train.
    ###################################################################

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
