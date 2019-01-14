import tensorflow as tf
import argparse
from imageio import imwrite
import numpy as np
from skimage.transform import resize
import os
import shutil
from importlib import import_module
import pickle


def write_image(filename,
                img,
                real_image_shape,
                output_image_shape):

    real_shape = list(real_image_shape)
    output_shape = list(output_image_shape)

    real_shape[0], real_shape[2] = real_shape[2], real_shape[0]
    output_shape[0], output_shape[2] = output_shape[2], output_shape[0]

    img = img.reshape(real_shape)
    img = resize(img, output_shape)
    img = (img*0.5 + 0.5)*255
    imwrite(filename, np.uint8(img))


def get_input_fn(noise_shape, gan_image_shape):

    def input_fn():

        gen_labels = tf.one_hot(indices=[9],
                                depth=10,
                                dtype=tf.float32)

        return {
            'noise': tf.random_normal(shape=[1, *noise_shape]),
            'gen_labels': gen_labels,
            'images': tf.random_normal(shape=[1, *gan_image_shape])
        }

    return input_fn


def main():

    ###################################################################
    # Define inputs.
    ###################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('model_module',
                        help=('The absolute path to the module which '
                              'defines the model_fn of the GAN. For '
                              'example, '
                              '"dnn.tensorflow.models.gan.fcgan".'))

    parser.add_argument('model_dir',
                        help='The training output folder.')

    parser.add_argument('output_image_dir',
                        default='pics',
                        help=('The directory into which the generated'
                              'images are saved.'))

    parser.add_argument('--num_images', default=25, type=int)

    parser.add_argument('--noise_shape',
                        default=[100],
                        nargs='+',
                        type=int)

    parser.add_argument('--real_image_shape',
                        help='The shape of the real images.',
                        default=[1, 28, 28],
                        nargs='+',
                        type=int)

    parser.add_argument('--gan_image_shape',
                        help=('The shape of the images the generator '
                              'produces.'),
                        default=[28*28],
                        nargs='+',
                        type=int)

    parser.add_argument('--output_image_shape',
                        help='The shape of the saved images',
                        default=[1, 280, 280],
                        nargs='+',
                        type=int)

    params = parser.parse_args()

    ###################################################################
    # Create the estimator.
    ###################################################################

    # Load the module which holds the model_fn.
    model_module = import_module(params.model_module)

    # Load a file holding the hyperparams of the GAN.
    with open(f'{params.model_dir}/hyperparams.pkl', 'rb') as f:
        hparams = pickle.load(f)

    # Create the esimator.
    estimator = tf.estimator.Estimator(model_fn=model_module.model_fn,
                                       model_dir=params.model_dir,
                                       params=hparams)

    ###################################################################
    # Sample images from generator.
    ###################################################################

    input_fn = get_input_fn(noise_shape=params.noise_shape,
                            gan_image_shape=params.gan_image_shape)
    prediction = estimator.predict(input_fn=input_fn,
                                   predict_keys='g_images')

    if os.path.exists(params.output_image_dir):
        shutil.rmtree(params.output_image_dir)
    os.mkdir(params.output_image_dir)

    ###################################################################
    # Create the output folder.
    ###################################################################

    if os.path.exists(params.output_image_dir):
        shutil.rmtree(params.output_image_dir)
    os.mkdir(params.output_image_dir)

    ###################################################################
    # Save the generated images.
    ###################################################################

    for i in range(1, params.num_images+1):
        write_image(filename=f'{params.output_image_dir}/{i}.jpg',
                    img=next(prediction)['g_images'],
                    real_image_shape=params.real_image_shape,
                    output_image_shape=params.output_image_shape)


if __name__ == '__main__':
    main()
