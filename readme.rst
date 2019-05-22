:author: Lukas Turcani

.. image::

This repo provides a number of network architectures, implemented in
PyTorch, as well as scripts used to run them.

To train a network, run the appropriate run script. For example::

    $ python -m dnn.train_scripts.mnist.progressive_growing_gan

Notice that the scripts are run from the module level using the
``python -m`` flag.

Networks can be modified using command line arguments, for example::

    $ python -m dnn.train_scripts.mnist.simple_cnn --conv_in_channels 1 20 50 --conv_out_channels 20 50 60 --conv_kernel_size 5 5 5 --conv_strides 1 1 1 --conv_paddings 0 0 0 --conv_dilations 1 1 1 --pool_kernel_sizes 2 2 2 --pool_strides 2 2 2 --pool_paddings 0 0 0 --pool-dilations 1 1 1 --train_batch_size 100 --label_smoothing 0.5 --epochs 10

changes the default values for training batch size, label smoothing and
the number of training epochs and adds a new convolution layer and pooling
layer vs the default network. Any number of layers can be added / removed
via the command line in this fashion.

Each script can have options viewed by::

    $ python -m path.to.train.script --help


Results
=======

This is a short summary of some of the nice results from this repo.
Not all implemented architectures are listed here.

* `Progressive Growing GAN (PGGAN)`_
* `Image Inpainting`_
* `DCGAN`_
* `FCGAN`_

Progressive Growing GAN (PGGAN)
-------------------------------

This is a somewhat complex GAN, largely due to a unique training
procedure where the resolution of the GAN is slowly increased during
training.

Run with::

    $ python -m dnn.train_scripts.mnist.progressive_growing_gan

Results:

.. image:: images/pggan.gif

Image Inpainting
----------------

This is a task where the generator is provided with an image that
is missing some pixels and it is asked to fill them in. In this
example, I cover up the bottom half of MNIST images and get the
generator to fill them in. The generator used an autoencoder
architecture.

Run with::

    $ python -m dnn.train_scripts.mnist.image_inpainting

Results:

.. image:: images/mnist_inpainting.jpg

DCGAN
-----

A more advanced GAN architecture, which is fully convolutional.

Run with::

    $ python -m dnn.train_scripts.mnist.dcgan

Results:

.. image:: images/mnist_dcgan.jpg

FCGAN
-----

This is a vanilla GAN using feed-forward networks as both the
generator and discriminator.

Run with::

    $ python -m dnn.train_scripts.mnist.fcgan

Results:

.. image:: images/mnist_fcgan.jpg
