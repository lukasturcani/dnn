"""
A GAN where the generator is an autoencoder.

"""

import tensorflow as tf


def discriminator(ilayer, mode, params, reuse=False):
    """

    """

    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('discriminator', reuse=reuse):
        d_params = zip(params.d_depths,
                       params.d_kernel_sizes,
                       params.d_strides,
                       params.d_padding)

        prev_layer = ilayer
        for i, (depth, ksize, stride, padding) in enumerate(d_params):

            prev_layer = tf.layers.conv2d(
                                            inputs=prev_layer,
                                            filters=depth,
                                            kernel_size=ksize,
                                            strides=stride,
                                            padding=padding,
                                            use_bias=False,
                                            name=f'conv_{i}')

            # Don't do batch norm or activation on the first or last
            # layer.
            if i != 0 and i != len(params.d_depths)-1:
                bnorm = tf.layers.batch_normalization(
                                                inputs=prev_layer,
                                                training=training,
                                                name=f'bnorm_{i}')

                prev_layer = tf.nn.leaky_relu(features=bnorm,
                                              alpha=params.lrelu_alpha,
                                              name=f'lrelu_{i}')

        return tf.reshape(prev_layer, [-1, 1])


def generator(ilayer, mode, params):
    """

    """

    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('generator'):
        g_params = zip(params.g_depths,
                       params.g_kernel_sizes,
                       params.g_strides,
                       params.g_padding)

        prev_layer = ilayer
        for i, (depth, ksize, stride, padding) in enumerate(g_params):
            layer = (tf.layers.conv2d if i < params.g_nconvs
                     else tf.layers.conv2d_transpose)
            name = f'tconv_{i}' if i < params.g_nconvs else f'conv_{i}'
            prev_layer = layer(
                               inputs=prev_layer,
                               filters=depth,
                               kernel_size=ksize,
                               strides=stride,
                               padding=padding,
                               use_bias=False,
                               name=name)

            # Don't do batch norm and activation on the last layer.
            if i != len(params.g_depths)-1:
                bnorm = tf.layers.batch_normalization(
                                                inputs=prev_layer,
                                                training=training)

                if i+1 in params.g_dropout_layers:
                    bnorm = tf.layers.dropout(inputs=bnorm,
                                              training=training)

                prev_layer = tf.nn.leaky_relu(features=bnorm,
                                              alpha=params.lrelu_alpha)

        return tf.nn.tanh(prev_layer)


def model_fn(features, labels, mode, params):
    """

    """

    g_images = generator(
                        ilayer=features['masked_images'],
                        mode=mode,
                        params=params)

    d_real_logits = discriminator(
                        ilayer=features['images'],
                        mode=mode,
                        params=params)
    d_real_predictions = tf.sigmoid(d_real_logits)

    d_fake_logits = discriminator(
                        ilayer=g_images,
                        mode=mode,
                        params=params,
                        reuse=True)
    d_fake_predictions = tf.sigmoid(d_fake_logits)

    predictions = {
        'g_images': g_images,
        'd_real_predictions': d_real_predictions,
        'd_fake_predictions': d_fake_predictions
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    g_gen_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[0]]*params.batch_size,
                    logits=d_fake_logits)

    g_mse_loss = tf.losses.mean_squared_error(
                    labels=features['images'],
                    predictions=g_images)

    # When pre-training the generator - only care about the mse loss.
    g_loss = (g_mse_loss if params.training_network == 'g' else
              g_gen_loss + g_mse_loss)

    d_real_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[0]]*params.batch_size,
                    logits=d_real_logits,
                    label_smoothing=params.label_smoothing)

    d_fake_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[1]]*params.batch_size,
                    logits=d_fake_logits)

    d_loss = d_real_loss + d_fake_loss

    trainables = tf.trainable_variables()
    g_vars = [t for t in trainables if t.name.startswith('generator')]
    d_vars = [t for t in trainables if t.name.startswith('discriminator')]

    g_trainer = tf.train.AdamOptimizer(
                            learning_rate=params.learning_rate,
                            beta1=params.beta1)
    d_trainer = tf.train.AdamOptimizer(
                            learning_rate=params.learning_rate,
                            beta1=params.beta1)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_train_op = g_trainer.minimize(
                                loss=g_loss,
                                var_list=g_vars,
                                global_step=tf.train.get_global_step())
        d_train_op = d_trainer.minimize(
                                loss=d_loss,
                                var_list=d_vars,
                                global_step=tf.train.get_global_step())

    train_op = tf.group(g_train_op, d_train_op)

    eval_metric_ops = {
        'd_real_accuracy': tf.metrics.accuracy(
                    labels=[[0]]*params.batch_size,
                    predictions=tf.round(d_real_predictions)),
        'd_fake_accuracy': tf.metrics.accuracy(
                    labels=[[1]]*params.batch_size,
                    predictions=tf.round(d_fake_predictions))
    }

    # Depending of if the generator or discriminator is being
    # pre-trained or if the entire GAN is being trained - select
    # different train ops and losses.
    train_ops = {
        'd': d_train_op,
        'g': g_train_op,
        'both': train_op
    }
    losses = {
        'd': d_loss,
        'g': g_mse_loss,
        'both': d_loss
    }
    return tf.estimator.EstimatorSpec(
                      mode=mode,
                      predictions=predictions,
                      loss=losses[params.training_network],
                      train_op=train_ops[params.training_network],
                      eval_metric_ops=eval_metric_ops)
