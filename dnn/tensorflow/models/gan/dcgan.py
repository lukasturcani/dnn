import tensorflow as tf
import numpy as np


def discriminator(ilayer, mode, params, reuse=False):
    """

    """

    training = mode == tf.estimator.ModeKeys.TRAIN
    labels_logits = None

    with tf.variable_scope('discriminator', reuse=reuse):
        d_params = zip(params.d_depths,
                       params.d_kernel_sizes,
                       params.d_strides,
                       params.d_padding)

        prev_layer = ilayer
        for i, (depth, ksize, stride, padding) in enumerate(d_params):
            if i == len(params.d_depths)-1 and params.labels:
                pixels = np.prod(prev_layer.shape.as_list()[1:])
                flat = tf.reshape(prev_layer, [-1, pixels])
                labels_logits = tf.layers.dense(inputs=flat,
                                                units=10,
                                                name='labels_logits')

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

        auth_logits = tf.reshape(prev_layer, [-1, 1])
        return auth_logits, labels_logits


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
            prev_layer = tf.layers.conv2d_transpose(
                                       inputs=prev_layer,
                                       filters=depth,
                                       kernel_size=ksize,
                                       strides=stride,
                                       padding=padding,
                                       use_bias=False,
                                       name=f'tconv_{i}')

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

    if params.labels:
        gen_labels = tf.expand_dims(features['gen_labels'], 1)
        gen_labels = tf.expand_dims(gen_labels, 1)
        g_ilayer = tf.concat(values=[features['noise'], gen_labels],
                             axis=3)
    else:
        g_ilayer = features['noise']

    g_images = generator(
                        ilayer=g_ilayer,
                        mode=mode,
                        params=params)

    d_real_auth_logits, d_real_labels_logits = discriminator(
                        ilayer=features['images'],
                        mode=mode,
                        params=params)

    d_fake_auth_logits, d_fake_labels_logits = discriminator(
                        ilayer=g_images,
                        mode=mode,
                        params=params,
                        reuse=True)

    predictions = {
        'g_images': g_images
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    g_auth_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[0]]*params.batch_size,
                    logits=d_fake_auth_logits)

    g_loss = g_auth_loss

    d_real_auth_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[0]]*params.batch_size,
                    logits=d_real_auth_logits,
                    label_smoothing=params.label_smoothing)

    d_fake_auth_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=[[1]]*params.batch_size,
                    logits=d_fake_auth_logits)

    d_loss = d_real_auth_loss + d_fake_auth_loss

    if params.labels:
        g_labels_loss = tf.losses.softmax_cross_entropy(
                          onehot_labels=features['gen_labels'],
                          logits=d_fake_labels_logits)

        g_loss += g_labels_loss

        d_real_labels_loss = tf.losses.softmax_cross_entropy(
                        onehot_labels=labels,
                        logits=d_real_labels_logits)

        d_fake_labels_loss = tf.losses.softmax_cross_entropy(
                        onehot_labels=features['gen_labels'],
                        logits=d_fake_labels_logits)

        d_loss += d_real_labels_loss + d_fake_labels_loss

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

    eval_metric_ops = {}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=d_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
