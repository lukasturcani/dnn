import tensorflow as tf


def discriminator(ilayer, mode, params, reuse=False):
    """
    """

    labels_logits = None

    with tf.variable_scope('discriminator', reuse=reuse):

        prev_layer = ilayer
        for i, size in enumerate(params.d_fc_layers):
            prev_layer = tf.layers.dense(
                     inputs=prev_layer,
                     units=size,
                     activation=lambda x:
                     tf.nn.leaky_relu(x, alpha=params.lrelu_alpha),
                     name=f'fc_{i}')

        auth_logits = tf.layers.dense(inputs=prev_layer,
                                      units=1,
                                      name='auth_logits')
        if params.labels:
            labels_logits = tf.layers.dense(inputs=prev_layer,
                                            units=10,
                                            name='labels_logits')
        return auth_logits, labels_logits


def generator(ilayer, mode, params):
    """
    """

    with tf.variable_scope('generator'):

        prev_layer = ilayer
        for i, size in enumerate(params.g_fc_layers):
            prev_layer = tf.layers.dense(
                             inputs=prev_layer,
                             units=size,
                             activation=lambda x:
                             tf.nn.leaky_relu(x, alpha=params.lrelu_alpha),
                             name=f'fc_{i}')

        return tf.layers.dense(inputs=prev_layer,
                               units=28*28,
                               activation=tf.nn.tanh,
                               name='g_images')


def model_fn(features, labels, mode, params):
    """

    """

    if params.labels:
        g_ilayer = tf.concat(values=[features['noise'],
                                     features['gen_labels']],
                             axis=1)
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
