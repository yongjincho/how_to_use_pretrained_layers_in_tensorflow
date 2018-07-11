import tensorflow as tf

import mnist


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer("hidden_size2", 10, "")
flags.DEFINE_string("model_ext_dir", "./model_ext", "")
flags.DEFINE_bool("freeze", True, "")


def model_fn(features, labels, mode):
    with tf.variable_scope("mnist"):
        # The pretrained layer is frozen by setting trainable to False.
        x = mnist.hidden_layer(features, trainable=(not FLAGS.freeze))

    with tf.variable_scope("mnist_ext"):
        outputs = tf.layers.dense(x, FLAGS.hidden_size)
        logits = tf.layers.dense(outputs, 10)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    predictions = tf.argmax(logits, -1)
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = tf.contrib.layers.optimize_loss(loss, None, None, optimizer)

    return tf.estimator.EstimatorSpec(
        loss=loss,
        predictions=predictions,
        train_op=train_op,
        mode=mode)


def main(argv):
    warmup = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.model_dir,
            vars_to_warm_start=[
                "mnist/hidden/kernel",
                "mnist/hidden/bias",
            ])

    estimator = tf.estimator.Estimator(
            model_fn,
            FLAGS.model_ext_dir,
            warm_start_from=warmup)

    estimator.train(mnist.input_fn, max_steps=100)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
