import tensorflow as tf


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("model_dir", "./model", "")
flags.DEFINE_integer("hidden_size", 20, "")


def input_fn():
    train, test = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(train)
    def transform(feature, label):
        feature = tf.reshape(feature, [-1])
        feature = tf.to_float(feature) / 255.
        label = tf.to_int32(label)
        return feature, label
    dataset = dataset.map(transform)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def hidden_layer(inputs, trainable=True):
    """This layer is pretrained and is used in the mnist_ext model"""
    outputs = tf.layers.dense(inputs, FLAGS.hidden_size, trainable=trainable, name="hidden")
    return outputs


def model_fn(features, labels, mode):
    with tf.variable_scope("mnist"):
        outputs = hidden_layer(features)
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
    estimator = tf.estimator.Estimator(model_fn, FLAGS.model_dir)
    estimator.train(input_fn, max_steps=100)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
