import tensorflow as tf


def jaccard(y_true, y_pred):
    y_true_ = tf.reshape(y_true, [-1])
    y_pred_ = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_, y_pred_))
    return (intersection + 1.0) / (tf.reduce_sum(y_true_) + tf.reduce_sum(y_pred_) - intersection + 1.0)
