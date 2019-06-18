from tensorflow.python.keras import backend as K
import tensorflow as tf

def acc(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def pre(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    TP = K.sum(tf.multiply(y_true, K.round(y_pred)))
    S = K.sum(K.round(y_pred))
    return TP / S

def rec(y_true, y_pred):
    y_pred = tf.sigmoid(y_pred)
    TP = K.sum(tf.multiply(y_true, K.round(y_pred)))
    S = K.sum(y_true)
    return TP / S

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)



