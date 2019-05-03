import tensorflow as tf
from keras import backend as K
from keras.backend.common import epsilon

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def logit(inputs):
    _epsilon = _to_tensor(epsilon(), inputs.dtype.base_dtype)
    inputs = tf.clip_by_value(inputs, _epsilon, 1 - _epsilon)
    inputs = tf.log(inputs / (1 - inputs))
    return inputs

def tfLaplace(x):
    laplace = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32)
    laplace = tf.reshape(laplace, [3, 3, 1, 1])
    edge = tf.nn.conv2d(x, laplace, strides=[1, 1, 1, 1], padding='SAME')
    edge = tf.nn.relu(tf.tanh(edge))
    return edge

def EdgeLoss(y_true, y_pred):
    y_true_edge = tfLaplace(y_true)
    edge_pos = 2.
    edge_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true_edge,y_pred,edge_pos), axis=-1)
    return edge_loss

def EdgeHoldLoss(y_true, y_pred):
    y_pred2 = tf.sigmoid(y_pred)
    y_true_edge = tfLaplace(y_true)
    y_pred_edge = tfLaplace(y_pred2)
    y_pred_edge = logit(y_pred_edge)
    edge_loss = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_edge,logits=y_pred_edge), axis=-1)
    saliency_pos = 1.12
    saliency_loss = K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,saliency_pos), axis=-1)
    return 0.7*saliency_loss+0.3*edge_loss



