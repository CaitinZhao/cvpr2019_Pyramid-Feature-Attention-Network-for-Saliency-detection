import tensorflow as tf
from keras.engine import Layer
from keras.layers import *
from bilinear_upsampling import BilinearUpsampling

class BatchNorm(BatchNormalization):
    def call(self, inputs, training=None):
          return super(self.__class__, self).call(inputs, training=True)

def BN(input_tensor,block_id):
    bn = BatchNorm(name=block_id+'_BN')(input_tensor)
    a = Activation('relu',name=block_id+'_relu')(bn)
    return a

def l1_reg(weight_matrix):
    return K.mean(weight_matrix)

class Repeat(Layer):
    def __init__(self,repeat_list, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.repeat_list = repeat_list

    def call(self, inputs):
        outputs = tf.tile(inputs, self.repeat_list)
        return outputs
    def get_config(self):
        config = {
            'repeat_list': self.repeat_list
        }
        base_config = super(Repeat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = [None]
        for i in xrange(1,len(input_shape)):
            output_shape.append(input_shape[i]*self.repeat_list[i])
        return tuple(output_shape)

def SpatialAttention(inputs,name):
    k = 9
    H, W, C = map(int,inputs.get_shape()[1:])
    attention1 = Conv2D(C // 2, (1, k), padding='same', name=name+'_1_conv1')(inputs)
    attention1 = BN(attention1,'attention1_1')
    attention1 = Conv2D(1, (k, 1), padding='same', name=name + '_1_conv2')(attention1)
    attention1 = BN(attention1, 'attention1_2')
    attention2 = Conv2D(C // 2, (k, 1), padding='same', name=name + '_2_conv1')(inputs)
    attention2 = BN(attention2, 'attention2_1')
    attention2 = Conv2D(1, (1, k), padding='same', name=name + '_2_conv2')(attention2)
    attention2 = BN(attention2, 'attention2_2')
    attention = Add(name=name+'_add')([attention1,attention2])
    attention = Activation('sigmoid')(attention)
    attention = Repeat(repeat_list=[1, 1, 1, C])(attention)
    return attention

def ChannelWiseAttention(inputs,name):
    H, W, C = map(int, inputs.get_shape()[1:])
    attention = GlobalAveragePooling2D(name=name+'_GlobalAveragePooling2D')(inputs)
    attention = Dense(C // 4, activation='relu')(attention)
    attention = Dense(C, activation='sigmoid',activity_regularizer=l1_reg)(attention)
    attention = Reshape((1, 1, C),name=name+'_reshape')(attention)
    attention = Repeat(repeat_list=[1, H, W, 1],name=name+'_repeat')(attention)
    attention = Multiply(name=name + '_multiply')([attention, inputs])
    return attention
