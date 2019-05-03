from keras.models import *
from attention import *
from bilinear_upsampling import BilinearUpsampling


class BatchNorm(BatchNormalization):
    def call(self, inputs, training=None):
          return super(self.__class__, self).call(inputs, training=True)

class Copy(Layer):
    def call(self, inputs, **kwargs):
        copy = tf.identity(inputs)
        return copy
    def compute_output_shape(self, input_shape):
        return input_shape

class layertile(Layer):
    def call(self, inputs, **kwargs):
        image = tf.reduce_mean(inputs, axis=-1)
        image = tf.expand_dims(image, -1)
        image = tf.tile(image, [1, 1, 1, 32])
        return image

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)[:-1] + [32]
        return tuple(output_shape)


def BN(input_tensor,block_id):
    bn = BatchNorm(name=block_id+'_BN')(input_tensor)
    a = Activation('relu',name=block_id+'_relu')(bn)
    return a

def AtrousBlock(input_tensor, filters, rate, block_id, stride=1):
    x = Conv2D(filters, (3, 3), strides=(stride, stride), dilation_rate=(rate, rate),
               padding='same', use_bias=False, name=block_id + '_dilation')(input_tensor)
    return x


def CFE(input_tensor, filters, block_id):
    rate = [3, 5, 7]
    cfe0 = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=block_id + '_cfe0')(
        input_tensor)
    cfe1 = AtrousBlock(input_tensor, filters, rate[0], block_id + '_cfe1')
    cfe2 = AtrousBlock(input_tensor, filters, rate[1], block_id + '_cfe2')
    cfe3 = AtrousBlock(input_tensor, filters, rate[2], block_id + '_cfe3')
    cfe_concat = Concatenate(name=block_id + 'concatcfe', axis=-1)([cfe0, cfe1, cfe2, cfe3])
    cfe_concat = BN(cfe_concat, block_id)
    return cfe_concat


def VGG16(img_input, dropout=False, with_CPFE=False, with_CA=False, with_SA=False, droup_rate=0.3):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    C1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    if dropout:
        x = Dropout(droup_rate)(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    C2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    if dropout:
        x = Dropout(droup_rate)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    C3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    if dropout:
        x = Dropout(droup_rate)(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    C4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    if dropout:
        x = Dropout(droup_rate)(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    if dropout:
        x = Dropout(droup_rate)(x)
    C5 = x
    C1 = Conv2D(64, (3, 3), padding='same', name='C1_conv')(C1)
    C1 = BN(C1, 'C1_BN')
    C2 = Conv2D(64, (3, 3), padding='same', name='C2_conv')(C2)
    C2 = BN(C2, 'C2_BN')
    if with_CPFE:
        C3_cfe = CFE(C3, 32, 'C3_cfe')
        C4_cfe = CFE(C4, 32, 'C4_cfe')
        C5_cfe = CFE(C5, 32, 'C5_cfe')
        C5_cfe = BilinearUpsampling(upsampling=(4, 4), name='C5_cfe_up4')(C5_cfe)
        C4_cfe = BilinearUpsampling(upsampling=(2, 2), name='C4_cfe_up2')(C4_cfe)
        C345 = Concatenate(name='C345_aspp_concat', axis=-1)([C3_cfe, C4_cfe, C5_cfe])
        if with_CA:
            C345 = ChannelWiseAttention(C345, name='C345_ChannelWiseAttention_withcpfe')
    C345 = Conv2D(64, (1, 1), padding='same', name='C345_conv')(C345)
    C345 = BN(C345,'C345')
    C345 = BilinearUpsampling(upsampling=(4, 4), name='C345_up4')(C345)

    if with_SA:
        SA = SpatialAttention(C345, 'spatial_attention')
        C2 = BilinearUpsampling(upsampling=(2, 2), name='C2_up2')(C2)
        C12 = Concatenate(name='C12_concat', axis=-1)([C1, C2])
        C12 = Conv2D(64, (3, 3), padding='same', name='C12_conv')(C12)
        C12 = BN(C12, 'C12')
        C12 = Multiply(name='C12_atten_mutiply')([SA, C12])
    fea = Concatenate(name='fuse_concat',axis=-1)([C12, C345])
    sa = Conv2D(1, (3, 3), padding='same', name='sa')(fea)

    model = Model(inputs=img_input, outputs=sa, name="BaseModel")
    return model

