from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization


def conv2d_bn(x, num_filters, filter_size=(3, 3), transpose=False):
    if transpose:
        x = Conv2DTranspose(num_filters, kernel_size=filter_size, strides=(2, 2), padding='same')(x)
    else:
        x = Conv2D(num_filters, kernel_size=filter_size, padding='same')(x)
    x = BatchNormalization(axis=3, scale=True)(x)
    x = Activation('relu')(x)
    return x


def unet():
    img_size = 240
    num_filters = 64
    X = Input((img_size, img_size, 3))

    # Encode level 1
    encode1 = conv2d_bn(X, num_filters)
    encode1 = conv2d_bn(encode1, num_filters)

    # Encode level 2
    encode2 = MaxPool2D((2, 2))(encode1)
    encode2 = conv2d_bn(encode2, num_filters * 2)
    encode2 = conv2d_bn(encode2, num_filters * 2)

    # Encode level 3
    encode3 = MaxPool2D((2, 2))(encode2)
    encode3 = conv2d_bn(encode3, num_filters * 4)
    encode3 = conv2d_bn(encode3, num_filters * 4)

    # Encode level 4
    encode4 = MaxPool2D((2, 2))(encode3)
    encode4 = conv2d_bn(encode4, num_filters * 8)
    encode4 = conv2d_bn(encode4, num_filters * 8)

    # Encode level 5
    encode5 = MaxPool2D((2, 2))(encode4)
    encode5 = conv2d_bn(encode5, num_filters * 16)
    encode5 = conv2d_bn(encode5, num_filters * 8)

    # Decode level 4
    decode4 = conv2d_bn(encode5, num_filters * 8, transpose=True)
    decode4 = Concatenate()([encode4, decode4])
    decode4 = conv2d_bn(decode4, num_filters * 8)
    decode4 = conv2d_bn(decode4, num_filters * 8)

    # Decode level 3
    decode3 = conv2d_bn(decode4, num_filters * 4, transpose=True)
    decode3 = Concatenate()([encode3, decode3])
    decode3 = conv2d_bn(decode3, num_filters * 4)
    decode3 = conv2d_bn(decode3, num_filters * 4)

    # Decode level 2
    decode2 = conv2d_bn(decode3, num_filters * 2, transpose=True)
    decode2 = Concatenate()([encode2, decode2])
    decode2 = conv2d_bn(decode2, num_filters * 2)
    decode2 = conv2d_bn(decode2, num_filters * 2)

    # Decode level 1
    decode1 = conv2d_bn(decode2, num_filters, transpose=True)
    decode1 = Concatenate()([encode1, decode1])
    decode1 = conv2d_bn(decode1, num_filters)
    decode1 = conv2d_bn(decode1, num_filters / 2)
    decode1 = Conv2D(1, kernel_size=(1, 1))(decode1)
    decode1 = Activation('sigmoid')(decode1)

    mdl = Model(X, decode1)
    return mdl


def small_unet():
    img_size = 240
    num_filters = 32
    X = Input((img_size, img_size, 3))

    # Encode level 1
    encode1 = conv2d_bn(X, num_filters)
    encode1 = conv2d_bn(encode1, num_filters)

    # Encode level 2
    encode2 = MaxPool2D((2, 2))(encode1)
    encode2 = conv2d_bn(encode2, num_filters * 2)
    encode2 = conv2d_bn(encode2, num_filters)

    # Decode level 1
    decode1 = conv2d_bn(encode2, num_filters, transpose=True)
    decode1 = Concatenate()([encode1, decode1])
    decode1 = conv2d_bn(decode1, num_filters)
    decode1 = conv2d_bn(decode1, num_filters / 2)
    decode1 = Conv2D(1, kernel_size=(1, 1))(decode1)
    decode1 = Activation('sigmoid')(decode1)

    mdl = Model(X, decode1)
    return mdl
