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
    X = Input((img_size, img_size, 3))

    # Encode level 1
    encode1 = conv2d_bn(X, 64)
    encode1 = conv2d_bn(encode1, 64)

    # Encode level 2
    encode2 = MaxPool2D((2, 2))(encode1)
    encode2 = conv2d_bn(encode2, 128)
    encode2 = conv2d_bn(encode2, 128)

    # Encode level 3
    encode3 = MaxPool2D((2, 2))(encode2)
    encode3 = conv2d_bn(encode3, 256)
    encode3 = conv2d_bn(encode3, 256)

    # Encode level 4
    encode4 = MaxPool2D((2, 2))(encode3)
    encode4 = conv2d_bn(encode4, 512)
    encode4 = conv2d_bn(encode4, 512)

    # Encode level 5
    encode5 = MaxPool2D((2, 2))(encode4)
    encode5 = conv2d_bn(encode5, 1024)
    encode5 = conv2d_bn(encode5, 512)

    # Decode level 4
    decode4 = conv2d_bn(encode5, 512, transpose=True)
    decode4 = Concatenate()([encode4, decode4])
    decode4 = conv2d_bn(decode4, 512)
    decode4 = conv2d_bn(decode4, 512)

    # Decode level 3
    decode3 = conv2d_bn(decode4, 256, transpose=True)
    decode3 = Concatenate()([encode3, decode3])
    decode3 = conv2d_bn(decode3, 256)
    decode3 = conv2d_bn(decode3, 256)

    # Decode level 2
    decode2 = conv2d_bn(decode3, 128, transpose=True)
    decode2 = Concatenate()([encode2, decode2])
    decode2 = conv2d_bn(decode2, 128)
    decode2 = conv2d_bn(decode2, 128)

    # Decode level 1
    decode1 = conv2d_bn(decode2, 64, transpose=True)
    decode1 = Concatenate()([encode1, decode1])
    decode1 = conv2d_bn(decode1, 64)
    decode1 = conv2d_bn(decode1, 32)
    decode1 = Conv2D(1, kernel_size=(1, 1))(decode1)
    decode1 = Activation('sigmoid')(decode1)

    mdl = Model(X, decode1)
    return mdl


def small_unet():
    img_size = 240
    filter_size = 32
    X = Input((img_size, img_size, 3))

    # Encode level 1
    encode1 = conv2d_bn(X, filter_size)
    encode1 = conv2d_bn(encode1, filter_size)

    # Encode level 2
    encode2 = MaxPool2D((2, 2))(encode1)
    encode2 = conv2d_bn(encode2, filter_size * 2)
    encode2 = conv2d_bn(encode2, filter_size)

    # Decode level 1
    decode1 = conv2d_bn(encode2, filter_size, transpose=True)
    decode1 = Concatenate()([encode1, decode1])
    decode1 = conv2d_bn(decode1, filter_size)
    decode1 = conv2d_bn(decode1, filter_size / 2)
    decode1 = Conv2D(1, kernel_size=(1, 1))(decode1)
    decode1 = Activation('sigmoid')(decode1)

    mdl = Model(X, decode1)
    return mdl
