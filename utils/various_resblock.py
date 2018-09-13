from keras.layers import BatchNormalization,Activation,Add
from keras.layers.convolutional import Conv2D
import keras_resnet

def convolution_block(x, filters, size, strides=(1,1), activation=True):
    x = Conv2D(filters, size, strides=strides, use_bias=False, padding="same")(x)
    x = keras_resnet.layers.BatchNormalization(epsilon=1e-5, freeze=True)(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

def residual_block(blockInput, num_filters=16, short_cut=False):
    if short_cut == True:
        strides = (2,2)
    else:
        strides = (1,1)

    x = convolution_block(blockInput, num_filters, (3,3), strides=strides)
    x = convolution_block(x, num_filters, (3,3), activation=False)

    if short_cut == True:
        short_input = Conv2D(num_filters, (1, 1), strides=(2,2), use_bias=False)(blockInput)
        short_input = keras_resnet.layers.BatchNormalization(epsilon=1e-5, freeze=True)(short_input)
    else:
        short_input = blockInput
    x = Add()([x, short_input])
    x = Activation('relu')(x)
    return x


def simple_conv_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

def simple_residual_block(blockInput, num_filters=16):
    x = Activation('relu')(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x
