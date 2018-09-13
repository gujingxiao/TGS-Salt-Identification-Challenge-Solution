from keras.layers import Activation, Add, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.models import Model
import keras_resnet
from various_resblock import residual_block, simple_residual_block

def selfdesigned_encoder(input_layer, start_channels):
    encoder_outputs = []

    # Start layers 128 -> 128
    x = Conv2D(start_channels, (3, 3), activation=None, use_bias=False, padding="same")(input_layer)
    x = keras_resnet.layers.BatchNormalization(epsilon=1e-5, freeze=True)(x)
    x = Activation('relu')(x)
    encoder_outputs.append(x)

    # 128 -> 64 -> 32 -> 16 -> 8
    for index in range(4):
        start_channels = start_channels * 2
        x = residual_block(x, start_channels, short_cut=True)
        x = residual_block(x, start_channels)
        encoder_outputs.append(x)

    return encoder_outputs

def simpledesigned_encoder(input_layer, start_channels):
    encoder_outputs = []
    x = input_layer

    for index in range(4):
        x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
        x = residual_block(x, start_channels)
        x = residual_block(x, start_channels)
        x = Activation('relu')(x)
        encoder_outputs.append(x)

        x = MaxPooling2D((2, 2))(x)

        start_channels = start_channels * 2

    # Middle
    x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
    x = residual_block(x, start_channels)
    x = residual_block(x, start_channels)
    x = Activation('relu')(x)
    encoder_outputs.append(x)

    return encoder_outputs

def selfdesigned_decoder(encoder_outputs, start_channels):
    start_channels = start_channels * 8
    x = encoder_outputs[4]

    for index in range(4):
        x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding="same")(x)
        x = keras_resnet.layers.BatchNormalization(epsilon=1e-5, freeze=True)(x)
        x = Activation('relu')(x)
        #concat = concatenate([x, encoder_outputs[3 - index]])
        concat = Add()([x, encoder_outputs[3 - index]])
        x = residual_block(concat, start_channels)
        x = residual_block(x, start_channels)

        start_channels = int(start_channels / 2)

    decoder_outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return decoder_outputs

def simpledesigned_decoder(encoder_outputs, start_channels):
    start_channels = start_channels * 8
    x = encoder_outputs[4]

    for index in range(4):
        if index % 2 == 1:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), padding="same")(x)
        else:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), padding="same")(x)
        x = concatenate([x, encoder_outputs[3 - index]])
        x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
        x = simple_residual_block(x, start_channels)
        x = simple_residual_block(x, start_channels)
        x = Activation('relu')(x)

        start_channels = int(start_channels / 2)

    decoder_outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return decoder_outputs


def encoder_model(input_layer, start_channels=32):
    return simpledesigned_encoder(input_layer, start_channels)
    #return selfdesigned_encoder(input_layer, start_channels)

def decoder_model(input_layer, start_channels=32):
    return simpledesigned_decoder(input_layer, start_channels)
    #return selfdesigned_decoder(input_layer, start_channels)

# Build model
def build_model(input_layer, start_channels):
    encoder_outputs = encoder_model(input_layer, start_channels)
    decoder_outputs = decoder_model(encoder_outputs, start_channels)

    return decoder_outputs

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i