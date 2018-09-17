from keras.layers import Activation, MaxPooling2D, BatchNormalization, Dropout, Add, SeparableConv2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from various_resblock import residual_block, simple_residual_block
from configs import FLAGS

def resnetX_encoder(input_layer, start_channels, blocks):
    encoder_outputs = []

    # Start layers 128 -> 128
    x = Conv2D(start_channels, (3, 3), activation=None, use_bias=False, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoder_outputs.append(x)

    # 128 -> 64 -> 32 -> 16 -> 8
    for index in range(4):
        start_channels = start_channels * 2
        x = residual_block(x, start_channels, short_cut=True)
        for iter in range(blocks[index] - 1):
            x = residual_block(x, start_channels)
        x = Dropout(FLAGS.dropout)(x)
        encoder_outputs.append(x)
    return encoder_outputs

def simResnet_encoder(input_layer, start_channels, blocks):
    encoder_outputs = []
    x = input_layer

    for index in range(4):
        x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
        for iter in range(blocks[index] - 1):
            x = simple_residual_block(x, start_channels)
        x = Activation('relu')(x)
        encoder_outputs.append(x)

        x = MaxPooling2D((2, 2))(x)
        x = Dropout(FLAGS.dropout)(x)

        start_channels = start_channels * 2

    # Middle
    x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
    for iter in range(blocks[4] - 1):
        x = simple_residual_block(x, start_channels)
    x = Activation('relu')(x)
    encoder_outputs.append(x)
    return encoder_outputs

def resnetX_decoder(encoder_outputs, start_channels, blocks):
    start_channels = start_channels * 8
    x = encoder_outputs[4]

    for index in range(4):
        if index % 2 == 1 and FLAGS.img_size_target == 101:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding="valid")(x)
        else:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        concat = concatenate([x, encoder_outputs[3 - index]])
        x = residual_block(concat, start_channels * 2)
        for iter in range(blocks[index] - 1):
            x = residual_block(x, start_channels * 2)
        x = Dropout(FLAGS.dropout)(x)
        start_channels = int(start_channels / 2)

    decoder_outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return decoder_outputs

def simResnet_decoder(encoder_outputs, start_channels, blocks):
    start_channels = start_channels * 8
    x = encoder_outputs[4]

    for index in range(4):
        if index % 2 == 1 and FLAGS.img_size_target == 101:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), padding="valid")(x)
        else:
            x = Conv2DTranspose(start_channels, (3, 3), strides=(2, 2), padding="same")(x)
        x = concatenate([x, encoder_outputs[3 - index]])
        x = Dropout(FLAGS.dropout)(x)
        x = Conv2D(start_channels, (3, 3), activation=None, padding="same")(x)
        for iter in range(blocks[index] - 1):
            x = simple_residual_block(x, start_channels)
        x = Activation('relu')(x)
        start_channels = int(start_channels / 2)

    decoder_outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return decoder_outputs


def encoder_model(input_layer, start_channels=32):
    # self-designed simple resnet
    if FLAGS.encoder_type == 1:
        return simResnet_encoder(input_layer, start_channels, blocks=[3, 4, 6, 4, 3])
    # self-designed resnet18
    elif FLAGS.encoder_type == 2:
        return resnetX_encoder(input_layer, start_channels, blocks=[2, 2, 2, 2])
    # self-designed resnet34
    elif FLAGS.encoder_type == 3:
        return resnetX_encoder(input_layer, start_channels, blocks=[3, 4, 6, 3])
    #default simple resnet
    else:
        return simResnet_encoder(input_layer, start_channels, blocks=[3, 4, 6, 4, 3])

def decoder_model(input_layer, start_channels=32):
    # self-designed simple resnet
    if FLAGS.decoder_type == 1:
        return simResnet_decoder(input_layer, start_channels, blocks=[4, 6, 3, 2])
    # self-designed resnet18
    elif FLAGS.decoder_type == 2:
        return resnetX_decoder(input_layer, start_channels, blocks=[2, 2, 2, 2])
    # self-designed resnet34
    elif FLAGS.decoder_type == 3:
        return resnetX_decoder(input_layer, start_channels, blocks=[3, 6, 4, 3])
    #default simple resnet
    else:
        return simResnet_decoder(input_layer, start_channels, blocks=[4, 6, 4, 3])

# Build model
def build_model(input_layer, start_channels):
    encoder_outputs = encoder_model(input_layer, start_channels)
    decoder_outputs = decoder_model(encoder_outputs, start_channels)

    return decoder_outputs

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i
