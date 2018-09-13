import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models import selfdesigned_encoder
from image_preprocess import upsample


# Set some parameters
path_train_images = '../data/train/images/'
path_train_masks = '../data/train/masks/'
path_test_images = '../data/test/images/'
img_size_ori = 101
img_size_target = 128

# Set GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

def build_resnet34_encoder(input_layer, start_channels):
    encoder_outputs = selfdesigned_encoder(input_layer, start_channels=start_channels)

    x = keras.layers.GlobalAveragePooling2D(name="pool5")(encoder_outputs[4])
    x = keras.layers.Dense(2, activation="softmax", name="fc2")(x)

    return x

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

"""
#-1  Loading of training/testing ids and depths
"""
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
labels = np.zeros((4000, 2), dtype=np.int8)

for index in range(train_df["masks"].shape[0]):
    sum_value = np.sum(train_df["masks"][index])

    if sum_value > 0:
        labels[index] = [0, 1]
    else:
        labels[index] = [1, 0]

"""
#-2  Create train/validation split stratified by salt coverage
"""
ids_train, ids_valid, x_train, x_valid, y_train, y_valid= train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    labels,
    test_size=0.1, random_state= 1234)

"""
#-3  Data augmentation
"""
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, y_train, axis=0)

"""
#-4  Setup model and train
"""
epochs = 200
batch_size = 32

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_resnet34_encoder(input_layer, start_channels=32)
print(output_layer.shape)
model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor='val_acc', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("../models/pretrained_encoder.model",monitor='val_acc',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max',factor=0.4, patience=3, min_lr=0.00001, verbose=1)
history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=epochs, batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=2, shuffle=True)