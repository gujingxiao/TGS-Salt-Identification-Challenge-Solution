import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from IOU_matrix import challenge_iou_metric, iou_metric_batch, predict_result
from rle_code import rle_encode
from loss_functions import weighted_bce_dice_loss
from image_preprocess import upsample, downsample
from models import build_model, cov_to_class
from keras.optimizers import Adam

# Set some parameters
path_train_images = '../data/train/images/'
path_train_masks = '../data/train/masks/'
path_test_images = '../data/test/images/'

# Set image size
img_size_ori = 101
img_size_target = 128
img_channels = 1
img_augmentation = 0

# Set train parameters
epochs = 200
batch_size = 32
start_features = 4
dropout = 0.2
val_ratio = 0.1
random_seed = 1234

# Set GPU ratio
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = False
set_session(tf.Session(config=config))

"""
#-1  Loading of training/testing ids and depths
"""
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

if img_channels == 1:
    train_df["images"] = [np.array(load_img(path_train_images + "{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
elif img_channels == 3:
    train_df["images"] = [np.array(load_img(path_train_images + "{}.png".format(idx))) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img(path_train_images + "{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

"""
#-2  Create train/validation split stratified by salt coverage
"""
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, img_channels),
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
    train_df.coverage.values, train_df.z.values,
    test_size=val_ratio, stratify=train_df.coverage_class, random_state= random_seed)

"""
#-3  Data augmentation
"""
if img_augmentation == 1:
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

"""
#-4  Setup model and train
"""
input_layer = Input((img_size_target, img_size_target, img_channels))
output_layer = build_model(input_layer, start_features)
model = Model(input_layer, output_layer)

model.compile(loss=weighted_bce_dice_loss, optimizer="adam", metrics=[challenge_iou_metric])
model.summary()

early_stopping = EarlyStopping(monitor='val_challenge_iou_metric', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("../models/unet_resnet.model",monitor='val_challenge_iou_metric',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_challenge_iou_metric', mode = 'max',factor=0.5, patience=3, min_lr=0.00001, verbose=1)
history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=epochs, batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=2, shuffle=True)

"""
#-5  Find out best Threshold
"""
model.load_weights("../models/unet_best1.model")

preds_valid = predict_result(model,x_valid,img_size_target)
preds_valid2 = np.array([downsample(x, img_size_ori, img_size_target) for x in preds_valid])
y_valid2 = np.array([downsample(x, img_size_ori, img_size_target) for x in y_valid])

## Scoring for last model
thresholds = np.linspace(0.2, 0.8, 60)
ious = np.array([iou_metric_batch(y_valid2, np.int32(preds_valid2 > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print("Best threshold: ", threshold_best, "Best Val IOU: ", iou_best)

"""
#-6  Generate submission
"""
if img_channels == 1:
    x_test = np.array([(upsample(np.array(load_img(path_test_images + "{}.png".format(idx), color_mode="grayscale")), img_size_ori, img_size_target)) / 255
                   for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, img_channels)
elif img_channels == 3:
    x_test = np.array([(upsample(np.array(load_img(path_test_images + "{}.png".format(idx))), img_size_ori, img_size_target)) / 255
                   for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, img_channels)

preds_test = predict_result(model,x_test,img_size_target)
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i], img_size_ori, img_size_target) > threshold_best))
             for i, idx in enumerate(tqdm_notebook(test_df.index.values))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('../submission/submission.csv')



