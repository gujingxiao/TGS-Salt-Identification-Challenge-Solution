import numpy as np
import pandas as pd

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from IOU_matrix import my_iou_metric, iou_metric_batch, predict_result
from rle_code import rle_encode
from loss_functions import weighted_bce_dice_loss
from image_preprocess import upsample, downsample
from models import cov_to_class, build_model
from configs import FLAGS

"""
#-1  Loading of training/testing ids and depths
"""
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(FLAGS.img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


"""
#-2  Create train/validation split stratified by salt coverage
"""
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, FLAGS.img_size_target, FLAGS.img_size_target, 1),
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, FLAGS.img_size_target, FLAGS.img_size_target, 1),
    train_df.coverage.values, train_df.z.values,
    test_size=FLAGS.val_ratio, stratify=train_df.coverage_class, random_state=FLAGS.random_seed)


"""
#-3  Data augmentation
"""
if FLAGS.augmentation is True:
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)


"""
#-4  Setup model and train
"""
input_layer = Input((FLAGS.img_size_target, FLAGS.img_size_target, 1))
output_layer = build_model(input_layer, FLAGS.start_channels)
model = Model(input_layer, output_layer)

model.compile(loss=weighted_bce_dice_loss, optimizer=FLAGS.optimizer, metrics=[my_iou_metric])
model.summary()

early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max', patience=FLAGS.final_patience, verbose=1)
model_checkpoint = ModelCheckpoint("../models/" + FLAGS.model_name, monitor='val_my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=FLAGS.factor, patience=FLAGS.factor_patience, min_lr=FLAGS.min_lr, verbose=1)
history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid], epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=2, shuffle=FLAGS.shuffle)


"""
#-5  Find out best Threshold
"""
model.load_weights("../models/" + FLAGS.model_name)

preds_valid = predict_result(model,x_valid,FLAGS.img_size_target)
preds_valid2 = np.array([downsample(x, FLAGS.img_size_ori, FLAGS.img_size_target) for x in preds_valid])
y_valid2 = np.array([downsample(x, FLAGS.img_size_ori, FLAGS.img_size_target) for x in y_valid])

thresholds = np.linspace(0.3, 0.7, 41)
ious = np.array([iou_metric_batch(y_valid2, np.int32(preds_valid2 > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print("Best threshold: ", threshold_best, "Best Val IOU: ", iou_best)


"""
#-6  Generate submission
"""
if FLAGS.generate_submission is True:
    x_test = np.array([upsample(np.array(load_img("../data/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255
                       for idx in tqdm_notebook(test_df.index)]).reshape(-1, FLAGS.img_size_target, FLAGS.img_size_target, 1)

    preds_test = predict_result(model, x_test, FLAGS.img_size_target)
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('../submission/' + FLAGS.submission_name)




