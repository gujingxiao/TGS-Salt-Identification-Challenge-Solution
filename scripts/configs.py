from tensorflow import flags

FLAGS = flags.FLAGS

# Image related flags
flags.DEFINE_integer("img_size_ori", 101, "Size of original images")
flags.DEFINE_integer("img_size_target", 128, "Size of input images")

# Training flags
flags.DEFINE_integer("num_epochs", 200, "Maximum of epochs")
flags.DEFINE_integer("batch_size", 32, "Number of batch size")
flags.DEFINE_integer("start_channels", 16, "Number of the start channels of the newwork")
flags.DEFINE_integer("random_seed", 1234, "Number of random seed")
flags.DEFINE_float("dropout", 0.2, "The ratio of dropout")
flags.DEFINE_float("val_ratio", 0.1, "The ratio of train/validation split")
flags.DEFINE_bool("shuffle", True, "Shuffle the dataset every epoch")
flags.DEFINE_bool("augmentation", True, "Do data augmentation")
flags.DEFINE_bool("random_crop", True, "Do data random cropping")

# Network flags
flags.DEFINE_integer("encoder_type", 1, "1 - simResnet, 2 - Resnet18, 3 - Resnet34")
flags.DEFINE_integer("decoder_type", 1, "1 - simResnet, 2 - Resnet18, 3 - Resnet34")

# Optimizer flags
flags.DEFINE_string("optimizer", "Adam", "Type of used optimizer")
flags.DEFINE_float("min_lr", 0.00001, "Minimum of learning rate")
flags.DEFINE_float("factor", 0.4, "Value of recuding factor")
flags.DEFINE_integer("factor_patience", 4, "Multiply factor in every factor patience")
flags.DEFINE_integer("final_patience", 12, "Stop training after final patience")

# Model flags
flags.DEFINE_string("model_name", "unet_best.model", "Name of snapshots")
flags.DEFINE_bool("generate_submission", False, "Generate submission")
flags.DEFINE_string("submission_name", "submission.csv", "Name of submission file")


