from yacs.config import CfgNode as CN

_C = CN()

# 
# Network parameters
#
_C.NETWORK = CN()

# the size of the temporal filter in the conv1 layer
_C.NETWORK.C1_TEMP_FILTER = 5

# the number of filters produced by the first convolutional layer
_C.NETWORK.C1_CHANNELS = 12

# Whether to scale the width of Res2, default is false.
_C.NETWORK.SCALE_RES2 = False

# the network width expansion factor
_C.NETWORK.WIDTH_FACTOR = 1.0

# the network depth expansion factor
_C.NETWORK.DEPTH_FACTOR = 1.0

# the nework bottleneck width factor
_C.NETWORK.BOTTLENECK_WIDTH_FACTOR = 1.0

# the number of classes
_C.NETWORK.NUM_CLASSES = 400

# dropout rate for the dropout layer before the final fully-connected layer
_C.NETWORK.DROPOUT_RATE = 0.0

# whether to use mixed precision or not
_C.NETWORK.MIXED_PRECISION = False

#
# paramters for batch normalization layers
#
_C.NETWORK.BN = CN()

# the momentum parameter for all batch norm layers
_C.NETWORK.BN.MOMENTUM = 0.9

# the epsilon parameter for all batch norm layers
_C.NETWORK.BN.EPS = 1e-5

#
# configuration of the data layer
#
_C.DATA = CN()

# the rate at which to sample the input
_C.DATA.FRAME_RATE = 1

# the temporal duration or number of frames of the input
_C.DATA.TEMP_DURATION = 1

# the number of channels in the input
_C.DATA.NUM_INPUT_CHANNELS = 3

# the minimum and maximum scale for image resize operation
_C.DATA.TRAIN_JITTER_SCALES = [182, 228]

# the spatial resolution of the input
_C.DATA.TRAIN_CROP_SIZE = 112

# path to .txt file containing path_to_video and label
# for training set
_C.DATA.TRAIN_LABEL_PATH = " "

# path to .txt file containing path_to_video and label
# for test set
_C.DATA.TEST_LABEL_PATH = " "

_C.DATA.TEST_CROP_SIZE = 160

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The standard deviation of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

#
# configuration for training
#
_C.TRAIN = CN()

# whether to use multiple gpus or single gpu
_C.TRAIN.MULTI_GPU = False

# number of examples in the training set
_C.TRAIN.DATASET_SIZE = 0

# batch size
_C.TRAIN.BATCH_SIZE = 1

# number of training epochs
_C.TRAIN.EPOCHS = 1

# directory for logging/saving model
_C.TRAIN.MODEL_DIR = " "

# loss function
_C.TRAIN.OPTIMIZER = "SGD"

# momentum for optimizer
_C.TRAIN.MOMENTUM = 0.9

# base learning rate
_C.TRAIN.BASE_LR = 1.6

# number of steps to save checkpoints
_C.TRAIN.SAVE_CHECKPOINTS_EVERY = None

#
# configuration for inference
#
_C.TEST = CN()

# number of spatial crops
_C.TEST.NUM_SPATIAL_CROPS = 3

# number of temporal views
_C.TEST.NUM_TEMPORAL_VIEWS = 1

# number of examples in the training set
_C.TEST.DATASET_SIZE = 0

# batch size
_C.TEST.BATCH_SIZE = 1

#
# configurations for weights and biases
#
_C.WANDB = CN()

# whether to log to weights and biases
_C.WANDB.ENABLE = False

# name of the project
_C.WANDB.PROJECT_NAME = "X3D-tf"

# group name
_C.WANDB.GROUP_NAME = " "

# Can be "online", "offline" or "disabled". 
_C.WANDB.MODE = "online"

# whether to sync tensorboard
_C.WANDB.TENSORBOARD = True

def get_default_config():
    return _C.clone()

