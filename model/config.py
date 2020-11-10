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

# the spatial resolution of the input
_C.DATA.SPATIAL_RES = 112
'''TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  # CHECKPOINT_FILE_PATH: 'x3d_s.pyth' # 73.50% top1 30-view accuracy to download from the model zoo (optional).
  # NUM_SPATIAL_CROPS: 1
  NUM_SPATIAL_CROPS: 3
DATA:
  NUM_FRAMES: 13
  SAMPLING_RATE: 6
  TRAIN_JITTER_SCALES: [182, 228]
  TRAIN_CROP_SIZE: 160
  # TEST_CROP_SIZE: 160 # use if TEST.NUM_SPATIAL_CROPS: 1
  TEST_CROP_SIZE: 182 # use if TEST.NUM_SPATIAL_CROPS: 3
  INPUT_CHANNEL_NUM: [3]
'''
def get_default_config():
    return _C.clone()

