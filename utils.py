import os
import math
import tensorflow as tf
from absl import logging
from wandb.keras import WandbCallback

def round_width(width, multiplier, min_depth=8, divisor=8):
  """
  Round width of filters based on width multiplier
  from: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py

  Args:
      width (int): the channel dimensions of the input.
      multiplier (float): the multiplication factor.
      min_width (int, optional): the minimum width after multiplication.
          Defaults to 8.
      divisor (int, optional): the new width should be dividable by divisor.
          Defaults to 8.
  """
  if not multiplier:
      return width

  width *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(
      min_depth, int(width + divisor / 2) // divisor * divisor
  )
  if new_filters < 0.9 * width:
      new_filters += divisor
  return int(new_filters)

def round_repeats(repeats, multiplier):
  """
  Round number of layers based on depth multiplier.
  Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
  """
  multiplier = multiplier
  if not multiplier:
      return repeats
  return int(math.ceil(multiplier * repeats))

@tf.function
def normalize(clips, mean, std, norm_value=255):
  """
  Standardizes an n-dimensional tensor of images by first 
    normalizing with the given norm_value, then subtracting
    the mean and dividing by the standard deviation channelwise.

  Args:
    clips (tf.Tensor): video clips to normalize
    mean (list): mean value of the video raw pixels across the R G B channels.
    std (list): standard deviation of the video raw pixels across the R G B channels.
    norm_value (int, optional): value to normalize raw pixels by.
      Defaults to 255.

  Returns:
    tf.Tensor: tensor of the same shape as `clips`
  """
  shapes = tf.shape(clips)
  all_frames = tf.reshape(clips, [-1, shapes[-3], shapes[-2], shapes[-1]])
  all_frames /= norm_value

  def _normalize(frame):
    frame = frame - mean
    return frame / std

  all_frames = tf.vectorized_map(
      lambda x: _normalize(x),
      all_frames
  )

  return tf.reshape(all_frames, tf.shape(clips))

@tf.function
def denormalize(clips, mean, std, norm_value=255, out_dtype=tf.uint8):
  """
  Reverses the standardization operation for an n-dimensional tensor
    of images.

  Args:
    clips (tf.Tensor): video clips to denormalize
    mean (list): mean value of the video raw pixels across the R G B channels.
    std (list): standard deviation of the video raw pixels across the R G B channels.
    norm_value (int, optional): value to denormalize raw pixels by.
      Defaults to 255.
    out_dtype (tf.dtypes, optional): the data type of the output tensor.
      Defaults to tf.unit8.

  Returns:
      tf.Tensor: tensor of the same shape as `clips`.

  """
  shapes = tf.shape(clips)
  all_frames = tf.reshape(clips, [-1, shapes[-3], shapes[-2], shapes[-1]])

  def _denormalize(frame):
    frame = frame * std
    return frame + mean

  all_frames = tf.vectorized_map(
      lambda x: _denormalize(x),
      all_frames
  )

  all_frames *= norm_value
  all_frames = tf.cast(all_frames, out_dtype)

  return tf.reshape(all_frames, tf.shape(clips))

def get_callbacks(cfg, lr_schedule, flags):
  callbacks = []
  lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule, 1),
  tb = tf.keras.callbacks.TensorBoard(
      log_dir=flags.model_dir,
      profile_batch=10, #flags.debug
      write_images=True,
      write_graph=True,
      histogram_freq=10,
      update_freq=flags.save_checkpoints_step or 'epoch'
  )
  ckpt = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(flags.model_dir, 'ckpt-{epoch:d}'),
      verbose=1,
      monitor='val_acc',
      save_freq=flags.save_checkpoints_step or 'epoch',
  )
  callbacks.extend([lr, tb, ckpt])
  wandb = WandbCallback(
      verbose=1,
      save_weights_only=True,
  )
  if cfg.WANDB.ENABLE:
    callbacks.append(wandb)
  
  return callbacks

def get_strategy(num_gpus):
  # training strategy setup
  avail_gpus = tf.config.list_physical_devices('GPU')
  for gpu in avail_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  if num_gpus > 1 and len(avail_gpus) > 1:
    devices = []
    for num in range(num_gpus):
      if num < len(avail_gpus):
        id = int(avail_gpus[num].name.split(':')[-1])
        devices.append(f'/gpu:{id}')
    assert len(devices) > 1
    strategy = tf.distribute.MirroredStrategy(devices)
  elif len(avail_gpus) == 1 and num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    logging.warn('Using CPU')
    strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  return strategy

def get_precision(mixed_precision):
  # mixed precision
  precision = 'float32'
  if mixed_precision:
    if tf.config.list_physical_devices('GPU'):
      precision = 'mixed_float16'
  return precision
