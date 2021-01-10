import os
import tensorflow as tf
from absl import logging
from wandb.keras import WandbCallback

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

def get_callbacks(cfg, lr_schedule, debug):
  callbacks = []
  lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule, 1),
  tb = tf.keras.callbacks.TensorBoard(
      log_dir=cfg.TRAIN.MODEL_DIR,
      profile_batch=debug,
      write_images=True,
      update_freq=cfg.TRAIN.SAVE_CHECKPOINTS_EVERY or 'epoch'
  )
  ckpt = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(cfg.TRAIN.MODEL_DIR, 'ckpt_{epoch:d}'),
      verbose=1,
      save_best_only=True,
      save_freq=cfg.TRAIN.SAVE_CHECKPOINTS_EVERY or 'epoch',
  )
  callbacks.append(lr, tb, ckpt)
  wandb = WandbCallback(
      verbose=1,
      save_weights_only=True,
  )
  if cfg.WANDB.ENABLE:
    callbacks.append(wandb)
  
  return callbacks

def get_strategy(cfg):
  # training strategy setup
  avail_gpus = tf.config.list_physical_devices('GPU')
  for gpu in avail_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
  if len(avail_gpus) > 1 and cfg.TRAIN.MULTI_GPU:
    devices = []
    for num in range(cfg.TRAIN.NUM_GPUS):
      if num < len(avail_gpus):
        id = int(avail_gpus[num].name.split(':')[-1])
        devices.append(f'/gpu:{id}')
    assert len(devices) > 1
    strategy = tf.distribute.MirroredStrategy(devices)
  elif len(avail_gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    logging.warn('Using CPU')
    strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  return strategy