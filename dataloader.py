import os
import glob
import numpy as np
import tensorflow as tf
from yacs.config import CfgNode
from decord import VideoReader, bridge, cpu

from transforms import TemporalTransforms, SpatialTransforms
import utils

class InputReader:
  def __init__(self, cfg: CfgNode, is_training, use_tfrecord,
              mixed_precision=False):
    """__init__()

    Args:
      cfg (CfgNode): the model configurations
      is_training (bool): boolean flag to indicate if
        reading training dataset
      use_tfrecord (bool): whether data is in tfrecord
        format.
      mixed_precision (bool): whether to use mixed precision.
    """
    self._cfg = cfg
    self._is_training = is_training
    self._use_tfrecord = use_tfrecord
    self._mixed_prec = mixed_precision
  
  def decode_video(self, line):
    """Given a line from a text file containing the link
    to a video and the numerical label, process the line
    and decode the video.

    Args:
      line (tf.Tensor): a string tensor containing the
        path to a video file and the label of the video.

    Returns:
      tf.uint8, tf.int32: the decoded video (with all its
        frames intact), the label of the video
    """
    line = tf.strings.strip(line)
    split = tf.strings.split(line, " ")

    # convert byte tensor to python string object
    path = tf.compat.as_str_any(split[0].numpy())

    # convert label to integer
    label = tf.strings.to_number(split[1], out_type=tf.int32)

    # decode video frames
    # if unsuccessful, replace with a tensor of zeros.
    try:
      bridge.set_bridge("tensorflow")
      vr = VideoReader(path, ctx=cpu(0))
      num_frames = len(vr)
      video = vr.get_batch(range(num_frames))
    except Exception as e:
      tf.compat.v1.logging.warn(
        f"\nFailed to decode video {path}. Replacing with zeros...")
      video = tf.zeros([100, 240, 144, 3], tf.uint8)

    return video, label

  @tf.function
  def parse_and_decode(self, serialized_example):
    """parse and decode the contents of a serialized
    tf.train.SequenceExample object.

    Args:
      serialized_example (tf.train.Example): A

    Returns:
      (tf.uint8, tf.int64): (video tensor, category id of video)
    """
    sequence_features = {
        "video": tf.io.FixedLenSequenceFeature([], dtype=tf.string)}

    context_features = {
        "video/num_frames": tf.io.FixedLenFeature([], tf.int64, -1),
        "video/class/label": tf.io.FixedLenFeature([], tf.int64, -1)}

    context, sequence = tf.io.parse_single_sequence_example(
        serialized_example, context_features, sequence_features)

    indices = tf.range(0, context["video/num_frames"])
    video = tf.map_fn(lambda i: tf.image.decode_jpeg(sequence["video"][i]),
            indices, fn_output_signature=tf.uint8)
    label = tf.cast(context["video/class/label"], tf.int32)

    return video, label

  @tf.function
  def process_batch(self, videos, label,  batch_size):
    """Reshapes the video tensor to be of the format
    `batch_size x H x W x C`
    """
    if self._is_training:
      videos = tf.squeeze(videos)
      videos.set_shape((
          batch_size,
          self._cfg.DATA.TEMP_DURATION,
          self._cfg.DATA.TRAIN_CROP_SIZE,
          self._cfg.DATA.TRAIN_CROP_SIZE,
          self._cfg.DATA.NUM_INPUT_CHANNELS
      ))
    else:
      shapes = tf.shape(videos)
      videos = tf.reshape(videos, shape=[-1, shapes[-4], shapes[-3], shapes[-2], shapes[-1]])
      videos.set_shape((
          batch_size * self._cfg.TEST.NUM_TEMPORAL_VIEWS * self._cfg.TEST.NUM_SPATIAL_CROPS,
          self._cfg.DATA.TEMP_DURATION,
          self._cfg.DATA.TEST_CROP_SIZE,
          self._cfg.DATA.TEST_CROP_SIZE,
          self._cfg.DATA.NUM_INPUT_CHANNELS
      ))

    if self._mixed_prec:
      dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
      videos = tf.cast(videos, dtype)
    return videos, label

  @property
  def dataset_options(self):
    """Returns set options for td.data.Dataset API"""
    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = not self._is_training
    options.experimental_optimization.parallel_batch = True
    return options

  def __call__(self, file_pattern, batch_size=None):
    """Loads, transforms and batches data"""
    temporal_transform = TemporalTransforms(
        sample_rate=self._cfg.DATA.FRAME_RATE,
        num_frames=self._cfg.DATA.TEMP_DURATION,
        is_training=self._is_training,
        num_views=self._cfg.TEST.NUM_TEMPORAL_VIEWS)

    spatial_transform = SpatialTransforms(
        jitter_min=self._cfg.DATA.TRAIN_JITTER_SCALES[0],
        jitter_max=self._cfg.DATA.TRAIN_JITTER_SCALES[1],
        crop_size=self._cfg.DATA.TRAIN_CROP_SIZE if self._is_training else self._cfg.DATA.TEST_CROP_SIZE,
        is_training=self._is_training,
        num_crops=self._cfg.TEST.NUM_SPATIAL_CROPS,
        random_hflip=self._is_training)

    if self._use_tfrecord:
      dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
      dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(
        filename,
        compression_type="GZIP",
        num_parallel_reads=tf.data.experimental.AUTOTUNE).prefetch(1),
      deterministic=False,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if self._is_training:
        dataset = dataset.shuffle(batch_size * 16 if batch_size else 1024)
    else:
      dataset = tf.data.TextLineDataset(file_pattern).cache()
      if self._is_training:
        dataset = dataset.shuffle(self._cfg.TRAIN.DATASET_SIZE,
          reshuffle_each_iteration=True)

    dataset = dataset.with_options(self.dataset_options)

    if self._use_tfrecord:
      dataset = dataset.map(lambda value: self.parse_and_decode(value),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.map(
          lambda x: tf.py_function(self.decode_video, [x], [tf.uint8, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._is_training:
      dataset = dataset.repeat()

    dataset = dataset.map(lambda *args: temporal_transform(*args),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda *args: spatial_transform(
            *args,
            self._cfg.DATA.MEAN,
            self._cfg.DATA.STD),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch_size is not None:
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.map(
          lambda *args: self.process_batch(*args, batch_size),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
