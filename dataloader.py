import tensorflow as tf
import decord

from transforms import TemporalTransforms, SpatialTransforms
from yacs.config import CfgNode


class InputReader:
  def __init__(self, 
              cfg: CfgNode,
              is_training: bool):
    """__init__()

    Args:
        cfg (CfgNode): the model configurations
        is_training (bool): boolean flag to indicate if
          reading training dataset
    """
    self._cfg = cfg
    self._is_training = is_training
    if self._is_training:
      self._label_path = self._cfg.DATA.TRAIN_LABEL_PATH
    else:
      self._label_path = self._cfg.DATA.TEST_LABEL_PATH
  
  def decode_video(self, line):
    """
    Given a line from a text file containing the link
    to a video and the numerical label, process the line
    and decode the video.

    Args:
        line (tf.Tensor): a string tensor containing the
          path to a video file and the label of the video.

    Returns:
        tf.uint8, tf.int32: the decoded video (with all its
          frames intact), the label of the video
    """
    # remove trailing and leading whitespaces
    line = tf.strings.strip(line)

    # split the (byte) string by space
    split = tf.strings.split(line, ' ')

    # convert byte tensor to python string object
    path = tf.compat.as_str_any(split[0].numpy())

    # convert label to integer
    label = tf.strings.to_number(split[1], out_type=tf.int32)

    try:
      # decode video frames
      decord.bridge.set_bridge('tensorflow')
      vr = decord.VideoReader(path)
      num_frames = len(vr)
      video = vr.get_batch(range(num_frames))
    except Exception as e:
      print(f"Failed to decode video {path} with exception: {e}")
      video = tf.zeros([16, 112, 112, 3], tf.uint8)

    return video, label
  
  def process_batch(self, videos, label):
    if self._is_training:
      videos = tf.squeeze(videos)
    else:
      shapes = tf.shape(videos)
      videos = tf.reshape(videos, shape=[-1, shapes[-4], shapes[-3], shapes[-2], shapes[-1]])

    return videos, label

  @property
  def dataset_options(self):
    """Returns set options for td.data.Dataset API"""
    options = tf.data.Options()
    options.experimental_deterministic = not self._is_training
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    return options

  def __call__(self, batch_size=None, num_views=1):
    """Loads, transforms and batches data"""
    temporal_transform = TemporalTransforms(
        sample_rate=self._cfg.DATA.FRAME_RATE,
        num_frames=self._cfg.DATA.TEMP_DURATION,
        is_training=self._is_training,
        num_views=num_views
    )

    spatial_transform = SpatialTransforms(
        jitter_min=self._cfg.DATA.TRAIN_JITTER_SCALES[0],
        jitter_max=self._cfg.DATA.TRAIN_JITTER_SCALES[1],
        crop_size=self._cfg.DATA.TRAIN_CROP_SIZE if self._is_training else self._cfg.DATA.TEST_CROP_SIZE,
        is_training=self._is_training,
        num_crops=self._cfg.TEST.NUM_SPATIAL_CROPS,
        random_hflip=self._is_training
    )

    dataset = tf.data.TextLineDataset(self._label_path).prefetch(1)

    if self._is_training:
      dataset = dataset.shuffle(64, seed=None).repeat()
    
    dataset = dataset.with_options(self.dataset_options)
    
    dataset = dataset.map(
        lambda x: tf.py_function(self.decode_video, [x], [tf.uint8, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(
        lambda *args: temporal_transform(*args),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(
        lambda *args: spatial_transform(
            *args,
            self._cfg.DATA.MEAN,
            self._cfg.DATA.STD
          ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if batch_size is not None:
      dataset = dataset.batch(batch_size, drop_remainder=False)
      dataset = dataset.map(
          lambda *args: self.process_batch(*args),
          num_parallel_calls=tf.data.experimental.AUTOTUNE
      )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
