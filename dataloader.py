import tensorflow as tf
import decord

from model.config import get_default_config

def decode_video(line):
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

  # decode video frames
  decord.bridge.set_bridge('tensorflow')
  vr = decord.VideoReader(path)
  num_frames = len(vr)
  video = vr.get_batch(range(num_frames))

  return video, label

@tf.function
def temporal_sampling(frame_rate, temporal_duration, video, label):
  """
  Temporally sample a clip from the given video by selecting
  a random start frame and looping the video until the number
  of frames given by :param temporal_duration is achieved at the
  given frame rate.

  Args:
      frame_rate (int): temporal stride
      temporal_duration (int): number of frames
      video (tf.Tensor): Full video
      label (tf.Tensor): integer representing class of video

  Returns:
      tuple (tf.Tensor, tf.Tensor): clip from video, clip label
  """
  size = tf.shape(video)[0]
  indices = tf.range(size)
  # randomly select start index from uniform distribution
  start_index = tf.random.uniform([1], 0, size, tf.int32)

  # calulate end_index so that the number of frames selected
  # will be equal to the temporal duration. The formular here
  # is simply the inverse of one used by tf.strided_slice to
  # to calculate the size of elements to extract: 
  # (end-begin)/stride
  end_index = start_index + (temporal_duration * frame_rate)
  end_index = tf.cast(end_index, tf.int32)

  # loop the indices to enusre that enough frames are available
  # to fulfil the temporal_duration
  num_loops = tf.math.ceil(end_index / size)
  num_loops = tf.cast(num_loops, tf.int32)
  indices = tf.tile(indices, multiples=num_loops)

  indices = tf.strided_slice(indices, start_index, end_index, [frame_rate])
  clip = tf.gather(video, indices, axis=0)
    
  return clip, label

cfg = get_default_config()
cfg.merge_from_file('configs/kinetics/X3D_M.yaml')
cfg.freeze()

dataset = tf.data.TextLineDataset('dataset/kinetics400/train.txt')

options = tf.data.Options()
options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
dataset = dataset.with_options(options)

dataset = dataset.map(
    lambda x: tf.py_function(decode_video, [x], [tf.uint8, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

dataset = dataset.map(
    lambda *args: temporal_sampling(
        cfg.DATA.FRAME_RATE,
        cfg.DATA.TEMP_DURATION,
        *args),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

dataset = dataset.prefetch(1)

import time
start = time.time()
list(dataset.take(10).as_numpy_iterator())
print('Runtime:', time.time() - start, 's')