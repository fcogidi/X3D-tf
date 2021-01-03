import tensorflow as tf

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