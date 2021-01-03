import tensorflow as tf

class SpatialTransforms:
  def __init__(self,
              jitter_min: int,
              jitter_max: int,
              crop_size: int,
              is_training: bool,
              num_crops: int=1,
              random_hflip: bool = False):
    """__init__()

    Args:
      jitter_min (int): minimum size to scale frames to
      jitter_max (int): maximum size to scale frames to
      crop_size (int): final size of frames after cropping
      is_training (bool): whether transformation is being applied
        on training data
      num_crops (int, optional): number of crops to take. Only for
        non-training data. Defaults to 1.
      random_hflip (bool, optional): whether to perform horizontal flip
        on frames (with probability of 0.5). Defaults to True.
    """
    self._is_training = is_training
    self._num_crops = num_crops
    self._crop_size = crop_size
    self._min_size = jitter_min
    self._max_size = jitter_max
    self._random_hflip = random_hflip

  def random_short_side_resize(self, clips, min_size, max_size):
    """
    Randomly scale the short side of frames in `clips`.
      Reference: https://github.com/facebookresearch/SlowFast/blob/a521bc407fb4d58e05c51bde1126cddec3081841/slowfast/datasets/transform.py#L9

    Args:
      clips (tf.Tensor): a tensor of rank 5 with dimensions
        `num_clips` x `num frames` x `height` x `width` x `channel`.
      min_size (int): minimum scale size
      max_size (int): maximum scale size

    Returns:
      tf.Tensor: transformed clips scaled to new height and width
    """
    size = tf.random.uniform([], min_size, max_size, tf.float32)
    num_views = tf.shape(clips)[0]

    height = tf.cast(tf.shape(clips)[2], tf.float32)
    width = tf.cast(tf.shape(clips)[3], tf.float32)

    if (width <= height and width == size) or (
        height <= width and height == size):
        return clips
    new_width = size
    new_height = size
    if width < height:
      new_height = tf.math.floor((height / width) * size)
    else:
      new_width = tf.math.floor((width / height) * size)
    new_height = tf.cast(new_height, tf.int32)
    new_width = tf.cast(new_width, tf.int32)
    frames = [tf.image.resize(clips[i], [new_height, new_width])
              for i in range(num_views)
    ]
    frames = tf.stack(frames, 0)
    return tf.cast(frames, clips.dtype)

  @tf.function
  def uniform_crop(self, clips, size, spatial_idx):
    """
    Perform uniform spatial sampling on the images.
    Reference: https://github.com/facebookresearch/SlowFast/blob/a521bc407fb4d58e05c51bde1126cddec3081841/slowfast/datasets/transform.py#L151
    
    Args:
      clips (tf.Tensor): images to perform uniform crop. The dimension is
          `num_clips` x `num frames` x `height` x `width` x `channel`.
      size (int): size of height and weight to crop the images.
      spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
          is larger than height. Or 0, 1, or 2 for top, center, and bottom
          crop if height is larger than width.
    Returns:
      cropped (tensor): images with dimension of
          `num_clips` x `num frames` x `size` x `size` x `channel`.
    """
    assert spatial_idx in [0, 1, 2]

    height = tf.shape(clips)[2]
    width = tf.shape(clips)[3]

    y_offset = tf.math.ceil((height - size) / 2)
    x_offset = tf.math.ceil((width - size) / 2)

    y_offset = tf.cast(y_offset, tf.int32)
    x_offset = tf.cast(x_offset, tf.int32)

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = clips[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size, :
    ]

    return cropped

  def __call__(self, clips, label, per_channel_mean, per_channel_std):
    tf.assert_rank(clips, 5, 'clips must be 5-dimensional tensor')

    if self._is_training:
      frames = tf.numpy_function(
          func=self.random_short_side_resize,
          inp=[clips, self._min_size, self._max_size],
          Tout=clips.dtype)
      # random crop
      # NOTE: frames is changed to a 4-D tensor
      frames = tf.image.random_crop(
          frames[0],
          size=[tf.shape(frames)[1], self._crop_size, self._crop_size, tf.shape(frames)[-1]])
      if self._random_hflip:
        frames = tf.image.flip_left_right(frames)
      
      # change rank of `frames` to 5
      frames = tf.expand_dims(frames, axis=0)
    else:
      frames = tf.numpy_function(
          func=self.random_short_side_resize,
          inp=[clips, self._crop_size, self._crop_size],
          Tout=clips.dtype)
      # uniform crop
      frames = [self.uniform_crop(
                  frames,
                  self._crop_size,
                  i%3 if self._num_crops > 1 else 1) # LeftCenterRight vs Center crop
                for i in range(self._num_crops)
      ]
      frames = tf.concat(frames, 0)
      label = tf.repeat(label, self._num_crops)

    # normalize pixel values
    frames = normalize(frames, per_channel_mean, per_channel_std)
    
    return frames, label

class TemporalTransforms:
  def __init__(self,
              sample_rate: int,
              num_frames: int,
              is_training: bool,
              num_views: int=1):
    self._sample_rate = sample_rate
    self._is_training = is_training
    self._num_frames = num_frames
    self._num_views = num_views

  @tf.function
  def get_temporal_sample(self, video, num_views=1):
    """
    Temporally sample a clip from the given video by selecting
      looping the video until the desired number of frames 
      is achieved.

    Args:
      video (tf.Tensor): Full video
      num_views (int): number of clips to sample from the video

    Returns:
      tuple (tf.Tensor, tf.Tensor): clip from video, clip label
    """
    size = tf.shape(video)[0]
    indices = tf.range(size)

    if self._is_training:
      # randomly select start index from uniform distribution
      start_index = tf.random.uniform([1], 0, size, tf.int32)
    else:
      start_index = tf.constant([0])

    # calulate end_index so that the number of frames selected
    # will be equal to the temporal duration. The formular here
    # is simply the inverse of one used by tf.strided_slice to
    # to calculate the size of elements to extract: 
    # (end-begin)/stride
    end_index = start_index + (self._num_frames * self._sample_rate * num_views)
    end_index = tf.cast(end_index, tf.int32)

    # loop the indices to enusre that enough frames are available
    # to fulfil the temporal_duration
    num_loops = tf.math.ceil(end_index / size)
    num_loops = tf.cast(num_loops, tf.int32)
    indices = tf.tile(indices, multiples=num_loops)

    indices = tf.strided_slice(indices, start_index, end_index, [self._sample_rate])
    clip = tf.gather(video, indices, axis=0)

    if not self._is_training: 
      return tf.reshape(clip, 
      [num_views, self._num_frames, tf.shape(video)[1], tf.shape(video)[2], tf.shape(video)[3]])
    return tf.expand_dims(clip, axis=0)

  def __call__(self, video, label):
    if self._is_training:
      clips = self.get_temporal_sample(video)
    else:
      clips = self.get_temporal_sample(video, self._num_views)
      label = tf.repeat(label, self._num_views)
    return clips, label

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
  
