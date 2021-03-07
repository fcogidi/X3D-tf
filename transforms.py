import tensorflow as tf
from utils import normalize

class TemporalTransforms:
  def __init__(self,
              is_training: bool,
              sample_rate: int,
              num_frames: int,
              num_views: int=1):
    self._sample_rate = sample_rate
    self._is_training = is_training
    self._num_frames = num_frames
    self._num_views = num_views

  @tf.function
  def get_temporal_sample(self, video):
    """
    Temporally sample a clip from the given video by
      looping the video until the desired number of frames 
      is achieved.

    Args:
      video (tf.Tensor): Full video

    Returns:
      tuple (tf.Tensor, tf.Tensor): clip from video, clip label
    """
    size = tf.shape(video)[0]
    indices = tf.range(size)      

    if self._is_training:
      # randomly select start index from uniform distribution
      start_index = tf.random.uniform([1], 0, size, tf.int32)

      # calulate end_index so that the number of frames selected
      # will be equal to the temporal duration. The formular here
      # is simply the inverse of one used by tf.strided_slice to
      # to calculate the size of elements to extract: 
      # (end-begin)/stride
      end_index = start_index + (self._num_frames * self._sample_rate)

      # loop the frames
      num_loops = TemporalTransforms._get_num_loops(size, end_index)
      indices = tf.tile(indices, multiples=num_loops)

      # get the indices of frames
      indices = tf.strided_slice(indices, start_index, end_index, [self._sample_rate])
    else:
      start_index = tf.constant([0]) # start from the beginning
      sample_rate = tf.maximum(1, size//self._num_frames)

      end_index = start_index + (self._num_frames * sample_rate * self._num_views)

      # loop the frames
      num_loops = TemporalTransforms._get_num_loops(size, end_index)
      indices = tf.tile(indices, multiples=num_loops)[0:end_index[0]]

      # get the indices of frames
      indices = tf.strided_slice(indices, start_index, end_index, [sample_rate])
    
    clip = tf.gather(video, indices, axis=0)

    if not self._is_training:
      return tf.reshape(clip, [self._num_views, self._num_frames,
        tf.shape(video)[1], tf.shape(video)[2], tf.shape(video)[3]])

    return tf.expand_dims(clip, axis=0)

  @staticmethod
  def _get_num_loops(size, end_index):
    """
    Determines the number of times to loop a video to enusre
      that enough frames are available to fulfil the temporal_duration.

    Args:
      size (tf.Tensor): number of frames in the video
      end_index (tf.Tensor): the desired index of the last frame

    Returns:
      tf.Tensor: the number of times to loop the video
    """
    num_loops = tf.math.ceil(end_index / size)
    return tf.cast(num_loops, tf.int32)
  
  def __call__(self, video, label):
    clips = self.get_temporal_sample(video)
    return clips, label

class SpatialTransforms:
  def __init__(self, jitter_min, jitter_max, crop_size, is_training,
              num_crops=1, random_hflip=False):
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
        on frames (with probability of 0.5). Defaults to False.
    """
    self._is_training = is_training
    self._num_crops = num_crops
    self._crop_size = crop_size
    self._min_size = float(jitter_min)
    self._max_size = float(jitter_max)
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
      # NOTE: `frames` is changed to a 4-D tensor
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
      frames = [
          self.uniform_crop(
              frames,
              self._crop_size,
              i%3 if self._num_crops > 1 else 1) # LeftCenterRight vs Center crop
          for i in range(self._num_crops)]
      frames = tf.convert_to_tensor(frames)

    # normalize pixel values
    frames = normalize(frames, per_channel_mean, per_channel_std)
    
    return frames, label
  
