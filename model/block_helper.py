from yacs.config import CfgNode
import tensorflow as tf
import tensorflow.keras as K

from model.utils import round_width

class Bottleneck(K.Layer):
  '''
  X3D Bottleneck block: 1x1x1, Tx3x3, 1x1x1 with squeeze-excitation
      added at every 2 stages.
  '''
  def __init__(self,
              channels: tuple,
              bn_cfg: CfgNode,
              stride: int = 1,
              block_index: int = 0,
              se_ratio: float = 0.0625,
              temp_kernel_size: int = 3):
    '''
    Constructs a single X3D bottleneck block.

    Args:
        channels (tuple): number of channels for each layer in the bottleneck
            order: (number of channles in first two layers, number of channels in last layer)
        bn_cfg (CfgNode): holds the parameters for the batch
        stride (int, optional): stride in the spatial dimension.
            Defaults to 1
        block_index (int): the index of the current block
        se_ratio (float, optional): the width multiplier for the squeeze-excitation
            operation. Defaults to 0.0625
        temp_kernel_size (int, optional): the filter size for the
            temporal dimension of the 3x3x3 convolutinal layer.
            Defaults to 3.
    '''
    super(Bottleneck, self).__init__()
    self.block_index = block_index
    self._bn_cfg = bn_cfg

    self.a = K.layers.Conv3D(
        filters=channels[0],
        kernel_size=1,
        strides=(1, 1, 1),
        padding='same',
        use_bias=False,
        data_format='channels_last')
    self.bn_a = K.layers.BatchNormalization(
        axis=-1,
        epsilon=self._bn_cfg.EPS,
        momentum=self._bn_cfg.MOMENTUM)
    self.relu = K.layers.ReLU()
    self.b = K.layers.Conv3D(
        filters=channels[0],
        kernel_size=(temp_kernel_size, 3, 3),
        strides=(1, stride, stride), # spatial downsampling
        padding='same',
        use_bias=False,
        groups=channels[0], # turns out to be necessary to reduce model params
        data_format='channels_last')
    self.bn_b = K.layers.BatchNormalization(
        axis=-1,
        epsilon=self._bn_cfg.EPS,
        momentum=self._bn_cfg.MOMENTUM)
    self.swish = K.layers.Activation('swish')

    # Squeeze-and-Excite operation
    if ((self.block_index + 1) % 2 == 0):
      width = round_width(channels[0], se_ratio)
      self.se_pool = AdaptiveAvgPool3D((1, 1, 1))
      self.se_fc1 = K.layers.Conv3D(
          filters=width,
          kernel_size=1,
          strides=1,
          use_bias=True,
          activation='relu')
      self.se_fc2 = K.layers.Conv3D(
          filters=channels[0],
          kernel_size=1,
          strides=1,
          use_bias=True,
          activation='sigmoid')

    self.c = K.layers.Conv3D(
        filters=channels[1],
        kernel_size=1,
        strides=(1, 1, 1),
        padding='same',
        use_bias=False,
        data_format='channels_last')
    self.bn_c = K.layers.BatchNormalization(
        axis=-1,
        epsilon=self._bn_cfg.EPS,
        momentum=self._bn_cfg.MOMENTUM)

  def call(self, input, training=False):
    out = self.a(input)
    out = self.bn_a(out, training=training)
    out = self.relu(out)
    out = self.b(out)
    out = self.bn_b(out, training=training)
    if ((self.block_index + 1) % 2 == 0):
        se = self.se_pool(out)
        se = self.se_fc1(se)
        se = self.se_fc2(se)
        out = out * se
    out = self.swish(out)
    out = self.c(out)
    out = self.bn_c(out, training=training)

    return out

class ResBlock(K.Layer):
  '''
  X3D residual stage: a single residual block
  '''
  _block_index = 0
  def __init__(self,
              channels: tuple,
              bn_cfg: CfgNode,
              stride: int = 1,
              se_ratio: float = 0.0625,
              temp_kernel_size: int = 3) -> None:
    '''
    Constructs a single X3D residual block.

    Args:
        channels (tuple): (input_channels, inner_channels, output_channels)
        bn_cfg (CfgNode) containing parameters for the batch
            normalization layer
        stride (int, optional): stride in the spatial dimension.
            Defaults to 1
        se_ratio (float, optional): the width multiplier for the squeeze-excitation
            operation. Defaults to 0.0625
        temp_kernel_size (int, optional): the filter size for the
            temporal dimension of the 3x3x3 convolutinal layer.
            Defaults to 3.
    '''
    super(ResBlock, self).__init__(name='ResBlock_%u' %ResBlock._block_index)
    ResBlock._block_index += 1

    self.in_channels = channels[0]
    self.inner_channels = channels[1]
    self.out_channels = channels[2]
    self._bn_cfg = bn_cfg

    # handles residual connection after downsampling
    if (self.in_channels != self.out_channels or stride != 1):
      self.residual = K.layers.Conv3D(
          filters=self.out_channels,
          kernel_size=(1, 1, 1),
          strides=(1, stride, stride),
          padding='valid',
          use_bias=False,
          data_format='channels_last')
      self.bn_r = K.layers.BatchNormalization(
          axis=-1,
          epsilon=self._bn_cfg.EPS,
          momentum=self._bn_cfg.MOMENTUM)

    self.bottleneck = Bottleneck(
        channels=channels[1:],
        stride=stride,
        bn_cfg=self._bn_cfg,
        block_index=ResBlock._block_index,
        se_ratio=se_ratio,
        temp_kernel_size=temp_kernel_size)
    self.add_op = K.layers.Add()
    self.relu = K.layers.Activation('relu')

  def call(self, input, training=False):
    out = self.bottleneck(input)
    if hasattr(self, 'residual'):
      res = self.residual(input)
      res = self.bn_r(res, training=training)
      out = self.add_op([res, out])
    else:
      out = self.add_op([input, out])
    out = self.relu(out)

    return out

class ResStage(K.Layer):
  '''
  Constructs a residual stage of given depth
      for the X3D network
  '''
  _stage_index = 2 # following the convention in the paper
  def __init__(self,
              in_channels: int,
              inner_channels: int,
              out_channels: int,
              depth: int,
              bn_cfg: CfgNode,
              se_ratio: float = 0.0625,
              temp_kernel_size: int = 3):
    '''
    Args:
        in_channels (int): number of channels at the input of
            a stage
        inner_channels (int): the number of channels at the bottleneck
            layers of the stage
        out_channels (int): the number of channels at the output of the
            stage
        depth (int): the depth of the stage or number of times stage
            is repeated
        bn_cfg (CfgNode) containing parameters for the batch
            normalization layer
        se_ratio (float, optional): the width multiplier for the squeeze-excitation
            operation. Defaults to 0.0625
        temp_kernel_size (int, optional): the filter size for the
            temporal dimension of the 3x3x3 convolutinal layer.
            Defaults to 3.
    '''
    super(ResStage, self).__init__(name='res_stage_%u' %ResStage._stage_index)
    ResStage._stage_index += 1

    self._bn_cfg = bn_cfg
    self.stage = K.Sequential()
    self._inner_channels = inner_channels

    # the second layer of the first block of each stage
    # does spatial downsampling with a stride of 2
    # the input for subsequent layers is the output
    # from the preceeding layer
    for i in range(depth):
      self.stage.add(
          ResBlock(
            bn_cfg=self._bn_cfg,
            se_ratio=se_ratio,
            temp_kernel_size=temp_kernel_size,
            stride=2 if i == 0 else 1,
            channels=(
              in_channels if i == 0 else out_channels,
              inner_channels,
              out_channels)))

  def call(self, input, training=False):
    return self.stage(input, training=training)

class AdaptiveAvgPool3D(K.Layer):
  '''
  Implementation of AdaptiveAvgPool3D as used in pyTorch impl.
  '''
  def __init__(self,
              spatial_out_shape=(1, 1, 1),
              data_format='channels_last',
              **kwargs) -> None:
    super(AdaptiveAvgPool3D, self).__init__(**kwargs)
    assert len(spatial_out_shape) == 3, "Please specify 3D shape"
    assert data_format in ('channels_last', 'channels_first')

    self.data_format = data_format
    self.out_shape = spatial_out_shape
    self.avg_pool = K.layers.GlobalAveragePooling3D()

  def call(self, input):
    out = self.avg_pool(input)
    if self.data_format == 'channels_last':
      return tf.reshape(
          out,
          shape=(
            -1,
            self.out_shape[0],
            self.out_shape[1],
            self.out_shape[2],
            out.shape[1]))
    else:
      return tf.reshape(
          out,
          shape=(
            -1,
            out.shape[1],
            self.out_shape[0],
            self.out_shape[1],
            self.out_shape[2]))
