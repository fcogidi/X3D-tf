from tensorflow import reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.python.ops import data_flow_ops

class Bottleneck(Layer):
    '''
    X3D Bottleneck block: 1x1x1, Tx3x3, 1x1x1 with squeeze-excitation
        added at every 2 stages.
    '''
    def __init__(self,
                channels: tuple,
                stride: int = 1,
                eps: float = 1e-5,
                bn_mnt: float = 0.1,
                block_index: int = 0,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3) -> None:
        '''Constructs a single X3D bottleneck block.

        Args:
            channels (tuple): number of channels for each layer in the bottleneck
                order: (number of channles in first two layers, number of channels in last layer)
            stride (int): stride in the spatial dimension
            eps (float): epsilon for batch norm (default: 1e-5)
            bn_mnt (float): momentum for batch norm (default: 0.1)
            block_index (int): the index of the current block
            se_ratio (float): the width multiplier for the squeeze-excitation
                operation (default: 0.0625)
            temp_kernel_size (int): number of filters to use in the temporal 
                dimension for 3x3x3 conv (default: 3)
        '''
        super(Bottleneck, self).__init__()
        assert len(channels) == 2, "Please provide two channels for the bottleneck block"
        self.block_index = block_index
        self.a = Conv3D(filters=channels[0], 
                        kernel_size=1,
                        strides=(1, 1, 1),
                        padding='same',
                        use_bias=False,
                        data_format='channels_last')
        self.bn_a = BatchNormalization(axis=-1,
                                        momentum=bn_mnt, 
                                        epsilon=eps)
        self.relu1 = Activation('relu')
        self.b = Conv3D(filters=channels[0], 
                        kernel_size=(temp_kernel_size, 3, 3),
                        strides=(1, stride, stride), # why?
                        padding='same',
                        use_bias=False,
                        data_format='channels_last')
        self.bn_b = BatchNormalization(axis=-1,
                                        momentum=bn_mnt, 
                                        epsilon=eps)
        self.relu2 = Activation('relu')

        # Squeeze-and-Excite operation
        if (self.block_index % 2 == 0):
            width = self._round_width(channels[0], se_ratio)
            self.se_pool = AdaptiveAvgPool3D((1, 1, 1)) 
            self.se_fc1 = Conv3D(filters=width,
                                kernel_size=1,
                                strides=1, 
                                use_bias=True, 
                                activation='relu')
            self.se_fc2 = Conv3D(filters=channels[0],
                                kernel_size=1, 
                                strides=1, 
                                use_bias=True, 
                                activation='sigmoid')

        self.c = Conv3D(filters=channels[1], 
                        kernel_size=1,
                        strides=(1, 1, 1),
                        padding='same',
                        use_bias=False,
                        data_format='channels_last')
        self.bn_c = BatchNormalization(axis=-1)

    def call(self, input, training=False):
        out = self.a(input)
        out = self.bn_a(out, training=training)
        out = self.relu1(out)
        out = self.b(out)
        out = self.bn_b(out, training=training)
        out = self.relu2(out)
        if (self.block_index % 2 == 0):
            se = self.se_pool(out)
            se = self.se_fc1(se)
            se = self.se_fc2(se)
            out = out * se
        out = self.c(out)
        out = self.bn_c(out, training=training)

        return out

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        from: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/operators.py
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(
            min_width, int(width + divisor / 2) // divisor * divisor
        )
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

class ResBlock(Layer):
    '''
    X3D residual stage: a single residual block
    '''
    _block_index = 1
    def __init__(self,
                channels: tuple,
                stride: int = 1,
                eps: float = 1e-5,
                bn_mnt: float = 0.1,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3) -> None:
        '''
        Constructs a single X3D residual block.

        Args:
            channels (tuple): (input_channels, inner_channels, output_channels) 
            stride (int): stride in the spatial dimension
            eps (float): epsilon for batch norm (default: 1e-5)
            bn_mnt (float): momentum for batch norm (default: 0.1)
            se_ratio (float): the width multiplier for the squeeze-excitation
                operation (default: 0.0625)
            temp_kernel_size (int): number of filters to use in the temporal 
                dimension for 3x3x3 conv (default: 3)
        '''
        super(ResBlock, self).__init__(name='ResBlock_%u' %ResBlock._block_index)
        assert len(channels) == 3, "Please provide three channels for the residual block"
        self.in_channels = channels[0]
        self.inner_channels = channels[1]
        self.out_channels = channels[2]
        ResBlock._block_index += 1

        if (self.in_channels != self.out_channels or stride != 1):
            self.residual = Conv3D(filters=self.out_channels,
                                    kernel_size=(1, 1, 1),
                                    strides=(1, stride, stride),
                                    padding='valid',
                                    use_bias=False,
                                    data_format='channels_last')
            self.bn_r = BatchNormalization(axis=-1,
                                            epsilon=eps,
                                            momentum=bn_mnt)
        self.bottleneck = Bottleneck(channels[1:],
                                    stride,
                                    eps,
                                    bn_mnt,
                                    ResBlock._block_index,
                                    se_ratio,
                                    temp_kernel_size)
        self.add_op = Add()
        self.relu = Activation('relu')
        
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

class AdaptiveAvgPool3D(Layer):
    '''
    Implementation of AdaptiveAvgPool3D is used in pyTorch impl.
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
        self.avg_pool = GlobalAveragePooling3D()

    def call(self, input):
        out = self.avg_pool(input)
        if self.data_format == 'channels_last':
            return reshape(out,
                            shape=(-1, 
                                    self.out_shape[0],
                                    self.out_shape[1],
                                    self.out_shape[2],
                                    out.shape[1]))
        else:
            return reshape(out,
                            shape=(-1, 
                                    out.shape[1],
                                    self.out_shape[0],
                                    self.out_shape[1],
                                    self.out_shape[2]))