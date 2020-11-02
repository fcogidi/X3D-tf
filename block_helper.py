from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization


class Bottleneck(Layer):
    '''
    X3D Bottleneck block: 1x1x1, Tx3x3, 1x1x1 with squeeze-excitation
        added at every 2 stages.
    '''
    def __init__(self,
                out_channels: int,
                stride: int = 1,
                eps: float = 1e-5,
                bn_mnt: float = 0.1,
                block_index: int = 0,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3) -> None:
        '''Constructs a single X3D bottleneck block.

        Args:
            out_channels (int): number of output channels
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
        self.block_index = block_index
        self.a = Conv3D(filters=out_channels, 
                        kernel_size=1,
                        strides=(1, 1, 1),
                        padding='same',
                        use_bias=False,
                        data_format='channels_last')
        self.bn_a = BatchNormalization(axis=-1,
                                        momentum=bn_mnt, 
                                        epsilon=eps)
        self.relu1 = Activation('relu')
        self.b = Conv3D(filters=out_channels, 
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
            width = self._round_width(out_channels, se_ratio)
            self.se_pool = AveragePooling3D(pool_size=(1, 1, 1))
            self.se_fc1 = Conv3D(filters=width,
                                kernel_size=1,
                                strides=1, 
                                use_bias=True, 
                                activation='relu')
            self.se_fc2 = Conv3D(filters=out_channels,
                                kernel_size=1, 
                                strides=1, 
                                use_bias=True, 
                                activation='sigmoid')

        self.c = Conv3D(filters=out_channels, 
                        kernel_size=1,
                        strides=(1, 1, 1),
                        padding='same',
                        use_bias=False,
                        data_format='channels_last')
        self.bn_c = BatchNormalization(axis=-1)

    def call(self, input, training=False):
        out = self.a(input)
        out = self.bn_a(out)
        out = self.relu1(out)
        out = self.b(out)
        out = self.bn_b(out)
        out = self.relu2(out)
        if (self.block_index % 2 == 0):
            se = self.se_pool(out)
            se = self.se_fc1(se)
            se = self.se_fc2(se)
            out = out * se
        out = self.c(out)
        out = self.bn_c(out)

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
    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: int = 1,
                eps: float = 1e-5,
                bn_mnt: float = 0.1,
                block_index: int = 0,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3) -> None:
        '''
        Constructs a single X3D residual block.

        Args:
            out_channels (int): number of output channels
            stride (int): stride in the spatial dimension
            eps (float): epsilon for batch norm (default: 1e-5)
            bn_mnt (float): momentum for batch norm (default: 0.1)
            block_index (int): the index of the current block
            se_ratio (float): the width multiplier for the squeeze-excitation
                operation (default: 0.0625)
            temp_kernel_size (int): number of filters to use in the temporal 
                dimension for 3x3x3 conv (default: 3)
        '''
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if (self.in_channels != self.out_channels and stride != 1):
            self.residual = Conv3D(filters=out_channels,
                                    kernel_size=(1, 1, 1),
                                    strides=(1, stride, stride),
                                    padding='valid',
                                    use_bias=False,
                                    data_format='channels_last')
            self.bn_r = BatchNormalization(axis=-1,
                                            epsilon=eps,
                                            momentum=bn_mnt)
        self.bottleneck = Bottleneck(out_channels,
                                    stride,
                                    eps,
                                    bn_mnt,
                                    block_index,
                                    se_ratio,
                                    temp_kernel_size)
        self.add_op = Add()
        self.relu = Activation('relu')
        
    def call(self, input, training=False):
        out = self.bottleneck(input)
        if hasattr(self, 'residual'):
            res = self.residual(input)
            res = self.bn_r(res)
            out = self.add_op([res, out])
        else:
            out = self.add_op([input, out])
        out = self.relu(out)

        return out
