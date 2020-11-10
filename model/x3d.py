import math
from yacs.config import CfgNode
from block_helper import ResStage
from block_helper import AdaptiveAvgPool3D

from tensorflow import pad
from tensorflow import constant
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

class X3D(Model):
    '''
    Constructs the X3D model, given the model configurations
    See: https://arxiv.org/abs/2004.04730v1
    '''
    def __init__(self,
                cfg: CfgNode):
        '''
        Initialize an instance of the model given the model configurations

        Args:
            cfg (CfgNode): the model configurations
        '''
        super(X3D, self).__init__()
        self.num_classes = cfg.NETWORK.NUM_CLASSES
        self._bn_cfg = cfg.NETWORK.BN

        # this block deals with the casw where the width expansion factor is not 1.
        # In the paper, a width factor of 1 results in 24 features from the conv_1
        # layer. If NETWORK.SCALE_RES2 is True, then the width expansion factor is
        # applied to a channel size of 12
        if cfg.NETWORK.SCALE_RES2: # apply width_factor directly to channel dimension
            self._conv1_dim = self._round_width(cfg.NETWORK.C1_CHANNELS,
                                                cfg.NETWORK.WIDTH_FACTOR)
            self._multiplier = 1 # increase the width by 2 for the other blocks
        else: # increase the number of channels by a factor of 2
            self._conv1_dim = self._round_width(cfg.NETWORK.C1_CHANNELS, 2)
            self._multiplier = 2 

        # [depth, num_channels] for each residual stage
        self._block_basis = [
                            [1, cfg.NETWORK.C1_CHANNELS * self._multiplier],
                            [2, self._round_width(cfg.NETWORK.C1_CHANNELS * self._multiplier, 2)],
                            [5, self._round_width(cfg.NETWORK.C1_CHANNELS * self._multiplier, 4)],
                            [3, self._round_width(cfg.NETWORK.C1_CHANNELS * self._multiplier, 8)]
                            ]

        # the first layer of the model before the residual stages
        self.conv1 = X3D_Stem(bn_cfg=cfg.NETWORK.BN,
                            out_channels=self._conv1_dim,
                            temp_filter_size=cfg.NETWORK.C1_TEMP_FILTER)

        self.stages = []
        out_dim = self._conv1_dim

        for block in self._block_basis:
            # the output of the previous block 
            # is the input of the current block
            in_dim = out_dim 

            # apply expansion factors
            out_dim = self._round_width(block[1], cfg.NETWORK.WIDTH_FACTOR)
            inner_dim = int(out_dim * cfg.NETWORK.BOTTLENECK_WIDTH_FACTOR)
            block_depth = self._round_repeats(block[0], cfg.NETWORK.DEPTH_FACTOR)

            stage = ResStage(in_channels=in_dim,
                            inner_channels=inner_dim,
                            out_channels=out_dim,
                            depth = block_depth,
                            bn_cfg=self._bn_cfg)
            self.stages.append(stage)

        self.conv5 = Sequential(name='conv_5')
        self.conv5.add(Conv3D(filters=self.stages[-1]._inner_channels,
                            kernel_size=(1, 1, 1),
                            strides=(1, 1, 1),
                            padding='valid',
                            use_bias=False,
                            data_format='channels_last'))
        self.conv5.add(BatchNormalization(axis=-1,
                                        epsilon=self._bn_cfg.EPS,
                                        momentum=self._bn_cfg.MOMENTUM))
        self.conv5.add(Activation('relu'))
        self.pool5 = AdaptiveAvgPool3D((1, 1, 1), name='pool_5') 
        self.fc1 = Conv3D(filters=2048,
                        kernel_size=1,
                        strides=1, 
                        use_bias=False, 
                        activation='relu',
                        name='fc_1')
        self.dropout = Dropout(rate=cfg.NETWORK.DROPOUT_RATE)
        self.fc2 = Dense(units=self.num_classes,
                        activation='softmax',
                        use_bias=True,
                        name='fc_2') 

    def call(self, input, training=False):
        out = self.conv1(input)
        for stage in self.stages:
            out = stage(out, training=training)
        out = self.conv5(out, training=training)
        out = self.pool5(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def summary(self, input_shape):
        x = Input(shape=input_shape)
        model = Model(inputs=x, outputs=self.call(x), name='X3D')
        return model.summary()

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """
        Round width of filters based on width multiplier
        from: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py

        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int, optional): the minimum width after multiplication.
                Defaults to 8.
            divisor (int, optional): the new width should be dividable by divisor.
                Defaults to 8.        
        """
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(
            min_depth, int(width + divisor / 2) // divisor * divisor
        )
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """
        Round number of layers based on depth multiplier.
        Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
        """
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

class X3D_Stem(Layer):
    '''
    X3D_Stem: the first layer of the X3D network, connecting
    the data layer and the residual stages. Applies channel-wise
    separable convolution.
    '''
    def __init__(self,
                bn_cfg: CfgNode,
                out_channels: int = 24,
                temp_filter_size: int = 5):
        '''
        Args:
            bn_cfg (CfgNode) containing parameters for the batch
                normalization layer
            out_channels (int): number of filters to use in
                the convolutional layers
            temp_filter_size (int): the filter size for the
                temporal convolution
        '''
        super(X3D_Stem, self).__init__(name='conv_1')

        self.bn_momentum = bn_cfg.MOMENTUM
        self.bn_eps = bn_cfg.EPS

        # represents spatial padding of size (0, 1, 1)
        self._spatial_paddings = constant([
                                        [0, 0],
                                        [0, 0],
                                        [1, 1],
                                        [1, 1],
                                        [0, 0]])

        # represents temporal padding of size
        # (temp_filter_size//2, 0, 0)
        self._temp_paddings = constant([
                                    [0, 0],
                                    [temp_filter_size//2, temp_filter_size//2],
                                    [0, 0], 
                                    [0, 0], 
                                    [0, 0]])

        # spatial convolution
        self.conv_s = Conv3D(filters=out_channels, 
                            kernel_size=(1, 3, 3),
                            strides=(1, 2, 2),
                            use_bias=False,
                            data_format='channels_last')
        
        # temporal convolution
        self.conv_t = Conv3D(filters=out_channels, 
                            kernel_size=(temp_filter_size, 1, 1),
                            strides=(1, 1, 1),
                            use_bias=False,
                            groups=out_channels,
                            data_format='channels_last')
                                
        self.bn = BatchNormalization(axis=-1, 
                                    momentum=self.bn_momentum,
                                    epsilon=self.bn_eps)
        self.relu = Activation('relu')

    def call(self, input, training=False):
        out = pad(input, self._spatial_paddings)
        out = self.conv_s(out)
        out = pad(out, self._temp_paddings)
        out = self.conv_t(out)
        out = self.bn(out, training=training)
        out = self.relu(out)

        return out
