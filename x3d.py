import math
from six import u
from tensorflow import pad
from tensorflow import constant
from block_helper import ResBlock
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D
from block_helper import AdaptiveAvgPool3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

class X3D(Model):
    def __init__(self,
                width_factor: int = 1,
                depth_factor: int = 1,
                bottleneck_factor: int = 1,
                spatial_res_factor: int = 1,
                num_classes: int = 400,
                frame_rate: int = 1,
                eps: float = 1e-5,
                bn_mnt: float = 0.9):
        super(X3D, self).__init__()
        self.frame_rate = frame_rate
        self.num_classes = num_classes
        self.res2_dim = 3
        self.res3_dim = 5
        self.res4_dim = 11
        self.res5_dim = 7

        self.conv1 = Stem(out_channels=24,
                        temp_filter_size=5)
        self.res2 = [ResBlock(channels=(24, 54, 24), stride=2) if i == 0 
                    else ResBlock(channels=(24, 54, 24)) 
                    for i in range(self.res2_dim)]
        self.res3 = [ResBlock(channels=(24, 108, 48),  stride=2) if i == 0
                    else ResBlock(channels=(48, 108, 48))
                    for i in range(self.res3_dim)]
        self.res4 = [ResBlock(channels=(48, 216, 96), stride=2) if i == 0
                    else ResBlock(channels=(96, 216, 96))
                    for i in range(self.res4_dim)]
        self.res5 = [ResBlock(channels=(96, 432, 192), stride=2) if i == 0
                    else ResBlock(channels=(192, 432, 192))
                    for i in range(self.res5_dim)]
        self.conv5 = Sequential(name='conv_5')
        self.conv5.add(Conv3D(filters=432,
                            kernel_size=(1, 1, 1),
                            strides=(1, 1, 1),
                            padding='valid',
                            use_bias=False,
                            data_format='channels_last'))
        self.conv5.add(BatchNormalization(axis=-1,
                                        momentum=bn_mnt, 
                                        epsilon=eps))
        self.conv5.add(Activation('relu'))
        self.pool5 = AdaptiveAvgPool3D((1, 1, 1), name='pool_5') 
        self.fc1 = Conv3D(filters=2048,
                                kernel_size=1,
                                strides=1, 
                                use_bias=False, 
                                activation='relu',
                                name='fc_1')
        self.fc2 = Dense(units=self.num_classes,
                        activation='softmax',
                        use_bias=True,
                        name='fc_2') 

    def call(self, input, training=False):
        out = self.conv1(input)
        for block in self.res2:
            out = block(out)
        for block in self.res3:
            out = block(out)
        for block in self.res4:
            out = block(out)
        for block in self.res5:
            out = block(out)
        out = self.conv5(out, training=training)
        out = self.pool5(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def summary(self, input_shape):
        x = Input(shape=input_shape)
        model = Model(inputs=x, outputs=self.call(x), name='X3D')
        return model.summary()

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """
        Round width of filters based on width multiplier.
        Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
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

class Stem(Layer):
    def __init__(self,
                out_channels: int = 24,
                temp_filter_size: int = 5):
        super(Stem, self).__init__(name='conv_1')
        self.spatial_paddings = constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        self.temp_paddings = constant([[0, 0], [temp_filter_size//2, temp_filter_size//2], [0, 0], [0, 0], [0, 0]])
                
        self.conv_s = Conv3D(filters=out_channels, 
                                kernel_size=(1, 3, 3),
                                strides=(1, 2, 2),
                                use_bias=False,
                                data_format='channels_last')
        
        self.conv_t = Conv3D(filters=out_channels, 
                                kernel_size=(temp_filter_size, 1, 1),
                                strides=(1, 1, 1),
                                use_bias=False,
                                groups=out_channels,
                                data_format='channels_last')
                                
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
        self.relu = Activation('relu')

    def call(self, input, training=False):
        out = pad(input, self.spatial_paddings)
        out = self.conv_s(out)
        out = pad(out, self.temp_paddings)
        out = self.conv_t(out)
        out = self.bn(out, training=training)
        out = self.relu(out)

        return out
