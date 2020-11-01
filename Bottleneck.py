import tensorflow as tf

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, 
                filters1: int, 
                filters2: int, 
                filters3: int) -> None:
        super(Bottleneck, self).__init__()
        self.block1 = tf.keras.layers.Conv3D(filters=filters1, 
                                            kernel_size=1,
                                            strides=(1, 1, 1),
                                            padding='same',
                                            use_bias=False,
                                            data_format='channels_last')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.block2 = tf.keras.layers.Conv3D(filters=filters2, 
                                            kernel_size=3,
                                            strides=(1, 1, 1),
                                            padding='same',
                                            use_bias=False,
                                            data_format='channels_last')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.Activation('relu')
        self.block3 = tf.keras.layers.Conv3D(filters=filters3, 
                                            kernel_size=1,
                                            strides=(1, 1, 1),
                                            padding='same',
                                            use_bias=False,
                                            data_format='channels_last')
        self.batch_norm3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_op = tf.keras.layers.Add()
        self.swish = tf.keras.layers.Activation('swish')

    def call(self, input, training=False):
        layer = self.block1(input)
        layer = self.batch_norm1(layer)
        layer = self.relu1(layer)
        layer = self.block2(layer)
        layer = self.batch_norm2(layer)
        layer = self.relu2(layer)
        layer = self.block3(layer)
        layer = self.batch_norm3(layer)
        
        output = self.add_op([input, layer])
        output = self.swish(output)

        return output
