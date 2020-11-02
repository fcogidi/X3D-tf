from Bottleneck import Bottleneck
import tensorflow as tf
import time

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)

tf.random.set_seed(42069)

if __name__ == "__main__":
    input = tf.random.uniform(shape=(1, 8, 112, 112, 3), maxval=256)
    print("Input shape: ", input.shape)

    spatial_paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
    temp_paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0], [0, 0]])

    spatial_input = tf.pad(input, spatial_paddings)
    
    spatial_conv = tf.keras.layers.Conv3D(filters=24, 
                            kernel_size=(1, 3, 3),
                            strides=(1, 2, 2),
                            use_bias=False,
                            data_format='channels_last')(spatial_input)
    print("Spatial conv shape: ", spatial_conv.shape)

    temp_input = tf.pad(spatial_conv, temp_paddings)

    temp_conv = tf.keras.layers.Conv3D(filters=24, 
                            kernel_size=(5, 1, 1),
                            strides=(1, 1, 1),
                            use_bias=False,
                            data_format='channels_last')(temp_input)
    print("Temp. conv shape: ", temp_conv.shape)
    conv1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(temp_conv)

    res_block = Bottleneck(out_channels=24)

    start_time = time.time()
    output = res_block(conv1)

    print("Output shape: ", output.shape)

    add_op = tf.keras.layers.Add()
    swish = tf.keras.layers.Activation('swish')
    print("Runtime: %.4fs" %(time.time() - start_time))
    #tf.print('Input: ', input)
    #tf.print('Output: ', output)