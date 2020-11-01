from Bottleneck import Bottleneck
import tensorflow as tf
import time

if __name__ == "__main__":
    input = tf.ones(shape=(1, 16, 224, 224, 3))
    input = tf.keras.activations.swish(input)
    res_block = Bottleneck(24, 24, 3)

    start_time = time.time()
    output = res_block(input)
    print("Runtime: %.4fs" %(time.time() - start_time))
    #tf.print('Input: ', input)
    #tf.print('Output: ', output)