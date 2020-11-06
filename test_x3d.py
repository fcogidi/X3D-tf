import time
from x3d import X3D
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

input = tf.random.uniform(shape=(1, 13, 160, 160, 3), maxval=256)

model = X3D()
start_time = time.time()
#model.build(input_shape=(input.shape))
print(model.summary(input.shape[1:]))
print("Runtime: %.4fs" %(time.time() - start_time))