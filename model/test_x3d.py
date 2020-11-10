import time
import tensorflow as tf

from x3d import X3D
from config import get_default_config

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

cfg = get_default_config()
cfg.merge_from_file('configs/kinetics/X3D_M.yaml')
cfg.freeze()

input = tf.random.uniform(shape=(1, cfg.DATA.TEMP_DURATION, cfg.DATA.SPATIAL_RES, cfg.DATA.SPATIAL_RES, cfg.DATA.NUM_INPUT_CHANNELS), maxval=256)
model = X3D(cfg)
start_time = time.time()
summary = model.summary(input.shape[1:])
runtime = time.time() - start_time
print(summary)
print("Runtime: %.4fs" %runtime)

'''with open('docs/model_summary/X3D_M.txt', 'w') as f:
    f.write(str(summary))'''