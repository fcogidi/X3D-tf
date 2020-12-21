import time

from transforms import TemporalTransforms, SpatialTransforms
from model.config import get_default_config
import dataloader

cfg = get_default_config()
cfg.merge_from_file('configs/kinetics/X3D_M.yaml')
cfg.freeze()

start = time.time()
train_data = dataloader.InputReader(cfg, True)
data = train_data(4)
print('Loading training data...')
#list(data.take(1000).as_numpy_iterator())
for video, label in data.take(1000):
  print(video.shape)

print('Runtime:', time.time() - start, 's')

print('\nLoading validation data...')
start = time.time()
train_data = dataloader.InputReader(cfg, False)
data = train_data(None, 3)
list(data.take(100).as_numpy_iterator())

'''for video, label in data.take(10):
  print(video.shape)'''
print('Runtime:', time.time() - start, 's')
