"""loads a TFRecord file and writes the serialized frames to disk
using the class label as part of the file name. This is intended
for verifying the contents of a TFRecord file. 
"""
import time
import json
import random
import imageio
import numpy as np
import tensorflow as tf
from absl import app, flags

import utils
import dataloader
from configs.default import get_default_config

flags.DEFINE_string('cfg_file', None,
                    'Number of videos to sample from the dataset.')
flags.DEFINE_string('file_pattern', None,
                    'Number of videos to sample from the dataset.')
flags.DEFINE_string('label_map_file', None,
                    'Number of videos to sample from the dataset.')
flags.DEFINE_integer('num_samples', 10,
                    'Number of videos to sample from the dataset.')
flags.DEFINE_bool('eval', False,
                    'Number of videos to sample from the dataset.')

flags.mark_flags_as_required(['cfg_file', 'label_map_file', 'file_pattern'])
FLAGS = flags.FLAGS

def main(_):
  assert '.yaml' in FLAGS.cfg_file, 'Please provide path to yaml file.'
  cfg = get_default_config()
  cfg.merge_from_file(FLAGS.cfg_file)
  cfg.freeze()

  assert '.json' in FLAGS.label_map_file, 'Please provide path to yaml file.'
  with open(FLAGS.label_map_file, 'r') as f:
    text_to_id = json.load(f)
    id_to_text = {id:text for text, id in text_to_id.items()}

  assert FLAGS.file_pattern is not None, 'Please provide a file pattern for tfrecord files.'
  loader = dataloader.InputReader(cfg, not FLAGS.eval, True)
  dataset = loader(FLAGS.file_pattern)

  tik = time.time()
  dataset = list(dataset.take(FLAGS.num_samples).as_numpy_iterator())
  print(f'Reading {FLAGS.num_samples} files took {time.time() - tik}s')

  views = cfg.TEST.NUM_TEMPORAL_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

  tik = time.time()
  for example in dataset:
    if FLAGS.eval:
      shapes = example[0].shape
      frames = np.reshape(example[0],
                (shapes[0] * shapes[1] * shapes[2], shapes[3], shapes[4], shapes[5]))
    else:
      frames = np.squeeze(example[0])
    label = id_to_text[example[1]]
    file_index = random.randint(100, 10000)
    with imageio.get_writer(f'{label}_{file_index}.mp4', fps=25) as writer:
      # NOTE for kinetics dataset: the frame rate used to write the
      # video may not correspond to the original video frame rate.
      # This can be rectified by including the average fps in the tfrecord file.
      frames = utils.denormalize(frames, cfg.DATA.MEAN, cfg.DATA.STD)
      for frame in frames:
        writer.append_data(frame.numpy())
  print(f'Writing {FLAGS.num_samples} files took {time.time() - tik}s')

if __name__ == "__main__":
  app.run(main)