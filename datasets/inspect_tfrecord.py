"""loads a TFRecord file and writes the serialized frames to disk
using the class label as part of the file name. This is intended
for verifying the contents of a TFRecord file. 
"""
import time
import json
import random
import imageio
import tensorflow as tf
from absl import app, flags

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
  loader = dataloader.InputReader(cfg, FLAGS.eval, True)
  dataset = loader(FLAGS.file_pattern, FLAGS.num_samples)

  tik = time.time()
  dataset = dataset.take(1)
  print(f'Reading {FLAGS.num_samples} files took {time.time() - tik}s')

  tik = time.time()
  for elem in list(dataset.as_numpy_iterator()):
    frames = elem[0]
    label = id_to_text[elem[1]]
    file_index = random.randint(100, 1000)
    with imageio.get_writer(f'{label}_{file_index}.mp4', fps=25) as writer:
      # NOTE for kinetics dataset: the frame rate used to write the
      # video may not correspond to the original video frame rate.
      # This can be rectified by including the average fps in the tfrecord file.
      for frame in frames:
        writer.append(frame)
  print(f'Writing {FLAGS.num_samples} files took {time.time() - tik}s')

if __name__ == "__main__":
  app.run(main)