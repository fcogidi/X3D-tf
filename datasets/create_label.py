"""creates a .txt file containing the path to videos in a folder
and thier corresponding class id. eg.
  path/to/video.mp4 6
  path/to/video.mkv 5
  path/to/video.avi 0

The program expects the folder containing the videos to have the following
structure:
  -- class_name_1
      -- video_1.mp4
      -- video_2.mp4
  -- class_name_2
      -- video_1.mp4
      -- video_2.mp4
A mapping of the class names and id needs to be provided as well.
"""
import os
import json
import glob
from absl import app, flags, logging

SUPPORTED_FILETYPES = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}

flags.DEFINE_string('data_dir', None,
                    'Name of directory containing dataset.')
flags.DEFINE_string('path_to_label_map', None,
                    'Path to .json file containing class label mapping to class id.')
flags.DEFINE_string('output_path', None,
                    'Path to to .txt file to write output.')
flags.DEFINE_string('test_json_file', None,
                    'Path to .json file containing Kinetics-400 test labels.')
flags.DEFINE_list('file_extensions', list(SUPPORTED_FILETYPES),
                  'List of video formats to search for and decode.')

flags.mark_flags_as_required(['data_dir', 'path_to_label_map', 'output_path'])
FLAGS = flags.FLAGS

def main(_):
  data_dir = FLAGS.data_dir
  if not data_dir or not os.path.isdir(data_dir):
    raise ValueError('Please provide valid directory for the annotation files.')

  label_path = FLAGS.path_to_label_map
  if not label_path or not '.json' in label_path:
    raise ValueError('Please provide valid path to label map.')

  out_path = FLAGS.output_path
  if not out_path or len(out_path.split('.')) < 1:
    raise ValueError('Please provide valid path to output file.')

  test_file = FLAGS.test_json_file
  if test_file is not None and '.json' not in test_file:
    raise ValueError('Please provide valid path to JSON test file.')

  if test_file:
    with open(test_file, 'r') as j:
        annotations = json.load(j)
  else:
    annotations = None
  
  with open(label_path, 'r') as f:
    label_map = json.load(f)

  # get files with supported extension
  file_paths = []
  for ext in FLAGS.file_extensions:
    if ext in SUPPORTED_FILETYPES:
      file_paths.extend(glob.glob(os.path.join(data_dir, '**', '*' + ext),
        recursive=True))
    else:
      logging.info(f'{ext} format not supported. Skipping...')

  # open output file
  with open(out_path, 'w') as writer:
    for file_path in file_paths:
      filename = os.path.basename(file_path).split('.')[0]
      if annotations: # annotations for test set is provide
        # NOTE: this is mainly coded for the kinetics400 dataset
        try:
          class_label = annotations[filename]['annotations']['label']
          class_label = class_label.replace(' ', '_') # replace space with underscore
          class_id = label_map[class_label] # get the integer label for the video
        except KeyError:
          logging.info(f'{filename} not found! Skipping...')
          continue
      else:
        class_label = os.path.basename(os.path.dirname(file_path))
        class_id = label_map[class_label]
      writer.write('{} {}\n'.format(file_path, class_id))

if __name__ == "__main__":
  app.run(main)