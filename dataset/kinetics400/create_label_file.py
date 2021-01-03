import os
from absl import app, flags, logging

flags.DEFINE_string('data_dir', None,
                    'Name of directory containing data files.')
flags.DEFINE_string('path_to_label_map', None,
                    'Path to .txt file containing labels')
flags.DEFINE_string('output_path', None,
                    'Name of output file.')
flags.DEFINE_integer('sample_size', None,
                    'Number of samples to include from each category.')

flags.mark_flags_as_required(['data_dir', 'path_to_label_map', 'output_path'])

FLAGS = flags.FLAGS

def main(_):
  data_dir = FLAGS.data_dir
  if not data_dir or not os.path.isdir(data_dir):
    raise ValueError('Please provide valid directory for the annotation files.')

  label_path = FLAGS.path_to_label_map
  if not label_path or not '.txt' in label_path:
    raise ValueError('Please provide valid path to label map.')

  out_path = FLAGS.output_path
  if not out_path or len(out_path.split('.')) < 1:
    raise ValueError('Please provide valid path to output file.')
  
  label_map = {}
  with open(label_path, 'r') as f:
    for index, line in enumerate(f):
      label = line.strip().replace(' ', '_')
      label_map[label] = index

  # open output file
  with open(out_path, 'w') as writer:
    # walk the top-level folders
    for dirpath, sub_dirs, _ in os.walk(data_dir):
      for sub_dir in sub_dirs:
        # get the path of each sub-directory
        index = label_map[sub_dir]
        sub_dirpath = os.path.join(dirpath, sub_dir)
        for path, _, files in os.walk(sub_dirpath):
          count = 0
          for file in files:
            if FLAGS.sample_size is None or count < FLAGS.sample_size:
              filepath = os.path.join(path, file)
              writer.write('{} {}\n'.format(filepath, index))
            count += 1

if __name__ == "__main__":
  app.run(main)