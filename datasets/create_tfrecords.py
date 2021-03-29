"""decodes and serializes frames from vidoes in a given directory
    into TFRecord files to improve parallelized I/O and provide
    prefetching benefits.

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
import math
import imageio
import numpy as np
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
from absl import app, flags, logging

SET = {'train', 'val', 'test'}
SUPPORTED_FILETYPES = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}

flags.DEFINE_string('video_dir', None,
                    'Name of directory containing video dataset.')
flags.DEFINE_string('label_map', None,
                    'Path to .json file containing mapping between class name and id.')
flags.DEFINE_string('output_dir', None,
                    'Path to folder to write tfrecord files.')
flags.DEFINE_string('set', 'train',
                    'The subset of the dataset to write to tfrecord format (train, val or test).')
flags.DEFINE_list('extensions', list(SUPPORTED_FILETYPES),
                  'Video formats to search for and decode.')
flags.DEFINE_string('test_annotations', None,
                    'Path to .json file containing test labels (designed for Kinetics dataset).')
flags.DEFINE_integer('videos_per_record', 32,
                    'Number of videos to decode and store in a single tfrecord file.')

flags.mark_flags_as_required(['video_dir', 'label_map', 'output_dir'])
FLAGS = flags.FLAGS

def to_tf_example(frames, class_id):
  """converts a list of frames and corresponding class id to bytes,
  represented in a tf.train.SequenceExample object.

  Args:
      frames (list): A list of numpy arrays representing indiviual
        frames of a video.
      class_id (int): an integer value representing the label of the
        video.

  Returns:
      tf.train.SequenceExample: the frames and class id encoded in a
        ProtocolMessage.
  """
  # encode the frames a JPEG as a way to compress them
  encoded_frames = [tf.image.encode_jpeg(frame, format='rgb', quality=90, optimize_size=True)
                    for frame in frames]
  frame_bytes = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame.numpy()]))
                for frame in encoded_frames]

  sequence = tf.train.FeatureLists(
    feature_list = {
      'video': tf.train.FeatureList(feature=frame_bytes)
  })

  context = tf.train.Features(
    feature = {
      'video/num_frames':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[frames.shape[0]])),
      'video/class/label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
  })

  tf_example = tf.train.SequenceExample(context=context, feature_lists=sequence)

  return tf_example
  
def write_tfrecord(paths, label_map, annotations, process_id, num_shards):
  """writes a list of videos to a tfrecord file.

  Args:
      paths (np.array): a list of paths to video files
      label_map (dict): a mapping of class labels and class ids
      annotations (dict): the groundtruth annotations for the test dataset.
      process_id (int): the index of the process that is currently writing
        the tfrecord file.
      num_shards (int): the total number of tfrecord files that will be created
        for the dataset.

  Returns:
      int: 1 if the process runs successfully.
  """
  tfr_options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)
  tfr_path = FLAGS.output_dir + '-{}-{}-of-{}.tfrecord'.format(FLAGS.set, process_id, num_shards)

  with tf.io.TFRecordWriter(tfr_path, tfr_options) as writer:
    for path in paths:
      logging.debug(f'writing {path}')
      filename = os.path.basename(path).split('.')[0]

      # get class label (string) and class id (integer)
      if annotations: # test set
        try:
          class_label = annotations[filename]['annotations']['label']
          class_label = class_label.replace(' ', '_') # replace space with underscore
          class_id = label_map[class_label] # get the integer label for the video
        except KeyError:
          logging.info(f'{filename} not found! Skipping...')
          continue
      else: # training or validation set
        class_label = os.path.basename(os.path.dirname(path))
        class_id = label_map[class_label]

      # decode video (skip videos that cannot be decoded)
      # NOTE: this removes any guarantees that
      # `videos_per_record` videos will fit into a single
      # tfrecord file.
      try:
        vr = imageio.get_reader(path, 'ffmpeg') # read the video

        # get all the frames
        fps = math.ceil(vr.get_meta_data()['fps'])
        frames = np.stack(list(vr.iter_data()))

        # trim video down to 10 seconds, if possible
        num_frames = min(frames.shape[0], fps*10)
        frames = frames[:num_frames, :, :, :]
      except Exception as e:
        logging.info(e)
        continue
      
      tf_example = to_tf_example(frames, class_id)
      writer.write(tf_example.SerializeToString())
  return 1

def main(_):
  video_dir = FLAGS.video_dir
  if not video_dir or not os.path.isdir(video_dir):
    raise ValueError('Please provide valid directory for videos.')

  label_path = FLAGS.label_map
  if not label_path or not '.json' in label_path:
    raise ValueError('Please provide valid path to label map.')
  
  with open(label_path, 'r') as f:
    label_map = json.load(f)

  # create tfrecord output directory if it does not exist
  output_path = os.path.dirname(FLAGS.output_dir)
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  test_file = FLAGS.test_annotations
  if test_file is not None and '.json' not in test_file:
    raise ValueError('Please provide valid path to JSON test file.')

  assert FLAGS.set in SET
  annotations = None
  if FLAGS.set == 'test':
    with open(test_file, 'r') as j:
      annotations = json.load(j)

  videos_per_record = max(1, FLAGS.videos_per_record)

  # get files with supported extension
  files = []
  for ext in FLAGS.extensions:
    if ext in SUPPORTED_FILETYPES:
      files.extend(glob.glob(os.path.join(video_dir, '**', '*' + ext),
        recursive=True))
    else:
      logging.info(f'{ext} format not supported. Skipping...')
  np.random.shuffle(files)

  returns = []
  process_id = 0
  num_files = len(files)
  num_workers = multiprocessing.cpu_count() # set to lower number if running out of memory

  # The list of file paths is split twice
  # so that the number of videos in a file is roughly
  # equal to the `videos_per_record parameter`.
  # This is based on https://gebob19.github.io/tfrecords/
  # Splitting all the files into smaller chunks improves
  # writing speed.
  # NOTE: splitting further does not guarantee that
  # `videos_per_record` videos will fit into a single
  # tfrecord, especially for small datasets.
  num_splits = round(num_files / (num_workers * videos_per_record))
  num_shards = num_workers * num_splits # total number of tfrecord files for dataset
  file_split = np.array_split(files, num_splits) # first split

  pbar = tqdm(total=num_files, desc=f'Writing {FLAGS.set} set to TFRecord')

  def update(*a):
    """updates the progress bar after a process exits successfully"""
    pbar.update(videos_per_record)
      
  for big_chunk in file_split:
    smaller_chunk = np.array_split(big_chunk, num_workers) # second split
    pool = multiprocessing.Pool(num_workers)
    for chunk in smaller_chunk:
      r = pool.apply_async(write_tfrecord,
          args=(chunk, label_map, annotations, process_id, num_shards),
          callback=update)
      process_id += 1
      returns.append(r)
    pool.close()
    for r in returns: r.get()
    pool.join()
  pbar.close()

if __name__ == "__main__":
  app.run(main)
