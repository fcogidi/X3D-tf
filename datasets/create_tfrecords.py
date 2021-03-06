import os
import json
import glob
import math
import decord
import imageio
import numpy as np
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
from absl import app, flags, logging

SET = {'train', 'val', 'test'}
SUPPORTED_FILETYPES = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}

flags.DEFINE_string('video_dir', None,
                    'Name of directory containing video files.')
flags.DEFINE_string('label_map', None,
                    'Path to .json file containing mapping between class name and id.')
flags.DEFINE_string('output_dir', None,
                    'Name of output directory.')
flags.DEFINE_string('set', 'train',
                    'Dataset to use (train, val or test')
flags.DEFINE_list('extensions', list(SUPPORTED_FILETYPES),
                  'Video formats to search for and decode.')
flags.DEFINE_string('test_annotations', None,
                    'Path to JSON file containing test labels (mainly for Kinetics dataset).')
flags.DEFINE_integer('files_per_record', 32,
                    'Number of files to store in a single TFRecord.')

flags.mark_flags_as_required(['video_dir', 'label_map', 'output_dir'])

FLAGS = flags.FLAGS

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_tf_example(frames, class_id) -> tf.train.Example:  
  encoded_frames = [tf.image.encode_jpeg(frame, format='rgb', quality=90, optimize_size=True)
                    for frame in frames]
  frame_bytes = [bytes_feature(frame.numpy()) for frame in encoded_frames]

  sequence = tf.train.FeatureLists(
    feature_list = {
      'video': tf.train.FeatureList(feature=frame_bytes)
  })

  context = tf.train.Features(
    feature = {
      'video/num_frames':
        int64_feature(frames.shape[0]),
      'video/class/label':
        int64_feature(class_id),
  })

  tf_example = tf.train.SequenceExample(context=context, feature_lists=sequence)

  return tf_example
  
def write_tfrecord(paths, label_map, annotations, process_id, num_shards):
  tfr_options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)
  tfr_path = FLAGS.output_dir + '-{}-{}-of-{}.tfrecord'.format(FLAGS.set, process_id, num_shards)

  with tf.io.TFRecordWriter(tfr_path, tfr_options) as writer:
    for path in paths:
      filename = os.path.basename(path).split('.')[0]

      # get class label (string) and class id (integer)
      if annotations:
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
      # NOTE: this further elimnates any guarantees that
      # `files_per_record` videos will fit into a single
      # tfrecord file.
      try:
        '''decord.bridge.set_bridge('tensorflow')
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        fps = math.ceil(vr.get_avg_fps())
        num_frames = min(len(vr), fps*10) # get at most 10 seconds of video
        frames = vr.get_batch(range(num_frames))'''
        vr = imageio.get_reader(path, 'ffmpeg')
        fps = math.ceil(vr.get_meta_data()['fps'])
        frames = np.stack(list(vr.iter_data()))
        num_frames = min(frames.shape[0], fps*10) # get at most 10 seconds of video
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

  # create TFRecord output directory if it does not exist
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

  files_per_record = max(1, FLAGS.files_per_record)

  # get files
  files = []
  for ext in FLAGS.extensions:
    if ext in SUPPORTED_FILETYPES:
      files.extend(glob.glob(os.path.join(video_dir, '**', '*' + ext),
        recursive=True))
    else:
      logging.info(f'{ext} format not supported. Skipping...')

  returns = []
  process_id = 0
  num_files = len(files)
  num_workers = multiprocessing.cpu_count() # set to lower number if running out of memory
  num_splits = round(num_files / (num_workers * files_per_record))

  num_shards = num_workers * num_splits
  file_split = np.array_split(files, num_splits)
  pbar = tqdm(total=num_files, desc=f'Writing {FLAGS.set} set to TFRecord')

  def update(*a):
    pbar.update(files_per_record)
      
  for big_chunk in file_split:
    # NOTE: splitting further does not guarantee that
    # `files_per_record` videos will fit into a single
    # tfrecord, especially for small datasets
    # TODO: fix this issue.
    smaller_chunk = np.array_split(big_chunk, num_workers)
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
