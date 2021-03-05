import tensorflow as tf
from PIL import Image
import imageio
import time
import json

def decode_frames(serialized_example):
    seq_feat = {
        'video': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }

    ctx_feat = {
        'video/num_frames': tf.io.FixedLenFeature([], tf.int64, -1),
        'video/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
    }
    ctx, seq = tf.io.parse_single_sequence_example(
        serialized_example, context_features=ctx_feat, sequence_features=seq_feat)

    # extract the expected shape 
    indices = tf.range(0, ctx['video/num_frames'])
    video = tf.map_fn(lambda i: tf.image.decode_jpeg(seq['video'][i]),
                    indices, fn_output_signature=tf.uint8)
    label = ctx['video/class/label']
    
    return video, label

def decode_serialized_tensor(example):
    tf_features = {
        'video': tf.io.VarLenFeature(tf.string),
        'video/num_frames': tf.io.FixedLenFeature([], tf.int64, -1),
        'video/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
    }
    parsed_tensors = tf.io.parse_single_example(example, tf_features)
    video = tf.io.parse_tensor(parsed_tensors['video'], out_type=tf.uint8)
    label = parsed_tensors['video/class/label']

    return video, label

dataset = tf.data.Dataset.list_files('tfrecord/kin400-val*', shuffle=True)
dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(
            filename,
            compression_type='GZIP',
            num_parallel_reads=tf.data.experimental.AUTOTUNE).prefetch(1),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

options = tf.data.Options()
options.experimental_optimization.map_vectorization.enabled = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
dataset = dataset.with_options(options)

dataset = dataset.shuffle(128)

dataset = dataset.map(lambda value: decode_frames(value),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

tik = time.time()
for idx, i in enumerate(list(dataset.take(10).as_numpy_iterator())):
    print(i[1])
    with open('datasets/kinetics400/label_map.json', 'r') as f:
        label_map = json.load(f)
    if (idx % 10) == 0:
        for label, id in label_map.items():
            if id == i[1]:
                writer = imageio.get_writer(f'{label}_{idx}.mp4', fps=25)

        for im in i[0]:
            writer.append_data(im)
        writer.close()
print(f'Reading files took {time.time() - tik}s')