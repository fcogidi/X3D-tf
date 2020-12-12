from decord import VideoReader
from decord import bridge
import tensorflow as tf

def data_gen(label_file='dataset/kinetics400/train.txt'):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split(' ')
            link = parts[0]
            label = int(parts[1])
            bridge.set_bridge('tensorflow')
            vr = VideoReader(link, num_threads=8)
            video = vr.get_batch(range(0, 31, 2))
            yield (video, label)

dataset = tf.data.Dataset.from_generator(data_gen,
                                        (tf.uint8, tf.int32),
                                        (tf.TensorShape((None, None, None, 3)), tf.TensorShape([])))

print(list(dataset.take(1).as_numpy_iterator()))
