import os
import math
from absl import app
import tensorflow as tf

from model.config import get_default_config
from model import x3d
import dataloader

def setup_model(model, cfg):
  input_shape = (
      None,
      cfg.DATA.TEMP_DURATION,
      cfg.DATA.TRAIN_CROP_SIZE,
      cfg.DATA.TRAIN_CROP_SIZE,
      cfg.DATA.NUM_INPUT_CHANNELS
  )
  model.build(input_shape)
  model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.0, momentum=0.9),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[
          tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=1,
              name='top_1_acc'
          ),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=5,
              name='top_5_acc'
          )
      ]
  )

  return model

def main(_):
  cfg = get_default_config()
  cfg.merge_from_file('configs/kinetics/X3D_XS.yaml')
  cfg.freeze()

  # DEBUG OPTIONS

  # MULTI_GPU setup

  # MIXED PRECISION setup
  train_batch_size = 2
  test_batch_size = 2

  train_data = dataloader.InputReader(cfg, True)(batch_size=train_batch_size)

  val_data = dataloader.InputReader(cfg, False)(batch_size=test_batch_size)

  #print(list(train_data.take(1).as_numpy_iterator()))

  model = x3d.X3D(cfg)
  model = setup_model(model, cfg)

  num_epochs = 100

  def lr_schedule(epoch, lr):
    base_lr = 1.6
    cosine = tf.math.cos(tf.constant(math.pi) * (epoch/num_epochs))
    lr = base_lr * (0.5 * cosine + 1)

    return lr

  # Allow restarting from checkpoint

  model.fit(
      train_data,
      epochs=num_epochs,
      steps_per_epoch=234619/train_batch_size,
      validation_data=val_data,
      validation_steps=19761/test_batch_size,
      validation_freq=1,
      verbose=1,
      callbacks=[
          tf.keras.callbacks.LearningRateScheduler(lr_schedule, 1),
          tf.keras.callbacks.TensorBoard(
              log_dir='trial',
              histogram_freq=5,
              update_freq='epoch'
          ),
          tf.keras.callbacks.ModelCheckpoint(
              os.path.join('trial', 'ckpt_{epoch:d}'),
              verbose=1,
              save_weights_only=True
          )
      ]

  )

if __name__ == "__main__":
  app.run(main)