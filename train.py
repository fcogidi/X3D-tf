import os
import math
import wandb
import tensorflow as tf
from absl import app, flags, logging
from wandb.keras import WandbCallback


from model.config import get_default_config
from model import x3d
import dataloader

flags.DEFINE_string('config_file_path', None,
                    '(Relative) path to config (.yaml) file.')

flags.mark_flag_as_required('config_file_path')

FLAGS = flags.FLAGS

def setup_model(model, cfg):
  """Compile model with loss function, model optimizers and metrics."""
  loss_fn = tf.keras.optimizers.SGD(lr=0.0, momentum=0.9)
  if cfg.NETWORK.MIXED_PRECISION:
    loss_fn = tf.keras.mixed_precision.experimental.LossScaleOptimizer(loss_fn, 'dynamic')
  model.compile(
      optimizer=loss_fn,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[
          tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=5,
              name='top_5_acc'
          )
      ]
  )

  return model

def main(_):
  cfg_path = FLAGS.config_file_path
  if not cfg_path:
    raise ValueError('Please provide valid path to config file.')
  assert '.yaml' in cfg_path, 'Must be a .yaml file'

  cfg = get_default_config()
  cfg.merge_from_file(cfg_path)
  cfg.freeze()

  # init wandb
  wandb.init(
      job_type='train',
      project='X3D-tf',
      group=cfg_path.split('/')[-1],
      sync_tensorboard=True,
      config=dict(cfg)
  )

  # DEBUG OPTIONS

  # training strategy setup
  avail_gpus = tf.config.list_physical_devices('GPU')
  
  for gpu in avail_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
  if len(avail_gpus) > 1 and cfg.TRAIN.MULTI_GPU:
    train_strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    logging.info('Availalbe GPU devices: %s', avail_gpus)
  elif len(avail_gpus) == 1:
    train_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    train_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  # mixed precision
  precision = 'float32'
  if cfg.NETWORK.MIXED_PRECISION:
    # only set to float16 if gpu is available
    if avail_gpus:
      precision = 'mixed_float16'
      tf.config.optimizer.set_jit(True)
  policy = tf.keras.mixed_precision.experimental.Policy(precision)
  tf.keras.mixed_precision.experimental.set_policy(policy)

  def get_dataset(cfg, is_training):
    return dataloader.InputReader(
        cfg,
        is_training
    )(batch_size=cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE)
  
  with train_strategy.scope():
    def lr_schedule(epoch, lr):
      base_lr = 1.6
      cosine = tf.math.cos(
          tf.constant(math.pi) * (epoch/cfg.TRAIN.EPOCHS))
      lr = base_lr * (0.5 * cosine + 1)

      return lr

    model = x3d.X3D(cfg)
    model = setup_model(model, cfg)

    # allow restarting from checkpoint
    ckpt_path = tf.train.latest_checkpoint(cfg.TRAIN.MODEL_DIR)
    if ckpt_path and (tf.train.list_variables(ckpt_path)[0][0] ==
      '_CHECKPOINTABLE_OBJECT_GRAPH'):
      logging.info(f'Loading from checkpoint {ckpt_path}')
      model.load_weights(ckpt_path)

    model.fit(
        get_dataset(cfg, True),
        epochs=cfg.TRAIN.EPOCHS,
        steps_per_epoch=cfg.TRAIN.DATASET_SIZE/cfg.TRAIN.BATCH_SIZE,
        validation_data=get_dataset(cfg, False),
        validation_steps=cfg.TEST.DATASET_SIZE/cfg.TEST.BATCH_SIZE,
        validation_freq=1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(lr_schedule, 1),
            tf.keras.callbacks.TensorBoard(
                log_dir=cfg.TRAIN.MODEL_DIR,
                profile_batch=0,
                histogram_freq=5,
                update_freq=5000#'epoch'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(cfg.TRAIN.MODEL_DIR, 'ckpt_{epoch:d}'),
                verbose=1,
                save_freq=5000,
                save_weights_only=True
            ),
            WandbCallback(
                verbose=1,
                save_model=True,
                log_evaluation=True,
            )
        ]

    )

if __name__ == "__main__":
  app.run(main)
