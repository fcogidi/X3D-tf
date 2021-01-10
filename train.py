import os
import math
import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from absl import app, flags, logging

from configs.default import get_default_config
from model import x3d
import dataloader
import utils

flags.DEFINE_string('config_file_path', None,
                    '(Relative) path to config (.yaml) file.')
flags.DEFINE_bool('debug', False,
                  'Whether to run in debug mode.')

flags.mark_flag_as_required('config_file_path')

FLAGS = flags.FLAGS

def load_model(model, cfg, ckpt_path):
  """Compile model with loss function, model optimizers and metrics."""
  if ckpt_path:
    logging.info(f'Loading model from checkpoint {ckpt_path}')
    model = tf.keras.models.load_model(ckpt_path)
  else:
    opt_str = cfg.TRAIN.OPTIMIZER.lower()
    if opt_str == 'sgd':
      opt = tfa.optimizers.SGDW(
          learning_rate=cfg.TRAIN.WARMUP_LR,
          weight_decay=cfg.TRAIN.WEIGHT_DECAY,
          momentum=cfg.TRAIN.MOMENTUM)
    elif opt_str == 'adam':
      opt = tfa.optimizers.AdamW(
          learning_rate=cfg.TRAIN.WARMUP_LR,
          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
      raise NotImplementedError

    if cfg.NETWORK.MIXED_PRECISION:
      opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          opt, 'dynamic')
    
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name='top_5_acc')])

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
  if cfg.WANDB.ENABLE:
    wandb.tensorboard.patch(root_logdir=cfg.TRAIN.MODEL_DIR)
    wandb.init(
        job_type='train',
        group=cfg.WANDB.GROUP_NAME,
        project=cfg.WANDB.PROJECT_NAME,
        sync_tensorboard=cfg.WANDB.TENSORBOARD,
        mode=cfg.WANDB.MODE,
        config=dict(cfg)
    )

  if FLAGS.debug:
    tf.config.run_functions_eagerly(True)
    tf.debugging.set_log_device_placement(True)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(1111)
    logging.set_verbosity(logging.DEBUG)

  # mixed precision
  precision = 'float32'
  if cfg.NETWORK.MIXED_PRECISION:
    # only set to float16 if gpu is available
    if tf.config.list_physical_devices('GPU'):
      precision = 'mixed_float16'
      # TODO: tf.config.optimizer.set_jit(True) # xla
  policy = tf.keras.mixed_precision.experimental.Policy(precision)
  tf.keras.mixed_precision.experimental.set_policy(policy)

  def get_dataset(cfg, is_training):
    """Returns a tf.data dataset"""
    return dataloader.InputReader(
        cfg,
        is_training
    )(cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE)
  
  # learning rate schedule
  def lr_schedule(epoch, lr):
    """
    Implements the learning rate schedule used in
      https://arxiv.org/abs/2004.04730      
    """
    if epoch > cfg.TRAIN.WARMUP_EPOCHS:
      cosine = tf.math.cos(
          tf.constant(math.pi) * (epoch/cfg.TRAIN.EPOCHS))
      new_lr = cfg.TRAIN.BASE_LR * (0.5 * cosine + 1)
    else:
      new_lr = cfg.TRAIN.WARMUP_LR + (epoch *
          (cfg.TRAIN.BASE_LR - cfg.TRAIN.WARMUP_LR) / cfg.TRAIN.WARMUP_EPOCHS)
    return new_lr

  strategy = utils.get_strategy(cfg)

  with strategy.scope():
    model = x3d.X3D(cfg)

    # allow restarting from checkpoint
    ckpt_path = tf.train.latest_checkpoint(cfg.TRAIN.MODEL_DIR)
    model = load_model(model, cfg, ckpt_path)

    model.fit(
        get_dataset(cfg, True),
        epochs=cfg.TRAIN.EPOCHS,
        steps_per_epoch=cfg.TRAIN.DATASET_SIZE/cfg.TRAIN.BATCH_SIZE,
        validation_data=get_dataset(cfg, False),
        verbose=1,
        callbacks=utils.get_callbacks(cfg, lr_schedule, FLAGS.debug)

    )

if __name__ == "__main__":
  app.run(main)
