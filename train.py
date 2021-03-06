import os
import math
import wandb
import tensorflow as tf
from absl import app, flags, logging

from configs.default import get_default_config
from model import X3D
import dataloader
import utils

flags.DEFINE_string('config', None,
    '(Relative) path to config (.yaml) file.')
flags.DEFINE_string('train_file_pattern', None,
    'Path to .txt file containing paths to video and integer label for training dataset.')
flags.DEFINE_string('val_file_pattern', None,
    'Path to .txt file containing paths to video and integer label for validation dataset.')
flags.DEFINE_string('model_dir', None,
    'Path to directory where model info, like checkpoints are (to be) stored.')
flags.DEFINE_string('pretrained_ckpt', None,
    'Path to directory where pretraining checkpoints are stored.')
flags.DEFINE_integer('num_gpus', 1,
    'Number of gpus to use for training.', lower_bound=0)
flags.DEFINE_integer('save_checkpoints_step', None,
    'Number of training steps to save checkpoints.', lower_bound=0)
flags.DEFINE_bool('mixed_precision', False,
    'Whether to use mixed precision during training.')
flags.DEFINE_bool('debug', False,
    'Whether to run in debug mode.')

flags.register_multi_flags_validator(
    ['config', 'train_file_pattern', 'val_file_pattern'],
    lambda flags: '.yaml' in flags['config'],
    message='File extension validation failed',)

flags.mark_flags_as_required(['config', 'train_file_pattern', 'model_dir'])

FLAGS = flags.FLAGS

def get_dataset(cfg, file_pattern, is_training, mixed_precision=False):
  """Returns a tf.data dataset"""
  return dataloader.InputReader(
      cfg,
      is_training,
      mixed_precision,
  )(file_pattern, cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE)

def load_model(model, cfg, mixed_precision=False):
  """Compile model with loss function, model optimizers and metrics."""
  opt_str = cfg.TRAIN.OPTIMIZER.lower()
  if opt_str == 'sgd':
    opt = tf.optimizers.SGD(
        learning_rate=cfg.TRAIN.WARMUP_LR,
        momentum=cfg.TRAIN.MOMENTUM,
        nesterov=True)
  elif opt_str == 'adam':
    opt = tf.optimizers.Adam(
        learning_rate=cfg.TRAIN.WARMUP_LR)
  else:
    raise NotImplementedError(f'{opt_str} not supported')

  if mixed_precision:
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
  
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
  cfg = get_default_config()
  cfg.merge_from_file(FLAGS.config)
  cfg.freeze()

  model_dir = FLAGS.model_dir
  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

  # init wandb
  if cfg.WANDB.ENABLE:
    wandb.tensorboard.patch(root_logdir=model_dir)
    run_id = wandb.util.generate_id()
    wandb.init(
        job_type='train',
        group=cfg.WANDB.GROUP_NAME,
        project=cfg.WANDB.PROJECT_NAME,
        sync_tensorboard=cfg.WANDB.TENSORBOARD,
        mode=cfg.WANDB.MODE,
        config=dict(cfg),
        resume='allow',
        id=run_id
    )

  if FLAGS.debug:
    tf.config.run_functions_eagerly(True)
    tf.debugging.set_log_device_placement(True)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(1111)
    logging.set_verbosity(logging.DEBUG)
    tf.debugging.experimental.enable_dump_debug_info(model_dir,
      tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

  strategy = utils.get_strategy(FLAGS.num_gpus)
  
  # mixed precision
  precision = utils.get_precision(FLAGS.mixed_precision)
  policy = tf.keras.mixed_precision.Policy(precision)
  tf.keras.mixed_precision.set_global_policy(policy)
  
  # learning rate schedule
  def lr_schedule(epoch, lr):
    """
    Implements the learning rate schedule used in
      https://arxiv.org/abs/2004.04730
    """
    if epoch > cfg.TRAIN.WARMUP_EPOCHS:
      new_lr = cfg.TRAIN.BASE_LR * (
          0.5 * (tf.math.cos(tf.constant(math.pi) * (epoch/cfg.TRAIN.EPOCHS)) + 1))
    else:
      new_lr = cfg.TRAIN.WARMUP_LR + (
          epoch * (cfg.TRAIN.BASE_LR - cfg.TRAIN.WARMUP_LR) / cfg.TRAIN.WARMUP_EPOCHS)
    return new_lr

  with strategy.scope():
    model = X3D(cfg)
    model = load_model(model, cfg, FLAGS.mixed_precision)

    # resume training from checkpoint, if available
    current_epoch = 0
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    if ckpt_path:
      current_epoch = int(os.path.basename(ckpt_path).split('-')[1])
      logging.info(f'Found checkpoint {ckpt_path} at epoch {current_epoch}')
      model.load_weights(ckpt_path)
    elif FLAGS.pretrained_ckpt:
      logging.info(f'Loading model from pretrained weights at {FLAGS.pretrained_ckpt}')
      if tf.io.gfile.isdir(FLAGS.pretrained_ckpt):
        model.load_weights(tf.train.latest_checkpoint(FLAGS.pretrained_ckpt))
      else:
        model.load_weights(FLAGS.pretrained_ckpt)

    model.fit(
        get_dataset(cfg, FLAGS.train_file_pattern, True, FLAGS.mixed_precision),
        verbose=1,
        epochs=cfg.TRAIN.EPOCHS,
        initial_epoch = current_epoch,
        steps_per_epoch=cfg.TRAIN.DATASET_SIZE/cfg.TRAIN.BATCH_SIZE,
        validation_data=get_dataset(cfg, FLAGS.val_file_pattern, False,
          FLAGS.mixed_precision) if FLAGS.val_file_pattern else None,
        callbacks=utils.get_callbacks(
            cfg, lr_schedule, FLAGS))

if __name__ == "__main__":
  app.run(main)
