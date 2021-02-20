import os
import wandb
import tensorflow as tf
from absl import app, flags, logging

from configs.default import get_default_config
from dataloader import InputReader
from model import X3D
import utils

flags.DEFINE_string('cfg', None,
    '(Relative) path to config (.yaml) file.')
flags.DEFINE_string('test_label_file', None,
    'Path to .txt file containing paths to video and integer label for test dataset.')
flags.DEFINE_string('model_folder', None,
    'Path to directory where checkpoint(s) are stored.')
flags.DEFINE_integer('gpus', 1,
    'Number of gpus to use for training.', lower_bound=0)
'''flags.DEFINE_bool('debug', False,
    'Whether to run in debug mode.')'''

flags.register_multi_flags_validator(
    ['cfg', 'test_label_file'],
    lambda flags: '.yaml' in flags['cfg'] and '.txt' in flags['test_label_file'],
    message='File extension validation failed',)

flags.mark_flags_as_required(['cfg', 'test_label_file', 'model_folder'])

FLAGS = flags.FLAGS

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
  cfg.merge_from_file(FLAGS.cfg)
  cfg.freeze()

  model_dir = FLAGS.model_folder
  if not tf.io.gfile.exists(model_dir):
    raise NotADirectoryError

  # init wandb
  if cfg.WANDB.ENABLE:
    wandb.init(
        job_type='eval',
        group=cfg.WANDB.GROUP_NAME,
        project=cfg.WANDB.PROJECT_NAME,
        sync_tensorboard=cfg.WANDB.TENSORBOARD,
        mode=cfg.WANDB.MODE,
        config=dict(cfg)
    )

  strategy = utils.get_strategy(FLAGS.gpus)

  with strategy.scope():
    model = X3D(cfg)
    model = load_model(model, cfg)
    
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    if ckpt_path:
      current_epoch = int(os.path.basename(ckpt_path).split('-')[1])
      logging.info(f'Found checkpoint {ckpt_path} at epoch {current_epoch}')
      model.load_weights(ckpt_path).expect_partial()

      model.evaluate(
        InputReader(cfg, False
        )(FLAGS.test_label_file, cfg.TEST.BATCH_SIZE),
        verbose=1,
        callbacks=[tf.keras.callbacks.TensorBoard(
          log_dir=model_dir,
          profile_batch=0)])

if __name__ == "__main__":
  app.run(main)
