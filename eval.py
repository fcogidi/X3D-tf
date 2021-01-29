import os
import wandb
import tensorflow as tf
from absl import app, flags, logging

from configs.default import get_default_config
from model import X3D
import utils
from train import load_model, get_dataset

flags.DEFINE_string('config', None,
    '(Relative) path to config (.yaml) file.')
flags.DEFINE_string('file_pattern', None,
    'File pattern of video files to test on.')
flags.DEFINE_string('model_dir', None,
    'Path to directory where model info, like checkpoints are (to be) stored.')
flags.DEFINE_string('pretrained_ckpt', None,
    'Path to directory where pretraining checkpoints are stored.')
flags.DEFINE_integer('num_gpus', 1,
    'Number of gpus to use for training.', lower_bound=0)
flags.DEFINE_bool('mixed_precision', False,
    'Whether to use mixed precision during training.')
flags.DEFINE_bool('debug', False,
    'Whether to run in debug mode.')

flags.register_multi_flags_validator(
    ['config', 'val_label_file'],
    lambda flags: '.yaml' in flags['config'],
    message='File extension validation failed',)

flags.mark_flags_as_required(['config', 'model_dir'])

FLAGS = flags.FLAGS

def main(_):
  cfg = get_default_config()
  cfg.merge_from_file(FLAGS.config)
  cfg.freeze()

  model_dir = FLAGS.model_dir
  if not tf.io.gfile.exists(model_dir):
    raise NotADirectoryError

  val_label_file = FLAGS.val_label_file
  if val_label_file is not None:
    assert '.txt' in val_label_file, 'File extension validation failed'

  # init wandb
  if cfg.WANDB.ENABLE:
    wandb.tensorboard.patch(root_logdir=model_dir)
    wandb.init(
        job_type='eval',
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

  strategy = utils.get_strategy(cfg, FLAGS.num_gpus)

  # mixed precision
  precision = utils.get_precision(FLAGS.mixed_precision)
  policy = tf.keras.mixed_precision.experimental.Policy(precision)
  tf.keras.mixed_precision.experimental.set_policy(policy)

  with strategy.scope():
    model = X3D(cfg)
    model = load_model(model, cfg)

    # resume training from checkpoint, if available
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

    model.evaluate(
        get_dataset(cfg, FLAGS.train_label_file, True),
        verbose=1,
        epochs=cfg.TRAIN.EPOCHS,
        steps_per_epoch=cfg.TRAIN.DATASET_SIZE/cfg.TRAIN.BATCH_SIZE,
        validation_data=get_dataset(cfg, val_label_file, False) if val_label_file else None,
        callbacks=[]

if __name__ == "__main__":
  app.run(main)