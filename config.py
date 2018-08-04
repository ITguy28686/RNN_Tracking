import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('max_steps', 100000, 'Number of max_steps')
tf.app.flags.DEFINE_integer('esize', 50, 'Size of examples')
tf.app.flags.DEFINE_integer('estep', 20, 'Length of step for grouping frames into examples')
tf.app.flags.DEFINE_integer('height', 240, 'Height of frames')
tf.app.flags.DEFINE_integer('width', 320, 'Width of frames')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size of frames')
tf.app.flags.DEFINE_float('lrate', 1e-5, 'Learning rate')
tf.app.flags.DEFINE_string('logdir', 'network/logs', 'Path to store logs and checkpoints')
tf.app.flags.DEFINE_string('conv', 'standard', 'Type of CNN block')
tf.app.flags.DEFINE_string('rnn', 'GRU', 'Type of RNN block (LSTM/GRU)')
# tf.app.flags.DEFINE_string('chkpt_file', './network/logs/old_logs/03/model.ckpt-54000', 'checkpoint file path')
tf.app.flags.DEFINE_string('chkpt_file', './network/logs/model.ckpt-40000', 'checkpoint file path')
tf.app.flags.DEFINE_boolean('update', False, 'Generate TFRecords')
tf.app.flags.DEFINE_boolean('download', False, 'Download dataset')
tf.app.flags.DEFINE_boolean('restore', True, 'Restore from previous checkpoint')
tf.app.flags.DEFINE_boolean('test', False, 'Test evaluation')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', True, 'Test evaluation')

data_dir = os.path.join('train_tf/')
# exclusion_vars = ["coord2_fc1","coord2_fc2","coord2_final","coord_conv1"]
exclusion_vars = []