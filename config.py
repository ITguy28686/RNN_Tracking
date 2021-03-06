import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('max_steps', 200000, 'Number of max_steps')
tf.app.flags.DEFINE_integer('esize', 50, 'Size of examples')
tf.app.flags.DEFINE_integer('estep', 20, 'Length of step for grouping frames into examples')
tf.app.flags.DEFINE_integer('height', 240, 'Height of frames')
tf.app.flags.DEFINE_integer('width', 320, 'Width of frames')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size of frames')
tf.app.flags.DEFINE_float('lrate', 1e-4, 'Learning rate')
tf.app.flags.DEFINE_string('logdir', 'network/logs', 'Path to store logs and checkpoints')
tf.app.flags.DEFINE_string('conv', 'standard', 'Type of CNN block')
tf.app.flags.DEFINE_string('rnn', 'GRU', 'Type of RNN block (LSTM/GRU)')
tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/ass_epsilon_version/model.ckpt-191000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/logs/model.ckpt-13000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/match_prev_version/model.ckpt-11000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/7x7_patch/model.ckpt-100000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/7x7_GRU2048/model.ckpt-100000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/9x9_GRU2048/model.ckpt-18000', 'checkpoint file path')
# tf.app.flags.DEFINE_string('chkpt_file', './network/old_logs/9x9_GRU1620/model.ckpt-3000', 'checkpoint file path')
tf.app.flags.DEFINE_boolean('update', False, 'Generate TFRecords')
tf.app.flags.DEFINE_boolean('download', False, 'Download dataset')
tf.app.flags.DEFINE_boolean('restore', True, 'Restore from previous checkpoint')
tf.app.flags.DEFINE_boolean('test', False, 'Test evaluation')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', True, 'Test evaluation')

data_dir = os.path.join('train_tf/')
# exclusion_vars = ["coord2_fc1","coord2_fc2","coord2_final","coord_conv1"]
# exclusion_vars = ["coord_final","fc_12_coord","fc_12_association"]
# exclusion_vars = ["coord_final","association_final"]
# exclusion_vars = ["fc_11-2_coord","fc_12_coord","fc_11-2_association","fc_12_association","fc_11","GRU
# exclusion_vars = ["coord_final","association_final"]
exclusion_vars = []