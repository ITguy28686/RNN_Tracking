import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('epoch', 1, 'Number of epoch')
tf.app.flags.DEFINE_integer('esize', 50, 'Size of examples')
tf.app.flags.DEFINE_integer('estep', 20, 'Length of step for grouping frames into examples')
tf.app.flags.DEFINE_integer('height', 240, 'Height of frames')
tf.app.flags.DEFINE_integer('width', 320, 'Width of frames')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size of frames')
tf.app.flags.DEFINE_float('lrate', 1e-4, 'Learning rate')
tf.app.flags.DEFINE_string('logdir', 'network/logs', 'Path to store logs and checkpoints')
tf.app.flags.DEFINE_string('conv', 'standard', 'Type of CNN block')
tf.app.flags.DEFINE_string('rnn', 'GRU', 'Type of RNN block (LSTM/GRU)')
tf.app.flags.DEFINE_boolean('update', False, 'Generate TFRecords')
tf.app.flags.DEFINE_boolean('download', False, 'Download dataset')
tf.app.flags.DEFINE_boolean('restore', False, 'Restore from previous checkpoint')
tf.app.flags.DEFINE_boolean('test', False, 'Test evaluation')

data_dir = os.path.join('train_tf/')