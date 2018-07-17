import tensorflow as tf
from dataset import Reader
from network.network import Network
import numpy as np

import config

FLAGS = tf.app.flags.FLAGS


class Learning:
    def __init__(self):
        self.train_reader = Reader.Reader("train_tf/*.tfrecord")

        self.logs_dir = FLAGS.logdir
        self.train_logs_path = self.logs_dir + '/train_logs'
        
        self.chkpt_file = self.logs_dir + "/model.ckpt"
        self.h_state_init = np.zeros(768).reshape(1,768).astype(np.float32)
        self.cell_state_init = np.zeros(768).reshape(1,768).astype(np.float32)

        self.is_training = True
        self._evaluate_train()

    def _train_step(self, sess, run_options=None, run_metadata=None):
        if run_options is not None:
            _, total_loss, summary, global_step = sess.run(
                [self.net.train_step, self.net.total_loss, self.net.summary_op, self.net.global_step],
                feed_dict=self.next_example(), options=run_options, run_metadata=run_metadata)
            self.train_writer.add_run_metadata(run_metadata, 'step{}'.format(global_step), global_step)
            print('Loss for the %d step: %s' % (global_step ,total_loss))
            
        else:
            _, total_loss, summary, global_step = sess.run(
                [self.net.train_step, self.net.total_loss, self.net.summary_op, self.net.global_step],
                feed_dict=self.next_example())
            print('Loss for the %d step: %s' % (global_step ,total_loss))
                
                
        self.train_writer.add_summary(summary, global_step)
        return global_step

    def next_example(self):
        frame_gt_batch, frame_x_batch = self.train_reader.get_random_example()
        
        # frame_x_batch = np.transpose(frame_x_batch, [3,0,1,2])
        # frame_x_batch = frame_x_batch[0:3]
        # frame_x_batch = np.transpose(frame_x_batch, [1,2,3,0])
        
        # print(np.array(frame_gt_batch).shape)
        # print(np.array(frame_x_batch).shape)
        
        return {self.net.x: frame_x_batch,
                self.net.track_y: frame_gt_batch,
                self.net.h_state_init: self.h_state_init,
                self.net.cell_state_init: self.cell_state_init
                }

    def _restore_checkpoint_or_init(self, sess):
        import os
        if FLAGS.restore:
            self.net.saver.restore(sess, self.chkpt_file)
            print("Model restored.")
        else:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('Parameters were initialized')
        self.net.print_model()

    def _evaluate_train(self):
        self.keep_prob = 0.75
        self.is_training = True
        self.net = Network(self.is_training)
        self.train_writer = tf.summary.FileWriter(self.train_logs_path, graph=self.net.graph)
        
        with tf.device("/gpu:0"):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            
            with tf.Session(graph=self.net.graph,config=config) as sess:
                self._restore_checkpoint_or_init(sess)

                step_num = 1
                max_steps = FLAGS.max_steps
                while step_num <= max_steps:
                    if step_num % 1000 == 0:
                        gs = self._train_step(sess,
                                           tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                           tf.RunMetadata())
                        save_path = self.net.saver.save(sess, self.chkpt_file)
                        print("Model saved in file: %s" % save_path)

                    else:
                        gs = self._train_step(sess)
                    step_num += 1
