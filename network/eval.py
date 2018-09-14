import tensorflow as tf
from dataset import Reader
from network.network import Network
import numpy as np
import tensorflow.contrib.slim as slim

import config

FLAGS = tf.app.flags.FLAGS


class Learning:
    def __init__(self):
        self.train_reader = Reader.Reader("train_tf/*.tfrecord")

        self.logs_dir = FLAGS.logdir
        self.train_logs_path = self.logs_dir + '/train_logs'
        self.GRU_SIZE = 1620
        self.cell_size = 9
        
        #self.chkpt_file = self.logs_dir + "/model.ckpt-54000"
        self.h_state_init_1 = np.zeros((1,self.GRU_SIZE), np.float32)
        self.h_state_init_2 = np.zeros((1,self.GRU_SIZE), np.float32)
        #self.cell_state_init = np.zeros(4096).reshape(1,4096).astype(np.float32)

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
        frame_gt_batch, frame_x_batch, file_name = self.train_reader.get_random_example()
        frame_gt_batch2 = np.array(frame_gt_batch).reshape(-1, self.cell_size, self.cell_size, 5+ self.cell_size* self.cell_size +1)
        # frame_gt_batch2 = frame_gt_batch2[..., 0:5].reshape(-1,self.cell_size * self.cell_size * 5)
        
        # frame_x_batch = np.transpose(frame_x_batch, [3,0,1,2])
        # frame_x_batch = frame_x_batch[0:3]
        # frame_x_batch = np.transpose(frame_x_batch, [1,2,3,0])
        
        # print(np.array(frame_gt_batch).shape)
        # print(np.array(frame_x_batch).shape)
        
        # print(file_name)
        # print(frame_gt_batch2[0, :, :, 0:5])
        # sys.exit(0)
        
        return {self.net.x: frame_x_batch,
                self.net.det_anno: frame_gt_batch2[..., 0:5],
                self.net.track_y: frame_gt_batch,
                self.net.h_state_init_1: self.h_state_init_1,
                self.net.h_state_init_2: self.h_state_init_2
                #self.net.cell_state_init: self.cell_state_init
                }

    def _restore_checkpoint_or_init(self, sess):
        import os
        if FLAGS.restore:
            
            # if len(config.exclusion_vars) == 0 :
                # self.net.saver.restore(sess, FLAGS.chkpt_file)
            
            # else:
            init_fn = get_init_fn(FLAGS.chkpt_file, config.exclusion_vars, FLAGS.ignore_missing_vars)
            
            init_fn(sess)
            initialize_uninitialized(sess)
                   
            global_step_init = self.net.global_step.assign(0)
            sess.run(global_step_init)
            
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
            # config.gpu_options.allow_growth = True
            
            with tf.Session(graph=self.net.graph,config=config) as sess:
                
                self._restore_checkpoint_or_init(sess)

                step_num = 1
                max_steps = FLAGS.max_steps
                while step_num <= max_steps:
                    if step_num % 1000 == 0:
                        gs = self._train_step(sess,
                                           tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                           tf.RunMetadata())
                        save_path = self.net.saver.save(sess, self.logs_dir + "/model.ckpt", global_step=self.net.global_step)
                        print("Model saved in file: %s" % save_path)

                    else:
                        gs = self._train_step(sess)
                    step_num += 1

                    
def get_init_fn(checkpoint_path, exclusions, ignore_missing_vars=True):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """

    # exclusions = []
    # if checkpoint_exclude_scopes:
        # exclusions = [scope.strip()
                      # for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_trainable_variables():
        # print("VAR: " + str(var))
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                print("exclude: " + str(var))
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    # print("/////////////restore vars")
    # for temp in variables_to_restore:
        # print(temp)
    # print("/////////////\n")
        
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path
    #print('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
                    
                    
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print("************ uninitialized vars")
    for i in not_initialized_vars: # only for testing
       print(i.name)
    print("************\n")

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

        