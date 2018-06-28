import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
from network.model import Model

FLAGS = tf.app.flags.FLAGS


class Network:
    def __init__(self, is_training):
        self.graph = tf.Graph()
        self.is_training = is_training
        self.eval()

    def eval(self):
        with self.graph.as_default():
            self.det_x = tf.placeholder(dtype=tf.float32, shape=(None, 64*4), name="det_inputs")
            self.img_x = tf.placeholder(dtype=tf.float32, shape=(None,300, 300, 3), name="img_inputs")
            self.img_x_nchw = tf.transpose(self.img_x, perm=[0,3,1,2])
            
            self.track_y = tf.placeholder(dtype=tf.float32, shape=(None, 64*6), name='track_label')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

            logits = Model(self.det_x, self.img_x_nchw, self.is_training, self.keep_prob).logits
            #self._calc_accuracy(logits, self.y)

            with tf.name_scope('Cost'):
                delta = self.track_y  - logits[1]
                loss1 = tf.reduce_mean(
                            tf.reduce_sum(tf.square(delta))) 
                            
                delta2 = self.track_y  - logits[0]
                loss2 = tf.reduce_mean(
                            tf.reduce_sum(tf.square(delta2)))
                            
                total_loss = loss1 + loss2
                            
                #cross_entropy = (self.track_y - logits[1]) * (self.track_y - logits[1])
                tf.summary.scalar("total_loss", total_loss)
                
            with tf.name_scope('Optimizer'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #optimizer = tf.train.GradientDescentOptimizer(FLAGS.lrate)
                optimizer = tf.train.AdamOptimizer(FLAGS.lrate)
                # optimizer = tf.train.MomentumOptimizer(FLAGS.lrate, 0.9, use_nesterov=True)
                # optimizer = tf.train.RMSPropOptimizer(FLAGS.lrate)
                self.train_step = slim.learning.create_train_op(total_loss, optimizer, self.global_step,
                                                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
    
    def loss_function(self, tensor_x, label_y):
        
        
        
        return 
    
    @staticmethod
    def print_model():
        def get_nb_params_shape(shape):
            nb_params = 1
            for dim in shape:
                nb_params = nb_params * int(dim)
            return nb_params

        tot_nb_params = 0
        for trainable_variable in slim.get_trainable_variables():
            print(trainable_variable.name, trainable_variable.shape)
            vshape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(vshape)
            tot_nb_params = tot_nb_params + current_nb_params
        print('Total number of trainable params', tot_nb_params)
