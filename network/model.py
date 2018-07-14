import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#from network.LSTM import LSTMcell
#from network.LSTM import RNN

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, mat_x, is_training, keep_prob):
    
        self.cell_size = 8
    
        self.mat_x = mat_x
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.logits, self.coord_logits = self._init_model()

    def _init_model(self):

        logits, coord_logits = self.mynet(self.mat_x, data_format='NCHW')
        return logits, coord_logits
    
    
    def mynet(self, mat_x, data_format='NCHW') :
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            trainable=self.is_training,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):

            #triple conv_pool
            img_flow = slim.repeat(mat_x, 1, slim.conv2d, 64, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv1_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 64, [3, 3], data_format=data_format, scope='conv2')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 128, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv3_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 128, [3, 3], data_format=data_format, scope='conv4')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv5_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], data_format=data_format, scope='conv6')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv7_pool')
            coord_flow = img_flow
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], data_format=data_format, scope='conv8')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv9_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], data_format=data_format, scope='conv10')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 64, [3, 3], data_format=data_format, scope='conv11')
            
            #LSTM_layer
            # tensro_size (w*h*channel)
            # (batch_size, channel, w, h) -> (1, batch_size, tensor_size)
            img_flow = tf.reshape(img_flow, ( 1, -1, np.prod(img_flow.get_shape().as_list()[1:])))
            img_flow = self._lstm_layer(input = img_flow, num_units = 768, scope='LSTM_1')
            
            #a fc layer 
            img_flow = slim.fully_connected(img_flow, 512, scope='fc_1')
            
            #second output layer
            img_flow = slim.fully_connected(img_flow, self.cell_size*self.cell_size*7, scope='final', activation_fn=None)
            
            coord_flow = slim.repeat(coord_flow, 1, slim.conv2d, 256, [3, 3], data_format=data_format, scope='coord_conv1')
            coord_flow = slim.repeat(coord_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='coord_conv2_pool')
            coord_flow = tf.reshape(coord_flow, (-1,np.prod(coord_flow.get_shape().as_list()[1:])))
            coord_flow = slim.fully_connected(coord_flow, 512, scope='coord_fc')
            coord_flow = slim.fully_connected(coord_flow, self.cell_size*self.cell_size*5, scope='coord_final', activation_fn=None)

            return img_flow, coord_flow

    def _lstm_layer(self, input, num_units,scope='LSTM'):
        with tf.variable_scope(scope):

            #將數據從[n_samples, n_steps, D_input]，轉換成[n_steps, n_samples, D_input]
            # rnn_inputs = tf.transpose(pre_rnn_inputs, perm=[1,0,2])
            
            # cell = LSTMcell(rnn_inputs, shape[1], num_units, tf.initializers.orthogonal)
            # pre_rnn_outputs = RNN(cell)
            
            # shape = pre_rnn_outputs.get_shape()
            # rnn_outputs = tf.reshape(pre_rnn_outputs, (-1 , shape[2]))
            
            
            cell = tf.contrib.rnn.LSTMCell(num_units = num_units, initializer=tf.initializers.orthogonal(), activation=tf.nn.relu6)
            h0_state = cell.zero_state(1,tf.float32)
            outputs, state = tf.nn.dynamic_rnn(cell, input, initial_state=h0_state)
            
            out_shape = outputs.get_shape()
            rnn_outputs = tf.reshape(outputs, (-1, out_shape[2]))

            return rnn_outputs
            
    def tenor_img_concat(self, det_prepared_concat, img_flow, scope='Concat'):
        with tf.variable_scope(scope):
            img_shape = img_flow.get_shape()
            det_shape = det_prepared_concat.get_shape()
            
            img_prepared_concat = tf.reshape(img_flow, ( -1, img_shape[1], img_shape[2] * img_shape[3]))
            det_prepared_concat = tf.reshape(det_prepared_concat, ( -1, 1, img_shape[2] * img_shape[3]))
            img_concated = tf.concat([img_prepared_concat, det_prepared_concat], 1)
            
            img_flow = tf.reshape(img_concated, ( -1, img_shape[1]+1, img_shape[2], img_shape[3]))
            
            return img_flow

    @staticmethod
    def _dense(output):
        with tf.name_scope('Dense'):
            return slim.fully_connected(output, 6, scope="dense")
       
    # def orthogonal_initializer_customed(self, shape,scale = 1.0):

        # flat_shape = (shape[0], np.prod(shape[1:]))

        # a = np.random.normal(0.0, 1.0, flat_shape)
        # u, _, v = np.linalg.svd(a, full_matrices=False)
        # q = u if u.shape == flat_shape else v
        # q = q.reshape(shape) #this needs to be corrected to float32
        
        # return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32,trainable=True)
        
