import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#from network.LSTM import LSTMcell
#from network.LSTM import RNN

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, mat_x, h_state_init, cell_state_init, is_training, keep_prob, data_format='NCHW'):
    
        self.cell_size = 9
        self.track_num = 30
    
        self.mat_x = mat_x
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.h_state_init = h_state_init
        self.cell_state_init = cell_state_init
        
        self.coord_flow, self.association_flow, self.lstm_state = self.mynet(self.mat_x, self.h_state_init, self.cell_state_init, data_format)
    
    def mynet(self, mat_x, h_state_init, cell_state_init, data_format='NCHW') :
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            trainable=self.is_training,
                            weights_initializer=tf.keras.initializers.he_normal(),
                            weights_regularizer=slim.l2_regularizer(0.01)):

            #triple conv_pool
            img_flow = slim.repeat(mat_x, 1, slim.conv2d, 128, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv1_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 128, [1, 1], data_format=data_format, scope='conv2')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv3_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [1, 1], data_format=data_format, scope='conv4')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv5_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [1, 1], data_format=data_format, scope='conv6')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv7_pool')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [1, 1], data_format=data_format, scope='conv8')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv9_pool')
            
            if data_format == 'NHWC' :
                img_flow = tf.transpose(img_flow, perm=[0,3,1,2])
                
            tensor_flow = slim.flatten(img_flow, scope='flat_10')
            tensor_flow = slim.fully_connected(tensor_flow, 2048, scope='fc_11' )
            
            # LSTM_layer
            # tensro_size (w*h*channel)
            # (batch_size, tensor_size) -> (1, batch_size, tensor_size) 
            tensor_flow = tf.reshape(tensor_flow, ( 1, -1, tensor_flow.get_shape()[1]))
            tensor_flow, lstm_state = self._lstm_layer(input = tensor_flow, num_units = 2048, h_state_init = h_state_init, cell_state_init = cell_state_init, scope='LSTM_1')

            coord_flow = slim.fully_connected(tensor_flow, 4096, scope='fc_12_coord')
            coord_flow = slim.fully_connected(coord_flow, self.cell_size*self.cell_size*5, scope='coord_final', activation_fn=None)
            
            association_flow = slim.fully_connected(tensor_flow, 4096, scope='fc_12_association')
            association_flow = slim.fully_connected(association_flow, self.cell_size*self.cell_size*self.track_num, scope='association_final', activation_fn=None)
            
            # coord_flow2 = slim.repeat(coord_flow2, 1, slim.conv2d, 512, [1, 1], data_format=data_format, scope='coord_conv1')
            
            # if data_format == 'NHWC' :
                # coord_flow2 = tf.transpose(coord_flow2, perm=[0,3,1,2])
            
            # coord_flow2 = tf.reshape(coord_flow2, (-1,np.prod(coord_flow2.get_shape().as_list()[1:])))
            # coord_flow2 = slim.fully_connected(coord_flow2, 512, scope='coord2_fc1' )
            # coord_flow2 = slim.fully_connected(coord_flow2, 4096, scope='coord2_fc2' )
            # coord_flow2 = slim.fully_connected(coord_flow2, self.cell_size*self.cell_size*(5+self.track_num), scope='coord2_final', activation_fn=None)

            return coord_flow, association_flow, lstm_state

    def _lstm_layer(self, input, num_units, h_state_init, cell_state_init, scope='LSTM'):
        with tf.variable_scope(scope):

            #將數據從[n_samples, n_steps, D_input]，轉換成[n_steps, n_samples, D_input]
            # rnn_inputs = tf.transpose(pre_rnn_inputs, perm=[1,0,2])
            
            # cell = LSTMcell(rnn_inputs, shape[1], num_units, tf.initializers.orthogonal)
            # pre_rnn_outputs = RNN(cell)
            
            # shape = pre_rnn_outputs.get_shape()
            # rnn_outputs = tf.reshape(pre_rnn_outputs, (-1 , shape[2]))
            
            
            cell = tf.contrib.rnn.LSTMCell(num_units = num_units, initializer=tf.initializers.orthogonal(), activation=tf.nn.relu6)
            
            lstm_init = tf.contrib.rnn.LSTMStateTuple(cell_state_init,h_state_init)
            outputs, state = tf.nn.dynamic_rnn(cell, input, initial_state=lstm_init)
            
            out_shape = outputs.get_shape()
            rnn_outputs = tf.reshape(outputs, (-1, out_shape[2]))

            return rnn_outputs, state
            
    def tenor_img_concat(self, det_prepared_concat, img_flow, scope='Concat'):
        with tf.variable_scope(scope):
            img_shape = img_flow.get_shape()
            det_shape = det_prepared_concat.get_shape()
            
            img_prepared_concat = tf.reshape(img_flow, ( -1, img_shape[1], img_shape[2] * img_shape[3]))
            det_prepared_concat = tf.reshape(det_prepared_concat, ( -1, 1, img_shape[2] * img_shape[3]))
            img_concated = tf.concat([img_prepared_concat, det_prepared_concat], 1)
            
            img_flow = tf.reshape(img_concated, ( -1, img_shape[1]+1, img_shape[2], img_shape[3]))
            
            return img_flow

  
