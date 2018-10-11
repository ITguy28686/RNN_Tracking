import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#from network.LSTM import LSTMcell
#from network.LSTM import RNN

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, mat_x, det_anno, prev_asscoia, h_state_init_1, h_state_init_2, is_training, keep_prob, data_format='NCHW'):
    
        self.cell_size = 9
        self.boxes_per_cell = 3
        self.track_num = 30
        self.GRU_SIZE = 1620
        self.record_N = 256
    
        self.mat_x = mat_x
        self.det_anno = det_anno
        self.prev_asscoia = prev_asscoia
        
        self.is_training = is_training
        self.h_state_init_1 = tuple([h_state_init_1])
        self.h_state_init_2 = tuple([h_state_init_2])
        # self.cell_state_init = cell_state_init
        
        self.coord_flow, self.epsilon_flow, self.associa_flow, self.rnn_coord_state, self.rnn_associa_state = self.mynet(self.mat_x, self.det_anno, self.prev_asscoia, self.h_state_init_1, self.h_state_init_2, data_format, keep_prob)
    
    def mynet(self, mat_x, det_anno, prev_asscoia, h_state_init_1, h_state_init_2, data_format='NCHW', keep_prob=0.5) :
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
            # img_flow = slim.repeat(img_flow, 1, slim.conv2d, 256, [1, 1], data_format=data_format, scope='conv10')
            # img_flow = slim.repeat(img_flow, 1, slim.conv2d, 512, [3, 3], stride=2, data_format=data_format, padding='SAME', scope='conv11_pool')
            
            if data_format == 'NHWC' :
                img_flow = tf.transpose(img_flow, perm=[0,3,1,2])
                
            tensor_flow = slim.flatten(img_flow, scope='flat_10')
            tensor_flow = slim.fully_connected(tensor_flow, self.GRU_SIZE, scope='fc_12' )
            
            with tf.variable_scope("det_anno"):
                # det_anno = tf.reshape(det_anno, (-1, self.cell_size * self.cell_size * 5))
                tensor_flow = tf.concat([tensor_flow, det_anno, prev_asscoia], axis = 1)
                tensor_flow = slim.fully_connected(tensor_flow, self.GRU_SIZE*2, scope='anno_fc' )
                tensor_flow = slim.fully_connected(tensor_flow, self.GRU_SIZE, scope='anno_fc2' )
                
            
            # LSTM_layer
            # tensro_size (w*h*channel)
            # (batch_size, tensor_size) -> (1, batch_size, tensor_size) 
            # tensor_flow, rnn_state = self._lstm_layer(input = tensor_flow, num_units = 2048, h_state_init = h_state_init, cell_state_init = cell_state_init, scope='LSTM_1')
            with tf.variable_scope("coord_gru"):
                tensor_flow = tf.reshape(tensor_flow, ( 1, -1, tensor_flow.get_shape()[1]))
                rnn_outputs, rnn_coord_state = self._gru_layer(input = tensor_flow, num_units = self.GRU_SIZE, h_state_init = h_state_init_1, scope='GRU_coord_epsilon')
                tensor_flow = tf.reshape(rnn_outputs, (-1, self.GRU_SIZE))
            
            # coord_flow = slim.fully_connected(tensor_flow, 512, scope='fc_11-2_coord')
            tensor_flow = slim.fully_connected(tensor_flow, self.GRU_SIZE, scope='fc_13')
            
            associa_input = tensor_flow
            
            tensor_flow = slim.dropout(
                    tensor_flow, keep_prob=keep_prob, is_training=self.is_training,
                    scope='dropout_coord')
            coord_flow = slim.fully_connected(tensor_flow, self.cell_size*self.cell_size*self.boxes_per_cell*5, scope='coord_final', activation_fn=None)
            
            epsilon_flow = slim.fully_connected(tensor_flow, self.record_N, scope='epsilon_final', activation_fn=None)
            epsilon_flow = tf.clip_by_value(epsilon_flow, 0.0001, 0.9999)
            
            
            with tf.variable_scope("associa_gru"):
                
                associa_input = tf.reshape(associa_input, ( -1, 1, associa_input.get_shape()[1]))
                associa_input = tf.tile(associa_input, [1, self.record_N, 1])
                associa_flow, rnn_associa_state = self._gru_layer(input = associa_input, num_units = self.GRU_SIZE, h_state_init = h_state_init_2, scope='GRU_associa')
                
                # associa_flow = tf.transpose(associa_flow, perm=[1,0,2])
                associa_flow = slim.fully_connected(associa_flow, self.cell_size * self.cell_size + 1, scope='associa_softmax', activation_fn= tf.nn.softmax )
                # associa_flow = tf.clip_by_value(associa_flow, 0.0001, 1)

            return coord_flow, epsilon_flow, associa_flow, rnn_coord_state, rnn_associa_state
            
    """
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
            
            rnn_outputs = tf.reshape(outputs, (-1, num_units))

            return rnn_outputs, state
    """
    
    def _gru_layer(self, input, num_units, h_state_init, scope='GRU'):
        with tf.variable_scope(scope):
            
            cell = tf.contrib.rnn.GRUCell(num_units = num_units, activation=tf.nn.relu6)
            rnn_cells = tf.contrib.rnn.MultiRNNCell([cell] , state_is_tuple=True)
            
            gru_init = h_state_init
            #gru_init = rnn_cells.zero_state(1, tf.float32)
            
            outputs, state = tf.nn.dynamic_rnn(rnn_cells, input, initial_state=gru_init)
            
            

            return outputs, state

    
    # def tenor_img_concat(self, det_prepared_concat, img_flow, scope='Concat'):
        # with tf.variable_scope(scope):
            # img_shape = img_flow.get_shape()
            # det_shape = det_prepared_concat.get_shape()
            
            # img_prepared_concat = tf.reshape(img_flow, ( -1, img_shape[1], img_shape[2] * img_shape[3]))
            # det_prepared_concat = tf.reshape(det_prepared_concat, ( -1, 1, img_shape[2] * img_shape[3]))
            # img_concated = tf.concat([img_prepared_concat, det_prepared_concat], 1)
            
            # img_flow = tf.reshape(img_concated, ( -1, img_shape[1]+1, img_shape[2], img_shape[3]))
            
            # return img_flow

  
