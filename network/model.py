import config
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#from network.LSTM import LSTMcell
#from network.LSTM import RNN

FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self, det_x, img_x, is_training, keep_prob):
    
        self.cell_size = 8
    
        self.det_x = det_x
        self.img_x = img_x
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.logits = self._init_model()

    def _init_model(self):
        

        # if FLAGS.conv == 'inception':
            # print('Using Inception model')
            # net = self._inception_cnn(self.inputs)
        # elif FLAGS.conv == 'vgg16':
            # print('Using VGG16 model')
            # net = self._vgg16(self.inputs)
        # else:
            # print('Using common cnn block')
            # net = self._cnn(self.inputs)

        logits = self.mynet(self.det_x, self.img_x)
        return logits
    
    
    def mynet(self, det_x, img_x) :
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            trainable=self.is_training,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            #one fc layers           
            det_flow = slim.fully_connected(det_x, 512, scope='fc1')  
            
            #LSTM layer
            det_flow = tf.reshape(det_flow, ( 1, -1, det_flow.shape[1]))
            det_flow = self._lstm_layer(input = det_flow, num_units = 512, scope='LSTM_1')
            
            #a fc layer 
            det_flow = slim.fully_connected(det_flow, 1444, scope='fc_2')
            
            #prepare the lstm1_output to concat
            det_prepared_concat = det_flow
            
            #first output layer
            det_flow = slim.fully_connected(det_flow, self.cell_size*self.cell_size*6, scope='final_1')

            #triple conv_pool
            img_flow = slim.repeat(img_x, 1, slim.conv2d, 32, [3, 3], stride=2, data_format='NCHW', padding='SAME', scope='conv1')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 64, [3, 3], stride=2, data_format='NCHW', padding='SAME', scope='conv2')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 64, [3, 3], stride=2, data_format='NCHW', padding='SAME', scope='conv3')


            #concat the img and lstm1_output flow
            img_flow = self.tenor_img_concat(det_prepared_concat, img_flow, scope='Concat')
            
            #twice conv
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 32, [3, 3], data_format='NCHW', scope='conv4')
            img_flow = slim.repeat(img_flow, 1, slim.conv2d, 16, [3, 3], data_format='NCHW', scope='conv5')
            
            #LSTM_layer
            # tensro_size (w*h*channel)
            # (batch_size, channel, w, h) -> (1, batch_size, tensor_size)
            img_flow = tf.reshape(img_flow, ( 1, -1, np.prod(img_flow.get_shape().as_list()[1:])))
            img_flow = self._lstm_layer(input = img_flow, num_units = 512, scope='LSTM_2')
            
            #a fc layer 
            img_flow = slim.fully_connected(img_flow, 1024, scope='fc_3')
            
            #second output layer
            img_flow = slim.fully_connected(img_flow, self.cell_size*self.cell_size*6, scope='final_2')

            return [det_flow,img_flow]

    def _cnn(self, input):
        with slim.arg_scope([slim.conv2d], stride=1,
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            trainable=self.is_training):
            with tf.variable_scope('Convolution', [input]):
                net = slim.conv2d(input, 32, [1, 1], stride=2, scope='Conv1',
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params={'is_training': self.is_training})
                net = slim.max_pool2d(net, [3, 3], scope='Pool1', stride=1)
                net = slim.conv2d(net, 32, [3, 3], scope='Conv2')
                net = slim.dropout(net, self.keep_prob, scope='Dropout')
                net = slim.max_pool2d(net, [3, 3], scope='Pool2', stride=1)
                net = slim.conv2d(net, 32, [3, 3], stride=2, scope='Conv3')
                return net

    def _inception_cnn(self, inputs):
        conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
        conv2 = slim.conv2d(conv1, 32, [3, 3], stride=2, padding='VALID', scope='Conv2d_2a_3x3')
        inc_inputs = slim.conv2d(conv2, 64, [3, 3], scope='Conv2d_2b_3x3')
        with slim.arg_scope([slim.conv2d], trainable=self.is_training, stride=1, padding='SAME'):
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
                with tf.variable_scope('BlockInceptionA', [inc_inputs]):
                    with tf.variable_scope('IBranch_0'):
                        ibranch_0 = slim.conv2d(inc_inputs, 96, [1, 1], scope='IConv2d_0a_1x1')
                    with tf.variable_scope('IBranch_1'):
                        ibranch_1_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                        ibranch_1 = slim.conv2d(ibranch_1_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                    with tf.variable_scope('IBranch_2'):
                        ibranch_2_conv1 = slim.conv2d(inc_inputs, 64, [1, 1], scope='IConv2d_0a_1x1')
                        ibranch_2_conv2 = slim.conv2d(ibranch_2_conv1, 96, [3, 3], scope='IConv2d_0b_3x3')
                        ibranch_2 = slim.conv2d(ibranch_2_conv2, 96, [3, 3], scope='IConv2d_0c_3x3')
                    with tf.variable_scope('IBranch_3'):
                        ibranch_3_pool = slim.avg_pool2d(inc_inputs, [3, 3], scope='IAvgPool_0a_3x3')
                        ibranch_3 = slim.conv2d(ibranch_3_pool, 96, [1, 1], scope='IConv2d_0b_1x1')
                    inception = tf.concat(axis=3, values=[ibranch_0, ibranch_1, ibranch_2, ibranch_3])
                with tf.variable_scope('BlockReductionA', [inception]):
                    with tf.variable_scope('RBranch_0'):
                        rbranch_0 = slim.conv2d(inception, 384, [3, 3], stride=2, padding='VALID',
                                                scope='RConv2d_1a_3x3')
                    with tf.variable_scope('RBranch_1'):
                        rbranch_1_conv1 = slim.conv2d(inception, 192, [1, 1], scope='RConv2d_0a_1x1')
                        rbranch_1_conv2 = slim.conv2d(rbranch_1_conv1, 224, [3, 3], scope='RConv2d_0b_3x3')
                        rbranch_1 = slim.conv2d(rbranch_1_conv2, 256, [3, 3], stride=2, padding='VALID',
                                                scope='RConv2d_1a_3x3')
                    with tf.variable_scope('RBranch_2'):
                        rbranch_2 = slim.max_pool2d(inception, [3, 3], stride=2, padding='VALID',
                                                    scope='RMaxPool_1a_3x3')
                return tf.concat(axis=3, values=[rbranch_0, rbranch_1, rbranch_2])

    def _vgg16(self, inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            trainable=self.is_training,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, self.keep_prob, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, self.keep_prob, scope='dropout7')
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
        return net

    def _lstm_layer(self, input, num_units,scope='LSTM'):
        with tf.variable_scope(scope):

            #將數據從[n_samples, n_steps, D_input]，轉換成[n_steps, n_samples, D_input]
            # rnn_inputs = tf.transpose(pre_rnn_inputs, perm=[1,0,2])
            
            # cell = LSTMcell(rnn_inputs, shape[1], num_units, tf.initializers.orthogonal)
            # pre_rnn_outputs = RNN(cell)
            
            # shape = pre_rnn_outputs.get_shape()
            # rnn_outputs = tf.reshape(pre_rnn_outputs, (-1 , shape[2]))
            
            
            cell = tf.contrib.rnn.LSTMCell(num_units = num_units, initializer=tf.initializers.orthogonal())
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
        
