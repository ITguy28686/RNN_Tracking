#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy  as np
class  TextCnnRnn(object):

    def __init__(self,config):
        self.config=config
        self.input_x=tf.placeholder(tf.int32,[None, self.config.seq_length],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None, self.config.num_classes],name="inpyt_y")
        self.keep_prob=tf.placeholder(tf.float32,None,name='keep_prob')
        self.pad = tf.placeholder(tf.float32, [None, 1, self.config.embedding_dim, 1], name='pad')
        self.l2_loss = tf.constant(0.0)
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')
        self.filter_sizes = list(map(int, self.config.multi_kernel_size.split(",")))
        self.cnnrnn()
        
    def input_embedding(self):
        """词嵌套"""
        with tf.device('/cpu:0'):
            embedding =tf.get_variable("embedding",[self.config.vocab_size,self.config.embedding_dim])
            _input = tf.nn.embedding_lookup(embedding, self.input_x)
            _input_expanded = tf.expand_dims(_input, -1)
        return _input_expanded
        
    def cnnrnn(self):
        emb=self.input_embedding()
        pooled_concat = []
        reduced = np.int32(np.ceil((self.config.seq_length) * 1.0 / self.config.max_pool_size))
        
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, emb, pad_post], 1)
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, self.config.max_pool_size, 1, 1], strides=[1, self.config.max_pool_size, 1, 1], padding='SAME',
                                        name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, self.config.num_filters])
                pooled_concat.append(pooled)
                
        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.keep_prob)
        # lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_unit)
        # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.hidden_unit)
        lstm_cell = tf.contrib.rnn.GRUCell(num_units=self.config.hidden_unit)
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        self._initial_state = lstm_cell.zero_state(self.config.batch_size, tf.float32)
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(pooled_concat, num_or_size_splits=int(reduced), axis=1)]
        # outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)
        #outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state,
        #                                           sequence_length=self.real_len)
        outputs, state=tf.nn.static_rnn( lstm_cell, inputs,self._initial_state,sequence_length=self.real_len)
        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        output = outputs[0]
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, self.config.hidden_unit], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))
        with tf.name_scope('score'):
            self.W = tf.Variable(tf.truncated_normal([self.config.hidden_unit, self.config.num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, self.W, b, name='scores')
            self.pred_y = tf.nn.softmax(self.scores, name="pred_y")
            tf.add_to_collection('pred_network', self.pred_y)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                             logits=self.scores)  # only named arguments accepted
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self.l2_loss
        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')
        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))

