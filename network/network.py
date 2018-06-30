import config
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from network.model import Model

FLAGS = tf.app.flags.FLAGS


class Network:
    def __init__(self, is_training):
        
        self.cell_size = 8
        self.coordloss_scale = 5
        self.confloss_scale = 2
        self.trackloss_scale = 1
        
        self.offset = np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size),
            (self.cell_size, self.cell_size))
            
        self.graph = tf.Graph()
        self.is_training = is_training
        self.eval()
            

    def eval(self):
        with self.graph.as_default():
            self.det_x = tf.placeholder(dtype=tf.float32, shape=(None, self.cell_size*self.cell_size*4), name="det_inputs")
            self.img_x = tf.placeholder(dtype=tf.float32, shape=(None,300, 300, 3), name="img_inputs")
            self.img_x_nchw = tf.transpose(self.img_x, perm=[0,3,1,2])
            
            self.track_y = tf.placeholder(dtype=tf.float32, shape=(None, self.cell_size*self.cell_size*6), name='track_label')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

            logits = Model(self.det_x, self.img_x_nchw, self.is_training, self.keep_prob).logits
            #self._calc_accuracy(logits, self.y)

            with tf.name_scope('Cost'):
                # delta = self.track_y  - logits[1]
                # loss1 = tf.reduce_mean(
                            # tf.reduce_sum(tf.square(delta))) 
                            
                # delta2 = self.track_y  - logits[0]
                # loss2 = tf.reduce_mean(
                            # tf.reduce_sum(tf.square(delta2)))
                            
                # total_loss = loss1 + loss2
                            
                loss1 = self.loss_function(logits[0],self.track_y)
                loss2 = self.loss_function(logits[1],self.track_y)
                self.total_loss = loss1 + loss2
                
                tf.summary.scalar("loss1", loss1)
                tf.summary.scalar("loss2", loss2)
                tf.summary.scalar("total_loss", self.total_loss)
                
            with tf.name_scope('Optimizer'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #optimizer = tf.train.GradientDescentOptimizer(FLAGS.lrate)
                optimizer = tf.train.AdamOptimizer(FLAGS.lrate)
                # optimizer = tf.train.MomentumOptimizer(FLAGS.lrate, 0.9, use_nesterov=True)
                # optimizer = tf.train.RMSPropOptimizer(FLAGS.lrate)
                self.train_step = slim.learning.create_train_op(self.total_loss, optimizer, self.global_step,
                                                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
    
    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.pack([boxes1[:, :, :, 0],
                          boxes1[:, :, :, 1],
                          boxes1[:, :, :, 0] + boxes1[:, :, :, 2],
                          boxes1[:, :, :, 1] + boxes1[:, :, :, 3]])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

        boxes2 = tf.pack([boxes2[:, :, :, 0],
                          boxes2[:, :, :, 1],
                          boxes2[:, :, :, 0] + boxes2[:, :, :, 2],
                          boxes2[:, :, :, 1] + boxes2[:, :, :, 3]])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :2], boxes2[:, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[:, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * \
            (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[:, :, :, 2] - boxes2[:, :, :, 0]) * \
            (boxes2[:, :, :, 3] - boxes2[:, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    
    
    def loss_function(self, tensor_x, label_y):
    
        tensors = tf.reshape(tensor_x,(-1,self.cell_size,self.cell_size,6))
        labels = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,6))
        batch_size = tf.shape(tensor_x)[0]
        
        predict_confidence = tensors[:,:,:,0]
        predict_boxes = tensors[:,:,:,1:5]
        predict_trackid = tensors[:,:,:,5]
        
        label_confidence = labels[:,:,:,0]
        label_boxes = labels[:,:,:,1:5]
        label_trackid = labels[:,:,:,5]
        
        with tf.name_scope('coord_loss'):
            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset,[1, self.cell_size, self.cell_size])
            offset = tf.tile(offset, [batch_size, 1, 1])
            
            # predict_boxes_tran = tf.pack([(predict_boxes[:, :, :, 0] + offset) / self.cell_size,
                                          # (predict_boxes[:, :, :, 1] + tf.transpose(offset, (0, 2, 1))) / self.cell_size,
                                          # tf.square(predict_boxes[:, :, :, 2]),
                                          # tf.square(predict_boxes[:, :, :, 3])])                        
            # predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 0])
            
            #iou_predict_truth = self.calc_iou(predict_boxes_tran, label_boxes)
            #object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
            
            label_boxes_tran = tf.stack([label_boxes[:, :, :, 0] * self.cell_size - offset,
                                  label_boxes[:, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1)),
                                  tf.sqrt(label_boxes[:, :, :, 2]),
                                  tf.sqrt(label_boxes[:, :, :, 3])])
            label_boxes_tran = tf.transpose(label_boxes_tran, [1, 2, 3, 0])
            
            boxes_delta = label_boxes_tran - predict_boxes
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), reduction_indices=[1, 2, 3])) * self.coordloss_scale
            
        with tf.name_scope('conf_loss'):
            conf_delta = label_confidence - predict_confidence
            conf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(conf_delta), reduction_indices=[1, 2])) * self.confloss_scale
        
        with tf.name_scope('track_loss'):
            track_delta = label_trackid - predict_trackid
            track_loss = tf.reduce_mean(tf.reduce_sum(tf.square(track_delta), reduction_indices=[1, 2])) * self.trackloss_scale
            
        return coord_loss + conf_loss + track_loss

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
