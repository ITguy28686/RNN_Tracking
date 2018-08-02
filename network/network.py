import config
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from network.model import Model

FLAGS = tf.app.flags.FLAGS


class Network:
    def __init__(self, is_training):
        
        self.cell_size = 9
        self.track_num = 30
        self.coord_scale = 5
        self.object_scale = 1
        self.noobject_scale = 1
        self.track_scale = 2
        
        self.offset = np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size),
            (self.cell_size, self.cell_size))
            
        self.graph = tf.Graph()
        self.is_training = is_training
        self.eval()
            

    def eval(self):
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=(None,300, 300, 4), name="img_inputs")
            self.x_nchw = tf.transpose(self.x, perm=[0,3,1,2])
            
            self.h_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 2048), name="h_state_init")
            self.cell_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 2048), name="cell_state_init")
            
            self.track_y = tf.placeholder(dtype=tf.float32, shape=(None, self.cell_size*self.cell_size*(5+self.track_num)), name='track_label')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            
            mynet = Model(self.x_nchw, self.h_state_init, self.cell_state_init, self.is_training, self.keep_prob)

            coord_flow = mynet.coord_flow
            association_flow = mynet.association_flow

            with tf.name_scope('Cost'):
                
                coord_loss = self.coord_loss_function(coord_flow,self.track_y,name='coord_loss')
                ass_loss = self.association_loss(association_flow,self.track_y)
                
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                
                self.total_loss = tf.add_n([coord_loss + ass_loss ] + reg_losses)
                # self.total_loss = tf.add_n(reg_losses)
                
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
            self.saver = tf.train.Saver(max_to_keep=None)
    
    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, 4]  ====> (x_center, y_center, w, h)
          boxes2: 4-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0],
                          boxes1[:, :, :, 1],
                          boxes1[:, :, :, 0] + boxes1[:, :, :, 2],
                          boxes1[:, :, :, 1] + boxes1[:, :, :, 3]])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

        boxes2 = tf.stack([boxes2[:, :, :, 0],
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
    
    def coord_loss_function(self, bbox_tensor, label_y, name='coord_loss'):

        tensors = tf.reshape(bbox_tensor,(-1,self.cell_size,self.cell_size,5))
        labels = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,5+self.track_num))
        batch_size = tf.shape(bbox_tensor)[0]
        
        predict_confidence = tensors[:,:,:,0]
        predict_boxes = tensors[:,:,:,1:5]
        
        label_confidence = labels[:,:,:,0]
        label_boxes = labels[:,:,:,1:5]
        
        with tf.name_scope(name):
            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset,[1, self.cell_size, self.cell_size])
            offset = tf.tile(offset, [batch_size, 1, 1])
            
            label_boxes_tran = tf.stack([label_boxes[:, :, :, 0] * self.cell_size - offset,
                                  label_boxes[:, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1)),
                                  tf.sqrt(label_boxes[:, :, :, 2]),
                                  tf.sqrt(label_boxes[:, :, :, 3])])
                                  
            label_boxes_tran = tf.transpose(label_boxes_tran, [1, 2, 3, 0])
            
            boxes_delta = label_boxes_tran - predict_boxes
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), reduction_indices=[1, 2, 3])) * self.coord_scale
            
            tf.summary.scalar(name, coord_loss)
            
        with tf.name_scope('conf_'+name):
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, 0] + offset) / self.cell_size,
                                      (predict_boxes[:, :, :, 1] + tf.transpose(offset, (0, 2, 1))) / self.cell_size,
                                      tf.square(predict_boxes[:, :, :, 2]),
                                      tf.square(predict_boxes[:, :, :, 3])])

            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, label_boxes)
        
            noobject_mask = tf.ones_like(label_confidence, dtype=tf.float32) - label_confidence
            
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(label_confidence * (predict_confidence - iou_predict_truth)),
                                        reduction_indices=[1, 2]), name='object_loss') * self.object_scale
                                        
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_mask * predict_confidence),
                                        reduction_indices=[1, 2]), name='noobject_loss') * self.noobject_scale
                                        
            tf.summary.scalar(name + '/object_loss', object_loss)
            tf.summary.scalar(name + '/noobject_loss', noobject_loss)
        
        return coord_loss + object_loss + noobject_loss
    
    def association_loss(self, tracking_tensor, label_y):

        track_class_predict = tf.reshape(tracking_tensor,(-1,self.cell_size,self.cell_size,self.track_num))
        
        labels = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,5+self.track_num))
        label_confidence = tf.expand_dims(labels[:,:,:,0], axis=3)
        
        track_class_label = labels[:, :, :, 5:]
        
        with tf.name_scope('track_loss'):
            track_loss = tf.reduce_mean(tf.reduce_sum(tf.square(label_confidence * (track_class_label - track_class_predict)),
            reduction_indices=[1, 2, 3])) * self.track_scale
            
            tf.summary.scalar("track_loss", track_loss)
            
        return track_loss
    
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
