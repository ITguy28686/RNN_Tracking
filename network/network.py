import config
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from network.model import Model

FLAGS = tf.app.flags.FLAGS


class Network:
    def __init__(self, is_training):
        
        self.cell_size = 9
        self.boxes_per_cell = 3
        self.track_num = 30
        self.img_size = 360
        self.GRU_SIZE = 1620
        
        self.coord_scale = 5
        self.object_scale = 20
        self.noobject_scale = 20
        self.track_scale = 2
        
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
            
        self.graph = tf.Graph()
        self.is_training = is_training
        self.eval()
            

    def eval(self):
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=(None,self.img_size, self.img_size, 4), name="img_inputs")
            self.x_nchw = tf.transpose(self.x, perm=[0,3,1,2])
            
            self.h_state_init_1 = tf.placeholder(dtype=tf.float32, shape=(1, self.GRU_SIZE), name="h_state_init1")
            self.h_state_init_2 = tf.placeholder(dtype=tf.float32, shape=(1, self.GRU_SIZE), name="h_state_init2")
            _h_state_init = tuple([self.h_state_init_1,self.h_state_init_2])
            
            self.det_anno = tf.placeholder(dtype=tf.float32, shape=(None, self.cell_size, self.cell_size, 5), name="det_anno")
            
            # self.cell_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 4096), name="cell_state_init")
            
            self.track_y = tf.placeholder(dtype=tf.float32, shape=(None, self.cell_size * self.cell_size * (5+self.cell_size*self.cell_size+1)), name='track_label')
            
            mynet = Model(self.x_nchw, self.det_anno, _h_state_init, is_training=True, data_format='NCHW', keep_prob=0.5)

            coord_flow = mynet.coord_flow
            association_flow = mynet.association_flow

            with tf.name_scope('Cost'):
                
                coord_loss, iou_predict_truth = self.coord_loss_function(coord_flow, self.track_y, name='coord_loss')
                ass_loss = self.association_loss(coord_flow, iou_predict_truth, association_flow, self.track_y)
                
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
          boxes1: 5-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_left, y_top, w, h)
          boxes2: 5-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_left, y_top, w, h)
        Return:
          iou: 4-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[..., 0],
                          boxes1[..., 1],
                          boxes1[..., 0] + boxes1[..., 2],
                          boxes1[..., 1] + boxes1[..., 3]], axis=-1)

        boxes2 = tf.stack([boxes2[..., 0],
                          boxes2[..., 1],
                          boxes2[..., 0] + boxes2[..., 2],
                          boxes2[..., 1] + boxes2[..., 3]], axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        rd = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
        
    def calc_rmse(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_left, y_top, w, h)
          boxes2: 5-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_left, y_top, w, h)
        Return:
          iou: 4-D tensor [BATCHSIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[..., 0],
                          boxes1[..., 1],
                          boxes1[..., 0] + boxes1[..., 2],
                          boxes1[..., 1] + boxes1[..., 3]], axis=-1)

        boxes2 = tf.stack([boxes2[..., 0],
                          boxes2[..., 1],
                          boxes2[..., 0] + boxes2[..., 2],
                          boxes2[..., 1] + boxes2[..., 3]], axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        rd = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
        
        
    
    def coord_loss_function(self, bbox_tensor, label_y, name='coord_loss'):

        tensors = tf.reshape(bbox_tensor,(-1,self.cell_size,self.cell_size,self.boxes_per_cell,5))
        label_tensor = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,5+self.cell_size*self.cell_size+1))
        batch_size = tf.shape(bbox_tensor)[0]
        
        predict_confidence = tensors[:,:,:,:,0] # shape = (batch_size,cell_size,cell_size,boxes_per_cell)
        predict_boxes = tensors[:,:,:,:,1:5] # shape = (batch_size,cell_size,cell_size,boxes_per_cell,4)
        
        label_confidence = tf.reshape(label_tensor[..., 0],
                [batch_size, self.cell_size, self.cell_size, 1])
        
        label_boxes = tf.reshape(label_tensor[..., 1:5],
                [batch_size, self.cell_size, self.cell_size, 1, 4])
        label_boxes = tf.tile(label_boxes, [1, 1, 1, self.boxes_per_cell, 1])
        
        offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        
        with tf.name_scope('conf_'+name):
            predict_boxes_tran = tf.stack([(predict_boxes[..., 0] + offset) / self.cell_size,
                                      (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                                      tf.square(predict_boxes[..., 2]),
                                      tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calc_iou(predict_boxes_tran, label_boxes)
            
            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            # object_mask = tf.cast(
                # (iou_predict_truth >= object_mask), tf.float32) * label_confidence
                
            object_mask = tf.cast(iou_predict_truth >= object_mask , tf.float32) * tf.cast(iou_predict_truth > tf.reshape(0.0,(1,1,1,1)) , tf.float32)
            
            object_mask_addconf = object_mask + label_confidence
            object_mask2 = tf.reduce_max(object_mask_addconf, 3, keep_dims=True)
            object_mask = tf.cast(object_mask_addconf >= object_mask2, tf.float32) * tf.cast(object_mask_addconf > tf.reshape(0.0,(1,1,1,1)) , tf.float32)

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask
            
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_mask * ( predict_confidence - iou_predict_truth)),
                                        reduction_indices=[1, 2, 3]), name='object_loss') * self.object_scale
                                        
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_mask * predict_confidence),
                                        reduction_indices=[1, 2, 3]), name='noobject_loss') * self.noobject_scale
                                        
            tf.summary.scalar(name + '/object_loss', object_loss)
            tf.summary.scalar(name + '/noobject_loss', noobject_loss)
        
        
        with tf.name_scope(name):
            
            label_boxes_tran = tf.stack([label_boxes[..., 0] * self.cell_size - offset,
                                  label_boxes[..., 1] * self.cell_size - offset_tran,
                                  tf.sqrt(label_boxes[..., 2]),
                                  tf.sqrt(label_boxes[..., 3])], axis=-1)
            
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - label_boxes_tran)
            
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), reduction_indices=[1, 2, 3, 4])) * self.coord_scale
            
            tf.summary.scalar(name, coord_loss)

        return coord_loss + object_loss + noobject_loss, iou_predict_truth
    
    def association_loss(self, bbox_tensor, iou_predict_truth, tracking_tensor, label_y):

        with tf.name_scope('track_loss'):
        
            # tensors = tf.reshape(bbox_tensor,(-1,self.cell_size,self.cell_size,self.boxes_per_cell,5))
            # predict_confidence = tf.reduce_max(tensors[:,:,:,:,0], 3, keep_dims=True) 
            # track_match_predict = tf.reshape(tracking_tensor,(-1, self.cell_size, self.cell_size, self.cell_size*self.cell_size+1))
            
            # label_tensor = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,5+self.cell_size*self.cell_size+1))
            # label_confidence = tf.expand_dims(label_tensor[:,:,:,0], axis=3)
            # track_match_label = label_tensor[:, :, :, 5:]
            # track_match_label *= tf.expand_dims(label_tensor[:,:,:,0],3)
            
            # predict_noobject_prob = tf.ones_like(
                    # predict_confidence, dtype=tf.float32) - predict_confidence
            # predict_track_tran = tf.concat([predict_noobject_prob, track_match_predict], axis=3)
            
            # label_noobject_prob = tf.ones_like(
                    # label_confidence, dtype=tf.float32) - label_confidence
            # label_track_tran = tf.concat([label_noobject_prob, track_match_label], axis=3)
            
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( logits=predict_track_tran, labels=label_track_tran, name='xentropy')
            
            # track_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, reduction_indices=[1, 2]), name='xentropy_mean')
            
            label_tensor = tf.reshape(label_y,(-1,self.cell_size,self.cell_size,5+self.cell_size*self.cell_size+1))
            
            boxes_delta = coord_mask * (tracking_tensor - label_boxes_tran)
            
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), reduction_indices=[1, 2, 3, 4])) * self.coord_scale
            
            
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
