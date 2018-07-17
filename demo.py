import os
import math
import random

import numpy as np
import cv2
import tensorflow as tf

slim = tf.contrib.slim
import sys

from network.model import Model

from utils.timer import Timer
import argparse


chkpt_file = "network/logs/model.ckpt"
test_tf = "train_tf/MOT16-02-0_train.tfrecord"

cell_size = 8
offset = np.reshape(np.array(
        [np.arange(cell_size)] * cell_size),
        (cell_size, cell_size))
        
offset_tran = np.transpose(offset, (1, 0))

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
img_input = tf.placeholder(dtype=tf.float32, shape=(1,300, 300, 4))
h_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 768))
cell_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 768))

graph = tf.Graph()
mynet = Model(img_input, h_state_init, cell_state_init, is_training=False, keep_prob=0, data_format='NHWC')

# Restore model.
ckpt_filename = './network/logs/model.ckpt'

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

def feature_decode(example):
 
        #decode concated mat
        frame_mat_shape = example.features.feature['frame_concate_mat_shape'].int64_list.value
        frame_mat_string = example.features.feature['frame_concat_mat'].bytes_list.value[0]
        frame_mat = np.fromstring(frame_mat_string, dtype=np.float32).reshape(frame_mat_shape)
        frame_mat = np.expand_dims(frame_mat, axis=0)

        #cv2.imshow("Image", frame_img)
        #cv2.waitKey(0)
        
        return frame_mat

def process_coord_logits(tensor_x):

    tensors = np.reshape(tensor_x,(cell_size,cell_size,5))
    
    confidence = tensors[:,:,0]
    predict_boxes = tensors[:,:,1:5]

    boxes = np.stack([(predict_boxes[..., 0] + offset) / cell_size * 300,
                 (predict_boxes[..., 1] + offset_tran) / cell_size * 300,
                 np.square(predict_boxes[..., 2]) * 300,
                 np.square(predict_boxes[..., 3]) * 300])
    boxes = boxes.astype(np.int32)
    #boxes = np.transpose(boxes, [1, 2, 0])
        
    return confidence, boxes
        
def process_logits(tensor_x):

    tensors = np.reshape(tensor_x,(cell_size,cell_size,7))
    
    confidence = tensors[:,:,0]
    predict_boxes = tensors[:,:,1:5]
    newtrack_conf = tensors[:,:,5]
    trackid = tensors[:,:,6].astype(np.int32)

    boxes = np.stack([(predict_boxes[..., 0] + offset) / cell_size * 300,
                 (predict_boxes[..., 1] + offset_tran) / cell_size * 300,
                 np.square(predict_boxes[..., 2]) * 300,
                 np.square(predict_boxes[..., 3]) * 300])
    boxes = boxes.astype(np.int32)
    #boxes = np.transpose(boxes, [1, 2, 0])
         
    return confidence, boxes, newtrack_conf, trackid
    
def check_point_inbound(point,width,height):
    if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height :
        return False
    return True
    
    

def draw_frame(img, confidence, boxes, newtrack_conf, trackid):
    #print(img.shape)
    for i in range(cell_size):
        for j in range(cell_size):
            if confidence[i][j] > 0.2 :
                #print(boxes[0][i][j],boxes[1][i][j],boxes[2][i][j],boxes[3][i][j])
                left_top = (boxes[0][i][j],boxes[1][i][j])
                right_bottom = (boxes[0][i][j]+boxes[2][i][j],boxes[1][i][j]+boxes[3][i][j])
                if check_point_inbound(left_top,300,300) and check_point_inbound(right_bottom,300,300) : 
                    cv2.rectangle(img, left_top, right_bottom, (0,255,0), 1)
                    cv2.putText(img, "ID: " + str(trackid[i][j]), (left_top[0]+5, left_top[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    
    
    return img
  
# Main image processing routine.
def process_image(concat_img,h_state,cell_state):
    # Run SSD network.
    logits, coord_logits, lstm_state = isess.run([mynet.logits, mynet.coord_logits, mynet.lstm_state],
                                                              feed_dict={img_input: concat_img, h_state_init: h_state, cell_state_init: cell_state})
    
    cell_state = lstm_state[0]
    h_state = lstm_state[1]
    
    confidence, boxes, newtrack_conf, trackid = process_logits(logits)
    confidence2, boxes2 = process_coord_logits(coord_logits)
    
    result_img = draw_frame(concat_img[0][...,0:3].copy(), confidence2, boxes2, newtrack_conf, trackid)
    cv2.imshow("result",result_img)
    cv2.waitKey(1)
    
    return h_state, cell_state
	

def tf_track():

    detect_timer = Timer()
    # cap = cv2.VideoCapture(video)
    # ret, _ = cap.read()

    h_state = np.zeros(768).reshape(1,768).astype(np.float32)
    cell_state = np.zeros(768).reshape(1,768).astype(np.float32)
    
    for string_record in tf.python_io.tf_record_iterator(path=test_tf):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        concat_img = feature_decode(example)
		
        detect_timer.tic()
        h_state, cell_state =  process_image(concat_img, h_state, cell_state)

        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        #visualization.display_video(frame, rclasses, rscores, rbboxes)
		
        detect_timer.toc()
        print('detecting time: {:.3f}s'.format(detect_timer.diff))
    print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))
        
	
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video', default="", type=str)
	
    # args = parser.parse_args()

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from video file
    tf_track()


if __name__ == '__main__':
    main()




