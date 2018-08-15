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
import re


chkpt_file = "network/logs/model.ckpt-100000"
# chkpt_file = "network/logs/old_logs/GRU_version/model.ckpt-40000"
tf_pattern = "train_tf/MOT16-02-*"
img_dir = "D:/DataSet/MOT16/train/MOT16-02/img1"

data_format='NCHW'

cell_size = 9
track_num = 30
boxes_per_cell = 3

offset = np.reshape(np.array(
        [np.arange(cell_size)] * cell_size * boxes_per_cell),
        (boxes_per_cell,cell_size, cell_size))
        
offset = np.transpose(offset, (1,2,0))
        
offset_tran = np.transpose(offset, (1, 0, 2))

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
img_input = tf.placeholder(dtype=tf.float32, shape=(1,300, 300, 4))
if data_format == 'NCHW' :
    img_input2 = tf.transpose(img_input, perm=[0,3,1,2])
else :
    img_input2 = img_input
    
h_state_init_1 = tf.placeholder(dtype=tf.float32, shape=(1, 4096))
h_state_init_2 = tf.placeholder(dtype=tf.float32, shape=(1, 4096))
_h_state_init = tuple([h_state_init_1,h_state_init_2])
cell_state_init = tf.placeholder(dtype=tf.float32, shape=(1, 768))




graph = tf.Graph()
mynet = Model(img_input2, _h_state_init, cell_state_init, is_training=False, keep_prob=1, data_format=data_format)

isess.run(tf.global_variables_initializer())

# Restore model.
saver = tf.train.Saver()
saver.restore(isess, chkpt_file)

def coord_loss(tensor_x, label_y):
    tensors = np.reshape(tensor_x,(cell_size,cell_size,5))
    labels = np.reshape(label_y,(cell_size,cell_size,7))
    
    predict_confidence = tensors[:,:,0]
    predict_boxes = tensors[:,:,1:]
    
    label_confidence = labels[:,:,0]
    label_boxes = labels[:,:,1:5]
    
    label_boxes_tran = np.stack([label_boxes[..., 0] * cell_size - offset,
                          label_boxes[..., 1] * cell_size - offset_tran,
                          np.sqrt(label_boxes[..., 2]),
                          np.sqrt(label_boxes[..., 3])])
    label_boxes_tran = np.transpose(label_boxes_tran, [1, 2, 0])
    
    boxes_delta = label_boxes_tran - predict_boxes
    coord_loss = np.sum(np.square(boxes_delta), axis =(0,1,2)) * 1

    conf_delta = label_confidence - predict_confidence
    conf_loss = np.sum(np.square(conf_delta), axis =(0, 1)) * 20
  
    return coord_loss + conf_loss
    
    



def feature_decode(example):
 
        frame_id = example.features.feature['frame_id'].int64_list.value[0]
 
        #decode concated mat
        frame_mat_shape = example.features.feature['frame_concate_mat_shape'].int64_list.value
        frame_mat_string = example.features.feature['frame_concat_mat'].bytes_list.value[0]
        frame_mat = np.fromstring(frame_mat_string, dtype=np.float32).reshape(frame_mat_shape)
        frame_mat = np.expand_dims(frame_mat, axis=0)
        
        frame_gt = example.features.feature['frame_gt'].float_list.value

        #cv2.imshow("Image", frame_img)
        #cv2.waitKey(0)
        
        return frame_id, frame_mat, frame_gt

def process_coord_logits(tensor_x,frame_gt,img_W,img_H):

    tensors = np.reshape(tensor_x,(cell_size,cell_size,boxes_per_cell,5))
    labels = np.reshape(frame_gt,(cell_size,cell_size,5+track_num))
    
    confidence = tensors[...,0]
    confidence = confidence.max(axis=2)
    #print(confidence)
    
    #sys.exit(0)
    # predict_boxes = tensors[:,:,1:5]
    det_boxes = labels[:,:,1:5]

    # boxes = np.stack([(predict_boxes[..., 0] + offset) / cell_size * img_W,
                 # (predict_boxes[..., 1] + offset_tran) / cell_size * img_H,
                 # np.square(predict_boxes[..., 2]) * img_W,
                 # np.square(predict_boxes[..., 3]) * img_H])
                 
    boxes = np.stack([det_boxes[..., 0] * img_W,
                 det_boxes[..., 1] * img_H,
                 det_boxes[..., 2] * img_W,
                 det_boxes[..., 3] * img_H])
                 
    boxes = boxes.astype(np.int32)
    # boxes = np.transpose(boxes, [2, 0, 1])
        
    return confidence, boxes
    
    
def process_trackid_logits(tensor_x):

    tensors = np.reshape(tensor_x,(cell_size,cell_size,track_num))

    #boxes = np.transpose(boxes, [1, 2, 0])
         
    return tensors
    
def check_point_inbound(point,width,height):
    if point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height :
        return False
    return True
    
    

def draw_frame(img, confidence, boxes, trackid):
    
    #print(img.shape)
    for i in range(cell_size):
        for j in range(cell_size):
            if confidence[i][j] > 0.1 :
                #print(boxes[0][i][j],boxes[1][i][j],boxes[2][i][j],boxes[3][i][j])
                left_top = (boxes[0][i][j],boxes[1][i][j])
                right_bottom = (boxes[0][i][j]+boxes[2][i][j],boxes[1][i][j]+boxes[3][i][j])
                if check_point_inbound(left_top,img.shape[1],img.shape[0]) and check_point_inbound(right_bottom,img.shape[1],img.shape[0]) : 
                    cv2.rectangle(img, left_top, right_bottom, (0,255,0), 3)
                    cv2.putText(img, "ID: " + str(np.argmax(trackid[i][j])) + ", %.2f%%" % (confidence[i][j]), (left_top[0]-5, left_top[1]-0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    return img


# Main image processing routine.
def process_image(img, concat_img, frame_gt, h_state_1, h_state_2):
    # Run SSD network.
    # coord_flow, association_flow, coord_flow2, lstm_state = isess.run([mynet.coord_flow, mynet.association_flow, mynet.coord_flow2, mynet.lstm_state],
                                                              # feed_dict={img_input: concat_img, h_state_init: h_state, cell_state_init: cell_state})
                                                              
    coord_flow, association_flow, rnn_state = isess.run([mynet.coord_flow, mynet.association_flow, mynet.rnn_state],feed_dict={img_input: concat_img, h_state_init_1: h_state_1, h_state_init_2: h_state_2})
    
    #print(lstm_state.c,lstm_state.h)
    # cell_state = lstm_state.c
    # h_state = lstm_state.h
    # print(rnn_state)
    h_state_1 = rnn_state[0]
    h_state_2 = rnn_state[1]
    
    # newtrack_conf = np.zeros(8*8)
    # trackid = np.zeros(8*8).reshape(8,8)
    
    trackid = process_trackid_logits(association_flow)
    confidence, boxes = process_coord_logits(coord_flow,frame_gt,img.shape[1],img.shape[0])
    #loss = coord_loss(coord_flow,frame_gt)
    #print("loss: " + str(loss))
    
    
    
    return confidence, boxes, trackid, h_state_1, h_state_2
	

def tf_track():

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 1280,768)
    detect_timer = Timer()
    # cap = cv2.VideoCapture(video)
    # ret, _ = cap.read()

    #cell_state = np.zeros(4096).reshape(1,4096).astype(np.float32)
    h_state_1 = np.zeros(4096).reshape(1,4096).astype(np.float32)
    h_state_2 = np.zeros(4096).reshape(1,4096).astype(np.float32)
    
    img_files = {
            int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)}
    
    tf_files = np.array(tf.gfile.Glob(tf_pattern)).tolist()
    
    r = re.compile("\d+")
    #print(r.findall(tf_files[1]))
    
    tf_files.sort(key=lambda x: int(r.findall(x)[2]))
    
    for test_tf in tf_files:
        for string_record in tf.python_io.tf_record_iterator(path=test_tf):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            frame_id, concat_img, frame_gt = feature_decode(example)
            img = cv2.imread(img_files[frame_id])
            
            detect_timer.tic()
            #print(h_state)
            confidence, boxes, trackid, h_state_1, h_state_2 = process_image(img,concat_img, frame_gt, h_state_1, h_state_2)
            # h_state_1 = np.full((1,4096), 100, dtype=np.float32)
            # h_state_2 = np.full((1,4096), 100, dtype=np.float32)
            result_img = draw_frame(img, confidence, boxes, trackid)
            
            #det_mask = concat_img[0][...,3].copy()
        
            #cv2.imshow("det_mask",det_mask)
            cv2.imshow("result",result_img)
            cv2.waitKey(1)
        
            
            # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
            #visualization.display_video(frame, rclasses, rscores, rbboxes)
            
            detect_timer.toc()
            #print('detecting time: {:.3f}s'.format(detect_timer.diff))
        #os.system("pause")
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




