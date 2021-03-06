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
from utils.recorder import Recorder

import argparse
import re


# chkpt_file = "network/logs/model.ckpt-191000"
chkpt_file = "./network/old_logs/ass_epsilon_version/model.ckpt-191000"

# chkpt_file = "network/logs/old_logs/GRU_version/model.ckpt-40000"
tf_pattern = "train_tf/MOT16-04-*"

train_set = ["MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"]
test_set = ["MOT16-01","MOT16-03","MOT16-06","MOT16-07","MOT16-08","MOT16-12","MOT16-14"]


train_dir = "D:/DataSet/MOT16/train/"
val_dir = "D:/DataSet/MOT16/test/"
#Video_name = "MOT16-14"

data_format='NCHW'

cell_size = 9
track_num = 30
boxes_per_cell = 3
GRU_SIZE = 1620
img_size = 360
record_N = 256

track_thresh = 0.4
object_tresh = 0.2
match_thresh = 0.4

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
img_input = tf.placeholder(dtype=tf.float32, shape=(1,img_size, img_size, 4))
if data_format == 'NCHW' :
    img_input2 = tf.transpose(img_input, perm=[0,3,1,2])
else :
    img_input2 = img_input
    
h_state_init_1 = tf.placeholder(dtype=tf.float32, shape=(1, GRU_SIZE))
# h_state_init_2 = tf.placeholder(dtype=tf.float32, shape=(1, GRU_SIZE))

det_anno = tf.placeholder(dtype=tf.float32, shape=(1, cell_size * cell_size * 5))

prev_asscoia = tf.placeholder(dtype=tf.float32, shape=(1, record_N * (cell_size*cell_size+1)), name="prev_asscoia")

track_record = []
max_track_id = 0

graph = tf.Graph()
mynet = Model(img_input2, det_anno, prev_asscoia, h_state_init_1, is_training=False, keep_prob=1, data_format=data_format)

isess.run(tf.global_variables_initializer())

# Restore model.
saver = tf.train.Saver()
saver.restore(isess, chkpt_file)

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

        
def get_coord_loss(confidence, boxes, frame_det):

    offset2 = np.reshape(np.array(
            [np.arange(cell_size)] * cell_size),
            (cell_size, cell_size))
            
    offset_tran2 = np.transpose(offset2, (1, 0))

    label_boxes_tran = np.stack([frame_det[..., 1] * cell_size - offset2,
                                  frame_det[..., 2] * cell_size - offset_tran2,
                                  np.sqrt(frame_det[..., 3]),
                                  np.sqrt(frame_det[..., 4])], axis=-1)
    
    
    # print("label_boxes_tran-------------")
    # print(label_boxes_tran)
    
    boxes_delta = np.expand_dims(frame_det[..., 0], axis=2) * (boxes - label_boxes_tran)
            
    coord_loss = np.square(boxes_delta).sum() * 5
    
    conf_delta = np.expand_dims(frame_det[..., 0], axis=2) * (confidence - frame_det[..., 0])
    
    conf_loss = np.square(conf_delta).sum() * 2
    
    return conf_loss, coord_loss
    

        
def process_coord_logits(tensor_x,frame_det,img_W,img_H):

    tensors = np.reshape(tensor_x,(cell_size,cell_size,boxes_per_cell,5))
    labels = np.reshape(frame_det,(cell_size,cell_size,5))
    
    # print(conf_max_arg.shape)
    # sys.exit(0)
    #print(confidence)
    
    #sys.exit(0)
    # predict_boxes = tensors[..., 1:5]
    # predict_boxes_copy = np.copy(predict_boxes)
    # print("predict_boxes_copy-------------")
    # print(predict_boxes_copy)

    
    # predict_boxes[..., 0] += offset
    # predict_boxes[..., 1] += offset_tran
    # predict_boxes[..., :2] = 1.0 * predict_boxes[..., 0:2] / cell_size
    # predict_boxes[..., 2:] = np.square(predict_boxes[..., 2:])

    # predict_boxes[..., 0] *= img_W
    # predict_boxes[..., 1] *= img_H
    # predict_boxes[..., 2] *= img_W
    # predict_boxes[..., 3] *= img_H
    
    # det_boxes = frame_det[:,:,1:5]
    boxes = np.zeros((cell_size,cell_size,4))
    # boxes_copy = np.zeros((cell_size,cell_size,4))
    
    # confidence = tensors[...,0]
    # conf_max_arg = confidence.argmax(axis=2)

    # confidence = confidence.max(axis=2, keepdims = False)
    
    confidence = labels[..., 0]
    
    # print("conf_max-------------")
    # print(confidence)
    
    
    
    # for i in range(cell_size):
        # for j in range(cell_size):
            # boxes[i][j][0] = predict_boxes[i, j, conf_max_arg[i][j], 0]
            # boxes[i][j][1] = predict_boxes[i, j, conf_max_arg[i][j], 1]
            # boxes[i][j][2] = predict_boxes[i, j, conf_max_arg[i][j], 2]
            # boxes[i][j][3] = predict_boxes[i, j, conf_max_arg[i][j], 3]
            
    # conf_loss, loss = get_coord_loss(confidence, boxes_copy, frame_det)
    # print("loss = " + str(conf_loss) +", " + str(loss))
    
    # boxes = np.stack([det_boxes[..., 0] * img_W,
                 # det_boxes[..., 1] * img_H,
                 # det_boxes[..., 2] * img_W,
                 # det_boxes[..., 3] * img_H])
                 
    boxes[..., 0] = labels[..., 1] * img_W
    boxes[..., 1] = labels[..., 2] * img_H
    boxes[..., 2] = labels[..., 3] * img_W
    boxes[..., 3] = labels[..., 4] * img_H
    
    boxes = boxes.astype(np.int32)
    
    # boxes = np.transpose(boxes, [2, 0, 1])
        
    return confidence, boxes
    
def cal_match(mask1, mask2):
    _outer = np.sum((mask1 | mask2))
    
    _inner = np.sum((mask1 & mask2))
    
    # print("\nmask1:---------")
    # print(mask1)
    # print("\nmask2:---------")
    # print(mask2)
    if ( _outer == 0 ):
        return 0
    
    return _inner/_outer

def box_to_mask(box,W,H):

    # print(box)
    mask = np.zeros((cell_size,cell_size), dtype = np.bool)
    
    x_lt = int(box[0]/W * cell_size)
    y_lt = int(box[1]/H * cell_size)
    x_rd = int((box[0]+box[2])/W * cell_size) + 1
    y_rd = int((box[1]+box[3])/H * cell_size) + 1
    
    # print("219999999")
    # print(x_lt,y_lt,x_rd,y_rd)
    
    x_lt = max(0, min(cell_size, x_lt))
    y_lt = max(0, min(cell_size, y_lt))
    x_rd = max(0, min(cell_size, x_rd))
    y_rd = max(0, min(cell_size, y_rd))
    
    # print("2277777777")
    # print(x_lt,y_lt,x_rd,y_rd)

    for i in range(y_lt, y_rd):
        for j in range(x_lt, x_rd):
            mask[i][j] = True
            
    return mask
    
    
    
    
def process_trackid_logits_and_draw(frame_idx, img, confidence, boxes, epsilon_flow, associa_flow):

    global track_record
    global max_track_id

    # tensors = np.reshape(tensor_x,(cell_size,cell_size,track_num))

    #boxes = np.transpose(boxes, [1, 2, 0])

    associa_prob = np.reshape(associa_flow,( record_N, cell_size*cell_size+1))
    
    # predict_noobject_prob = np.ones_like(
                    # confidence, dtype=np.float32) - confidence
    
    # predict_track_tran = np.concatenate((predict_noobject_prob, track_prob), axis=2)
    
    # max_prob_track = np.argmax(predict_track_tran, 2)
    # print(max_prob_track.dtype)
    
    record = []
    # print(confidence)
    
    column_index_of_max_value_every_row = np.argmax(associa_prob, axis=1)
    row_index_of_max_value_every_column = np.argmax(associa_prob, axis=0)
    print(row_index_of_max_value_every_column)
    print(column_index_of_max_value_every_row)
    # np.savetxt('numpy_out.txt', associa_prob, delimiter=',')   # X is an array
    
    for i in range(cell_size):
        for j in range(cell_size):
            
            if(confidence[i][j] < object_tresh):
                continue
            
            # print(210)
            
            index = i * cell_size + j
            
            max_id = row_index_of_max_value_every_column[index] + 1
                    
            draw_frame(img, i, j, boxes, max_id)
            
            # cv2.waitKey(0)
            record += [[frame_idx, max_id, boxes[i][j][0], boxes[i][j][1], boxes[i][j][2], boxes[i][j][3]]]
                
    return img, record
    
def check_point_inbound(point,width,height):
    if point[0] < 0:
        point[0] = 0
    elif point[0] > width:
        point[0] = width
        
        
    if point[1] < 0:
        point[1] = 0
    elif point[1] > width:
        point[1] = width
        
    return point
    
    

def draw_frame(img, i, j, boxes, trackid):
    
    trackid = int(trackid)
    
    #print(boxes[0][i][j],boxes[1][i][j],boxes[2][i][j],boxes[3][i][j])
    left_top = [boxes[i][j][0],boxes[i][j][1]]
    right_bottom = [boxes[i][j][0]+boxes[i][j][2],boxes[i][j][1]+boxes[i][j][3]]
    
    left_top = check_point_inbound(left_top,img.shape[1],img.shape[0])
    right_bottom = check_point_inbound(right_bottom,img.shape[1],img.shape[0])
    
    cv2.rectangle(img, (left_top[0],left_top[1]) , (right_bottom[0],right_bottom[1]), ((87*trackid)%255,(293*trackid)%255,(159*trackid)%255), 3)
    cv2.putText(img, "ID: " + str(trackid) , (left_top[0]-5, left_top[1]-0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    return img

def remove_outdate_record(timeout):
    global track_record
    remove_items = []
    
    for i in range(len(track_record)):
        if(track_record[i].hit == False):
            track_record[i].counter += 1
        
        if(track_record[i].counter == timeout):
            remove_items += [i]
            
    remove_items.sort(reverse=True)
    
    for index in remove_items:
        track_record.pop(index)
        
    for i in range(len(track_record)):
        track_record[i].hit = False
        
# Main image processing routine.
def process_image(frame_idx, img, concat_img, frame_det, prev_asscoia_tensor, h_state_1):
    # Run SSD network.
    # coord_flow, association_flow, coord_flow2, lstm_state = isess.run([mynet.coord_flow, mynet.association_flow, mynet.coord_flow2, mynet.lstm_state],
                                                              # feed_dict={img_input: concat_img, h_state_init: h_state, cell_state_init: cell_state})
                                                              
                                                              
    
    frame_det_tensor = frame_det.flatten()
    frame_det_tensor = np.expand_dims(frame_det_tensor, axis = 0)
    
    # print(frame_det_tensor)
    print(frame_det_tensor.shape)
    print(h_state_1.shape)
    
    # print(concat_img)
    # print(concat_img.shape)
    
    
    coord_flow, epsilon_flow, associa_flow, rnn_coord_state, rnn_associa_state = isess.run(
                                        [mynet.coord_flow, mynet.epsilon_flow, mynet.associa_flow, mynet.rnn_coord_state, mynet.rnn_associa_state],
                                          feed_dict={img_input: concat_img,
                                                    # prev_asscoia: prev_asscoia_tensor,
                                                    det_anno: frame_det_tensor,
                                                    h_state_init_1: h_state_1
                                                    })
    

    # sys.exit(0)
    
    h_state_1 = rnn_coord_state[0]
    # h_state_2 = rnn_state[1]
    
    # newtrack_conf = np.zeros(8*8)
    # trackid = np.zeros(8*8).reshape(8,8)
    
    confidence, boxes = process_coord_logits(coord_flow, frame_det, img.shape[1], img.shape[0])
    img, record = process_trackid_logits_and_draw(frame_idx, img, confidence, boxes, epsilon_flow, associa_flow)
    # img = gt_trackid_logits_and_draw(img, boxes, frame_gt)
    
    # draw_frame(img, confidence, boxes, trackid)
    
    #loss = coord_loss(coord_flow,frame_gt)
    #print("loss: " + str(loss))
    
    timeout = 150
    remove_outdate_record(timeout)
    
    # sys.exit(0)
    
    return img, associa_flow.reshape(1, record_N*(cell_size*cell_size+1)) , h_state_1, record


def encode_det(frame_idx, det_array, img):

    frame_indices = det_array[:, 0].astype(np.int)

    mask = (frame_indices == frame_idx) & ( det_array[:, 6] > 0.3)
    
    rows = det_array[mask]
    
    mask_img = np.zeros((img_size,img_size,1), np.float32)
    
    det_tensor = np.zeros((cell_size,cell_size, 5), dtype=np.float32)
    
    for i in range(rows.shape[0]):
    
        if(rows[i][2] < 0):
            rows[i][4] += rows[i][2]
            rows[i][2] = 0

        if(rows[i][3] < 0):
            rows[i][5] += rows[i][3]
            rows[i][3] = 0
            
        if (rows[i][2] + rows[i][4] >= 1):
            rows[i][4] = 0.999 - rows[i][2]
            
        if (rows[i][3] + rows[i][5] >= 1):
            rows[i][5] = 0.999 - rows[i][3]
    
    
        mask_x = int(rows[i][2]*img_size)
        mask_y = int(rows[i][3]*img_size)
        mask_w = int(rows[i][4]*img_size)
        mask_h = int(rows[i][5]*img_size)
        
        # if(mask_x >= 300 or mask_y>= 300):
            # print('%f,%f  %f,%f' % (rows[i][1],rows[i][2],mask_x,mask_y))
        
        # if(mask_x < 0):
            # mask_w += mask_x
            # mask_x = 0
                  
        # if(mask_y < 0):
            # mask_h += mask_y
            # mask_y = 0
        
        # if(mask_x+mask_w >= img_size):
            # mask_w = img_size - mask_x - 1
        
        # if(mask_y+mask_h >= img_size):
            # mask_h = img_size - mask_y - 1
        
        for y in range(mask_y,mask_y+mask_h):
            mask_img[y][mask_x] = 1
            mask_img[y][mask_x+mask_w] = 1
            
        for x in range(mask_x,mask_x+mask_w):
            mask_img[mask_y][x] = 1
            mask_img[mask_y+mask_h][x] = 1
        
        x_ind = int(rows[i][2] * cell_size)
        y_ind = int(rows[i][3] * cell_size)
        
        if det_tensor[y_ind, x_ind, 0] == 1:
            continue
        
        det_tensor[y_ind, x_ind, 0] = 1
        det_tensor[y_ind, x_ind, 1:5] = rows[i][2:6]    
            
    mask_img = mask_img.astype(np.float32)
        
    img_f = np.float32(img)
    img_f /= 255
    img_f = img_f.clip(0,1.)
    
    concat_img = np.concatenate((img_f, mask_img),axis = 2)
    concat_img = np.expand_dims(concat_img, axis = 0)
    
    # print(det_tensor)
    # cv2.imshow("mask",mask_img)
    # cv2.waitKey(0)
    # print("/////////////////////////////////")

    return concat_img, det_tensor


    
def det_track(set_dir, set_name):
    
    global track_record
    track_record = []
    
    global max_track_id
    max_track_id = 0

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 1280,768)
    detect_timer = Timer()
    # cap = cv2.VideoCapture(video)
    # ret, _ = cap.read()

    #cell_state = np.zeros(4096).reshape(1,4096).astype(np.float32)
    h_state_1 = np.zeros((1,GRU_SIZE), dtype=np.float32)
    prev_asscoia_tensor = np.zeros((1,record_N * (cell_size * cell_size+1)), dtype=np.float32)
    # h_state_2 = np.zeros((1,GRU_SIZE), dtype=np.float32)
    
    img_dir = os.path.join(set_dir, "img1")
    detfile = os.path.join(set_dir, "det/det.txt")
    
    img_files = {
            int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)}
    
    det_array = np.loadtxt(detfile, delimiter=',')
    
    frame_indices = det_array[:, 0].astype(np.int)
    
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    
    temp_img = cv2.imread(img_files[min_frame_idx])
    det_array[:, 2] /= temp_img.shape[1]
    det_array[:, 3] /= temp_img.shape[0]
    det_array[:, 4] /= temp_img.shape[1]
    det_array[:, 5] /= temp_img.shape[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    img_zero = cv2.imread(img_files[1])
    out = cv2.VideoWriter('output_'+ set_name +'.avi', fourcc, 20.0, (img_zero.shape[1],img_zero.shape[0]))
    
    video_record = []
    
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        
        img = cv2.imread(img_files[frame_idx])
        resized_img = cv2.resize(img, (img_size, img_size))
        
        concat_img, frame_det = encode_det(frame_idx, det_array, resized_img)
        
        detect_timer.tic()
        #print(h_state)
        result_img, prev_asscoia_tensor, h_state_1, record = process_image(frame_idx, img, concat_img, frame_det, prev_asscoia_tensor, h_state_1)
        video_record += record
        # h_state_1 = np.full((1,4096), 100, dtype=np.float32)
        # h_state_2 = np.full((1,4096), 100, dtype=np.float32)
        # result_img = draw_frame(img, confidence, boxes, trackid)
        
        #det_mask = concat_img[0][...,3].copy()
    
        #cv2.imshow("det_mask",det_mask)
        cv2.imshow("result",result_img)
        # sys.exit(0)
        cv2.waitKey(0)
    
        out.write(result_img)
        
        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        #visualization.display_video(frame, rclasses, rscores, rbboxes)
        
        detect_timer.toc()
        #print('detecting time: {:.3f}s'.format(detect_timer.diff))
    #os.system("pause")
    print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))
    print('Total time: {:.3f}s'.format(detect_timer.total_time))
    
    out.release()
    cv2.destroyAllWindows()
    
    f = open( set_name+".txt" , 'w')
    
    for row in video_record:
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()
    
	
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video', default="", type=str)
	
    # args = parser.parse_args()

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from video file
    # tf_track()
    
    # for set in train_set:
        # track_record = np.zeros((cell_size,cell_size),dtype=np.float32)
        # det_track(os.path.join(train_dir, set), set)
        
    for set in test_set:
        track_record = np.zeros((cell_size,cell_size),dtype=np.float32)
        det_track(os.path.join(val_dir, set), set)


if __name__ == '__main__':
    main()




