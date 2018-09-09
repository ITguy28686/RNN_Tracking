
import os
import sys
import argparse
import cv2

import tensorflow as tf
import numpy as np

from utils.dataset_utils import int64_feature, float_feature, bytes_feature
from utils.recorder import Recorder


# MOT_DIR = "D:/DataSet/2DMOT2015"
MOT_DIR = "D:/DataSet/MOT16"
img_size = 360
cell_size = 9

# track_record = np.zeros((cell_size,cell_size,5),dtype=np.float32) #(top_left_x, top_left_y, _w, _h, track_id)
track_record = []

def get_frame_gt(frame_idx, gt_array, last_trackid):

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]

    gt_tensor = np.zeros((cell_size,cell_size,5+cell_size*cell_size+1),dtype=np.float32)

    for i in range(rows.shape[0]):
        gt_tensor, last_trackid = encode_label(rows[i], gt_tensor, last_trackid)
        # gt_tensor[i] = rows[i][1]       #track_id
        # gt_tensor[i+64] = 1             #conf
        # gt_tensor[i+64*2] = rows[i][2]  #x
        # gt_tensor[i+64*3] = rows[i][3]  #y
        # gt_tensor[i+64*4] = rows[i][4]  #w
        # gt_tensor[i+64*5] = rows[i][5]  #h

    frame_gt = float_feature(gt_tensor.flatten().tolist())
    frame_id = int64_feature(frame_idx)
    
    return frame_gt, frame_id, last_trackid, gt_tensor
    
def get_frame_imgmask(frame_idx, gt_array, img_files, gt_tensor):

    #get mask
    # frame_indices = gt_array[:, 0].astype(np.int)
    # mask = frame_indices == frame_idx
    # rows = gt_array[mask]
    
    mask_img = np.zeros((img_size,img_size,1), np.float32)

    for i in range(cell_size):
        for j in range(cell_size):
            mask_x_lt = int(img_size * gt_tensor[i][j][1])
            mask_y_lt = int(img_size * gt_tensor[i][j][2])
            mask_x_rd = int(img_size * (gt_tensor[i][j][1] + gt_tensor[i][j][3]))
            mask_y_rd = int(img_size * (gt_tensor[i][j][2] + gt_tensor[i][j][4]))
            
            # if(mask_x >= 300 or mask_y>= 300):
                # print('%f,%f  %f,%f' % (rows[i][1],rows[i][2],mask_x,mask_y))
            
            if(mask_x_lt < 0):
                mask_x_lt = 0
                      
            if(mask_y_lt < 0):
                mask_y_lt = 0
            
            if(mask_x_rd >= img_size):
                mask_x_rd = img_size - 1
            
            if(mask_y_rd >= img_size):
                mask_y_rd = img_size - 1
            
            for y in range(mask_y_lt,mask_y_rd):
                mask_img[y][mask_x_lt] = 1
                mask_img[y][mask_x_rd] = 1
                
            for x in range(mask_x_lt,mask_x_rd):
                mask_img[mask_y_lt][x] = 1
                mask_img[mask_y_rd][x] = 1  
    mask_img = mask_img.astype(np.float32)  
    
    # cv2.imshow("mask_img",mask_img)
    # cv2.waitKey(0)
   
    
    #get image
    frame_raw = tf.gfile.FastGFile(img_files[frame_idx], 'rb').read()

    # decode image to jpeg
    img_data = tf.image.decode_jpeg(frame_raw)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)  
    resized_img_data = tf.image.resize_images(img_data, (img_size, img_size), method=0)
    #normalized_img_data = tf.image.per_image_standardization(resized_img_data)
    
    img_mat = resized_img_data.eval()
    
    #concat img and mask
    img_mat_concat = np.concatenate((img_mat,mask_img),axis=2)
    # print(img_mat_concat[0][0])
    # sys.exit(0)
    
    resized_raw = img_mat_concat.tobytes()
    frame_concat_mat = bytes_feature(resized_raw)
    
    shape = img_mat_concat.shape
    
    shape = np.array(shape)
    shape = shape.flatten()
    shape = shape.tolist()
    
    frame_concate_mat_shape = int64_feature(shape)
    
    return frame_concate_mat_shape, frame_concat_mat

def find_id_inrecord(track_id):
    global track_record
    
    index = 0
    for rec in track_record:
        if rec.track_id == track_id:
            return True, index
        index += 1
        
    return False, index

    
def find_prev_index_inrecord(track_id, y_ind, x_ind):
    global track_record
    
    for i in range(cell_size):
        for j in range(cell_size):
            if track_record[i][j] == track_id :
                track_record[i][j] = 0
                track_record[y_ind][x_ind] = track_id
                return True,i,j
    
    track_record[y_ind][x_ind] = track_id
    return False,None,None

    
def insert_track_mask(arr, x, y, w, h):
    x_lt = int(x * cell_size)
    y_lt = int(y * cell_size)
    x_rd = int((x+w) * cell_size)
    y_rd = int((y+h) * cell_size)

    for i in range(y_lt, y_rd+1):
        for j in range(x_lt, x_rd+1):
            arr[i][j] = True
            
            
def encode_label(row,gt_tensor,last_trackid):

    global track_record
    
    if(row[2] < 0):
        row[4] += row[2]
        row[2] = 0

    if(row[3] < 0):
        row[5] += row[3]
        row[3] = 0
        
    if (row[2] + row[4] >= 1):
        row[4] = 0.999 - row[2]
        
    if (row[3] + row[5] >= 1):
        row[5] = 0.999 - row[3]

    #boxes = [row[2] + row[4] / 2.0, row[3] + row[5] / 2.0, row[4], row[5]]
    boxes = row[2:6]

    track_id = int(row[1])
    
    x_ind = int(boxes[0] * cell_size)
    y_ind = int(boxes[1] * cell_size)
    
    if gt_tensor[y_ind, x_ind, 0] == 1:
        return gt_tensor, last_trackid
        
    gt_tensor[y_ind, x_ind, 0] = 1
    gt_tensor[y_ind, x_ind, 1:5] = boxes
    
    # match_prev_cell = gt_tensor[y_ind, x_ind, 5:5+cell_size*cell_size].reshape(cell_size,cell_size)
    # is_find,i,j = find_prev_index_inrecord(track_id, y_ind, x_ind)
    is_find, index = find_id_inrecord(track_id)
    
    # if is_find == False:
        # gt_tensor[y_ind, x_ind, 5+cell_size*cell_size] = 1
    
    # else:
        # match_prev_cell[i][j] = 1
    
    if is_find == False:
        new_record = Recorder(cell_size,track_id)
        insert_track_mask(new_record.mask, boxes[0], boxes[1], boxes[2], boxes[3])
        track_record += [new_record]
        gt_tensor[y_ind, x_ind, 5+cell_size*cell_size] = 1
    
    else:
        # print("(%d, %d)" % (track_id, track_record[index].track_id))
        gt_tensor[y_ind, x_ind, 5:5+cell_size*cell_size] = track_record[index].mask.flatten().astype(np.float32)
        # print(track_record[index].mask)
        track_record[index].mask = np.zeros((9,9), dtype=np.bool)
        insert_track_mask(track_record[index].mask, boxes[0], boxes[1], boxes[2], boxes[3])
    
    if(track_id > last_trackid):
        last_trackid += 1
        #print(last_trackid)

    return gt_tensor, last_trackid

def convert_to_example(frame_id, frame_gt, frame_concate_mat_shape, frame_concat_mat):

    example = tf.train.Example(features=tf.train.Features(feature={
            'frame_id': frame_id,
            'frame_concate_mat_shape': frame_concate_mat_shape,
            'frame_gt': frame_gt,
            'frame_concat_mat': frame_concat_mat,
            }))
            
    return example

    
def run(output_dir):
    """Runs the conversion operation.

      output_dir: Output directory.
    """
    
    
    
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    path = os.path.join(MOT_DIR, "train")    
    mot_dirs = os.listdir(path)
    #print(mot_dirs)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    for mot_dir_name in mot_dirs:
        
        mot_dir = os.path.join(path,mot_dir_name)
        img_dir = os.path.join(mot_dir,"img1")
        img_files = {
            int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)}
            
        train_data_dir = os.path.join(mot_dir,"train_data")
        
        index = 0
        while True:
            with tf.Graph().as_default() as graph:
                with tf.Session(config=config) as session:
        
                    det_file = os.path.join(train_data_dir,"det2_"+ str(index)+".npy")
                    gt_file = os.path.join(train_data_dir,"hypotheses.txt_"+ str(index))
                    
                    if not os.path.exists(gt_file):
                        break

                    tf_filename = '%s/%s-%d_train.tfrecord' % (output_dir,mot_dir_name,index)
                    print("Processing %s..." % tf_filename)
                    #det_array = np.load(det_file)
                    
                    gt_array = np.loadtxt(gt_file, delimiter=',')
                    
                    if not len(gt_array):
                        index += 1
                        continue
                    
                    frame_indices = gt_array[:, 0].astype(np.int)
                    min_frame_idx = frame_indices.astype(np.int).min()
                    max_frame_idx = frame_indices.astype(np.int).max()
                    
                    last_trackid = 0
                    global track_record
                    # track_record = np.zeros((cell_size,cell_size),dtype=np.float32)
                    track_record = []
                    
                    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                        for frame_idx in range(min_frame_idx+2, max_frame_idx + 1):

                            if frame_idx not in img_files:
                                continue
                            
                            #frame_det = get_frame_det(frame_idx, det_array, img_files)
                            frame_gt, frame_id, last_trackid, gt_tensor = get_frame_gt(frame_idx, gt_array, last_trackid)
                            
                            frame_concate_mat_shape, frame_concat_mat = get_frame_imgmask(frame_idx, gt_array, img_files, gt_tensor)
                            
                            example = convert_to_example(frame_id, frame_gt, frame_concate_mat_shape, frame_concat_mat)
                            tfrecord_writer.write(example.SerializeToString())
                        
                    if os.stat(tf_filename).st_size == 0 :
                        os.remove(tf_filename)
                                   
                    index += 1
                    print("------\n")
            
    print("Converting Finish")            


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")

    parser.add_argument(
        "--img_dir", required=True)
    parser.add_argument(
        "--det_file", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    run(output_dir = "train_tf")


if __name__ == "__main__":
    main()