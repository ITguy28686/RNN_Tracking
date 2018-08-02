
import os
import sys
import argparse
import cv2

import tensorflow as tf
import numpy as np

from utils.dataset_utils import int64_feature, float_feature, bytes_feature


MOT_DIR = "D:\DataSet\MOT16"
cell_size = 9
track_num = 30

def get_frame_gt(frame_idx, gt_array, last_trackid):

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]

    gt_tensor = np.zeros((cell_size,cell_size,5+track_num),dtype=np.float32)

    for i in range(rows.shape[0]):
        gt_tensor, last_trackid = encode_label(rows[i],gt_tensor, last_trackid)
        # gt_tensor[i] = rows[i][1]       #track_id
        # gt_tensor[i+64] = 1             #conf
        # gt_tensor[i+64*2] = rows[i][2]  #x
        # gt_tensor[i+64*3] = rows[i][3]  #y
        # gt_tensor[i+64*4] = rows[i][4]  #w
        # gt_tensor[i+64*5] = rows[i][5]  #h

    frame_gt = float_feature(gt_tensor.flatten().tolist())
    
    return frame_gt, last_trackid
    
def get_frame_imgmask(frame_idx, gt_array, img_files):

    #get mask
    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    rows = gt_array[mask]
    
    mask_img = np.zeros((300,300,1), np.float32)

    for i in range(rows.shape[0]):
        mask_x = int(300 * rows[i][2])
        mask_y = int(300 * rows[i][3])
        mask_w = int(300 * rows[i][4])
        mask_h = int(300 * rows[i][5])
        
        # if(mask_x >= 300 or mask_y>= 300):
            # print('%f,%f  %f,%f' % (rows[i][1],rows[i][2],mask_x,mask_y))
        
        if(mask_x < 0):
            mask_w += mask_x
            mask_x = 0
                  
        if(mask_y < 0):
            mask_h += mask_y
            mask_y = 0
        
        if(mask_x+mask_w >= 300):
            mask_w = 300 - mask_x - 1
        
        if(mask_y+mask_h >= 300):
            mask_h = 300 - mask_y - 1
        
        for y in range(mask_y,mask_y+mask_h):
            mask_img[y][mask_x] = 1
            mask_img[y][mask_x+mask_w] = 1
            
        for x in range(mask_x,mask_x+mask_w):
            mask_img[mask_y][x] = 1
            mask_img[mask_y+mask_h][x] = 1  
    mask_img = mask_img.astype(np.float32)  
    
    # cv2.imshow("mask_img",mask_img)
    # cv2.waitKey(0)
   
    
    #get image
    frame_raw = tf.gfile.FastGFile(img_files[frame_idx], 'rb').read()

    # decode image to jpeg
    img_data = tf.image.decode_jpeg(frame_raw)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)  
    resized_img_data = tf.image.resize_images(img_data, (300, 300), method=0)
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


def encode_label(row,gt_tensor,last_trackid):

    if(row[2] < 0):
        row[4] += row[2]
        row[2] = 0

    if(row[3] < 0):
        row[5] += row[3]
        row[3] = 0

    #boxes = [row[2] + row[4] / 2.0, row[3] + row[5] / 2.0, row[4], row[5]]
    boxes = row[2:6]

    track_id = int(row[1])
    
    x_ind = int(boxes[0] * cell_size)
    y_ind = int(boxes[1] * cell_size)
    
    if gt_tensor[y_ind, x_ind, 0] == 1:
        return gt_tensor, last_trackid
        
    gt_tensor[y_ind, x_ind, 0] = 1
    gt_tensor[y_ind, x_ind, 1:5] = boxes
    
    if(track_id > last_trackid):
        last_trackid += 1
        #print(last_trackid)
    
    gt_tensor[y_ind, x_ind, 5+track_id-1] = 1

    return gt_tensor, last_trackid

def convert_to_example(frame_gt, frame_concate_mat_shape, frame_concat_mat):

    example = tf.train.Example(features=tf.train.Features(feature={
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
                    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                        for frame_idx in range(min_frame_idx+2, max_frame_idx + 1):

                            if frame_idx not in img_files:
                                continue
                            
                            #frame_det = get_frame_det(frame_idx, det_array, img_files)
                            frame_gt, last_trackid = get_frame_gt(frame_idx, gt_array, last_trackid)
                            
                            frame_concate_mat_shape, frame_concat_mat = get_frame_imgmask(frame_idx, gt_array, img_files)
                            
                            example = convert_to_example(frame_gt, frame_concate_mat_shape, frame_concat_mat)
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