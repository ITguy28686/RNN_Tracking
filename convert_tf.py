
import os
import sys
import argparse
import cv2

import tensorflow as tf
import numpy as np

from utils.dataset_utils import int64_feature, float_feature, bytes_feature


MOT_DIR = "D:\DataSet\MOT16"
    
def get_frame_det(frame_idx, det_array, img_files):
    
    if frame_idx not in img_files:
        return None
    
    #print("do %d" % frame_idx)
    frame_indices = det_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    rows = det_array[mask]
    det_tensor = np.zeros((64*4),dtype=np.float32)
    
    #print(rows)
    for i in range(rows.shape[0]):
        if i >= 64:
            break
            
        det_tensor[i] = rows[i][1]       #x
        det_tensor[i+64] = rows[i][2]    #y
        det_tensor[i+64*2] = rows[i][3]  #w
        det_tensor[i+64*3] = rows[i][4]  #h
        
    #print("frame_det_shape" + str(det_tensor.shape))

    frame_det = float_feature(det_tensor.flatten().tolist())
    
    return frame_det
   

def get_frame_gt(frame_idx, gt_array):

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]

    if not len(rows):
        rows = np.array([[frame_idx,0,0,0,0,0]])
    
    gt_tensor = np.zeros((64*6),dtype=np.float32)
    
    for i in range(rows.shape[0]):
        if i >= 64:
            break
            
        gt_tensor[i] = rows[i][1]       #track_id
        gt_tensor[i+64] = 1             #conf
        gt_tensor[i+64*2] = rows[i][2]  #x
        gt_tensor[i+64*3] = rows[i][3]  #y
        gt_tensor[i+64*4] = rows[i][4]  #w
        gt_tensor[i+64*5] = rows[i][5]  #h

    frame_gt = float_feature(gt_tensor.flatten().tolist())
    
    return frame_gt
       

def get_frame_img(frame_idx, img_files):

    frame_raw = tf.gfile.FastGFile(img_files[frame_idx], 'rb').read()

    # decode image to jpeg
    img_data = tf.image.decode_jpeg(frame_raw)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)  
    resized_img_data = tf.image.resize_images(img_data, (300, 300), method=0)
    
    img_mat = resized_img_data.eval()
    
    resized_raw = img_mat.tobytes()
    frame_img = bytes_feature(resized_raw)
    
    shape = img_mat.shape
    
    shape = np.array(shape)
    shape = shape.flatten()
    shape = shape.tolist()
    
    frame_img_shape = int64_feature(shape)
    
    return frame_img_shape, frame_img


def convert_to_example(frame_det, frame_gt, frame_img_shape, frame_img):

    example = tf.train.Example(features=tf.train.Features(feature={
            'frame_img_shape': frame_img_shape,
            'frame_det': frame_det,
            'frame_gt': frame_gt,
            'frame_img': frame_img}))
            
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
                    
                    if not os.path.exists(det_file):
                        break
                    
                    tf_filename = '%s/%s-%d_train.tfrecord' % (output_dir,mot_dir_name,index)
                    print("Processing %s..." % tf_filename)
                    det_array = np.load(det_file)
                    
                    frame_indices = det_array[:, 0].astype(np.int)
                    min_frame_idx = frame_indices.astype(np.int).min()
                    max_frame_idx = frame_indices.astype(np.int).max()
                    
                    gt_array = np.loadtxt(gt_file, delimiter=',')

                    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
                        
                            frame_det = get_frame_det(frame_idx, det_array, img_files)
                            if frame_det is None:
                                continue
                                
                            frame_gt = get_frame_gt(frame_idx, gt_array)
                            
                            frame_img_shape, frame_img = get_frame_img(frame_idx, img_files)
                            
                            example = convert_to_example(frame_det, frame_gt, frame_img_shape, frame_img)
                            tfrecord_writer.write(example.SerializeToString())
                            
                                   
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