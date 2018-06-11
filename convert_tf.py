
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
        return None, None
    
    #print("do %d" % frame_idx)
    frame_indices = det_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    rows = det_array[mask]
    rows_shape = rows.shape
    
    rows_shape = np.array(rows_shape)
    rows_shape = rows_shape.flatten()
    rows_shape = rows_shape.tolist()
    
    #print("rows_shape: " + str(rows_shape))
    
    frame_det_shape = int64_feature(rows_shape)

    rows = rows.astype(np.uint8)
    rows = rows.tostring()
    frame_det = bytes_feature(rows)
    
    return frame_det_shape, frame_det
   

def get_frame_gt(frame_idx, gt_array):

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]

    if not len(rows):
        rows = np.array([[frame_idx,0,0,0,0,0]])
        #print("rows: " + str(rows))
        
    else:
        rows = np.delete(rows,np.s_[:4],1)
    
    rows_shape = rows.shape
    
    rows_shape = np.array(rows_shape)
    rows_shape = rows_shape.flatten()
    rows_shape = rows_shape.tolist()
    
    #if frame_idx == 1:
    #    print("rows_shape: " + str(rows_shape))
    
    frame_gt_shape = int64_feature(rows_shape)

    rows = rows.astype(np.uint8)
    rows = rows.tostring()
    frame_gt = bytes_feature(rows)
    
    return frame_gt_shape, frame_gt
       

def get_frame_img(frame_idx, img_files):

    frame_raw = tf.gfile.FastGFile(img_files[frame_idx], 'rb').read()
    img_mat = cv2.imread(img_files[frame_idx])
    
    frame_img = bytes_feature(frame_raw)
    
    shape = img_mat.shape
    
    shape = np.array(shape)
    shape = shape.flatten()
    shape = shape.tolist()
    
    frame_img_shape = int64_feature(shape)
    
    return frame_img_shape, frame_img


def convert_to_example(frame_det_shape, frame_det, frame_gt_shape, frame_gt, frame_img_shape, frame_img):

    example = tf.train.Example(features=tf.train.Features(feature={
            'frame_det_shape': frame_det_shape,
            'frame_gt_shape': frame_gt_shape,
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
    
    for mot_dir_name in mot_dirs:
        tf_filename = '%s/%s_train.tfrecord' % (output_dir,mot_dir_name)
        mot_dir = os.path.join(path,mot_dir_name)
        det_dir = os.path.join(mot_dir,"det")
        det_file = os.path.join(det_dir,"det_2.npy")
        gt_dir = os.path.join(mot_dir,"gt")
        gt_file = os.path.join(gt_dir,"hypotheses.txt")
        img_dir = os.path.join(mot_dir,"img1")
        img_files = {
            int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)}
            
        det_array = np.load(det_file)
        
        frame_indices = det_array[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        
        gt_array = np.loadtxt(gt_file, delimiter=',')

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            
                frame_det_shape, frame_det = get_frame_det(frame_idx, det_array, img_files)
                if frame_det_shape is None:
                    continue
                    
                frame_gt_shape, frame_gt = get_frame_gt(frame_idx, gt_array)
                
                frame_img_shape, frame_img = get_frame_img(frame_idx, img_files)
                
                example = convert_to_example(frame_det_shape, frame_det, frame_gt_shape, frame_gt, frame_img_shape, frame_img)
                tfrecord_writer.write(example.SerializeToString())
                    


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