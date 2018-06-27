import os
import sys
import config
import numpy as np
import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS

class Reader:
    def __init__(self, data_pattern):
        self.zip_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        self.data_pattern = data_pattern
        self.files = np.array(tf.gfile.Glob(self.data_pattern))
        self.init_dataset()

    def get_random_example(self):
        frame_det_batch, frame_gt_batch, frame_img_batch = self.read_tfrecord(np.random.choice(self.files))
        return frame_det_batch, frame_gt_batch, frame_img_batch

    def read_tfrecord(self, path):
        frame_det_batch = []
        frame_gt_batch = []
        frame_img_batch = []
        
        index = 0
        for string_record in tf.python_io.tf_record_iterator(path=path):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            frame_det, frame_gt, frame_img = self.feature_decode(example)
            frame_det_batch.append(frame_det)
            frame_gt_batch.append(frame_gt)
            frame_img_batch.append(frame_img)
            
            index += 1
        

        FLAGS.batch_size = index
        #det_x = self.gen_det_x(np.asarray(frame_det_batch)) #reshape the detection input to [batch_size][x,...,y...,w...,h...] batch_size * 4 * 64 tensor
        
        return frame_det_batch, frame_gt_batch, frame_img_batch
        
    def feature_decode(self,example):
        
        #decode detection
        frame_det = example.features.feature['frame_det'].float_list.value

        #decode tracking_gt
        frame_gt = example.features.feature['frame_gt'].float_list.value
        
        #decode img
        frame_img_shape = example.features.feature['frame_img_shape'].int64_list.value
        frame_img_string = example.features.feature['frame_img'].bytes_list.value[0]
        frame_img = np.fromstring(frame_img_string, dtype=np.float32).reshape(frame_img_shape)

        #cv2.imshow("Image", frame_img)
        #cv2.waitKey(0)
        
        return frame_det, frame_gt, frame_img
        
    def parse_tfr_filename(self, path):
        filename, ext = os.path.splitext(path)
        path, file = os.path.split(filename)
        return "{}/{}.tfr".format(path, file), np.array([int(file.split('_')[-1])])

    def normalize_images(self, data) -> np.ndarray:
        return data / 255

    def init_dataset(self):
        np.random.shuffle(self.files)
        self.iterator = np.nditer(self.files)
