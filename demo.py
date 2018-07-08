import os
import math
import random

import numpy as np
import cv2
import tensorflow as tf


slim = tf.contrib.slim
import sys

from network.model import Model

from timer import Timer
import argparse


chkpt_file = "network/logs/model.ckpt"
test_tf = "train_tf/MOT16-02-0_train.tfrecord"

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)

det_input = tf.placeholder(dtype=tf.float32, shape=(1, self.cell_size*self.cell_size*4))
img_input = tf.placeholder(dtype=tf.float32, shape=(1,300, 300, 3))

graph = tf.Graph()
logits = self.mynet(self.det_x, self.img_x)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_fpn_vgg_300.SSDNet()
#ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './log/model.ckpt-10000'
#ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
#ckpt_filename = './checkpoints/VGG_VOC0712_SSD_300x300_iter_120000.ckpt'

# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)



# Main image processing routine.
def process_image(img, select_threshold=0.4, nms_threshold=0.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
	

def video_detector(video):
    detect_timer = Timer()
    cap = cv2.VideoCapture(video)
    ret, _ = cap.read()

    while ret:
        ret, frame = cap.read()
		
        if frame is None:
            break
		
        detect_timer.tic()
        rclasses, rscores, rbboxes =  process_image(frame)

        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        visualization.display_video(frame, rclasses, rscores, rbboxes)
		
        detect_timer.toc()
    print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))
        
	
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default="", type=str)
	
    args = parser.parse_args()

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from video file
    video_detector(args.video)


if __name__ == '__main__':
    main()




