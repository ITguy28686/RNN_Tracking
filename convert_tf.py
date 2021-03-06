
import os
import sys
import argparse
import cv2
import collections

import tensorflow as tf
import numpy as np

from utils.dataset_utils import int64_feature, float_feature, bytes_feature
from utils.recorder import Recorder


# MOT_DIR = "D:/DataSet/2DMOT2015"
MOT_DIR = "D:/DataSet/MOT16"
img_size = 360
record_N = 256
cell_size = 9

# track_record = np.zeros((cell_size,cell_size,5),dtype=np.float32) #(top_left_x, top_left_y, _w, _h, track_id)
track_record = []

#row = (frame_index, track_id, x, y, w, h)
def get_associa_gt(frame_idx, gt_array, alive_dict, tackid_grid_vector):

    global track_record

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]

    ass_matrix = np.zeros((record_N+1,cell_size*cell_size+1),dtype=np.float32)
    ass_matrix[:, -1] = 1
    ass_matrix[record_N, :] = 1
    
    e_vector = np.zeros((record_N),dtype=np.float32)

    for i in range(rows.shape[0]):
        id = int(rows[i][1])
        if id not in track_record:
            track_record += [id]
            # ass_matrix[len(track_record)-1][tackid_grid_vector[id][0]*cell_size+tackid_grid_vector[id][1]] = 1
            # e_vector[len(track_record)-1] = 1
            
    # print(track_record)        
    for i in range(len(track_record)):
        # print(alive_dict[track_record[i]])
        if( track_record[i] != 0 and alive_dict[track_record[i]][1] < frame_idx):
            track_record[i] = 0
            
        if track_record[i] != 0 and track_record[i] in tackid_grid_vector.keys():
            id = track_record[i]
            true_id = alive_dict[track_record[i]][2]
            ass_matrix[true_id][-1] = 0
            # print(tackid_grid_vector[id][0], tackid_grid_vector[id][1])
            ass_matrix[true_id][tackid_grid_vector[id][0]*cell_size + tackid_grid_vector[id][1]] = 1
            ass_matrix[record_N][tackid_grid_vector[id][0]*cell_size + tackid_grid_vector[id][1]] = 0

            # print("track: " + str(i))
    
    for key, value in alive_dict.items():
        if(value[0] <= frame_idx and value[1] >= frame_idx):
            e_vector[value[2]] = 1
    
    
    # print(ass_matrix)
    # print(e_vector)
    # print('---------------\n')
    
    np.savetxt('numpy_out.txt', ass_matrix.astype(int), delimiter=',', fmt='%i')
    sys.exit(0)
    
    ass_matrix_gt = float_feature(ass_matrix.flatten().tolist())
    e_vector_gt = float_feature(e_vector.flatten().tolist())
    
    return ass_matrix_gt, e_vector_gt

def get_frame_gt(frame_idx, gt_array, last_trackid):

    frame_indices = gt_array[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    
    rows = gt_array[mask]
    if rows.size == 0:
        return None, None, last_trackid, None, None

    gt_tensor = np.zeros((cell_size,cell_size,5),dtype=np.float32)
    
    tackid_grid_vector = collections.defaultdict(list)
    
    for i in range(rows.shape[0]):
        last_trackid, tackid_grid = encode_label(rows[i], gt_tensor, last_trackid)
        
        if tackid_grid is not None:
            # print(tackid_grid)
            tackid_grid_vector[int(rows[i][1])].append(tackid_grid[0])
            tackid_grid_vector[int(rows[i][1])].append(tackid_grid[1])
        # gt_tensor[i] = rows[i][1]       #track_id
        # gt_tensor[i+64] = 1             #conf
        # gt_tensor[i+64*2] = rows[i][2]  #x
        # gt_tensor[i+64*3] = rows[i][3]  #y
        # gt_tensor[i+64*4] = rows[i][4]  #w
        # gt_tensor[i+64*5] = rows[i][5]  #h

    frame_gt = float_feature(gt_tensor.flatten().tolist())
    frame_id = int64_feature(frame_idx)
    
    # print(tackid_grid_vector)
    
    return frame_gt, frame_id, last_trackid, gt_tensor, tackid_grid_vector
    
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

    
# def find_prev_index_inrecord(track_id, y_ind, x_ind):
    # global track_record
    
    # for i in range(cell_size):
        # for j in range(cell_size):
            # if track_record[i][j] == track_id :
                # track_record[i][j] = 0
                # track_record[y_ind][x_ind] = track_id
                # return True,i,j
    
    # track_record[y_ind][x_ind] = track_id
    # return False,None,None

    
def insert_track_mask(arr, x, y, w, h):
    x_lt = int(x * cell_size)
    y_lt = int(y * cell_size)
    x_rd = int((x+w) * cell_size) + 1
    y_rd = int((y+h) * cell_size) + 1

    for i in range(y_lt, y_rd):
        for j in range(x_lt, x_rd):
            arr[i][j] = True
            
            
def encode_label(row, gt_tensor, last_trackid):

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
        return last_trackid, None
    
    gt_tensor[y_ind, x_ind, 0] = 1
    gt_tensor[y_ind, x_ind, 1:5] = boxes
    
    tackid_grid = (y_ind, x_ind)
    
    # match_prev_cell = gt_tensor[y_ind, x_ind, 5:5+cell_size*cell_size].reshape(cell_size,cell_size)
    # is_find,i,j = find_prev_index_inrecord(track_id, y_ind, x_ind)
    # is_find, index = find_id_inrecord(track_id)
    
    # if is_find == False:
        # gt_tensor[y_ind, x_ind, 5+cell_size*cell_size] = 1
    
    # else:
        # match_prev_cell[i][j] = 1
    
    # if is_find == False:
        # new_record = Recorder(cell_size,track_id)
        # insert_track_mask(new_record.mask, boxes[0], boxes[1], boxes[2], boxes[3])
        # track_record += [new_record]
        # gt_tensor[y_ind, x_ind, 5+cell_size*cell_size] = 1
    
    # else:
        # gt_tensor[y_ind, x_ind, 5:5+cell_size*cell_size] = track_record[index].mask.flatten().astype(np.float32)

        # track_record[index].mask = np.zeros((9,9), dtype=np.bool)
        # insert_track_mask(track_record[index].mask, boxes[0], boxes[1], boxes[2], boxes[3])
    
    if(track_id > last_trackid):
        last_trackid += 1
        #print(last_trackid)

    return last_trackid, tackid_grid

def convert_to_example(frame_id, frame_gt, frame_concate_mat_shape, frame_concat_mat, ass_matrix_gt, e_vector_gt):

    example = tf.train.Example(features=tf.train.Features(feature={
            'frame_id': frame_id,
            'frame_concate_mat_shape': frame_concate_mat_shape,
            'frame_gt': frame_gt,
            'frame_concat_mat': frame_concat_mat,
            'ass_matrix_gt': ass_matrix_gt,
            'e_vector_gt': e_vector_gt
            }))
            
    return example

def gen_alive_matrix(gt_array, min_frame_idx, max_frame_idx):
    
    # key: track id, value[0], value[1]: existing in what frame range, value[2]: true trackid
    alive_dict = collections.defaultdict(list)
    
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
    
        frame_indices = gt_array[:, 0].astype(np.int)
        mask = frame_indices == frame_idx
        
        rows = gt_array[mask]
        
        for row in rows:
            trackid = int(row[1])
            
            if trackid not in alive_dict.keys():
                alive_dict[trackid].append(frame_idx)
                alive_dict[trackid].append(frame_idx)
            
            else:
                alive_dict[trackid][1] = frame_idx
    
    max_N = max(alive_dict.keys())
    
    empty_indexes = []
    for i in range(1,max_N+1):
        if(i not in alive_dict.keys()):
            empty_indexes += [i]
            
    # print(empty_indexes)
        
    for key, value in alive_dict.items():
        n = sum(i < key for i in empty_indexes)
        alive_dict[key].append(key-n-1)
        
    # print(alive_dict)
    
    return alive_dict
        
    
    
    
    
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
        
                    # det_file = os.path.join(train_data_dir,"det2_"+ str(index)+".npy")
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
                    
                    alive_dict = gen_alive_matrix(gt_array, min_frame_idx, max_frame_idx)
                    # print(alive_dict)
                    
                    
                    
                    last_trackid = 0
                    global track_record
                    # track_record = np.zeros((cell_size,cell_size),dtype=np.float32)
                    track_record = []
                    
                    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                        for frame_idx in range(min_frame_idx, max_frame_idx + 1):

                            if frame_idx not in img_files:
                                continue
                            
                            frame_gt, frame_id, last_trackid, gt_tensor, tackid_grid_vector = get_frame_gt(frame_idx, gt_array, last_trackid)
                            if(frame_gt == None):
                                continue
                                
                            ass_matrix_gt, e_vector_gt = get_associa_gt(frame_idx, gt_array, alive_dict, tackid_grid_vector)
                            
                            frame_concate_mat_shape, frame_concat_mat = get_frame_imgmask(frame_idx, gt_array, img_files, gt_tensor)
                            
                            example = convert_to_example(frame_id, frame_gt, frame_concate_mat_shape, frame_concat_mat, ass_matrix_gt, e_vector_gt)
                            tfrecord_writer.write(example.SerializeToString())
                        
                    if os.stat(tf_filename).st_size == 0 :
                        os.remove(tf_filename)
                                   
                    index += 1
                    print("------\n")
        # sys.exit(0)    
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