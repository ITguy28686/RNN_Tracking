# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import time
import random
import sys

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

batch_size = 60*2

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    
    times = len(image_filenames)/batch_size
    if len(image_filenames)%batch_size > 0:
        times += 1
    times *= 5
    
    i = 0
    seq_info = []
    while i < times :
        
        start_num = random.randint(0,sys.maxsize) % (max_frame_idx - min_frame_idx + 1) + min_frame_idx
        end_num = start_num + batch_size
        
        if end_num > max_frame_idx:
            end_num = max_frame_idx
        
        # batch_max_frame_idx = min_frame_idx + (i+1)*batch_size - 1;
        # if batch_max_frame_idx > max_frame_idx:
            # batch_max_frame_idx = max_frame_idx
            
        frame_indices = detections[:, 0].astype(np.int)
        mask = (frame_indices >= start_num) & (frame_indices <= end_num)
    
        rows = detections[mask]   


        batch_image_filenames = dict((k, v) for k, v in image_filenames.items() if (k >= start_num) & (k <= end_num))
        
        
        #print(image_filenames[batch_size*i:batch_size*(i+1)])
        
        temp_seq_info = {
            "sequence_name": os.path.basename(sequence_dir),
            "image_filenames": batch_image_filenames,
            "detections": rows,
            "groundtruth": groundtruth,
            "image_size": image_size,
            "min_frame_idx": start_num,
            "max_frame_idx": end_num,
            "feature_dim": feature_dim,
            "update_ms": update_ms
        }
        
        seq_info.append(temp_seq_info)
        i += 1
        
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
	
    tStart = time.time()	

    batch_seq_info = gather_sequence_info(sequence_dir, detection_file)
    
    i = 0
    
    for seq_info in batch_seq_info:
        
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        results = []
        
        #dectection file through nmp and conf filter
        det_2 = []

        def frame_callback(vis, frame_idx):
            #print("Processing frame %05d" % frame_idx)

            # Load image and generate detections.
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
            detections = [d for d in detections if d.confidence >= min_confidence]
            

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            img_w = seq_info["image_size"][1]
            img_h = seq_info["image_size"][0]
            #print("img_w: " + str(img_w))
            
            #store det annotation
            for d in detections:
                det_temp = (frame_idx)
                det_temp = np.r_[det_temp,[d.tlwh[0]/img_w,d.tlwh[1]/img_h,d.tlwh[2]/img_w,d.tlwh[3]/img_h]]
                #det_temp = np.r_[det_temp,d.confidence]
                det_2.append(det_temp)
                #print("det: " + str(np.asarray(det_2).shape))
            
            
            # Update tracker.
            tracker.predict()
            
            #print("    detections: " + str([d.confidence for d in detections]))
            
            tracker.update(detections)

            # Update visualization.
            if display:
                image = cv2.imread(
                    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
                vis.set_image(image.copy())
                #vis.draw_detections(detections)
                vis.draw_trackers(tracker.tracks)

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                
                if ((frame_idx - seq_info["min_frame_idx"])%2 == 0):
                    results.append([
                        frame_idx, track.track_id, bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h])

        # Run tracker.
        if display:
            visualizer = visualization.Visualization(seq_info, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(seq_info)
        visualizer.run(frame_callback)
        
        train_data_dir = os.path.join(sequence_dir, "train_data")
        
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
            
        # train_det = os.path.join(train_data_dir, "det2_" + str(i))
        
        #save det_2
        #print("det_2 shape: " + str(np.asarray(det_2).shape))
        # np.save(train_det, np.asarray(det_2), allow_pickle=False)
        
        train_gt_hypotheses = os.path.join(train_data_dir, output_file + "_" + str(i))

        # Store results.
        f = open(train_gt_hypotheses, 'w')
        for row in results:
            print('%d,%d,%.4f,%.4f,%.4f,%.4f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
                
        i += 1
        
        print("min: " + str(seq_info["min_frame_idx"]))
        print("max: " + str(seq_info["max_frame_idx"]))
        print("----------------------------------------")
    
    # Time Counting
    tEnd = time.time()
	
    
	
    FPS = (seq_info["max_frame_idx"])/(tEnd - tStart)
    print("FPS = " + str(FPS))

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="./hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
