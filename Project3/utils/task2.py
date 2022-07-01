import numpy as np
from .task1 import label2corners
# from .vis import *


def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7) Careful, this must return the initial boxes, not the enlarged ones to pass the tests!
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    max_points = config['max_points']
    delta = config['delta']

    enlarged_pred = enlarge_boxes(pred, delta)
    boxes_corners = label2corners(enlarged_pred)
    
    valid_pred = []
    pooled_xyz = []
    pooled_feat = []
    
    for box_corners, box in zip(boxes_corners, pred):
        are_in_box = points_in_box(xyz, box_corners)
        indexes_in_box = np.where(are_in_box)[0]
        
        if  indexes_in_box.shape[0] > 0:
            valid_pred.append(box)
            
            indices = sample(indexes_in_box, max_points, method=config['sampling_method_roi'])

            xyz_samples = xyz[indices]
            feat_samples = feat[indices]

            pooled_xyz.append(xyz_samples)
            pooled_feat.append(feat_samples)
    
    # visualizer = Visualizer()
    # visualizer.update(np.array(pooled_xyz).reshape(-1, 3))
    # visualizer.update_boxes(label2corners(np.array(valid_pred)))
    # vispy.app.run()

    return np.array(valid_pred), np.array(pooled_xyz), np.array(pooled_feat)

def points_in_box(xyz, box_corners):
    '''
    input
        xyz (N,3) point coordinates in rectified reference frame
        box_corners (8, 3)
    output
        flag (bool) true if point is in bounding box
    '''
    # Corners 0 and 6 are the opposite, that's why we use them
    min_x = np.minimum(box_corners[0, 0], box_corners[6, 0])
    max_x = np.maximum(box_corners[0, 0], box_corners[6, 0])
    min_y = np.minimum(box_corners[0, 1], box_corners[6, 1])
    max_y = np.maximum(box_corners[0, 1], box_corners[6, 1])
    min_z = np.minimum(box_corners[0, 2], box_corners[6, 2])
    max_z = np.maximum(box_corners[0, 2], box_corners[6, 2])

    return (xyz[:, 0] >= min_x) & (xyz[:, 0] <= max_x) & (xyz[:, 1] >= min_y) & (xyz[:, 1] <= max_y) & (xyz[:, 2] >= min_z) & (xyz[:, 2] <= max_z)
   

def enlarge_boxes(pred, delta):
    """
    input
        pred (N,7) bounding box labels
        delta float margin we want to add in each direction
    output
        enlarged_pred (N, 7) boxes enlarged by delta in each direction. The center is corrected as well
    """
    enlarged_pred = pred.copy()
    enlarged_pred[:, 3:6]=enlarged_pred[:, 3:6] + 2*delta
    enlarged_pred[:, 1]=enlarged_pred[:, 1] + delta # Correct the center of the bottom face
    
    return enlarged_pred

def sample(indexes, required_size, method = "random"):
    '''
    input
        indexes, (L, ): index of 3d points being in box without duplicate
        required_size, int: number of points in the final sampled ROI (M)
        method, string: Method used to sample, must be either 'random' or 'uniform', default is 'random'
    output
        xyz_samples (M, 3) Sample of 3D points with possible duplicates
        feat_samples (M, C) Corresponding sample of features with possible duplicates
    '''
    indices = indexes

    if  indexes.shape[0]<required_size:
        if method == "random":
            indices = np.random.choice(indexes, size=required_size, replace=True)
        elif method == "uniform":
            number_tiles = int(required_size/indexes.shape[0]) if required_size%indexes.shape[0]==0 else int(required_size/indexes.shape[0])+1
            indices = np.tile(indexes, number_tiles)[:required_size]
        else:
            raise ValueError(f"Method '{method}' is not valid! Must be 'random' or 'uniform'.")
    elif indexes.shape[0]>required_size:
        indices = np.random.choice(indexes, size=required_size, replace=False)

    assert indices.shape[0] == required_size

    return indices