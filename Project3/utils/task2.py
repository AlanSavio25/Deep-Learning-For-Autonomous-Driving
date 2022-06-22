import numpy as np
import random
import math

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
        valid_pred (K',7)
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
    
    valid_pred = []
    pooled_xyz = []
    pooled_feat = []
    
    for box in enlarged_pred:
        are_in_box = points_in_box(xyz, box)
        indexes_in_box = np.where(are_in_box)[0]
        
        if  indexes_in_box.shape[0] > 0:
            valid_pred.append(box)
            
            indices = indexes_in_box

            if  indexes_in_box.shape[0]<max_points:
                indices = np.random.choice(indexes_in_box, size=max_points, replace=True)
            elif indexes_in_box.shape[0]>max_points:
                indices = np.random.choice(indexes_in_box, size=max_points, replace=False)

            pooled_xyz.append(xyz[indices])
            pooled_feat.append(feat[indices])

    return np.array(valid_pred), np.array(pooled_xyz), np.array(pooled_feat)

def points_in_box(xyz, box):
    '''
    input
        xyz (N,3) point coordinates in rectified reference frame
        box (7,) 3d bounding box label (x,y,z,h,w,l,ry)
    output
        flag (bool) true if point is in bounding box
    '''
    # We transform the points in the bounding box coordinate system so that the axis are aligned and we can simply check values along each axis
    # Should maybe inverse the sign
    Ry = np.array([[math.cos(box[6]),  0,  -math.sin(box[6])],
                   [0,                 1,                  0],
                   [math.sin(box[6]),  0,   math.cos(box[6])]])
    # We first shift from - the center and then use the inverse rotation
    t = Ry@np.array([-box[0], -(box[1]-box[3]/2), -box[2]])
    cam0_to_bbox = np.vstack((np.hstack((Ry, t.reshape(-1, 1))), [0, 0, 0, 1]))
    xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz_bbox = (cam0_to_bbox@xyz_homogeneous.T).T[:, :3]

    min_x = - box[5]/2
    max_x = box[5]/2
    min_y = -box[3]/2
    max_y = box[3]/2
    min_z = - box[4]/2
    max_z = box[4]/2

    return np.logical_and(xyz_bbox[:, 0] >= min_x, xyz_bbox[:, 0] <= max_x) & np.logical_and(xyz_bbox[:, 1] >= min_y, xyz_bbox[:, 1] <= max_y) &  np.logical_and(xyz_bbox[:, 2] >= min_z, xyz_bbox[:, 2] <= max_z)
    

# def points_in_boxes(xyz, boxes, max_points):
#     '''
#     input
#         xyz (N,3) points in rectified reference frame
#         boxes (K,7) 3d bounding box labels (x,y,z,h,w,l,ry)
#     output
#         valid_indices (K',M) indices of points that are in each k' bounding box
#         valid (K', 7) index vector showing valid bounding boxes, i.e. with at least
#                    one point within the box
#     '''
   

def enlarge_boxes(pred, delta):
    """
    input
        pred (N,7) bounding box labels
        delta float margin we want to add in each direction
    output
        enlarged_pred (N, 7) boxes enlarged by delta in each direction. The center is corrected as well
    """
    enlarged_pred = pred
    enlarged_pred[:, 3:6]=enlarged_pred[:, 3:6] + 2*delta
    enlarged_pred[:, 1]=enlarged_pred[:, 1] + delta # Correct the center of the bottom face
    
    return enlarged_pred

# pred (N,7) bounding box labels
# xyz (N,3) point cloud
# feat (N,C) features
# config (dict) data config
# pred = np.array([[1,2,3,1,1,5,math.pi/2], [1,20,3,1,1,5,math.pi], [1,2.5,3,1,1,5,math.pi]])
# xyz = np.array([[0,0,0], [1,2,3], [1,1.5,6]])
# feat = np.array([[0,0,0,0,0], [1,2,3,4,4], [1,1.5,2,5,5]])
# config = {"max_points":2, "delta":1}
# a,b,c = roi_pool(pred, xyz, feat, config)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(b)