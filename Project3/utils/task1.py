import numpy as np
import math
from shapely.geometry import Polygon

def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding boxes with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    N = label.shape[0]
    bboxes_corners_bbox_coordinates = np.ones((N, 8, 4)) # 8 corners of each bbox with size 4 for homogeneous coordinates
    bboxes_corners_cam = []
    for i, bbox_info in enumerate(label):
        bbox_x_center = bbox_info[0] 
        bbox_y_bottom = bbox_info[1]
        bbox_z_center = bbox_info[2]

        bbox_height = bbox_info[3]
        bbox_width = bbox_info[4]
        bbox_length = bbox_info[5]

        bbox_center = np.array([bbox_x_center, bbox_y_bottom-bbox_height/2, bbox_z_center])
        bbox_y_rotation = bbox_info[6]

        #Compute every corners in the bbox coordinate system
        bboxes_corners_bbox_coordinates[i][0][0] = bbox_length/2
        bboxes_corners_bbox_coordinates[i][0][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][0][2] = bbox_width/2

        bboxes_corners_bbox_coordinates[i][1][0] = - bbox_length/2
        bboxes_corners_bbox_coordinates[i][1][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][1][2] = bbox_width/2

        bboxes_corners_bbox_coordinates[i][2][0] = - bbox_length/2
        bboxes_corners_bbox_coordinates[i][2][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][2][2] = - bbox_width/2

        bboxes_corners_bbox_coordinates[i][3][0] = bbox_length/2
        bboxes_corners_bbox_coordinates[i][3][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][3][2] = - bbox_width/2

        bboxes_corners_bbox_coordinates[i][4][0] = bbox_length/2
        bboxes_corners_bbox_coordinates[i][4][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][4][2] = bbox_width/2

        bboxes_corners_bbox_coordinates[i][5][0] = - bbox_length/2
        bboxes_corners_bbox_coordinates[i][5][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][5][2] = bbox_width/2

        bboxes_corners_bbox_coordinates[i][6][0] = - bbox_length/2
        bboxes_corners_bbox_coordinates[i][6][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][6][2] = - bbox_width/2

        bboxes_corners_bbox_coordinates[i][7][0] = bbox_length/2
        bboxes_corners_bbox_coordinates[i][7][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][7][2] = - bbox_width/2
        
        Ry = np.array([[math.cos(bbox_y_rotation),  0,  math.sin(bbox_y_rotation)],
                       [0,                          1,                          0],
                       [-math.sin(bbox_y_rotation), 0, math.cos(bbox_y_rotation)]])
        
        t = bbox_center # We don't apply the rotation to it since its based on its center
        bbox_to_cam0 = np.vstack((np.hstack((Ry, t.reshape(-1, 1))), [0, 0, 0, 1]))

        bbox_corners_cam0 = (bbox_to_cam0@bboxes_corners_bbox_coordinates[i].T).T[:,:3]
        bboxes_corners_cam.append(bbox_corners_cam0)
    
    return np.array(bboxes_corners_cam)


def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding boxes with (x,y,z,h,w,l,ry)
        target (M,7) 3D bounding boxes with (x,y,z,h,w,l,ry)
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''

    iou = np.zeros((pred.shape[0], target.shape[0])) # (N,M)
    pred_labels = label2corners(pred) # (N, 8, 3)
    target_labels = label2corners(target) # (M, 8, 3)

    for i in range(pred_labels.shape[0]):
        for j in range(target_labels.shape[0]):
            pred_corners = pred_labels[i]
            target_corners = target_labels[j]
            pred_bbox_info = pred[i]
            target_bbox_info = target[j]

            iou[i,j] = get_iou_pairwise(pred_corners, target_corners, pred_bbox_info, target_bbox_info)

    return iou

def get_iou_pairwise(pred_corners, target_corners, pred_bbox_info, target_bbox_info):
    '''
    input
        pred_corners (8,3) corner coordinates in the rectified reference frame
        target_corners (8,3) corner coordinates in the rectified reference frame
        pred_bbox_info 3D bounding box with (x,y,z,h,w,l,ry)
        target_bbox_info 3D bounding box with (x,y,z,h,w,l,ry)
    output
        iou - float
    '''

    pred_poly = Polygon(pred_corners[:4, [0,2]])
    target_poly = Polygon(target_corners[:4, [0,2]])

    # check if the two boxes don't overlap
    if not pred_poly.intersects(target_poly):
        return 0.0

    inter_area = pred_poly.intersection(target_poly).area

    # iou = 0 if bbox is on top of the other with no overlap
    if pred_corners[7, 1] <= target_corners[0,1] or target_corners[7,1] <= pred_corners[0,1]:
        return 0.0

    min_intersection_y = np.maximum(pred_corners[0, 1], target_corners[0,1])
    max_intersection_y = np.minimum(pred_corners[7, 1], target_corners[7, 1])

    inter_volume = inter_area * (max_intersection_y - min_intersection_y)

    pred_volume = pred_bbox_info[3] * pred_bbox_info[4] * pred_bbox_info[5]
    target_volume = target_bbox_info[3] * target_bbox_info[4] * target_bbox_info[5]

    union_volume = (pred_volume + target_volume - inter_volume)

    iou = inter_volume / union_volume

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0.0

    return iou


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''

    iou = get_iou(pred, target) # N,M
    tp = len(iou[iou>=threshold])
    fn = np.sum(np.all(iou<=threshold, axis=0))
    if tp+fn == 0:
        return 0
    recall = tp/(tp + fn)
    return recall
