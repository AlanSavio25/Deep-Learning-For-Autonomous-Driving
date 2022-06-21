import numpy as np
import math

def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
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
        pred (N,7) 3D bounding box corners
        target (N,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    pass

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
    pass