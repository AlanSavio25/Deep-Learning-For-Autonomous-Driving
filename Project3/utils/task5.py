import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    N = pred.shape[0]
    s_p = pred.copy()
    c_p = score.copy()

    pred_on_the_ground = pred.copy()
    pred_on_the_ground[:, [1, 3]] = np.tile([0, 1], (N, 1))
    iou = get_iou(pred_on_the_ground, pred_on_the_ground)

    s_f = []
    c_f = []

    while s_p.shape[0] != 0:
        i = np.argmax(c_p)
        d_i = s_p[i, :]
        s_p = np.delete(s_p, i, axis=0)
        s_f.append([d_i])
        c_f.append(c_p[i])
        c_p = np.delete(c_p, i, axis=0)
        iou=np.delete(iou, i, axis=1)
        boxes_to_keep = iou[i, :]<threshold
        iou=np.delete(iou, i, axis=0)
        iou=iou[boxes_to_keep, :][:, boxes_to_keep]
        s_p = s_p[boxes_to_keep, :]
        c_p = c_p[boxes_to_keep]

    if len(s_f) == 0:
        return np.array([]), np.array([])

    s_f = np.squeeze(s_f, axis=1) # Reshape from (M, 1, 7) to (M, 7)
    c_f = np.array(c_f).reshape(-1, 1) # Reshape from (M, ) to (M, 1)
    
    return s_f, c_f