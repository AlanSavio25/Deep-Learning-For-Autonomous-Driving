# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import numpy as np
import wandb

from utils.task1 import get_iou, label2corners

def point_scene(points, pred, target, threshold=0.5, name='test'):
    '''
    points (N,3) point cloud
    pred (N,7) predicted bounding boxes (N,1) scores
    target (N,7) target bounding boxes
    threshold (float) when to consider a prediction correct
    '''
    all_boxes = []
    iou = get_iou(pred, target).max(axis=1)
    correct = iou >= threshold

    for i, p in enumerate(label2corners(pred)):
        all_boxes.append({'corners': p.tolist(),
                          'label': f'{int(100*iou[i])}',
                          'color': [0,255,0] if correct[i] else [255,0,0]})
    for i, t in enumerate(label2corners(target)):
        all_boxes.append({'corners': t.tolist(),
                          'label': '',
                          'color': [255,255,255]})

    return {name: wandb.Object3D({
                'type': 'lidar/beta',
                'points': points,
                'boxes': np.array(all_boxes)
           })}