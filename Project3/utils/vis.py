# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import numpy as np
import wandb
# import vispy
# from vispy.scene import visuals, SceneCanvas

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


### Taken from project 1 and slightly modified ###

# class Visualizer():
#     def __init__(self):
#         self.canvas = SceneCanvas(keys='interactive', show=True)
#         self.grid = self.canvas.central_widget.add_grid()
#         self.view = vispy.scene.widgets.ViewBox(border_color='white',
#                         parent=self.canvas.scene)
#         self.grid.add_widget(self.view, 0, 0)
#
#         # Point Cloud Visualizer
#         self.sem_vis = visuals.Markers()
#         self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
#         self.view.add(self.sem_vis)
#         visuals.XYZAxis(parent=self.view.scene)
#
#         # Object Detection Visualizer
#         self.obj_vis = visuals.Line()
#         self.obj_vis2 = visuals.Line()
#         self.view.add(self.obj_vis)
#         self.view.add(self.obj_vis2)
#         self.connect = np.asarray([[0,1],[0,3],[0,4],
#                                    [2,1],[2,3],[2,6],
#                                    [5,1],[5,4],[5,6],
#                                    [7,3],[7,4],[7,6]])
#
#     def update(self, points):
#         '''
#         :param points: point cloud data
#                         shape (N, 3)
#         Task 2: Change this function such that each point
#         is colored depending on its semantic label
#         '''
#         self.sem_vis.set_data(points, size=3, face_color=[1.,1.,1.,0.5])
#
#     def update_boxes(self, corners, isGroundTruth=True):
#         '''
#         :param corners: corners of the bounding boxes
#                         shape (N, 8, 3) for N boxes
#         (8, 3) array of vertices for the 3D box in
#         following order:
#             1 -------- 0
#            /|         /|
#           2 -------- 3 .
#           | |        | |
#           . 5 -------- 4
#           |/         |/
#           6 -------- 7
#
#         isGroundTruth: Boolean to say if these are the ground truth corners or predicted ones to put different colors
#         '''
#         for i in range(corners.shape[0]):
#             connect = np.concatenate((connect, self.connect+8*i), axis=0) \
#                       if i>0 else self.connect
#         if isGroundTruth:
#             self.obj_vis.set_data(corners.reshape(-1,3),
#                               connect=connect,
#                               width=2,
#                               color=[0,1,0,1])
#         else:
#             self.obj_vis2.set_data(corners.reshape(-1,3),
#                               connect=connect,
#                               width=2,
#                               color=[1,0,0,1])