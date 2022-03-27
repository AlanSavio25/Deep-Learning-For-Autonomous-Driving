# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import math
from load_data import load_data
#from task2 import compute_world_bbox_corners

def compute_world_bbox_corners(bboxes_info, cam_to_velo):
    '''
    This is a copy from the method with same name in task2
    '''
    bboxes_corners_bbox_coordinates = np.ones((bboxes_info.shape[0], 8, 4)) # 8 corners of each bbox with size 4 for homogeneous coordinates
    bboxes_corners_world = []
    
    for i, bbox_info in enumerate(bboxes_info):
        bbox_x_center = bbox_info[3] 
        bbox_y_bottom = bbox_info[4]
        bbox_z_center = bbox_info[5]

        bbox_height = bbox_info[0]
        bbox_width = bbox_info[1]
        bbox_length = bbox_info[2]

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

        bbox_corners_cam0 = (bbox_to_cam0@bboxes_corners_bbox_coordinates[i].T).T
        bbox_corners_velo = (cam_to_velo@bbox_corners_cam0.T).T[:, :3]
        bboxes_corners_world.append(bbox_corners_velo)
    
    return np.array(bboxes_corners_world)

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points, sem_labels, color_map):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        points_color_bgr = [color_map[sem_label[0]] for sem_label in sem_labels]
        points_color_rgb = np.array(points_color_bgr)
        points_color_rgb[:,[0, 2]] = points_color_rgb[:,[2, 0]] # Swap blue and red columns
        points_color_rgb_normalized = points_color_rgb/255 # Normalize in range [0-1]
        self.sem_vis.set_data(points, size=3, face_color=points_color_rgb_normalized)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('../data/data.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3], data["sem_label"], data["color_map"])

    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    cam0_to_velo = np.linalg.inv(data["T_cam0_velo"])
    bboxes_info = np.array(data["objects"])[:, 8:15].astype(float) # We only keep the dimensions, location and rotation y
    corners = compute_world_bbox_corners(bboxes_info, cam0_to_velo)
    visualizer.update_boxes(corners)
    vispy.app.run()




