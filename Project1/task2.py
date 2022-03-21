from load_data import load_data
import numpy as np
from matplotlib import pyplot as plt
import math

def filter_points_behind_camera(homogeneous_camera_points, arrays_to_filter=None):
    '''
    Filter points which are behind the camera (and their corresponding labels)

    Parameters
    ----------
    homogeneous_camera_points : np.array(N, 4)
        The homogeneous points in the camera coordinate.
    arrays_to_filter: optional, array of np.array(N, )
        Other arrays to filter with the same mask as the one used to filter the points

    Return
    ----------
    filtered_homogeneous_camera_points : np.array(M, 4)
        The filtered points which are in front of the image.
    filtered_arrays: array of np.array(M, )
        The other arrays filtered with the same mask as the one for the points
    '''
    mask = homogeneous_camera_points[:,2]>0 # Remove all points behind the camera

    filtered_homogeneous_camera_points = homogeneous_camera_points[mask, :] 
    filtered_arrays = None
    if arrays_to_filter is not None:
        filtered_arrays = []
        for array_to_filter in arrays_to_filter:
            filtered_arrays.append(array_to_filter[mask])

    return filtered_homogeneous_camera_points, filtered_arrays


def filter_points_not_in_camera(projected_points, image_shape, arrays_to_filter=None):
    '''
    Filter points which are outside the camera range (and on the same time keep only their corresponding labels and z coordinates):

    Parameters
    ----------
    projected_points : np.array(N, 2)
        The projected points (x,y) into the image coordinate
    image_shape : tuple
        Shape of the image we want to put the points on
    arrays_to_filter: optional, array of np.array(N, )
        Other arrays to filter with the same mask as the one used to filter the points

    Return
    ----------
    filtered_projected_points : np.array(M, 2)
        The filtered points which are inside the image
    filtered_arrays: array of np.array(M, )
        The other arrays filtered with the same mask as the one for the points
    '''
    #Remove all points not in the image range
    mask = np.all([projected_points[:, 0]>=0, projected_points[:, 0]<image_shape[1], projected_points[:, 1]>=0, projected_points[:, 1]<image_shape[0]], axis=0)
    
    filtered_projected_points = projected_points[mask, :]
    filtered_arrays = None
    if arrays_to_filter is not None:
        filtered_arrays = []
        for array_to_filter in arrays_to_filter:
            filtered_arrays.append(array_to_filter[mask])

    return filtered_projected_points, filtered_arrays


def compute_projected_points(world_points, K, extrinsic, image_shape, arrays_to_filter=None, filter_points_outside_camera = True):
    '''
    Compute the 2D projected points in the image coordinate system from the corresponding 3D points in the world using the intrinsic/extrinsic matrix of the camera
    and the shape of the image to remove points outside the range of the image.

    Parameters
    ----------
    world_points : np.array(N, 3)
        The inhomogeneous 3D points in the world coordinate system
    K : np.array(3, 4)
        The intrinsic parameters of the camera
    extrinsic: np.array(4, 4)
        The extrinsic parameters of the camera
    image_shape : tuple
        Shape of the image we want to project the points on
    arrays_to_filter: optional, array of np.array(N, )
        Other arrays to filter with the same mask as the one used to filter the points which don't appear in the image.
    filter_points_outside_camera: optional, Boolean
        If set to true, return only the points which are inside the image range. Otherwise, also return the points which lie outside the x,y range of the image.

    Return
    ----------
    filtered_projected_points : np.array(M, 2)
        The projected points in the image coordinate which are inside the image range
    filtered_arrays: array of np.array(M, )
        The other arrays filtered with the same mask as the one used for the points which don't appear in the image. If arrays_to_filter is not None, 
        add one array with the z-coordinates in the camera coordinate systems of the remaining points.
    '''
    homogeneous_world_points = np.hstack((world_points, np.ones(world_points.shape[0]).reshape(-1, 1))) #Add a column of 1 to have homogeneous points

    homogeneous_camera_points = (extrinsic@homogeneous_world_points.T).T # Go to the camera coordinate system
    homogeneous_camera_points, arrays_to_filter = filter_points_behind_camera(homogeneous_camera_points, arrays_to_filter)

    if arrays_to_filter is not None:
        z_cameras = homogeneous_camera_points[:, 2]/homogeneous_camera_points[:, 3] # Use to know which point to keep when two are on the same pixel in the image after projection 
        arrays_to_filter.append(z_cameras)

    homogeneous_projected_points = (K@homogeneous_camera_points.T).T # Go to the image coordinate system
    projected_points = (homogeneous_projected_points/homogeneous_projected_points[:, 2].reshape(-1,1))[:, :2] # Divide by z coordinates to have inhomogeneous points
    rounded_projected_points = np.around(projected_points).astype(int) # We round them to quantize them into pixels
    
    if filter_points_outside_camera:
        rounded_projected_points, arrays_to_filter = filter_points_not_in_camera(rounded_projected_points, image_shape, arrays_to_filter)
    

    return rounded_projected_points, arrays_to_filter


def put_semantic_color_on_image(filtered_projected_points, filtered_semantic_labels, filtered_z_cameras, image, color_map):
    '''
    Put the color corresponding to the semantic of the closest point in the point cloud in the image.

    Parameters
    ----------
    filtered_projected_points : np.array(N, 2)
        The filtered points which are inside the image
    filtered_semantic_labels : np.array(N, 1)
        Their corresponding semantic labels
    filtered_z_cameras: np.array(N, )
        Their corresponding depth (z coordinates in the camera coordinate). This is used in case two points correspond to the same pixel,
        where we will use the color of the closest point.
    image: np.array(W, H, 3)
        The RGB image where we want to put the semantic of the point clouds on top. 
    color_map: dict
        map from the semantic label to a bgr color


    Return
    ----------
    image_with_semantic: np.array(W, H, 3)
        the input image where we added the color of the semantic of the points.
    '''
    depth_assigned = np.full(image.shape[0:2], -1)

    image_with_semantic = image.copy()

    for i in range(filtered_projected_points.shape[0]):
        x, y = filtered_projected_points[i]
        depth = filtered_z_cameras[i]
        semantic_label = filtered_semantic_labels[i][0]

        if(depth_assigned[y, x] == -1 or depth_assigned[y, x]>depth):
            depth_assigned[y, x] = depth
            bgr = color_map[semantic_label]
            rgb = [bgr[2], bgr[1], bgr[0]]
            image_with_semantic[y, x, :] = rgb

    return image_with_semantic

def main():

    FILENAME = "./data/demo.p"
    data = load_data(FILENAME)
    K = data["P_rect_20"] #Intrinsic matrix, rectified (last column not 0), 3x4
    extrinsic = data["T_cam2_velo"]
    semantic_labels = data["sem_label"]
    color_map = data["color_map"]
    image = data["image_2"]
    world_points = data["velodyne"][:, :3] #We don't need reflectance

    
    filtered_projected_points, [filtered_semantic_labels, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [semantic_labels])

    image_with_semantic = put_semantic_color_on_image(filtered_projected_points, filtered_semantic_labels, filtered_z_cameras, image, color_map)
    
    #Part 2.2    
    cam0_to_velo = np.linalg.inv(data["T_cam0_velo"])

    bboxes_info = np.array(data["objects"])[:, 8:15].astype(float) # We only keep the dimensions, location and rotation y

    bboxes_corners_bbox_coordinates = np.ones((bboxes_info.shape[0], 8, 4)) # 8 corners of each bbox with 4 for homogeneous coordinates
    bboxes_projected_cam2 = []

    for i, bbox_info in enumerate(bboxes_info):
        bbox_x_center = bbox_info[3] 
        bbox_y_center = bbox_info[4]
        bbox_z_center = bbox_info[5]

        bbox_height = bbox_info[0]
        bbox_width = bbox_info[1]
        bbox_length = bbox_info[2]

        bbox_center = np.array([bbox_x_center, bbox_y_center-bbox_height/2, bbox_z_center])


        bbox_y_rotation = bbox_info[6]
        #bbox_y_rotation = math.pi/2

        print(f"Rotation in gradients {bbox_y_rotation} = {bbox_y_rotation*180/math.pi} degrees")

        bboxes_corners_bbox_coordinates[i][0][0] = bbox_width/2
        bboxes_corners_bbox_coordinates[i][0][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][0][2] = bbox_length/2

        bboxes_corners_bbox_coordinates[i][1][0] = - bbox_width/2
        bboxes_corners_bbox_coordinates[i][1][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][1][2] = bbox_length/2

        bboxes_corners_bbox_coordinates[i][2][0] = - bbox_width/2
        bboxes_corners_bbox_coordinates[i][2][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][2][2] = - bbox_length/2

        bboxes_corners_bbox_coordinates[i][3][0] = bbox_width/2
        bboxes_corners_bbox_coordinates[i][3][1] = bbox_height/2
        bboxes_corners_bbox_coordinates[i][3][2] = - bbox_length/2

        bboxes_corners_bbox_coordinates[i][4][0] = bbox_width/2
        bboxes_corners_bbox_coordinates[i][4][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][4][2] = bbox_length/2

        bboxes_corners_bbox_coordinates[i][5][0] = - bbox_width/2
        bboxes_corners_bbox_coordinates[i][5][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][5][2] = bbox_length/2

        bboxes_corners_bbox_coordinates[i][6][0] = - bbox_width/2
        bboxes_corners_bbox_coordinates[i][6][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][6][2] = - bbox_length/2

        bboxes_corners_bbox_coordinates[i][7][0] = bbox_width/2
        bboxes_corners_bbox_coordinates[i][7][1] = - bbox_height/2
        bboxes_corners_bbox_coordinates[i][7][2] = - bbox_length/2
        
        #print(bboxes_corners_bbox_coordinates[i])

        Ry = np.array([[math.cos(bbox_y_rotation),  0,  math.sin(bbox_y_rotation)],
                       [0,                          1,                          0],
                       [-math.sin(bbox_y_rotation), 0, math.cos(bbox_y_rotation)]])
        
        """
        Rx = np.array([[1,  0,  0],
                       [0, math.cos(bbox_y_rotation), -math.sin(bbox_y_rotation)],
                       [0, math.sin(bbox_y_rotation), math.cos(bbox_y_rotation)]])
        
        Rz = np.array([[math.cos(bbox_y_rotation),  -math.sin(bbox_y_rotation),  0],
                       [math.sin(bbox_y_rotation), math.cos(bbox_y_rotation), 0],
                       [0, 0, 1]])
        """
        
        #Ry = np.eye(3)
        #t = Ry @ bbox_center
        t = bbox_center # We don't apply the rotation to it since its based on its center
        #t = np.array([0, 0, 0])
        bbox_to_cam0 = np.vstack((np.hstack((Ry, t.reshape(-1, 1))), [0, 0, 0, 1]))

        bbox_corners_cam0 = (bbox_to_cam0@bboxes_corners_bbox_coordinates[i].T).T

        #print(bbox_corners_cam0)
        bbox_corners_velo = (cam0_to_velo@bbox_corners_cam0.T).T[:, :3]

        bbox_corners_cam2, _ = compute_projected_points(bbox_corners_velo, K, extrinsic, image.shape, filter_points_outside_camera=False)
        bboxes_projected_cam2.append(bbox_corners_cam2)
    
    #Draw the 12 lines of the bounding box:
    line_indices = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for bbox in bboxes_projected_cam2:
        for i1, i2 in line_indices:
            if i1 < len(bbox) and i2 < len(bbox):
                x = [bbox[i1][0], bbox[i2][0]]
                y = [bbox[i1][1], bbox[i2][1]]
                bgr = color_map[10] # 10 stands for car
                color = (bgr[2]/255, bgr[1]/255, bgr[0]/255)
                plt.plot(x, y, color=color, linewidth=1)

    plt.imshow(image_with_semantic)
    plt.show()

if __name__ == "__main__":
    main()