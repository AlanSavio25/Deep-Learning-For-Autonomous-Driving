from load_data import load_data
import numpy as np
from matplotlib import pyplot as plt

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


def compute_projected_points(world_points, K, extrinsic, image_shape, arrays_to_filter=None):
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
    filtered_homogeneous_camera_points, arrays_to_filter = filter_points_behind_camera(homogeneous_camera_points, arrays_to_filter)
    z_cameras = filtered_homogeneous_camera_points[:, 2]/filtered_homogeneous_camera_points[:, 3] # Use to know which point to keep when two are on the same pixel in the image after projection 
    
    if arrays_to_filter is not None:
        arrays_to_filter.append(z_cameras)

    homogeneous_projected_points = (K@filtered_homogeneous_camera_points.T).T # Go to the image coordinate system
    projected_points = (homogeneous_projected_points/homogeneous_projected_points[:, 2].reshape(-1,1))[:, :2] # Divide by z coordinates to have inhomogeneous points
    rounded_projected_points = np.around(projected_points).astype(int) # We round them to quantize them into pixels
    filtered_projected_points, filtered_arrays = filter_points_not_in_camera(rounded_projected_points, image_shape, arrays_to_filter)
    
    return filtered_projected_points, filtered_arrays


def put_semantic_color_on_image(filtered_projected_points, filtered_semantic_labels, filtered_z_cameras, image, color_map):
    '''
    Put the color corresponding to the semantic of the closest point in the point cloud in the image.

    Parameters
    ----------
    filtered_projected_points : np.array(N, 2)
        The filtered points which are inside the image
    filtered_semantic_labels : np.array(N, 1)
        Their corresponding semantic labels
    filtered_z_cameras: np.array(M, )
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
    plt.imshow(image_with_semantic)
    plt.show()



if __name__ == "__main__":
    main()