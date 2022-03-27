from load_data import load_data
import numpy as np
from matplotlib import pyplot as plt
#from task2 import compute_projected_points

def filter_points_behind_camera(homogeneous_camera_points, arrays_to_filter=None):
    '''
    This is a copy from the method with same name in task2
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
    This is a copy from the method with same name in task2    
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
    This is a copy from the method with same name in task2
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

def put_channel_id_on_image(filtered_projected_points, filtered_colors, filtered_z_cameras, image):
    '''
    Put the color corresponding to the LIDAR channel it belongs to.

    Parameters
    ----------
    filtered_projected_points : np.array(N, 2)
        The filtered points which are inside the image
    filtered_colors : np.array(N, 1)
        Their corresponding channel color
    filtered_z_cameras: np.array(N, )
        Their corresponding depth (z coordinates in the camera coordinate). This is used in case two points correspond to the same pixel,
        where we will use the color of the closest point.
    image: np.array(W, H, 3)
        The RGB image where we want to put the semantic of the point clouds on top.

    Return
    ----------
    image_with_channels: np.array(W, H, 3)
        the input image where we added the color of each channel.
    '''
    depth_assigned = np.full(image.shape[0:2], -1)

    image_with_channels = image.copy()
    
    
    for i in range(filtered_projected_points.shape[0]):
        x, y = filtered_projected_points[i]
        depth = filtered_z_cameras[i]
        channel_color = filtered_colors[i]

        if(depth_assigned[y, x] == -1 or depth_assigned[y, x]>depth):
            depth_assigned[y, x] = depth
            image_with_channels[y, x, :] = channel_color

    return image_with_channels

def main():

    FILENAME = "../data/data.p"

    data = load_data(FILENAME)
    K = data["P_rect_20"] #Intrinsic matrix, rectified (last column not 0), 3x4
    extrinsic = data["T_cam0_velo"]

    image = data["image_2"]
    world_points = data["velodyne"][:, :3] #We don't need reflectance

    velodyne_ranges = np.linalg.norm(world_points, axis=1)
    elevation_angles = np.arcsin(world_points[:, 2] / velodyne_ranges) * (180/np.pi)

    vertical_min, vertical_max = elevation_angles.min(), elevation_angles.max()
    Num_Channels = 64
    angular_resolution = (vertical_max - vertical_min) / Num_Channels
    
    four_colors = np.array([[255,0,0], [0,0,255], [0,255,0], [255,255,0]])
    channel_ids = (elevation_angles - vertical_min) // angular_resolution
    channel_ids[channel_ids == Num_Channels] = Num_Channels - 1 # Replaces the 64.0 value to 63.0. In most cases, only one point is affected.
    color_ids = np.mod(channel_ids, 4).astype(int)
    colors = four_colors[color_ids]

    filtered_projected_points, [filtered_colors, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [colors])
    image_with_channels = put_channel_id_on_image(filtered_projected_points, filtered_colors, filtered_z_cameras, image)
    plt.imshow(image_with_channels)
    plt.show()

if __name__ == "__main__":
    main()