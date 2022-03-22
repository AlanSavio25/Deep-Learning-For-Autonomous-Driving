from load_data import load_data
import numpy as np
from matplotlib import pyplot as plt
import math
from task2 import *

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

    FILENAME = "./data/demo.p"

    data = load_data(FILENAME)
    K = data["P_rect_20"] #Intrinsic matrix, rectified (last column not 0), 3x4
    extrinsic = data["T_cam2_velo"]

    image = data["image_2"]
    world_points = data["velodyne"][:, :3] #We don't need reflectance

    velodyne_ranges = np.linalg.norm(world_points, axis=1)
    elevation_angles = np.arcsin(world_points[:, 2] / velodyne_ranges) * (180/np.pi)

    vertical_min, vertical_max = elevation_angles.min(), elevation_angles.max()
    Num_Channels = 64
    angular_resolution = (vertical_max - vertical_min) / Num_Channels
    
    four_colors = np.array([[255,0,0], [0,0,255], [0,255,0], [250,50,50]])
    channel_ids = (elevation_angles - vertical_min) // angular_resolution
    color_ids = np.mod(channel_ids, 4).astype(int)
    colors = four_colors[color_ids]

    filtered_projected_points, [filtered_colors, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [colors])
    image_with_channels = put_channel_id_on_image(filtered_projected_points, filtered_colors, filtered_z_cameras, image)
    plt.imshow(image_with_channels)
    plt.show()

if __name__ == "__main__":
    main()