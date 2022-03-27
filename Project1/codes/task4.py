from data_utils import *
#from task2 import compute_projected_points
import cv2

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

def calib_imu2velo():
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert imu coordinates to velo coordinates
    """
    filepath = "../data/problem_4/calib_imu_to_velo.txt"
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)

    imu_to_velo = np.vstack((np.hstack((R, T)),[0, 0, 0, 1]))
    return imu_to_velo

def main():
    index = "0000000037"
    # index = "0000000043"
    # index = "0000000320"
    # index = "0000000077"
    # index = "0000000310"

    calib_velo2cam_mat = calib_velo2cam("../data/problem_4/calib_velo_to_cam.txt")
    R, T = calib_velo2cam_mat
    extrinsic = np.vstack((np.hstack((R, T)),[0, 0, 0, 1]))
    K = calib_cam2cam("../data/problem_4/calib_cam_to_cam.txt", mode='02')
    image = cv2.imread(f'../data/problem_4/image_02/data/{index}.png')
    world_points = load_from_bin(f"../data/problem_4/velodyne_points/data/{index}.bin")
    distances = np.linalg.norm(world_points, axis=1)
    depth_colors = depth_color(distances)
    # Visualize distorted point cloud
    filtered_projected_points, [filtered_depth_colors, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [depth_colors])
    image_with_colors = print_projection_plt(filtered_projected_points.T, filtered_depth_colors, image) 
    plt.imshow(image_with_colors)
    plt.show()

    velodyne_timestamp_start = compute_timestamps("../data/problem_4/velodyne_points/timestamps_start.txt", index)
    velodyne_timestamp_end = compute_timestamps("../data/problem_4/velodyne_points/timestamps_end.txt", index)
    # We can use either image_timestamps or velodyne trigger timestamps.
    # image_timestamp = compute_timestamps("../data/problem_4/image_02/timestamps.txt", index)
    image_timestamp = compute_timestamps("../data/problem_4/velodyne_points/timestamps.txt", index)
    oxts_timestamp = compute_timestamps("../data/problem_4/oxts/timestamps.txt", index)
    oxts_velocity = load_oxts_velocity(f"../data/problem_4/oxts/data/{index}.txt")
    oxts_angular_rate = load_oxts_angular_rate(f"../data/problem_4/oxts/data/{index}.txt")

    '''
    - We assume that each scan takes exactly 1/10th of a second. 
    - We calculate the time difference between image trigger and each 3d point's laser.
    - And then use this time to find the amount of distortion (by multiplying with the velocity).
    '''

    # Compute angles between 3D points and forward-facing axis
    world_point_angles = np.arctan2(world_points[:, 1], world_points[:, 0]) * (180/np.pi) # swapped x and y coords because of np.arctan2 requirements
    start_to_image_angle = ((image_timestamp - velodyne_timestamp_start) / 0.1) * 360

    # The time of laser is the time when the laser was fired
    time_of_laser = velodyne_timestamp_start + 0.1 * (( (start_to_image_angle + world_point_angles) % 360) / 360.0)
    image_to_laser = time_of_laser - image_timestamp # negative implies laser fired before image triggered

    # Convert to imu coords
    homogeneous_world_points = np.hstack((world_points, np.ones(world_points.shape[0]).reshape(-1, 1))) #Add a column of 1 to have homogeneous points
    imu_to_velo = calib_imu2velo()
    velo_to_imu = np.linalg.inv(imu_to_velo)
    imu_world_points = (velo_to_imu@homogeneous_world_points.T).T
    corrected_imu_world_points = np.zeros(imu_world_points.shape)

    # Useful reference for transformation: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html

    # Apply translation and rotation correction to imu_world_points
    T = -(image_to_laser * oxts_velocity.reshape(-1, 1)).T
    rotation_angle = image_to_laser * oxts_angular_rate[2] # We care only about the rotation along z axis.

    for i in range(homogeneous_world_points.shape[0]):
        R = np.array([
            [np.cos(rotation_angle[i]), np.sin(rotation_angle[i]),  0],
            [-np.sin(rotation_angle[i]), np.cos(rotation_angle[i]), 0],
            [0,                          0,                         1]
        ])
        P = np.vstack((np.hstack((R, T[i].reshape(-1, 1))),[0, 0, 0, 1]))
        corrected_imu_world_points[i, :] = (P @ imu_world_points[i, :].T).T

    # Revert to velodyne coordinates and visualize corrected image
    corrected_velo_world_points = (imu_to_velo@corrected_imu_world_points.T).T
    filtered_projected_points, [filtered_depth_colors, filtered_z_cameras] = compute_projected_points(corrected_velo_world_points[:, :3], K, extrinsic, image.shape, [depth_colors])
    corrected_image_with_colors = print_projection_plt(filtered_projected_points.T, filtered_depth_colors, image) 
    plt.imshow(corrected_image_with_colors)
    plt.show()

if __name__ == "__main__":
    main()