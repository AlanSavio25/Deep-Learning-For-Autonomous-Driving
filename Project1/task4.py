from data_utils import *
from task2 import *
import cv2



def calib_imu2velo():
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert imu coordinates to velo coordinates
    """
    filepath = "./data/problem_4/calib_imu_to_velo.txt"
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
    # index = "0000000077"

    calib_velo2cam_mat = calib_velo2cam("./data/problem_4/calib_velo_to_cam.txt")
    R, T = calib_velo2cam_mat
    extrinsic = np.vstack((np.hstack((R, T)),[0, 0, 0, 1]))
    K = calib_cam2cam("./data/problem_4/calib_cam_to_cam.txt", mode='02')
    image = cv2.imread(f'./data/problem_4/image_02/data/{index}.png')
    world_points = load_from_bin(f"./data/problem_4/velodyne_points/data/{index}.bin")
    distances = np.linalg.norm(world_points, axis=1)
    depth_colors = depth_color(distances)

    # Visualize distorted point cloud
    filtered_projected_points, [filtered_depth_colors, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [depth_colors])
    image_with_colors = print_projection_plt(filtered_projected_points.T, filtered_depth_colors, image) 
    plt.imshow(image_with_colors)
    plt.axis("off")
    plt.show()

    velodyne_timestamp_start = compute_timestamps("./data/problem_4/velodyne_points/timestamps_start.txt", index)
    velodyne_timestamp_end = compute_timestamps("./data/problem_4/velodyne_points/timestamps_end.txt", index)
    image_timestamp = compute_timestamps("./data/problem_4/velodyne_points/timestamps.txt", index)
    oxts_timestamp = compute_timestamps("./data/problem_4/oxts/timestamps.txt", index)
    oxts_velocity = load_oxts_velocity(f"./data/problem_4/oxts/data/{index}.txt")
    oxts_angular_rate = load_oxts_angular_rate(f"./data/problem_4/oxts/data/{index}.txt")

    scan_time = velodyne_timestamp_end - velodyne_timestamp_start

    # Compute angles between 3D points and forward-facing axis
    world_point_angles = np.arctan2(world_points[:, 1], world_points[:, 0]) * (180/np.pi) # swapped x and y coords because of np.arctan2 requirements
    start_to_image_angle = ((image_timestamp - velodyne_timestamp_start) / scan_time) * 360

    # The time of laser is the time when the laser was fired
    time_of_laser = velodyne_timestamp_start + scan_time * (( (start_to_image_angle + world_point_angles) % 360) / 360.0)
    image_to_laser = time_of_laser - image_timestamp # negative implies laser fired before image triggered

    # Convert to imu coords
    homogeneous_world_points = np.hstack((world_points, np.ones(world_points.shape[0]).reshape(-1, 1))) # Add a column of 1 to have homogeneous points
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
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()