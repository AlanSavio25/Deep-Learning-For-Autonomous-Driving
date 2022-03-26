from data_utils import *
from task2 import *
import cv2

def main():
    
    calib_velo2cam_mat = calib_velo2cam("./data/problem_4/calib_velo_to_cam.txt")
    R, T = calib_velo2cam_mat
    extrinsic = np.vstack((np.hstack((R, T)),[0, 0, 0, 1]))

    K = calib_cam2cam("./data/problem_4/calib_cam_to_cam.txt", mode='02')

    image = cv2.imread('./data/problem_4/image_02/data/0000000037.png')     #data["image_2"]
    world_points = load_from_bin("./data/problem_4/velodyne_points/data/0000000037.bin")

    distances = np.linalg.norm(world_points, axis=1)
    depth_colors = depth_color(distances)

    filtered_projected_points, [filtered_depth_colors, filtered_z_cameras] = compute_projected_points(world_points, K, extrinsic, image.shape, [depth_colors])
    image_with_colors = print_projection_plt(filtered_projected_points.T, filtered_depth_colors, image) 
    plt.imshow(image_with_colors)
    # plt.show()

    index = "0000000037"

    velodyne_timestamp_start = compute_timestamps("./data/problem_4/velodyne_points/timestamps_start.txt", index)
    velodyne_timestamp_end = compute_timestamps("./data/problem_4/velodyne_points/timestamps_end.txt", index)
    # image_timestamp = compute_timestamps("./data/problem_4/image_02/timestamps.txt", index)
    # We can use either image_timestamps or velodyne trigger timestamps.
    image_timestamp = compute_timestamps("./data/problem_4/velodyne_points/timestamps.txt", index)
    oxts_timestamp = compute_timestamps("./data/problem_4/oxts/timestamps.txt", index)
    oxts_velocity = load_oxts_velocity(f"./data/problem_4/oxts/data/{index}.txt")
    oxts_angular_rate = load_oxts_angular_rate(f"./data/problem_4/oxts/data/{index}.txt")

    # We assume that each scan takes exactly 1/10th of a second. 

    # Compute angles between 3D points and forward-facing axis
    world_point_angles = np.arctan2(world_points[:, 1], world_points[:, 0]) * (180/np.pi) # swapped x and y because of np.arctan2 requirements
    
    # We need to get the time difference between image trigger and each 3d point's laser.

    # The image_trigger_angle is the angle between velodyne start position and the point when the velodyne was facing forward
    image_trigger_angle = ((image_timestamp - velodyne_timestamp_start) / 0.1) * 360


    # The time of laser is the time when the laser was fired
    time_of_laser = velodyne_timestamp_start + 0.1*(( (image_trigger_angle + world_point_angles) % 360) /360)
    image_trigger_to_laser = time_of_laser - image_timestamp # negative implies laser fired before image triggered

    # TODO: Implement rotation and translation of world points
    # Remember to use calib_imu_to_velo matrix before working with imu data
    # Translation is (velocity * image_trigger_to_laser)
    # Rotation is (angular rate * image_trigger_to_laser)


if __name__ == "__main__":
    main()