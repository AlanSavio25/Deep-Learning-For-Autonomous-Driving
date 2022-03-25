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
    print(filtered_projected_points.shape)
    print(filtered_projected_points.shape, filtered_depth_colors.shape)
    image_with_colors = print_projection_plt(filtered_projected_points.T, filtered_depth_colors, image) 
    plt.imshow(image_with_colors)
    plt.show()


if __name__ == "__main__":
    main()