import numpy as np
import math

def compute_local_features(targets, xyz):
    """
    targets (64,7) target box for each prediction based on highest iou
    xyz (64,512,3) indices 
    """

    distances = np.linalg.norm(xyz, axis=2) #(64, 512)
    xyz_homogeneous = np.concatenate((xyz, np.ones((xyz.shape[0], xyz.shape[1], 1))), axis=2)
    xyz_bboxes = []
    for target, xyzs in zip(targets, xyz_homogeneous):
        Ry = np.array([[math.cos(target[6]),  0,  -math.sin(target[6])],
                       [0,                    1,                   0],
                       [math.sin(target[6]),  0,  math.cos(target[6])]])

        target_center = np.array([-target[0], -target[1]+target[3]/2, -target[2]])
        
        t = Ry@target_center
        cam0_to_bbox = np.vstack((np.hstack((Ry, t.reshape(-1, 1))), [0, 0, 0, 1]))
        xyzs_bbox = (cam0_to_bbox@xyzs.T).T[:, :3]
        xyz_bboxes.append(xyzs_bbox)

    local_features = np.concatenate((np.array(xyz_bboxes), distances.reshape(distances.shape[0], distances.shape[1], 1)), axis=2)
    return local_features


# boxes =np.tile(np.array([0,1,2,3,4,5,math.pi]), (64,1))
# t = np.tile(np.array([0,-0.5, 2]), (64,512,1))
# print(compute_local_features(boxes, t))