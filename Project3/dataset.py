# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from utils.task2 import roi_pool
from utils.task3 import sample_proposals

class DatasetLoader(Dataset):
    def __init__(self, config, split):
        self.config, self.split = config, split
        root_dir = config['root_dir']
        assert(os.path.isdir(root_dir))

        t_path = os.path.join(root_dir, f'project3_{split}.txt')
        self.frames = open(t_path).read().splitlines()

        h5_path = os.path.join(root_dir, f'project3_{split}.h5')
        self.hf = h5py.File(h5_path, 'r')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        points = self.get_data(idx, 'xyz')
        valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=self.get_data(idx, 'detections'),
                                                       xyz=points,
                                                       feat=self.get_data(idx, 'features'),
                                                       config=self.config)
        if self.split == 'test':
            return {'frame': frame, 'input': np.concatenate((pooled_xyz, pooled_feat),-1)}

        target = self.get_data(idx, 'target')
        assinged_target, xyz, feat, iou = sample_proposals(pred=valid_pred,
                                                           target=target,
                                                           xyz=pooled_xyz,
                                                           feat=pooled_feat,
                                                           config=self.config,
                                                           train=self.split=='train')
        sampled_frame = {
            'input': np.concatenate((xyz, feat),-1),
            'assinged_target': assinged_target,
            'iou': iou
        }
        if self.split == 'train':
            return sampled_frame
        
        sampled_frame.update({
            'frame': frame,
            'target': target,
            'points': points,
        })
        return sampled_frame

    def get_data(self, idx, key):
        '''
        Get numpy data from hdf5 file for a given frame and key
        Available keys are:
            detections: (K,15) K coarse detection results
            intensity: (N,) point intensity values
            features: (N,128) first stage decoder features
            xyz: (N,3) point coordinates
            target: (K',15) K' gt detections
        To easily collate data for processing, point clouds were randomly
        sampled to contain N=16384 points. If a point cloud contained less
        than N points, it was padded by random point repetitions.
        K=300 coarse detection are generated from each scene. The detections
        pass through a low threshold NMS, hence contain 0's that are filtered
        out.
        output
            detections: (K,7) K coarse detection results (x,y,z,h,w,l,ry)
            intensity: (N,) point intensity values
            features: (N,128) first stage decoder features
            seg: (N,) first stage binary segmentation results
            xyz: (N,3) point coordinates
            target: (K',7) K' gt detections (x,y,z,h,w,l,ry)
        '''
        frame = self.frames[idx]
        data = self.hf[frame][key][()]
        # Rearrange detection data to the format: (x,y,z,h,w,l,ry)
        if key in ['detections', 'target']:
            if len(data) == 0:
                return np.asarray([-1,-1,-1,-100,-100,-100,-1], dtype=np.float32).reshape(1,7)
            data = np.concatenate((data[...,10:13],               # (x,y,z)
                                   data[...,7:10],                # (h,w,l)
                                   data[...,13].reshape(-1,1)),   # (ry)
                                   axis=1)
        # Discard all zero detections
        if key == 'detections':
            data = data[(np.abs(data).sum(1)>0)]
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        return data

    def collate_batch(self, batch):
        '''
        Take each sampled proposal as an individual batch and collate
        all batches by concatenating over the primary axis
        Append batch index as 0th axis on target predictions
        '''
        batch_size = batch.__len__()
        ans_dict = batch[0]
        for key in ans_dict.keys():
            for b in range(1,batch_size):
                ans_dict[key] = np.concatenate((ans_dict[key], batch[b][key]),0,dtype=np.float32)
        for key in ans_dict.keys():
            if key not in ['target', 'points', 'frame']:
                ans_dict[key] = torch.from_numpy(ans_dict[key])
        return ans_dict