# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.__dict__.update(config)

        # Encoder
        channel_in = self.channel_in
        self.set_abstraction = nn.ModuleList()
        for k in range(len(self.npoint)):
            mlps = [channel_in] + self.mlps[k]
            npoint = self.npoint[k] if self.npoint[k]!=-1 else None
            self.set_abstraction.append(
                    PointnetSAModule(
                        npoint=npoint,
                        radius=self.radius[k],
                        nsample=self.nsample[k],
                        mlp=self.mlps[k],
                        use_xyz=True,
                        bn=False
                    )
                )
            channel_in = mlps[-1]

        # Classification head
        cls_layers = []
        pre_channel = channel_in
        for k in range(len(self.cls_fc)):
            cls_layers.extend([
                nn.Conv1d(pre_channel, self.cls_fc[k], kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.cls_fc[k]
        cls_layers.extend([
            nn.Conv1d(pre_channel, 1, kernel_size=1),
            nn.Sigmoid()
        ])
        self.cls_layers = nn.Sequential(*cls_layers)

        # Regression head
        det_layers = []
        pre_channel = channel_in
        for k in range(len(self.reg_fc)):
            det_layers.extend([
                nn.Conv1d(pre_channel, self.reg_fc[k], kernel_size=1),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.reg_fc[k]
        det_layers.append(nn.Conv1d(pre_channel, 7, kernel_size=1))
        self.det_layers = nn.Sequential(*det_layers)

    def forward(self, x):
        xyz = x[..., 0:3].contiguous()                      # (B,N,3)    
        feat = x[..., 3:].transpose(1, 2).contiguous()      # (B,C,N)

        for layer in self.set_abstraction:
            xyz, feat = layer(xyz, feat)
            
        pred_class = self.cls_layers(feat).squeeze(dim=-1)  # (B,1)
        pred_box = self.det_layers(feat).squeeze(dim=-1)    # (B,7)
        return pred_box, pred_class