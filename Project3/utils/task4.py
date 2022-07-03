import torch
import torch.nn as nn
import numpy as np

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        positive_samples = iou >= self.config['positive_reg_lb'] # boolean array (N, )

        # edge case where we have to return 0
        if torch.sum(positive_samples) == 0:
            return self.loss(torch.tensor([0.]), torch.tensor([0.]))

        positive_pred = pred[positive_samples]
        positive_target = target[positive_samples]
        
        loss_location = self.loss(positive_pred[:, :3], positive_target[:, :3])
        loss_size = self.loss(positive_pred[:, 3:6], positive_target[:, 3:6])
        loss_rotation = self.loss(positive_pred[:, 6], positive_target[:, 6])
        
        return loss_location + 3*loss_size + loss_rotation

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,1) confidence prediction
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        positive_samples = iou >= self.config['positive_cls_lb'] # boolean array (N, )
        negative_samples = iou <= self.config['negative_cls_ub'] # boolean array (N, )
        positive_pred = pred[positive_samples]
        negative_pred = pred[negative_samples]
        final_iou = torch.cat([negative_pred, positive_pred], dim=0)
        final_gt = torch.cat([torch.zeros_like(negative_pred), torch.ones_like(positive_pred)])

        return self.loss(final_iou, final_gt)

class BinBasedRegressionLoss(nn.Module):

    '''
    Adapted from PointRCNN: https://github.com/sshaoshuai/PointRCNN
    '''

    def __init__(self, config):
        super().__init__()
        self.loss = nn.SmoothL1Loss()
        self.config = config


    def forward(self, pred, target, iou):

        positive_samples = iou >= self.config['positive_reg_lb']  # boolean array (N, )

        # edge case where we have to return 0
        if torch.sum(positive_samples) == 0:
            return self.loss(torch.tensor([0.]), torch.tensor([0.]))

        positive_pred = pred[positive_samples]
        positive_target = target[positive_samples]

        anchor_size = positive_pred[3:6]
        loc_scope = 1.5
        loc_bin_size = 0.5

        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2

        loc_loss = 0

        # xz localization loss
        x_offset_label, y_offset_label, z_offset_label = positive_target[:, 0], positive_target[:, 1], positive_target[:, 2]
        x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        x_bin_label = (x_shift / loc_bin_size).floor().long()
        z_bin_label = (z_shift / loc_bin_size).floor().long()

        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        loss_x_bin = nn.CrossEntropyLoss(positive_pred[:, x_bin_l: x_bin_r], x_bin_label)
        loss_z_bin = nn.CrossEntropyLoss(positive_pred[:, z_bin_l: z_bin_r], z_bin_label)
        loc_loss += loss_x_bin + loss_z_bin

        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = nn.SmoothL1Loss((positive_pred[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = nn.SmoothL1Loss((positive_pred[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        loc_loss += loss_x_res + loss_z_res

        # y localization loss
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = nn.SmoothL1Loss(positive_pred[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)
        loc_loss += loss_y_offset

        num_head_bin = 9
        # angle loss
        ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

        ry_label = positive_target[:, 6]

        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
        ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
        loss_ry_bin = nn.CrossEntropyLoss(positive_pred[:, ry_bin_l:ry_bin_r], ry_bin_label)
        loss_ry_res = nn.SmoothL1Loss((positive_pred[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

        angle_loss = loss_ry_bin + loss_ry_res

        # size loss
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert positive_pred.shape[1] == size_res_r, '%d vs %d' % (positive_pred.shape[1], size_res_r)

        size_res_norm_label = (positive_target[:, 3:6] - anchor_size) / anchor_size

        size_res_norm = positive_target[:, size_res_l:size_res_r]
        size_loss = nn.SmoothL1Loss(size_res_norm, size_res_norm_label)

        # Total regression loss
        size_loss = 3 * size_loss  # consistent with old codes
        loss_reg = loc_loss + angle_loss + size_loss

        return loss_reg