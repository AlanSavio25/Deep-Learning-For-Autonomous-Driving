import torch
import torch.nn as nn

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