import numpy as np

# from .task1 import get_iou # TODO: change back to this line
from task1 import get_iou


def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''

    iou = np.asarray(get_iou(pred, target))  # returns N,M

    ## Part a) ##

    indices = [(x, y) for x, y in zip(np.arange(iou.shape[0]), np.argmax(iou, axis=1))]  # indices of the ground_truth closest to each pred



    # # assignment_indices = zip(np.arange(len(iou)), np.argmax(iou, axis=1)) # Returns a set of indices like [(0,2),(1,3)..]
    # per_pred_indices = np.argmax(iou, axis=1) # N,1 These are the closest ground_truth to each pred
    # # pred_indices = np.argmax(iou, axis=0) # M,1 These are the closest pred to each ground_truth
    # per_pred_labels = np.asarray(target)[per_pred_indices] # N,1
    # per_pred_iou = iou[np.arange(iou.shape[0]), per_pred_indices] # N,1
    #
    # per_target_indices = np.argmax(iou, axis=0) # index of prediction with max iou for each target
    # highest_iou_preds = np.asarray(pred)[per_target_indices]
    # per_target_iou = iou[per_target_indices, np.arange(iou.shape[1])]

    ## Part b) ##

    fg = [idx for idx in indices if iou[idx] >= 0.55]
    bg = [idx for idx in indices if iou[idx] < 0.45]
    easy_bg = [idx for idx in indices if iou[idx] < 0.05]
    hard_bg = [idx for idx in indices if 0.45 > iou[idx] >= 0.05]

    extended_indices = [(x, y) for x, y in zip(np.argmax(iou, axis=0), np.arange(iou.shape[1])) if iou[x, y] > 0]
    extended_fg = fg + extended_indices

    if train:
        if len(bg) == 0: # No background, only foreground
            # all samples should be foreground
            if len(extended_fg) < 64:
                sample_indices = extended_fg + np.random.choice(extended_fg, size=64-len(extended_fg), replace=True)
            else:
                sampled_indices = np.random.choice(extended_fg, size=64, replace=False)

        elif len(extended_fg) == 0: # No fg, only bg
            if len(easy_bg) > 0 and len(hard_bg) > 0:
                num_hard = np.floor(config['bg_hard_ratio'] * len(hard_bg))
                num_easy = 64 - num_hard
                easy_indices = easy_bg + np.random.choice(easy_bg, size=num_easy-len(easy_bg), replace=True)
                hard_indices = hard_bg + np.random.choice(hard_bg, size=num_hard-len(hard_bg), replace=True)
                sample_indices = easy_indices + hard_indices
            elif len(easy_bg) > 0:
                sample_indices = np.random.choice(easy_bg, size=64, replace=True)
            else:
                sample_indices = np.random.choice(hard_bg, size=64, replace=True)

        else: # both fg and bg exist in the scene
            if len(extended_fg) >= 32:
                sample_indices = np.random.choice(extended_fg, size=32, replace=False) #+ compute bg samples (32)
            else:
                sample_indices = extended_fg + compute bg samples (64-len(extended_fg))

        assert len(sampled_indices) == 64

    else:
        sample_indices = extended_fg + bg


    # background = np.where(labels_iou < 0.45)
    # easy_background = np.where(labels_iou < 0.05)
    # hard_background = np.where(labels_iou < 0.45 & labels_iou >= 0.05)
    # extended_foreground = np.where()

    assigned_targets = target[[t for p, t in sample_indices]]
    samples_xyz = xyz[[p for p, t in sample_indices]]
    samples_feat = feat[[p for p, t in sample_indices]]
    samples_iou = iou[sample_indices]

    return assigned_targets, samples_xyz, samples_feat, samples_iou
