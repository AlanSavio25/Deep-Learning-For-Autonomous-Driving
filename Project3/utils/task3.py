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
    iou = np.ones_like(iou)
    ## Part a) ##

    indices = [(x, y) for x, y in zip(np.arange(iou.shape[0]), np.argmax(iou, axis=1))]  # indices of the ground_truth closest to each pred

    ## Part b) ##

    fg = [idx for idx in indices if iou[idx] >= config['t_fg_lb']]
    bg = np.asarray([idx for idx in indices if iou[idx] < config['t_bg_up']])
    easy_bg = np.asarray([idx for idx in indices if iou[idx] < config['t_bg_hard_lb']])
    hard_bg = np.asarray([idx for idx in indices if config['t_bg_up'] > iou[idx] >= config['t_bg_hard_lb']])

    extended_indices = [(x, y) for x, y in zip(np.argmax(iou, axis=0), np.arange(iou.shape[1])) if iou[x, y] > 0]
    extended_fg = np.asarray(fg + extended_indices)

    if train:
        if len(bg) == 0: # No background, only foreground
            if len(extended_fg) < config['num_samples']:
                sample_indices = extended_fg[np.random.choice(len(extended_fg), size=config['num_samples'], replace=True)]
            else:
                sample_indices = extended_fg[np.random.choice(len(extended_fg), size=config['num_samples'], replace=False)]

        elif len(extended_fg) == 0: # No fg, only bg
            if len(easy_bg) > 0 and len(hard_bg) > 0:

                sample_indices = get_bg_sample_indices(required_samples=config['num_samples'], easy=easy_bg, hard=hard_bg, bg_hard_ratio=config['bg_hard_ratio'])

            elif len(easy_bg) > 0:
                sample_indices = easy_bg[np.random.choice(len(easy_bg), size=config['num_samples'], replace=True)]
            else:
                sample_indices = hard_bg[np.random.choice(len(hard_bg), size=config['num_samples'], replace=True)]

        else: # both fg and bg exist in the scene
            if len(extended_fg) >= config['num_fg_sample']:
                sample_indices = extended_fg[np.random.choice(len(extended_fg), size=config['num_fg_sample'], replace=False)]
            else:
                sample_indices = extended_fg
            sample_indices = np.concatenate((sample_indices, get_bg_sample_indices(required_samples=config['num_samples']-sample_indices.shape[0], easy=easy_bg, hard=hard_bg, bg_hard_ratio=config['bg_hard_ratio'])))
    else:
        sample_indices = np.concatenate((extended_fg, bg))

    assigned_targets = target[[t for p, t in sample_indices]]
    samples_xyz = xyz[[p for p, t in sample_indices]]
    samples_feat = feat[[p for p, t in sample_indices]]

    samples_iou = iou[[x for x,_ in sample_indices], [y for _,y in sample_indices]]
    return assigned_targets, samples_xyz, samples_feat, samples_iou


def get_bg_sample_indices(required_samples, easy, hard, bg_hard_ratio):
    num_hard = int(np.floor(bg_hard_ratio * required_samples))
    num_easy = int(required_samples - num_hard)
    easy_indices = easy[np.random.choice(len(easy), size=num_easy, replace=True)]
    hard_indices = hard[np.random.choice(len(hard), size=num_hard, replace=True)]
    sample_indices = np.concatenate((easy_indices, hard_indices))
    return sample_indices


