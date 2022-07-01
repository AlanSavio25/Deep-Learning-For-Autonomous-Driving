import numpy as np

# from task1 import get_iou  # Import libraries when testing locally
from .task1 import get_iou
# from .task2 import sample
# from task2 import sample


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

    iou = get_iou(pred, target)  # returns N,M

    indices = [(x, y) for x, y in
               zip(np.arange(iou.shape[0]), np.argmax(iou, axis=1))]  # N,1 indices of the ground_truth closest to each pred

    # Keeping all sets of indices in lists instead of np.arrays is simpler
    fg = [idx for idx in indices if iou[idx] >= config['t_fg_lb']]
    bg = [idx for idx in indices if iou[idx] < config['t_bg_up']]
    easy_bg = [idx for idx in indices if iou[idx] < config['t_bg_hard_lb']]
    hard_bg = [idx for idx in indices if config['t_bg_up'] > iou[idx] >= config['t_bg_hard_lb']]

    extended_indices = [(x, y) for x, y in zip(np.argmax(iou, axis=0), np.arange(iou.shape[1])) if iou[x, y] > 0]
    extended_fg = list(set().union(fg, extended_indices)) # We use a set to remove the duplicates which satisfy both criterion

    if train:
        if len(bg) == 0:  # No background, only foreground
            # fg_indices = sample(extended_fg, config['num_samples'], method=config['sampling_method'])
            # sample_indices = np.array(extended_fg)[fg_indices].tolist()
            sample_indices = sample(extended_fg, config['num_samples'], method=config['sampling_method'])

        elif len(extended_fg) == 0:  # No fg, only bg
            if len(easy_bg) > 0 and len(hard_bg) > 0:

                sample_indices = get_bg_sample_indices(required_samples=config['num_samples'], easy=easy_bg,
                                                       hard=hard_bg, bg_hard_ratio=config['bg_hard_ratio'], method=config['sampling_method'])

            elif len(easy_bg) > 0:
                # easy_bg_indices = sample(easy_bg, config['num_samples'], method=config['sampling_method'])
                # sample_indices = np.array(easy_bg)[easy_bg_indices].tolist()
                sample_indices = sample(easy_bg, config['num_samples'], method=config['sampling_method'])
            else:
                # hard_bg_indices = sample(hard_bg, config['num_samples'], method=config['sampling_method'])
                # sample_indices = np.array(hard_bg)[hard_bg_indices].tolist()
                sample_indices = sample(hard_bg, config['num_samples'], method=config['sampling_method'])

        else:  # both fg and bg exist in the scene
            if len(extended_fg) >= config['num_fg_sample']:
                # fg_indices = sample(extended_fg, config['num_fg_sample'], method=config['sampling_method'])
                # sample_indices = np.array(extended_fg)[fg_indices].tolist()
                sample_indices = sample(extended_fg, config['num_fg_sample'], method=config['sampling_method'])
            else:
                sample_indices = extended_fg

            required_bg_samples = config['num_samples'] - len(sample_indices)
            bg_sample_indices = get_bg_sample_indices(required_samples=required_bg_samples, easy=easy_bg,
                                                      hard=hard_bg, bg_hard_ratio=config['bg_hard_ratio'], method=config['sampling_method'])
            sample_indices = sample_indices + bg_sample_indices
    else:
        sample_indices = extended_fg + bg
    samples_targets = target[[t for _, t in sample_indices]]
    samples_xyz = xyz[[p for p, _ in sample_indices]]
    samples_feat = feat[[p for p, _ in sample_indices]]
    samples_iou = iou[[x for x, _ in sample_indices], [y for _, y in sample_indices]]

    return samples_targets, samples_xyz, samples_feat, samples_iou


def get_bg_sample_indices(required_samples, easy, hard, bg_hard_ratio, method="random"):
    if required_samples == 0:
        return []
    num_hard = int(np.floor(bg_hard_ratio * required_samples))
    num_easy = int(required_samples - num_hard)
    if len(easy) > 0:
        # easy_bg_indices = sample(easy, num_easy, method=method)
        # easy_indices = np.array(easy)[easy_bg_indices].tolist()
        easy_indices = sample(easy, num_easy, method=method)
    else:
        easy_indices = []
    if len(hard) > 0:
        # hard_bg_indices = sample(hard, num_hard, method=method)
        # hard_indices = np.array(hard)[hard_bg_indices].tolist()
        hard_indices = sample(hard, num_hard, method=method)
    else:
        hard_indices = []

    sample_indices = easy_indices + hard_indices
    return sample_indices

def sample(indexes, required_size, method="random"):
    '''
    input
        indexes, (K, 2): tuple indexes that point to a specific pred-target pair
        required_size, int: number of samples required
        method, string: Method used to sample, must be either 'random' or 'uniform' or 'uniform_targets', default is 'random'
    output
        selected_indices (required_size, ) Samples of indexes of boxes
    '''
    # indices = np.arange(len(indexes))
    ind_of_indexes = np.arange(len(indexes))
    indices = indexes

    if method == "uniform_targets":
        samples = []
        targets = list(set([y for _,y in indexes]))
        num_targets = len(targets)

        # allocations = [required_size // num_targets + (1 if x < required_size % num_targets else 0)
        #                for x in range(num_targets)] # eg: [13,13,13,13,12] for 64 required size and 5 targets
        proposals_per_target = [sum([1 for _,y in indexes if y==t]) for t in targets]
        sorted_proposals_per_target = sorted(zip(proposals_per_target, targets), key=lambda x: x[0])

        almost_equal_allocations = list(reversed([required_size // num_targets + (1 if x < required_size % num_targets else 0)
                       for x in range(num_targets)])) # eg: [13,13,13,13,12] for 64 required size and 5 targets


        zipped = list(zip(sorted_proposals_per_target, almost_equal_allocations))
        allocations = almost_equal_allocations.copy()
        # allocations = [alloc  if prop>=alloc else prop for (prop, _), alloc in zipped]
        for i in range(len(sorted_proposals_per_target)):
            prop = sorted_proposals_per_target[i][0]
            if prop < allocations[i]:
                allocations[i] = prop
            allocations[i+1:] = list(reversed([(required_size - sum(allocations[:i+1])) // (len(allocations[i+1:])) + (1 if x < (required_size - sum(allocations[:i+1])) % len(allocations[i+1:]) else 0)
                   for x in range((len(allocations[i+1:])))]))

        if sum(allocations) < required_size:
            multiplier = required_size / sum(allocations)
            allocations = [round(alloc * multiplier) for alloc in allocations]
            allocations[0] += required_size - sum(allocations)


        for (_, i) in sorted_proposals_per_target: # sorted
            all_preds_of_target = [(x,y) for x,y in indexes if y==i]
            samples_from_target = sample(all_preds_of_target, required_size=allocations.pop(0), method="uniform")
            samples += samples_from_target
    else:
        if  ind_of_indexes.shape[0] < required_size:
            if method == "random":
                indices = np.random.choice(ind_of_indexes, size=required_size, replace=True)
            elif method == "uniform":
                number_tiles = int(required_size/ind_of_indexes.shape[0]) if required_size%ind_of_indexes.shape[0]==0 else int(required_size/ind_of_indexes.shape[0])+1
                indices = np.tile(ind_of_indexes, number_tiles)[:required_size]
            else:
                raise ValueError(f"Method '{method}' is not valid! Must be 'random' or 'uniform'.")
        elif ind_of_indexes.shape[0] >= required_size:
            indices = np.random.choice(ind_of_indexes, size=required_size, replace=False)
        samples = np.array(indexes)[indices].tolist()

        assert indices.shape[0] == required_size
    assert len(samples) == required_size
    return samples #indices