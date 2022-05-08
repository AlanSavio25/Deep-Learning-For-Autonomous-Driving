import torch

def compute_contour_given_labels(y_semseg_lbl):
    """
    Compute the edges given the semantic labels of each pixels

    Parameters
    ----------
    y_semseg_lbl : torch.tensor(B, H, W)
        The semantic labels for each pixels for each image of the batch

    Return
    ----------
    output : torch.tensor(B, H, W)
        The edges of the images with 1 for the pixels being edges (change between 2 different classes, 
        the one being to the right or below the other is considered as an edge) or 0 otherwise.
    """
    B, H, W = y_semseg_lbl.shape

    output = torch.zeros((B, H, W))
    
    output[:, 1:, :] = torch.ne(y_semseg_lbl[:, 1:, :], y_semseg_lbl[:, :H-1, :])
    output[:, :, 1:] = torch.logical_or(output[:, :, 1:], torch.ne(y_semseg_lbl[:, :, 1:], y_semseg_lbl[:, :, :W-1]))

    return output

def compute_normals(depth_image):
    """
    Compute the normals given the depth of each pixel using cross-product of neighboring pixels. Set first row and column to same normal as 
    neighbors since it cannot be computed for them.
    Adapted from https://answers.opencv.org/question/82453/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-product/

    Parameters
    ----------
    depth_image : torch.tensor(B, H, W)
        The depth for each pixels for each image of the batch

    Return
    ----------
    output : torch.tensor(B, 3, H, W)
        The normal (3D) at each pixel for the whole batch
    """
    B, H, W = depth_image.shape

    output = torch.zeros(B, 3, H, W)
    for y in range(1, H):
        for x in range(1, W):
            """
            /* * * * *
             * * t * *
             * l c * *
             * * * * */
            """
            t = torch.concat([torch.tensor([x, y-1]).repeat(B, 1), depth_image[:, y-1, x].unsqueeze(0).T], dim=1)
            l = torch.concat([torch.tensor([x-1, y]).repeat(B, 1), depth_image[:, y, x-1].unsqueeze(0).T], dim=1)
            c = torch.concat([torch.tensor([x, y]).repeat(B, 1), depth_image[:, y, x].unsqueeze(0).T], dim=1)
            d = torch.cross(torch.sub(l, c), torch.sub(t, c), dim=1)
            
            n = torch.nn.functional.normalize(d, dim=1)

            output[:, :, y, x] = n
    
    # Put same normal in border of the images
    output[:, :, 0, :] = output[:, :, 1, :] 
    output[:, :, :, 0] = output[:, :, :, 1]

    return output