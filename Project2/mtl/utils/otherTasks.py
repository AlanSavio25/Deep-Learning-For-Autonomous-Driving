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
    edges : torch.tensor(B, H, W)
        The edges of the semantic images with 1 for the pixels being edges (change between 2 different classes, 
        the one being to the right or below the other is considered as an edge) or 0 otherwise.
    """
    B, H, W = y_semseg_lbl.shape

    edges = torch.zeros((B, H, W), device="cuda" if torch.cuda.is_available() else "cpu")
    
    edges[:, 1:, :] = torch.ne(y_semseg_lbl[:, 1:, :], y_semseg_lbl[:, :H-1, :])
    edges[:, :, 1:] = edges[:, :, 1:] + torch.ne(y_semseg_lbl[:, :, 1:], y_semseg_lbl[:, :, :W-1]) > 0

    return edges.to(torch.float)


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

    xvalues = torch.arange(0, W)
    yvalues = torch.arange(0, H)

    """
    /* * * * *
    * * t * *
    * l c * *
    * * * * */
    """
    
    tyy, txx = torch.meshgrid(yvalues[:H-1], xvalues[1:])
    t = torch.concat([torch.stack([txx, tyy], dim=2).unsqueeze(0).repeat(B, 1, 1, 1), depth_image[:, :H-1, 1:].unsqueeze(3)], dim=3)

    lyy, lxx = torch.meshgrid(yvalues[1:], xvalues[:W-1])
    l = torch.concat([torch.stack([lxx, lyy], dim=2).unsqueeze(0).repeat(B, 1, 1, 1), depth_image[:, 1:, :W-1].unsqueeze(3)], dim=3) 
    
    cyy, cxx = torch.meshgrid(yvalues[1:], xvalues[1:])
    c = torch.concat([torch.stack([cxx, cyy], dim=2).unsqueeze(0).repeat(B, 1, 1, 1), depth_image[:, 1:, 1:].unsqueeze(3)], dim=3)

    
    d = torch.cross(torch.sub(l, c), torch.sub(t, c), dim=3)
    n = torch.nn.functional.normalize(d, dim=3).transpose(2,3).transpose(1,2)
    
    output[:, :, 1:, 1:] = n
    output[:, :, 0, :] = output[:, :, 1, :] 
    output[:, :, :, 0] = output[:, :, :, 1]
    
    return output
