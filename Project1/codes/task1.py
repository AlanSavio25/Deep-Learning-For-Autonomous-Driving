from load_data import load_data
import numpy as np
from matplotlib import pyplot as plt

def show_bev(point_cloud, resolution):
    '''
    Show a bird eye view (2D view from above) of the given point clouds with the given resolution.
    If many points are placed in the same bin given the resolution, the highest reflectance will
    be shown.

    Parameters
    ----------
    point_cloud : np.array(N, 4)
        The point cloud with x,y,z coordinates and the reflectance
    resolution : float
        The resolution used to quantize the points
    '''
    #We use the resolution to round the values. Will make easier to create histogram
    point_cloud[:, 0] = np.around(point_cloud[:, 0]/resolution)*resolution
    point_cloud[:, 1] = np.around(point_cloud[:, 1]/resolution)*resolution

    min_x = min(point_cloud[:, 0])
    min_y = min(point_cloud[:, 1])
    max_x = max(point_cloud[:, 0])
    max_y = max(point_cloud[:, 1])

    number_pixels_x = round((max_x-min_x)/resolution)+1
    number_pixels_y = round((max_y-min_y)/resolution)+1

    bev = np.zeros((number_pixels_x, number_pixels_y))

    for point in point_cloud:
        x_loc = round((point[0]-min_x)/resolution) # Round because of floating point precision
        y_loc = round((point[1]-min_y)/resolution) # Round because of floating point precision
        bev[x_loc, y_loc] = max(bev[x_loc, y_loc], point[3]) # We keep the highest intensity of the bin

    plt.imshow(bev, cmap='gray')
    plt.show()


def main():
    FILENAME = "../data/data.p"
    RESOLUTION = 0.2

    data = load_data(FILENAME)
    point_cloud = data["velodyne"]

    show_bev(point_cloud, RESOLUTION)


if __name__ == "__main__":
    main()