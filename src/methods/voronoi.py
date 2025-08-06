from scipy.interpolate import NearestNDInterpolator
import numpy as np

def voronoi_average_scalar_field(grid, data):
    grid_flat = grid.reshape(2, -1).T
    scalar_field = NearestNDInterpolator(data[:,:2], data[:,-1])(grid_flat)
    scalar_field = scalar_field.reshape(*np.shape(grid[0]))
    return scalar_field