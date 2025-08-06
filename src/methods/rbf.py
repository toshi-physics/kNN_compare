from scipy.interpolate import RBFInterpolator
import numpy as np

def rbf_average_scalar_field(grid, data, neighbours=None):
    grid_flat = grid.reshape(2, -1).T
    scalar_field = RBFInterpolator(data[:,:2], data[:,-1], neighbours)(grid_flat)
    scalar_field = scalar_field.reshape(*np.shape(grid[0]))
    return scalar_field