from scipy.interpolate import NearestNDInterpolator
import numpy as np

def voronoi_average_scalar_field(grid, posdata, adata):
    grid_flat = grid.reshape(2, -1).T
    scalar_field = NearestNDInterpolator(posdata, adata)(grid_flat)
    scalar_field = scalar_field.reshape(*np.shape(grid[0]))
    return scalar_field