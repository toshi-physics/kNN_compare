import numpy as np
from scipy.spatial import KDTree

def kNN_average_density(grid, data, k): #data is Nx2 : xpos, ypos
    tree = KDTree(data)

    densities = np.zeros_like(grid[0], dtype=np.float32)
    nx, ny = np.shape(grid[0])
    for ix in range(nx):
        for iy in range(ny):
            p = grid[:, ix, iy]
            distances, indices = tree.query(p, k)
            densities[ix, iy] = k/(np.pi*max(distances)**2)

    return densities


def kNN_average_scalar_field(grid, data, field, k): #data is Nx3 : xpos, ypos, field_value
    tree = KDTree(data)

    scalar_field = np.zeros_like(grid[0], dtype=np.complex64)

    for ix in range(len(grid[0])):
        for iy in range(len(grid[1])):
            p = grid[:, ix, iy]
            distances, indices = tree.query(p, k)
            f = field[indices]
            weights = np.exp(-distances**2/(np.mean(distances)**2))
            scalar_field[ix, iy] = np.inner(f, weights)/np.sum(weights)

    return scalar_field


def kNN_average_scalar_field_adaptable(grid, data, field, n_cutoff, r_cutoff, return_neighbours=False): #data is Nx3 : xpos, ypos, field_value
    tree = KDTree(data)

    scalar_field = np.zeros_like(grid[0], dtype=np.complex64)
    neighbours = np.zeros_like(grid[0], dtype=np.int32)

    for ix in range(len(grid[0])):
        for iy in range(len(grid[1])):
            p = grid[:, ix, iy]
            no_cells=False
            indices = tree.query_ball_point(p, r_cutoff)
            if len(indices)>n_cutoff:
                distances, indices = tree.query(p, k=np.int32(np.power(len(indices), 2/3)))
            else: no_cells=True
            if not no_cells:
                f = field[indices]
                weights = np.exp(-distances**2/(np.mean(distances)**2))
                scalar_field[ix, iy] = np.inner(f, weights)/np.sum(weights)   
                neighbours[ix, iy] = len(indices)

    if return_neighbours:
        return scalar_field, neighbours
    else:
        return scalar_field