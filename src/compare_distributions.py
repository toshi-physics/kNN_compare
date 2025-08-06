import numpy as np
from scipy.spatial import KDTree

def compare_density_error(og_data, og_grid, cg_data, cg_grid): #original data and grid and coarsegrained data and grid
    og_nx, og_ny = np.shape(og_grid[0])
    cg_nx, cg_ny = np.shape(cg_grid[0])
    cdx, cdy     = cg_grid[:,-1,-1] - cg_grid[:, -2,-2]
    odx, ody     = og_grid[:,-1,-1] - og_grid[:, -2,-2]
    if (og_nx < cg_nx) or (og_ny < cg_ny) :
        print("Warning: coarse-grained grid is finer than original grid. Output will be blank.")
    
    elif (og_nx == cg_nx) and (og_ny == cg_ny):
        error = np.abs(og_data-cg_data)*cdx*cdy

    else:
        flat_og_data = og_data.flatten()
        flat_og_grid = og_grid.reshape(2, -1).T
        dr = og_grid[:, -1, -1]-og_grid[:,-2, -2]
        dr = np.sqrt(dr[0]**2 + dr[1]**2)
        tree = KDTree(flat_og_grid)
        error = np.zeros_like(cg_data)
        og_projected = np.zeros_like(cg_data)
        for ix in range(len(cg_grid[0])):
            for iy in range(len(cg_grid[1])):
                p = cg_grid[:, ix, iy]
                #indices = tree.query_ball_point(p, dr)
                distances, indices = tree.query(p, k=1)
                mean_og = np.sum(flat_og_data[indices])*odx*ody
                og_projected[ix, iy] = mean_og
        error = np.abs(og_projected/np.sum(og_projected*cdx*cdy)-cg_data)*cdx*cdy
                
    return error, og_projected/np.sum(og_projected*cdx*cdy)

def compare_field_error(og_data, og_grid, cg_data, cg_grid): #original data and grid and coarsegrained data and grid
    og_nx, og_ny = np.shape(og_grid[0])
    cg_nx, cg_ny = np.shape(cg_grid[0])
    cdx, cdy     = cg_grid[:,-1,-1] - cg_grid[:, -2,-2]
    odx, ody     = og_grid[:,-1,-1] - og_grid[:, -2,-2]
    if (og_nx < cg_nx) or (og_ny < cg_ny) :
        print("Warning: coarse-grained grid is finer than original grid. Output will be blank.")
    
    elif (og_nx == cg_nx) and (og_ny == cg_ny):
        error = np.abs(og_data-cg_data)/og_data

    else:
        flat_og_data = og_data.flatten()
        flat_og_grid = og_grid.reshape(2, -1).T
        dr = og_grid[:, -1, -1]-og_grid[:,-2, -2]
        dr = np.sqrt(dr[0]**2 + dr[1]**2)
        tree = KDTree(flat_og_grid)
        error = np.zeros_like(cg_data)
        og_projected = np.zeros_like(cg_data)
        for ix in range(len(cg_grid[0])):
            for iy in range(len(cg_grid[1])):
                p = cg_grid[:, ix, iy]
                #indices = tree.query_ball_point(p, dr)
                distances, indices = tree.query(p, k=1)
                mean_og = np.sum(flat_og_data[indices])
                og_projected[ix, iy] = mean_og
        error = np.abs(og_projected-cg_data)
                
    return error, og_projected