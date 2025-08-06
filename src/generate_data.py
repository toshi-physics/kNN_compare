import numpy as np
from scipy.ndimage import gaussian_filter

def generate_patchy_data(grid, n_patches, patch_radius, gaussian_smoothing_radius, n_data, seed, state):
    bg_noise = 0.01
    patch_strength = 10

    dx, dy = grid[:, -1, -1] - grid[:, -2, -2]
    lx, ly = grid[:, -1, -1] + np.array([dx, dy])
    if seed:
        np.random.seed(seed)
    prior_distribution = np.random.rand(*np.shape(grid[0]))*bg_noise
    rand_centers = np.random.rand(n_patches,2)*np.array([lx, ly])
    for n in np.arange(n_patches):
        prior_distribution = prior_distribution + patch_strength*np.exp((-(grid[0]-rand_centers[n,0])**2 -(grid[1]-rand_centers[n,1])**2 )/patch_radius)
    prior_distribution = gaussian_filter(prior_distribution, gaussian_smoothing_radius)

    prior_max = np.max(prior_distribution)

    angles = generate_angles(grid, dx, dy, prior_distribution, state)

    data   = []

    while len(data) < n_data:
        x = np.random.randint(0, lx)
        y = np.random.randint(0, ly)
        r = np.random.uniform(0, prior_max)

        if r < prior_distribution[np.int32(x), np.int32(y)]:
            data.append([x, y, angles[np.int32(x), np.int32(y)]])

    data = np.array(data)

    return data, prior_distribution, angles

def generate_angles(grid, dx, dy, distribution, state):
    angles = np.zeros_like(grid[0])

    if state == "isotropic":
        angles = np.random.rand(*np.shape(grid[0]))*2*np.pi
    elif state == "homogeneous":
        angles = np.ones_like(grid[0])*np.random.rand(1)*2*np.pi + np.random.rand(*np.shape(grid[0]))*0.1*np.pi
    elif state == "tangential":
        angles = np.arctan2(np.gradient(distribution, dx, dy)[1], np.gradient(distribution, dx, dy)[0]) + np.random.rand(*np.shape(grid[0]))*0.1*np.pi
    elif state == "homeotropic":
        angles = np.arctan2(-np.gradient(distribution, dx, dy)[0], np.gradient(distribution, dx, dy)[1]) + np.random.rand(*np.shape(grid[0]))*0.1*np.pi
    else:
        print("Warning: Provided state does not match available states: isotropic, homogeneous, tangential, or homeotropic. Returned array will be empty.")

    return angles