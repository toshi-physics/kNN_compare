import numpy as np

def box_average_density(lx, ly, data, box_size, sweep_size): #data is Nx2 : xpos, ypos

    #box_points_x = np.arange(box_size/2, lx-box_size/2, sweep_size)
    #box_points_y = np.arange(box_size/2, ly-box_size/2, sweep_size)
    box_points_x = np.arange(0, lx, sweep_size)
    box_points_y = np.arange(0, ly, sweep_size)
    box_points = np.array(np.meshgrid(box_points_x, box_points_y, indexing='ij'))
    nbox_x = len(box_points_x)
    nbox_y = len(box_points_y)
    nbox   = nbox_x * nbox_y
    densities = np.zeros((len(box_points_x), len(box_points_y)), dtype=np.float32)

    for box in np.arange(nbox):
        box_center_x = box_points[0] [box%nbox_x][box//nbox_x]
        box_center_y = box_points[1] [box%nbox_x][box//nbox_x]
        box_center = np.vstack((box_center_x, box_center_y)).T

        fdata = np.copy(data)
        fdata = fdata - box_center
        fdata = np.floor(fdata/box_size) # coordinates in box units

        cellsinbox = np.logical_and(fdata[:, -1] == 0, fdata[:, -2] == 0)

        densities[box%nbox_x, box//nbox_x] = np.sum(cellsinbox)/box_size**2

    return densities, box_points


def box_average_scalar_field(lx, ly, pos_data, field, box_size, sweep_size): #data is Nx2 : xpos, ypos
    box_points_x = np.arange(0, lx, sweep_size)
    box_points_y = np.arange(0, ly, sweep_size)
    box_points = np.array(np.meshgrid(box_points_x, box_points_y, indexing='ij'))
    nbox_x = len(box_points_x)
    nbox_y = len(box_points_y)
    nbox = nbox_x * nbox_y
    scalar_field = np.zeros((len(box_points_x), len(box_points_y)), dtype=np.complex64)
    empty_boxes=0
    for box in np.arange(nbox):
        box_center_x = box_points[0] [box%nbox_x][box//nbox_x]
        box_center_y = box_points[1] [box%nbox_x][box//nbox_x]
        box_center = np.vstack((box_center_x, box_center_y)).T

        posdata = np.copy(pos_data)
        posdata = posdata - box_center
        posdata = np.floor(posdata/box_size) # coordinates in box units

        cellsinbox = np.logical_and(posdata[:, -1] == 0, posdata[:, -2] == 0)
        if np.sum(cellsinbox)==0:
            empty_boxes+=1
        else:
            scalar_field[box%nbox_x, box//nbox_x] = np.mean(field[cellsinbox])
    return scalar_field
