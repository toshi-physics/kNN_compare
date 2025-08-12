import numpy as np
import json, argparse, os
from src.methods import box, kNN, rbf, voronoi
from src.compare_distributions import compare_density_error, compare_field_error

def main():

    initParser = argparse.ArgumentParser(description='create_data')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initParser.add_argument('-cdx','--coarse_dx', help='coarse_dx', default=0, type=int)
    initParser.add_argument('-cdy','--coarse_dy', help='coarse_dy', default=0, type=int)
    initParser.add_argument('-bs','--box_size', help='box_size', default=0, type=int)
    initParser.add_argument('-ss','--sweep_size', help='sweep_size', default=0, type=int)
    initParser.add_argument('-knn','--k_knn', help='k used in kNN', default=0, type=int)
    initParser.add_argument('-rbf','--rbf_bool', help='bool for rbf method', default=False, type=bool)
    initParser.add_argument('-v','--voronoi_bool', help='bool for voronoi method', default=False, type=bool)

    initargs = initParser.parse_args()
    savedir = initargs.save_dir

    if not os.path.exists(savedir+'/analysis/'):
        os.makedirs(savedir+'/analysis/')
    
    if os.path.isfile(savedir+"/../parameters.json"):
	    with open(savedir+"/../parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    lx = parameters["lx"]
    ly = parameters["ly"]
    dx = parameters["dx"]
    dy = parameters["dy"]
    ndata = parameters["ndata"]

    fine_grid_x = np.arange(0, lx, dx)
    fine_grid_y = np.arange(0, ly, dy)
    fine_grid   = np.array(np.meshgrid(fine_grid_x, fine_grid_y, indexing='ij'))

    cdx = initargs.coarse_dx
    cdy = initargs.coarse_dy

    prior_distribution = np.load(savedir+'/data/'+'og_distribution.npy')
    angle_field = np.load(savedir+'/data/'+'angle_field.npy')
    posdata = np.load(savedir+'/data/'+'posdata.npy')
    adata   = np.load(savedir+'/data/'+'adata.npy')

    if cdx:
        coarse_grid_x = np.arange(0, lx, cdx)
        coarse_grid_y = np.arange(0, ly, cdy)
        coarse_grid   = np.array(np.meshgrid(coarse_grid_x, coarse_grid_y, indexing='ij'))

        if initargs.k_knn:
            
            kNN_k = initargs.k_knn
            kNN_density = kNN.kNN_average_density(coarse_grid, posdata, k=kNN_k)
            #kNN_density = gaussian_filter(kNN_density, 1)
            kNN_density = kNN_density/np.sum(kNN_density*cdx*cdy)

            kNN_order   = kNN.kNN_average_scalar_field(coarse_grid, posdata, adata, k=kNN_k)
            kNN_angles  = np.angle(kNN_order)

            error_kNN, og_projected_kNN = compare_density_error(prior_distribution, fine_grid, kNN_density, coarse_grid)
            error_angle_kNN, _ = compare_field_error(angle_field, fine_grid, kNN_angles, coarse_grid)
            
            np.save(savedir+'/analysis/'+'kNN_density_k_{:d}_cdx_{:d}_cdy_{:d}'.format(kNN_k, cdx, cdy), kNN_density)
            np.save(savedir+'/analysis/'+'kNN_order_k_{:d}_cdx_{:d}_cdy_{:d}'.format(kNN_k, cdx, cdy), kNN_order)
            np.save(savedir+'/analysis/'+'kNN_density_error_k_{:d}_cdx_{:d}_cdy_{:d}'.format(kNN_k, cdx, cdy), error_kNN)
            np.save(savedir+'/analysis/'+'kNN_angle_error_k_{:d}_cdx_{:d}_cdy_{:d}'.format(kNN_k, cdx, cdy), error_angle_kNN)

        if initargs.rbf_bool:
            
            rbf_angles = rbf.rbf_average_scalar_field(coarse_grid, posdata, adata)
            rbf_angles = np.mod(rbf_angles, 2*np.pi)

            error_angle_rbf, _ = compare_field_error(angle_field, fine_grid, rbf_angles, coarse_grid)

            np.save(savedir+'/analysis/'+'rbf_angle_error_cdx_{:d}_cdy_{:d}'.format(cdx, cdy), error_angle_rbf)

        if initargs.voronoi_bool:

            v_angles = voronoi.voronoi_average_scalar_field(coarse_grid, posdata, adata)
            v_angles = np.mod(v_angles, 2*np.pi)

            error_angle_voronoi, _ = compare_field_error(angle_field, fine_grid, v_angles, coarse_grid)

            np.save(savedir+'/analysis/'+'voronoi_angle_error_cdx_{:d}_cdy_{:d}'.format(cdx, cdy), error_angle_voronoi)

    if initargs.box_size:

        box_size = initargs.box_size
        sweep_size = initargs.sweep_size
        box_density, box_points = box.box_average_density(lx, ly, posdata, box_size, sweep_size)
        box_density = box_density/np.sum(box_density*sweep_size*sweep_size)
        box_order   = box.box_average_scalar_field(lx, ly, posdata, adata, box_size, sweep_size)
        box_angles  = np.angle(box_order)
        error_angle_box, _ = compare_field_error(angle_field, fine_grid, box_angles, box_points+sweep_size)
        error_box, og_projected_box = compare_density_error(prior_distribution, fine_grid, box_density, box_points+sweep_size)

        np.save(savedir+'/analysis/'+'box_density_bs_{:d}_ss_{:d}'.format(box_size, sweep_size), box_density)
        np.save(savedir+'/analysis/'+'box_order_bs_{:d}_ss_{:d}'.format(box_size, sweep_size), box_order)
        np.save(savedir+'/analysis/'+'box_density_error_bs_{:d}_ss_{:d}'.format(box_size, sweep_size), error_box)
        np.save(savedir+'/analysis/'+'box_angle_error_bs_{:d}_ss_{:d}'.format(box_size, sweep_size), error_angle_box)

if __name__=="__main__":
    main()