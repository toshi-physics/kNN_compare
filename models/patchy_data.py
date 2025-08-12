import numpy as np
from src.generate_data import generate_patchy_data
import json, argparse, os


def main():

    initParser = argparse.ArgumentParser(description='create_data')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/../parameters.json"):
	    with open(savedir+"/../parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    lx = parameters["lx"]
    ly = parameters["ly"]
    dx = parameters["dx"]
    dy = parameters["dy"]
    n_patches = parameters["npatches"]
    patch_radius = parameters["patch_r"]
    snr = parameters["SNR"]
    ndata = parameters["ndata"]
    state = parameters["state"]
    fine_grid_x = np.arange(0, lx, dx)
    fine_grid_y = np.arange(0, ly, dy)
    fine_grid   = np.array(np.meshgrid(fine_grid_x, fine_grid_y, indexing='ij'))

    data, prior_distribution, angle_field, signal, noise = generate_patchy_data(fine_grid, n_patches=n_patches, patch_radius=patch_radius, SNR=snr, n_data=ndata, state=state)

    prior_distribution = prior_distribution/np.sum(prior_distribution*dx*dy)

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    np.save(savedir+'/data/'+'posdata', data[:,:2])
    np.save(savedir+'/data/'+'adata', data[:,-1])
    np.save(savedir+'/data/'+'angle_field', angle_field)
    np.save(savedir+'/data/'+'og_distribution', prior_distribution)
    np.save(savedir+'/data/'+'og_signal', signal)
    np.save(savedir+'/data/'+'og_noise', noise)

if __name__=="__main__":
    main()