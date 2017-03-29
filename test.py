
from neuro_seg_plot import NeuroSegPlot as nsp
import sys
import numpy as np

if __name__ == '__main__':

    from hdf5_slim_processing import Hdf5Processing as hp
    import h5py

    # Parameters:
    anisotropy = [10, 1, 1]
    interpolation_mode = 'nearest'
    transparent = True
    opacity = 0.25
    label = '44'
    pathid = '1'
    surface_source = 'gt'
    thresh_planes = {'low': 1}
    thresh_planes = None

    # # Specify the files
    # raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    # raw_file = 'cremi.splB.train.raw_neurons.crop.split_xyz.h5'
    # raw_skey = ['z', '1', 'raw']
    #
    # seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170208_avoid_reduncant_path_calculation_sample_b_slice_z_train01_predict10_full/intermed/'
    # seg_file = 'cremi.splB.train.segmlarge.crop.split_z.h5'
    # seg_skey = ['z', '1', 'beta_0.5']
    #
    # gt_path = seg_path
    # gt_file = 'cremi.splB.train.gtlarge.crop.split_z.h5'
    # gt_skey = ['z', '1', 'neuron_ids']
    #
    # paths_path = seg_path
    # paths_file = 'cremi.splB.train.paths.crop.split_z.h5'
    # pathlist_file = 'cremi.splB.train.pathlist.crop.split_z.pkl'
    # paths_skey = ['z_predict1', 'falsepaths', 'z', '1', 'beta_0.5']

    # DEVELOP ----
    # Specify the files
    raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    raw_file = 'cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
    raw_skey = ['x', '1', 'raw']

    seg_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    seg_file = 'cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    seg_skey = ['x', '1', 'beta_0.5']

    gt_path = seg_path
    gt_file = 'cremi.splA.train.gtlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    gt_skey = ['x', '1', 'neuron_ids']

    paths_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    paths_file = 'cremi.splA.train.paths.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    pathlist_file = 'cremi.splA.train.pathlist.crop.crop_x10_110_y200_712_z200_712.split_x.pkl'
    paths_skey = ['predict', 'falsepaths', 'x', '1', 'beta_0.5']

    # Load path
    paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[paths_skey])[paths_skey]
    print paths.keys()
    # label = paths.keys()[1]
    print 'Selected label = {}'.format(label)
    path = np.array(paths[label, pathid])

    import processing_lib as lib

    # Load segmentation
    seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
    seg_image[seg_image != int(label)] = 0

    # Load ground truth
    gt_image = np.array(hp(filepath=gt_path + gt_file, nodata=True, skeys=[gt_skey])[gt_skey])
    gt_labels = np.unique(lib.getvaluesfromcoords(gt_image, path))
    t_seg_image = np.array(gt_image)
    for l in gt_labels:
        t_seg_image[t_seg_image == l] = 0
    gt_image[t_seg_image > 0] = 0
    t_seg_image = None

    crop = lib.find_bounding_rect(seg_image + gt_image, s_=True)
    print 'crop = {}'.format(crop)
    seg_image = lib.crop_bounding_rect(seg_image, crop)
    gt_image = lib.crop_bounding_rect(gt_image, crop)

    path = np.swapaxes(path, 0, 1)
    path[0] = path[0] - crop[0].start
    path[1] = path[1] - crop[1].start
    path[2] = path[2] - crop[2].start

    # Load raw image
    raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])
    raw_image = lib.crop_bounding_rect(raw_image, crop)

    # Optional computations ---------------------------------

    # # Mask raw data
    # raw_image[seg_image == 0] = 0

    # -------------------------------------------------------

    nsp.start_figure()
    nsp.add_xyz_planes(raw_image, anisotropy=anisotropy, threshold=thresh_planes)
    nsp.add_iso_surfaces(seg_image, anisotropy, colormap='Spectral')
    nsp.add_iso_surfaces(gt_image, anisotropy, colormap='Spectral')
    nsp.add_path(path, anisotropy)
    nsp.show()