from neuro_seg_plot import NeuroSegPlot as nsp
import numpy as np


if __name__ == '__main__':

    from hdf5_slim_processing import Hdf5Processing as hp

    # Parameters:
    anisotropy = [10, 1, 1]
    interpolation_mode = 'nearest'
    transparent = True
    opacity = 0.25
    # label = '118'
    # label = '32'
    label='236'
    pathid = '0'
    surface_source = 'seg'

    # Specify the files
    raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    raw_file = 'cremi.splC.train.raw_neurons.crop.split_xyz.h5'
    raw_skey = ['z', '1', 'raw']

    seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170217_crossvalidation_spl_c_slc_z_train01_pred10/intermed/'
    seg_file = 'cremi.segmlarge.crop.split_z.h5'
    seg_skey = ['z', '1', 'beta_0.5']

    gt_path = raw_path
    gt_file = 'cremi.splC.train.raw_neurons.crop.split_xyz.h5'
    gt_skey = ['z', '1', 'neuron_ids']

    paths_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170217_crossvalidation_spl_c_slc_z_train01_pred10_recompute/intermed/'
    paths_file = 'cremi.paths.crop.split_z.h5'
    pathlist_file = 'cremi.pathlist.crop.split_z.pkl'
    false_paths_skey = ['z_predict1', 'falsepaths', 'z', '1', 'beta_0.5']
    true_paths_skey = ['z_predict1', 'truepaths', 'z', '1', 'beta_0.5']

    # # DEVELOP ----
    # # Specify the files
    # raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    # raw_file = 'cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
    # raw_skey = ['x', '1', 'raw']
    #
    # seg_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # seg_file = 'cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # seg_skey = ['x', '1', 'beta_0.5']
    #
    # gt_path = seg_path
    # gt_file = 'cremi.splA.train.gtlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # gt_skey = ['x', '1', 'neuron_ids']
    #
    # paths_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # paths_file = 'cremi.splA.train.paths.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # pathlist_file = 'cremi.splA.train.pathlist.crop.crop_x10_110_y200_712_z200_712.split_x.pkl'
    # false_paths_skey = ['predict', 'falsepaths', 'x', '1', 'beta_0.5']
    # true_paths_skey = ['predict', 'truepaths', 'x', '1', 'beta_0.5']

    # crop = np.s_[0:10, 0:100, 0:100]

    # Load all false paths
    false_paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[false_paths_skey])[false_paths_skey]
    # And select an object that contains at least one false path
    print false_paths.keys()
    if label == 'random':
        import random
        random.seed()
        label = random.choice(false_paths.keys())

    # Load true paths and false paths for this object
    true_paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[true_paths_skey])[true_paths_skey][label]
    false_paths = false_paths[label]

    # Create path list and respective class list
    paths = []
    classes = []
    for d, k, path, kl in true_paths.data_iterator():
        paths.append(np.array(path))
        classes.append(1)
    for d, k, path, kl in false_paths.data_iterator():
        paths.append(np.array(path))
        classes.append(0)

    print 'len(true_paths) = {}'.format(len(true_paths))
    print 'len(false_paths) = {}'.format(len(false_paths))

    # label = paths.keys()[1]
    print 'Selected label = {}'.format(label)

    import processing_lib as lib

    # Load raw image
    raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])

    # Load segmentation
    seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
    seg_image[seg_image != int(label)] = 0

    # Load ground truth
    gt_image = np.array(hp(filepath=gt_path + gt_file, nodata=True, skeys=[gt_skey])[gt_skey])
    gt_labels = np.array([])
    for path in paths:
        gt_labels = np.concatenate((gt_labels, np.unique(lib.getvaluesfromcoords(gt_image, path))), axis=0)
    t_gt_image = np.array(gt_image)
    for l in gt_labels:
        t_gt_image[t_gt_image == l] = 0
    gt_image[t_gt_image > 0] = 0
    t_gt_image = None

    import vigra
    vigra.analysis.relabelConsecutive(gt_image, start_label=0, out=gt_image)

    # Crop all images
    crop = lib.find_bounding_rect(seg_image, s_=True)
    print 'crop = {}'.format(crop)
    seg_image = lib.crop_bounding_rect(seg_image, crop)
    gt_image = lib.crop_bounding_rect(gt_image, crop)
    raw_image = lib.crop_bounding_rect(raw_image, crop)

    # Adjust path position to the cropped images
    for i in xrange(0, len(paths)):
        path = paths[i]
        path = np.swapaxes(path, 0, 1)
        path[0] = path[0] - crop[0].start
        path[1] = path[1] - crop[1].start
        path[2] = path[2] - crop[2].start
        path = np.swapaxes(path, 0, 1)
        paths[i] = path

    # Create custom colormap dictionary
    cdict_BlYlRd = {'red': ((0.0, 1.0, 1.0),  # Very likely merge position
                            (0.5, 1.0, 1.0),  #
                            (1.0, 0.0, 0.0)),  # Very likely non-merge position
                    #
                    'green': ((0.0, 0.0, 0.0),  # No green at the first stop
                              (0.5, 1.0, 1.0),
                              (1.0, 1.0, 1.0)),  # No green for final stop
                    #
                    'blue': ((0.0, 0.0, 0.0),
                             (0.5, 0.0, 0.0),
                             (1.0, 0.0, 0.0))}

    lut = nsp.lut_from_colormap(cdict_BlYlRd, 256)

    # Groundtruth iso surface and path prediction classes
    nsp.start_figure()
    nsp.add_iso_surfaces(gt_image, anisotropy=anisotropy, colormap='Spectral',
                         vmin=np.unique(gt_image)[1], vmax=np.unique(gt_image)[-1])
    nsp.add_xyz_planes(raw_image, anisotropy=anisotropy)
    nsp.plot_multiple_paths_with_mean_class(
        paths, classes_pred,
        custom_lut=lut,
        vmin=0, vmax=1,
        anisotropy=anisotropy
    )

    # Groundtruth iso surface and path training classes
    gt_figure = nsp()
    gt_figure.start_figure()
    gt_figure.add_iso_surfaces(gt_image, anisotropy=anisotropy, colormap='Spectral',
                               vmin=np.unique(gt_image)[1], vmax=np.unique(gt_image)[-1])
    gt_figure.add_xyz_planes(raw_image, anisotropy=anisotropy)
    gt_figure.plot_multiple_paths_with_mean_class(
        paths, classes,
        custom_lut=lut,
        vmin=0, vmax=1,
        anisotropy=anisotropy
    )

    # Segmentation iso surface and paths with groundtruth label
    seg_figure = nsp()
    seg_figure.start_figure()
    seg_figure.add_iso_surfaces(seg_image, anisotropy=anisotropy, color=(0.3, 0.5, 0.6))
    seg_figure.add_xyz_planes(raw_image, anisotropy=anisotropy)
    sub_paths = seg_figure.multiple_paths_for_plotting(paths, image=gt_image)
    seg_figure.add_multiple_paths(sub_paths, colormap='Spectral', anisotropy=anisotropy,
                                  vmin=np.unique(gt_image)[1], vmax=np.unique(gt_image)[-1])
    seg_figure.show()

