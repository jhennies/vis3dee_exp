from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra
import numpy as np


def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]

anisotropy = [1, 1, 10]
# cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170321_splB_z1/cache/'
# cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/'
cache_folder = '/media/julian/Daten/datasets/results/multicut_workflow/170328_test/cache/'

resolve_id = 15.0

# Load paths
with open(cache_folder + 'path_data/resolve_paths_{}.pkl'.format(resolve_id), mode='r') as f:
    paths = pickle.load(f)
with open(cache_folder + 'path_data/resolve_paths_probs_{}.pkl'.format(resolve_id), mode='r') as f:
    path_weights = pickle.load(f)

# Load initial segmentation
segmentation = vigra.readHDF5(cache_folder + '../result.h5', 'z/1/test')
# Mask segmentation
segmentation[segmentation != resolve_id] = 0

# Load corrected segmentation
segm_resolved = vigra.readHDF5(cache_folder + '../result_resolved.h5', 'z/1/test')
segm_resolved[segmentation != resolve_id] = 0

# Load gt
# gt = vigra.readHDF5(
#     '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.splB.train.raw_neurons.crop.axes_xyz.split_z.h5',
#     'z/1/neuron_ids'
# )
# gt = vigra.readHDF5(
#     '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5',
#     'z/1/neuron_ids'
# )
gt = vigra.readHDF5(
    '/media/julian/Daten/datasets/cremi_2016/resolve_merges/cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5',
    'z/1/neuron_ids'
)

def getvaluesfromcoords(im, path):
    p = np.swapaxes(path, 0, 1)
    return im[p[0], p[1], p[2]]
# gt[segmentation != 595] = 0
gt_labels = np.array([])
for path in paths:
    gt_labels = np.concatenate((gt_labels, np.unique(getvaluesfromcoords(gt, path))), axis=0)
t_gt_image = np.array(gt)
for l in gt_labels:
    t_gt_image[t_gt_image == l] = 0
gt[t_gt_image > 0] = 0
t_gt_image = None
gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0, keep_zeros=False)
# gt = remove_small_segments(gt)

# Crop all images
crop = find_bounding_rect(segmentation)
print 'crop = {}'.format(crop)
segmentation = segmentation[crop]
gt = gt[crop]
segm_resolved = segm_resolved[crop]

# Adjust path position to the cropped images
for i in xrange(0, len(paths)):
    path = paths[i]
    path = np.swapaxes(path, 0, 1)
    path[0] = path[0] - crop[0].start
    path[1] = path[1] - crop[1].start
    path[2] = path[2] - crop[2].start
    path = np.swapaxes(path, 0, 1)
    paths[i] = path

# # Create custom colormap dictionary
# cdict_BlYlRd = {'red': ((0.0, 1.0, 1.0),  # Very likely merge position
#                         (0.5, 1.0, 1.0),  #
#                         (1.0, 0.0, 0.0)),  # Very likely non-merge position
#                 #
#                 'green': ((0.0, 0.0, 0.0),  # No green at the first stop
#                           (0.5, 1.0, 1.0),
#                           (1.0, 1.0, 1.0)),  # No green for final stop
#                 #
#                 'blue': ((0.0, 0.0, 0.0),
#                          (0.5, 0.0, 0.0),
#                          (1.0, 0.0, 0.0))}

# Create custom colormap dictionary
cdict_BlYlRd = {'red': ((0.0, 0.0, 0.0),  # Very likely merge position
                        (0.5, 1.0, 1.0),  #
                        (1.0, 1.0, 1.0)),  # Very likely non-merge position
                #
                'green': ((0.0, 1.0, 1.0),  # No green at the first stop
                          (0.5, 1.0, 1.0),
                          (1.0, 0.0, 0.0)),  # No green for final stop
                #
                'blue': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0))}

lut = nsp.lut_from_colormap(cdict_BlYlRd, 256)

path_vmin = min(path_weights)
path_vmax = max(path_weights)
print 'path_vmin = {}'.format(path_vmin)
print 'path_vmax = {}'.format(path_vmax)
print 'path_weights = {}'.format(path_weights)

# Ground truth
nsp.start_figure()
nsp.add_iso_surfaces(gt, anisotropy=anisotropy, colormap='Spectral',
                     vmin=np.unique(gt)[1], vmax=np.unique(gt)[-1])
nsp.plot_multiple_paths_with_mean_class(
    paths, path_weights,
    custom_lut=lut,
    anisotropy=anisotropy,
    vmin=path_vmin, vmax=path_vmax
)

# Segmentation and path probs
nsp.start_figure()
nsp.add_iso_surfaces(segmentation, anisotropy=anisotropy, colormap='Spectral',
                     vmin=np.unique(segmentation)[1], vmax=np.unique(segmentation)[-1])
# nsp.plot_multiple_paths_with_mean_class(
#     paths, path_weights,
#     custom_lut=lut,
#     anisotropy=anisotropy,
#     vmin=path_vmin, vmax=path_vmax
# )
nsp.add_path(np.swapaxes(
    paths[-1], 0, 1), s=[path_weights[-1]] * path.shape[0], anisotropy=anisotropy,
    vmin=path_vmin, vmax=path_vmax, custom_lut=lut
)

# Paths information content
# nsp.start_figure()
# nsp.plot_multiple_paths_with_mean_class(
#     paths, [1/float(len(paths))] * len(paths),
#     custom_lut=lut,
#     method=np.sum,
#     anisotropy=anisotropy
# )

# Resolved segmentation
nsp.start_figure()
nsp.add_iso_surfaces(segm_resolved, anisotropy=anisotropy, colormap='Spectral',
                     vmin=np.unique(segm_resolved)[1], vmax=np.unique(segm_resolved)[-1])
# nsp.add_path(np.swapaxes(
#     paths[-1], 0, 1), s=[path_weights[-1]] * path.shape[0], anisotropy=anisotropy,
#     vmin=path_vmin, vmax=path_vmax, custom_lut=lut
# )
nsp.plot_multiple_paths_with_mean_class(
    paths, path_weights,
    custom_lut=lut,
    anisotropy=anisotropy,
    vmin=path_vmin, vmax=path_vmax
)

nsp.show()



import sys
sys.exit()

