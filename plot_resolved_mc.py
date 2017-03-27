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
cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/'

resolve_id = 4.0

# Load paths
with open(cache_folder + 'path_data/resolve_paths_{}'.format(resolve_id), mode='r') as f:
    paths = pickle.load(f)
with open(cache_folder + 'path_data/resolve_paths_weights_{}'.format(resolve_id), mode='r') as f:
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
gt = vigra.readHDF5(
    '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5',
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
gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0)
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

# Ground truth
nsp.start_figure()
nsp.add_iso_surfaces(gt, anisotropy=anisotropy, colormap='Spectral',
                     vmin=np.unique(gt)[1], vmax=np.unique(gt)[-1])
nsp.plot_multiple_paths_with_mean_class(
    paths, path_weights,
    custom_lut=lut,
    anisotropy=anisotropy,
    vmin=0, vmax=1
)

# Segmentation and path probs
nsp.start_figure()
nsp.add_iso_surfaces(segmentation, anisotropy=anisotropy, colormap='Spectral',
                     vmin=np.unique(segmentation)[1], vmax=np.unique(segmentation)[-1])
nsp.plot_multiple_paths_with_mean_class(
    paths, path_weights,
    custom_lut=lut,
    anisotropy=anisotropy,
    vmin=0, vmax=1
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
nsp.add_path(np.swapaxes(
    paths[-1], 0, 1), s=[path_weights[-1]] * path.shape[0], anisotropy=anisotropy,
    vmin=0, vmax=1, custom_lut=lut
)

nsp.show()



import sys
sys.exit()



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

# # Groundtruth iso surface and path training classes
# gt_figure = nsp()
# gt_figure.start_figure()
# gt_figure.add_iso_surfaces(gt_image, anisotropy=anisotropy, colormap='Spectral',
#                            vmin=np.unique(gt_image)[1], vmax=np.unique(gt_image)[-1])
# gt_figure.add_xyz_planes(raw_image, anisotropy=anisotropy)
# gt_figure.plot_multiple_paths_with_mean_class(
#     paths, classes,
#     custom_lut=lut,
#     vmin=0, vmax=1,
#     anisotropy=anisotropy
# )

# # Segmentation iso surface and paths with groundtruth label
# seg_figure = nsp()
# seg_figure.start_figure()
# seg_figure.add_iso_surfaces(seg_image, anisotropy=anisotropy, color=(0.3, 0.5, 0.6))
# seg_figure.add_xyz_planes(raw_image, anisotropy=anisotropy)
# sub_paths = seg_figure.multiple_paths_for_plotting(paths, image=gt_image)
# seg_figure.add_multiple_paths(sub_paths, colormap='Spectral', anisotropy=anisotropy,
#                               vmin=np.unique(gt_image)[1], vmax=np.unique(gt_image)[-1])
seg_figure.show()
