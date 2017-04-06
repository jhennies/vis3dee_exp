from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra
import numpy as np

# # Load path
# path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1/cache/path_data/path_splB_z1.pkl'
# with open(path_file, mode='r') as f:
#     path_data = pickle.load(f)
#
# paths = path_data['paths']
# paths_to_objs = path_data['paths_to_objs']


def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]


with open(
        '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/evaluation.pkl',
        mode='r'
) as f:
    eval_data = pickle.load(f)

# Load segmentation
print 'Loading segmentation...'
seg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/result.h5', 'z/1/test')
print 'Loading resolved segmentation...'
newseg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/result_resolved_thresh_0.5.h5', 'z/1/test')
print 'Loading gt...'
gt = vigra.readHDF5(
    '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
    'z/1/neuron_ids'
)



id = eval_data[0.5]['merged_split'][1]
print 'ID = {}'.format(id)

number_of_objects_to_plot = 5

# id = 101.0
# path_to_obj = paths_to_objs[id]
# print 'path_to_obj = {}'.format(path_to_obj)
# seg[seg != path_to_obj] = 0

mask = seg != id

seg[mask] = 0
newseg[mask] = 0
gt[mask] = 0

crop = find_bounding_rect(seg)
seg = seg[crop]
newseg = newseg[crop]
gt = gt[crop]

print seg.shape

print 'unique in seg: {}'.format(np.unique(seg))
print 'unique in newseg: {}'.format(np.unique(newseg))
print 'unique in gt: {}'.format(np.unique(gt))

# gt_unique, gt_counts = np.unique(gt, return_counts=True)
# for l in gt_unique[gt_counts < 1000]:
#     gt[gt == l] = 0


# Just keep the biggest branch
def largest_lbls(im, n):
    im_unique, im_counts = np.unique(im, return_counts=True)
    largest_ids = np.argsort(im_counts)[-n-1:-1]
    return im_unique[largest_ids]

# from copy import deepcopy
tgt = np.zeros(gt.shape, dtype=gt.dtype)
tnewseg = np.zeros(newseg.shape, dtype=newseg.dtype)
for l in largest_lbls(gt, number_of_objects_to_plot):
    tgt[gt == l] = l
for l in largest_lbls(newseg, number_of_objects_to_plot):
    tnewseg[newseg == l] = l
gt = tgt
newseg = tnewseg

# sorted_ids = np.argsort(gt_counts)
# sorted_uniques = gt_unique[sorted_ids]

seg, _, _ = vigra.analysis.relabelConsecutive(seg, start_label = 0)
newseg, _, _ = vigra.analysis.relabelConsecutive(newseg, start_label=0)
gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0)

print 'Starting to plot...'

nsp.start_figure()

# nsp.add_path(paths[id].swapaxes(0, 1), anisotropy=[1, 1, 10])
nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(seg), opacity=0.5)

nsp.start_figure()

nsp.add_iso_surfaces(newseg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(newseg), opacity=0.5)

nsp.start_figure()

nsp.add_iso_surfaces(gt, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(gt), opacity=0.5)
nsp.show()

