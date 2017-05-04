
# Do the following for objects ['merged_split'][3]
#                          and ['merged_split'][8]
#
# 1. Iso surface of mc segmentation
# 2. Iso surface of mc segmentation + Path of merge detection + raw data + xyz planes
#    Make sure it is indicated that the path runs from border to border
# 3. Iso surface of mc segmentation + sampled paths
# 4. Iso surface of resolved result (+ xyz planes ?)
#    Iso surface of gt (mask with segmentation) (+ xyz planes ?)

from neuro_seg_plot import NeuroSegPlot as nsp
import vigra
import numpy as np
import pickle
from mayavi import mlab


def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]


def synchronize_labels(gt, newseg):

    newseg = newseg + np.amax(np.unique(gt))
    newseg[newseg == np.amax(np.unique(gt))] = 0

    z1, z2 = np.unique(gt, return_counts=True)
    z1 = z1[np.argsort(z2)][::-1]
    list = [0]
    for i in z1:
        print "i: ", i

        if i == 0:
            continue

        a, b = np.unique(newseg[gt == i], return_counts=True)
        a = a[np.argsort(b)][::-1]
        c = 0

        if a[0] == 0 and len(a) == 1:
            continue

        else:
            for idx, elem in enumerate(a):
                if elem in list:
                    continue
                c = elem
                newseg[newseg == c] = i
                print "C: ", c
                break

        print "np.unique(newseg):", np.unique(newseg)
        list.append(i)

    for idx, elem in enumerate(np.unique(newseg)[np.unique(newseg) > np.amax(np.unique(gt))]):
        newseg[newseg == elem] = np.amax(np.unique(gt)) + idx + 1


with open(
        '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/evaluation.pkl',
        mode='r'
) as f:
    eval_data = pickle.load(f)

cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/cache/'

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
print 'Loading raw data...'
raw = vigra.readHDF5(
    '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
    'z/1/raw'
)

idx = eval_data[0.5]['merged_split'][3]
print 'ID = {}'.format(idx)

number_of_objects_to_plot = 4

plot_type = 4

mask = seg != idx
seg[mask] = 0
crop = find_bounding_rect(seg)
seg = seg[crop]

nsp.start_figure()

if plot_type == 1:
    # ---------------------------------
    # 1. Iso surface of mc segmentation
    seg[seg > 0] = 1
    # seg = seg[:200, :200, :100]
    nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=3, opacity=0.8)
    mlab.view(azimuth=135)

elif plot_type == 2:

    # ---------------------------------
    # 2. Iso surface of mc segmentation + Path of merge detection + raw data as xyz planes

    # Load path
    path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/cache/path_data/path_splB_z1.pkl'
    with open(path_file, mode='r') as f:
        path_data = pickle.load(f)

    paths = path_data['paths']
    paths_to_objs = path_data['paths_to_objs']

    path_id = 14

    # Get specific path from selected idx
    current_paths = np.array(paths)[paths_to_objs == idx]
    path = current_paths[path_id]

    raw = raw[crop]

    nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(seg), opacity=0.5)
    nsp.add_path(path.swapaxes(0, 1), anisotropy=[1, 1, 10])
    nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    mlab.view(azimuth=135)

elif plot_type ==3:

    # ---------------------------------
    # 3. Iso surface of mc segmentation + sampled paths

    raw = raw[crop]

    # Load paths
    with open(cache_folder + 'path_data/resolve_paths_{}.pkl'.format(idx), mode='r') as f:
        paths = pickle.load(f)
    with open(cache_folder + 'path_data/resolve_paths_probs_{}.pkl'.format(idx), mode='r') as f:
        path_weights = pickle.load(f)

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


    nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(seg), opacity=0.5)
    nsp.plot_multiple_paths_with_mean_class(
        paths, path_weights,
        custom_lut=lut,
        anisotropy=[1, 1, 10],
        vmin=path_vmin, vmax=path_vmax
    )
    nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    mlab.view(azimuth=135)

elif plot_type == 4:

    # ---------------------------------
    # 4. Iso surface of resolved result (+ xyz planes ?)
    #    Iso surface of gt (mask with segmentation) (+ xyz planes ?)

    newseg[mask] = 0
    newseg = newseg[crop]
    raw = raw[crop]
    gt[mask] = 0
    gt = gt[crop]

    # Just keep the biggest branch
    def largest_lbls(im, n):
        im_unique, im_counts = np.unique(im, return_counts=True)
        largest_ids = np.argsort(im_counts)[-n - 1:-1]
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

    newseg, _, _ = vigra.analysis.relabelConsecutive(newseg, start_label=0)
    gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0)

    synchronize_labels(gt, newseg)

    # nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    # nsp.add_iso_surfaces(newseg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(newseg), opacity=0.8)
    # mlab.view(azimuth=135)
    #
    # nsp.start_figure()

    nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    nsp.add_iso_surfaces(gt, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(gt), opacity=0.8)
    mlab.view(azimuth=135)

nsp.show()
