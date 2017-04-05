from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra
import numpy as np

# # Load path
# path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1/cache/path_data/path_splB_z1.pkl'
# with open(path_file, mode='r') as f:
#     path_data = pickle.load(f)

# Load path
path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170404_all_samples_lifted/paths_cache/train/path_ds_train_splB_z1.pkl'
with open(path_file, mode='r') as f:
    path_data = pickle.load(f)

paths = path_data['paths']
paths_to_objs = path_data['paths_to_objs']

# # Load segmentation
# seg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_splB_z1/result.h5', 'z/1/test')
# Load segmentation
seg = vigra.readHDF5('/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5', 'z/1/beta_0.45')

id = 20
path_to_obj = paths_to_objs[1][id]
print 'path_to_obj = {}'.format(path_to_obj)
seg[seg != seg[tuple(paths[1][id][0])]] = 0

nsp.start_figure()

nsp.add_path(paths[1][id].swapaxes(0, 1), anisotropy=[1, 1, 10])
nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10])

nsp.show()