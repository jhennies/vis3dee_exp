from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra
import numpy as np
from copy import deepcopy

# # Load path
# path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1/cache/path_data/path_splB_z1.pkl'
# with open(path_file, mode='r') as f:
#     path_data = pickle.load(f)

# Load path
# path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170404_all_samples_lifted/paths_cache/train/path_ds_train_splB_z1.pkl'
path_data_filepath = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170421_all_samples_lifted_more_trees/splB_z1/cache/path_data/paths_ds_splB_z1.h5'
#
# with open(path_file, mode='r') as f:
#     path_data = pickle.load(f)
loaded_paths = vigra.readHDF5(path_data_filepath, 'all_paths')
paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')

paths = []
for p in loaded_paths:
    paths.append(np.reshape(p, (len(p)/3, 3)))
# paths = path_data['paths']
# paths_to_objs = path_data['paths_to_objs']

# # Load segmentation
# seg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_splB_z1/result.h5', 'z/1/test')
# Load segmentation
seg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170421_all_samples_lifted_more_trees/splB_z1/result.h5', 'z/1/data')

for p_id, p in enumerate(paths):
    # id = 36
    if len(p) > 1000:

        t_seg = deepcopy(seg)

        path_to_obj = paths_to_objs[p_id]
        print 'path_to_obj = {}'.format(path_to_obj)
        # seg[seg != seg[tuple(paths[p_id][0])]] = 0
        t_seg[seg != path_to_obj] = 0

        nsp.start_figure()

        nsp.add_path(paths[p_id].swapaxes(0, 1), anisotropy=[1, 1, 10])
        nsp.add_iso_surfaces(t_seg, anisotropy=[1, 1, 10])

nsp.show()