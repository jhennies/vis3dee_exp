from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra

# Load path
path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_splB_z1/cache/path_data/path_splB_z1.pkl'
with open(path_file, mode='r') as f:
    path_data = pickle.load(f)

paths = path_data['paths']
paths_to_objs = path_data['paths_to_objs']

# Load segmentation
seg = vigra.readHDF5('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_splB_z1/result.h5', 'z/1/test')

id = 20
path_to_obj = paths_to_objs[id]
print 'path_to_obj = {}'.format(path_to_obj)
seg[seg != path_to_obj] = 0

nsp.start_figure()

nsp.add_path(paths[id].swapaxes(0, 1), anisotropy=[1, 1, 10])
nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10])

nsp.show()