import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_surface import read_data_from_surface, compute_shape_complementarity


# .py masif_ligand $one
if len(sys.argv) <= 1:
    print("Usage: {config} "+sys.argv[0]+" PDBID_A {masif_ppi_search | masif_site}")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)

print(sys.argv[1])
pdb=sys.argv[1]
fields = pdb.split('.')
pid=fields[0]
masif_app = sys.argv[2]


if masif_app == 'masif_ppi_search': 
    params = masif_opts['ppi_search']
elif masif_app == 'masif_site':
    params = masif_opts['site']
    params['ply_chain_dir'] = masif_opts['ply_chain_dir']
elif masif_app == 'masif_ligand':
    params = masif_opts['ligand']


np.random.seed(0)
print('Reading data from input ply surface files.')
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_protein.ply".format(pid)
my_precomp_dir = params['masif_precomputation_dir']
if not os.path.exists(my_precomp_dir):
    os.makedirs(my_precomp_dir)
ply_file = masif_opts["ply_chain_dir"] + "/{}_protein.ply".format(pid)
print(ply_file)

input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(ply_file, params, pid=pid)
      
pid=fields[0]
np.save(my_precomp_dir+pid+'_rho_wrt_center', rho) # shape = [N vertices X ]
np.save(my_precomp_dir+pid+'_theta_wrt_center', theta)
np.save(my_precomp_dir+pid+'_input_feat', input_feat)
np.save(my_precomp_dir+pid+'_mask', mask)
# np.save(my_precomp_dir+pid+'_list_indices', neigh_indices)
np.save(my_precomp_dir+pid+'_iface_labels', iface_labels)
# Save x, y, z
# print("####################################"+my_precomp_dir+pid, 'verts[pid][:,0]', verts.shape)
np.save(my_precomp_dir+pid+'_X.npy', verts[:,0])
np.save(my_precomp_dir+pid+'_Y.npy', verts[:,1])
np.save(my_precomp_dir+pid+'_Z.npy', verts[:,2])
