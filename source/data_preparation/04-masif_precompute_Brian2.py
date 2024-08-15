import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)
import pymesh

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_surface2 import read_data_from_surface, compute_shape_complementarity


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
params['sc_w']=0.015
params['sc_interaction_cutoff']=3.0
params['sc_radius']=params['max_distance']

my_precomp_dir = params['masif_precomputation_dir']
if not os.path.exists(my_precomp_dir+pid):
    os.makedirs(my_precomp_dir+pid)
if os.path.exists(my_precomp_dir+pid+'/'+pid+'_list_indices.npy'):
    print("file exists for",pid)

else:
    np.random.seed(0)
print('Reading data from input ply surface files.')

rho = {}
neigh_indices = {}
mask = {}
input_feat = {}
theta = {}
iface_labels = {}
verts = {}

suf="protein"
p_ply_file = masif_opts["ply_chain_dir"] + "{}_{}.ply".format(pid,suf)
print(p_ply_file)
input_feat[suf], rho[suf], theta[suf], mask[suf], neigh_indices[suf], iface_labels[suf], verts[suf] = read_data_from_surface(p_ply_file, params, pid=pid+'/'+pid+"_"+suf)
print(len(verts[suf]))

print("saving _rho_wrt_center")
np.save(my_precomp_dir+pid+'/'+pid+'_rho_wrt_center', rho[suf]) # shape = [N vertices X ]
print("saving _theta_wrt_center")
np.save(my_precomp_dir+pid+'/'+pid+'_theta_wrt_center', theta[suf])
print("saving _input_feat")
np.save(my_precomp_dir+pid+'/'+pid+'_input_feat', input_feat[suf])
print("saving _mask")
np.save(my_precomp_dir+pid+'/'+pid+'_mask', mask[suf])
pad=params["max_shape_size"]
neigh_indices[suf] = np.array([i + [-1]*(pad-len(i)) for i in neigh_indices[suf]])
print("saving _list_indices")
np.save(my_precomp_dir+pid+'/'+pid+'_list_indices', neigh_indices)
print("saving _iface_labels")
np.save(my_precomp_dir+pid+'/'+pid+'_iface_labels', iface_labels[suf])
# Save x, y, z
# print("####################################"+my_precomp_dir+pid, 'verts[pid][:,0]', verts.shape)
print("saving XYZ")
np.save(my_precomp_dir+pid+'/'+pid+'_X.npy', verts[suf][:,0])
np.save(my_precomp_dir+pid+'/'+pid+'_Y.npy', verts[suf][:,1])
np.save(my_precomp_dir+pid+'/'+pid+'_Z.npy', verts[suf][:,2])
print("theta",my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')
# os.remove(my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')

suf="gas"
g_ply_file = masif_opts["ply_chain_dir"] + "{}_{}.ply".format(pid,suf)
print(g_ply_file)
input_feat[suf], rho[suf], theta[suf], mask[suf], neigh_indices[suf], iface_labels[suf], verts[suf] = read_data_from_surface(g_ply_file, params, False, pid+'/'+pid+"_"+suf)
print(len(verts[suf]))



p1_sc_labels, p2_sc_labels = compute_shape_complementarity(p_ply_file, g_ply_file, neigh_indices['protein'],neigh_indices['gas'], rho['protein'], rho['gas'], mask['protein'], mask['gas'], params)
np.save(my_precomp_dir+pid+'/'+pid+'_prot_sc_labels', p1_sc_labels)
np.save(my_precomp_dir+pid+'/'+pid+'_gas_sc_labels', p2_sc_labels)

mylabels = p1_sc_labels[0]
labels = np.median(mylabels, axis=1)
print(pid, labels.max())
regular_mesh=pymesh.load_mesh(p_ply_file)
print(pid, "iface sum 1",(regular_mesh.get_attribute("vertex_iface")==1).sum())
print(pid, "iface sum 2",(regular_mesh.get_attribute("vertex_iface")==2).sum())

pos_labels = np.where(((labels < -0.2) & (labels > -0.6) | (labels > 0.1)))[0]
print(pid,"len(pos_labels)", len(pos_labels))


print(pid, "len_intersect_iface",len(np.intersect1d(
np.where(regular_mesh.get_attribute("vertex_iface")==1),
pos_labels
)))
print("theta",my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')
# os.remove(my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')