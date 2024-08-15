# Neural network ligand application specific parameters.
import os
import sys
from multiprocessing import Pool
import subprocess as sp
import itertools
# need to preprocess to only select O2 within 3.5 angstroms of protein
import numpy as np
import scipy
from subprocess import Popen, PIPE
import subprocess as sp
from functools import partial
import time
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)
import pymesh
# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts
np.random.seed(0)

from masif_modules.read_data_from_surface2 import read_data_from_surface, compute_shape_complementarity




def convert_to_string(binary):
    return binary.decode('utf-8')

def _run_command(cmd_info):
    """Helper function for submitting commands parallelized."""
    cmd, supress = cmd_info
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    output, err = p.communicate()
    if convert_to_string(err) != '' and not supress:
        print("\nERROR: " + convert_to_string(err))
        raise
    output = convert_to_string(output)
    p.terminate()
    return output


def run_commands(cmds, supress=False, n_procs=1):
    """Wrapper for submitting commands to shell"""
    if type(cmds) is str:
        cmds = [cmds]
    if n_procs == 1:
        outputs = []
        for cmd in cmds:
            outputs.append(_run_command((cmd, supress)))
    else:
        cmd_info = list(zip(cmds, itertools.repeat(supress)))
        pool = Pool(processes = n_procs)
        outputs = pool.map(_run_command, cmd_info)
        pool.terminate()
    return outputs



def precompute(pid, overwrite=False):
	try:
		my_precomp_dir = params['masif_precomputation_dir']
		if not os.path.exists(my_precomp_dir+pid):
			os.makedirs(my_precomp_dir+pid)
		if not overwrite and os.path.exists(my_precomp_dir+pid+'/'+pid+'_prot_sc_labels.npy'):
			print("file exists for",pid)
			return "file exists for %s" % pid
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



		np.save(my_precomp_dir+pid+'/'+pid+'_rho_wrt_center', rho[suf]) # shape = [N vertices X ]
		np.save(my_precomp_dir+pid+'/'+pid+'_theta_wrt_center', theta[suf])
		np.save(my_precomp_dir+pid+'/'+pid+'_input_feat', input_feat[suf])
		np.save(my_precomp_dir+pid+'/'+pid+'_mask', mask[suf])
		pad=params["max_shape_size"]
		neigh_indices[suf]= np.array([i + [-1]*(pad-len(i)) for i in neigh_indices[suf]])
		np.save(my_precomp_dir+pid+'/'+pid+'_list_indices', neigh_indices)
		np.save(my_precomp_dir+pid+'/'+pid+'_iface_labels', iface_labels[suf])
		# Save x, y, z
		# print("####################################"+my_precomp_dir+pid, 'verts[pid][:,0]', verts.shape)
		np.save(my_precomp_dir+pid+'/'+pid+'_X.npy', verts[suf][:,0])
		np.save(my_precomp_dir+pid+'/'+pid+'_Y.npy', verts[suf][:,1])
		np.save(my_precomp_dir+pid+'/'+pid+'_Z.npy', verts[suf][:,2])
		print("theta",my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')
        # takes to long to save
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
		# print("theta",my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')
		# os.remove(my_precomp_dir+pid+'/'+pid+"_"+suf+'_theta.npy')
		
		return "Success %s" % pid
	except Exception as e:
		print("error", e, pid)
		return "error %s" % pid
        
    

if __name__ == "__main__":
	# python get_close_pdbs.py 2 3 O2IF 2
	sys.path.insert(0, "/data/pompei/bw973/Oxygenases/masif/source")
	from default_config.masif_opts import masif_opts

	params = masif_opts['ligand']
	if not os.path.exists(params[ "masif_precomputation_dir"]):
		os.makedirs(params[ "masif_precomputation_dir"])

	print("params['max_distance']",params['max_distance'])
	params['sc_w']=0.015
	params['sc_interaction_cutoff']=3.0
	params['sc_radius']=params['max_distance']
	# params['training_list']=masif_opts['ppi_search']['training_list']


	processes = int(sys.argv[1])
	if len(sys.argv) > 2:
		overwrite=sys.argv[2]
	if len(sys.argv) > 3:
		params['training_list']=sys.argv[3]
		
	else:
		params['training_list']="lists/feat_train.txt"
	print("params['training_list']", params['training_list'])
	# l = int(sys.argv[2])
	pids = [x.rstrip() for x in open(params['training_list']).readlines()]
	print(len(pids))
	# pids = pids[9000:]
	print("pids", pids)
	pool = Pool(processes=processes)
	_ = pool.map(partial(precompute, overwrite=overwrite), pids)
	print(_)
