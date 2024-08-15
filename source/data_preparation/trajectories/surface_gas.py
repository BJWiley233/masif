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
import shutil
np.random.seed(0)

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS, computeMSMS2
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB, extract_protein
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh, assignAtomNamesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from biopandas.pdb import PandasPdb




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



def surface(pdb_id):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+"_gas.pdb")
    try:
        
        print(masif_opts['ply_chain_dir']+pdb_id+'_gas.ply')
        if os.path.exists(masif_opts['ply_chain_dir']+pdb_id+'_gas.ply'):
            print("file exists for",pdb_id)
            return "file exists for %s" % pdb_id
        
        tmp_dir= masif_opts['tmp_dir']
        
        out_filename1 = tmp_dir+"/"+pdb_id+'_gas'
        shutil.copy(pdb_filename, out_filename1+".pdb") 


        print(pdb_filename, out_filename1+".pdb")
        # exit()
        # Step 1: Compute MSMS of surface w/hydrogens, 
        probe='1.50'
        print("PROBE:", probe)
        try:
            print("Running MSMS")
            mesh_list=[]
            ppdb_df = PandasPdb().read_pdb(out_filename1+".pdb")
            rl = ppdb_df.df["HETATM"].residue_number.unique()
            rl
            for r in rl:
                ppdb_df = PandasPdb().read_pdb(out_filename1+".pdb")
                df=ppdb_df._df["HETATM"]
                df=df[df.residue_number==r]
                ppdb_df._df["HETATM"] = df
                ppdb_df.to_pdb(out_filename1+"%d.pdb"%r)
                vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+"%d.pdb"%r,
                protonate=True, probe=probe)
                mesh = pymesh.form_mesh(vertices1, faces1)
                mesh_list.append(mesh)
        except:
            print('error with probe=1.50')
            mesh_list=[]
            ppdb_df = PandasPdb().read_pdb(out_filename1+".pdb")
            rl = ppdb_df.df["HETATM"].residue_number.unique()
            rl
            for r in rl:
                ppdb_df = PandasPdb().read_pdb(out_filename1+".pdb")
                df=ppdb_df._df["HETATM"]
                df=df[df.residue_number==r]
                ppdb_df._df["HETATM"] = df
                ppdb_df.to_pdb(out_filename1+"%d.pdb"%r)
                vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+"%d.pdb"%r,
                protonate=True, probe='1.51')
                mesh = pymesh.form_mesh(vertices1, faces1)
                mesh_list.append(mesh)

        # Step 4: Fix the mesh.
        mesh = pymesh.merge_meshes(mesh_list)
        vertices2 = mesh.vertices
        faces2 = mesh.faces

        print('len(mesh.vertices)',len(mesh.vertices))
        regular_mesh = fix_mesh(mesh, 0.75)
        print("len vertices2", len(vertices2), "len(regular_mesh.vertices)", len(regular_mesh.vertices))

        # Step 5: Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)


        regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                                vertices1, names1)   
        np.save(masif_opts['ply_chain_dir']+pdb_id+'_ligand_names.npy',
                    {'names1':np.array(regular_names)})


        save_ply(out_filename1+".ply", regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal,
                            names=regular_names
                            )
        
        shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 

        return "Success %s" % pdb_id
    except Exception as e:
        print("error", e, pdb_id)
        return "error %s" % pdb_id


if __name__ == "__main__":
    # python get_close_pdbs.py 2 3 O2IF 2
    sys.path.insert(0, "/data/pompei/bw973/Oxygenases/masif/source")
    from default_config.masif_opts import masif_opts
    params = masif_opts['ligand']
    params['training_list']="pdbs_g.txt"
    if not os.path.exists(masif_opts['ply_chain_dir']):
        os.makedirs(masif_opts['ply_chain_dir'])


    # python surface_protein.py N2 Fe2 5 


    
    
    if (len(sys.argv)>1):
        gas=sys.argv[1]
        metal=sys.argv[2]
    else:
        gas='O2I'
        metal='Fe2'
    processes = int(sys.argv[1])
    if (len(sys.argv)>1):
        params['training_list']=sys.argv[2]
    else:
        params['training_list']=params['training_list']

    
    pdb_ids = [x.rstrip() for x in open(params['training_list']).readlines()]
    #pdb_ids = pdb_ids[0:10]
    print(pdb_ids)
    

    pool = Pool(processes=processes)
    _ = pool.map(partial(surface), pdb_ids)
    print(_)
