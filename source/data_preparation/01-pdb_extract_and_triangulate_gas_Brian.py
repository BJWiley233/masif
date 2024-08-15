#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
import re
import scipy
# from IPython.core.debugger import set_trace

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
from sklearn.neighbors import KDTree
from biopandas.pdb import PandasPdb

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)

if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
    

in_fields = sys.argv[1].split(".") # *.py Sim_2_traj_10_frame_1224.pdb masif_ligand
pdb_id = in_fields[0]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+"_gas.pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+"_gas.pdb"
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
