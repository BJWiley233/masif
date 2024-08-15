#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS, computeMSMS2
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh, assignAtomNamesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
# in_fields = sys.argv[1].split(".")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
print("pdb_filename", pdb_filename, protonated_file)
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
print(pdb_filename, out_filename1+".pdb", chain_ids1)
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

#Extact protein atoms
# extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Step 1: Compute MSMS of surface w/hydrogens, 
try:
    print("Running MSMS")
    vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
        protonate=True)
    import pandas as pd
    save="data_preparation/01-benchmark_MSMS/" + pdb_id+"_"+chain_ids1+'.npy'
    print(save)
    np.save(save,
            {"vertices1":vertices1,
             "faces1":faces1, 
             "normals1":normals1, 
             "names1":names1, 
             "areas1":areas1})
    # df = pd.DataFrame.from_dict({"vertices1":vertices1,
    #          "faces1":faces1, 
    #          "normals1":normals1, 
    #          "names1":names1, 
    #          "areas1":areas1})
    # print("PD3")
    # df.to_csv("data_preparation/01-benchmark_MSMS/" + pdb_id+"_"+chain_ids1)
    print("PD4")
except:
    set_trace()

# Step 2: Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    np.save(masif_opts['ply_chain_dir']+'test_chg_names.npy',
            {'vertex_hbond':vertex_hbond,'names1':np.array(names1)})

# Step 3: For each surface residue, assign the hydrophobicity of its amino acid. 
if masif_opts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Step 4: Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)

print('len(mesh.vertices)',len(mesh.vertices))
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
print("len vertices2", len(vertices2), "len(regular_mesh.vertices)", len(regular_mesh.vertices))

# Step 5: Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

# Step 6: Assign charges on new vertices based on charges of old vertices (nearest neighbor)

    # Step 6.A: why not just compute on new mesh vertices? if we have names, test this
print('HBOND assignChargesToNewMesh')
if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hbond, masif_opts)

regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                          vertices1, names1)   

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)

if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

iface = np.zeros(len(regular_mesh.vertices))
# we have to change this for any dioxygens within 3.5 angstroms of any residues
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS2(pdb_filename,\
        protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface, names=regular_names, csv=out_filename1+".csv")

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
                        names=regular_names, csv=out_filename1+".csv")
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
shutil.copy(out_filename1+'.csv', masif_opts['pdb_chain_dir']) 