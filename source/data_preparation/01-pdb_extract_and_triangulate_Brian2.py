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

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)

if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
    
# Save the chains as separate files. 
# in_fields = sys.argv[1].split("_")
in_fields = sys.argv[1].split(".") # *.py Sim_2_traj_10_frame_1224.pdb masif_ligand

pdb_id = in_fields[0]
# chain_ids1 = in_fields[1]

if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
print('tmp_dir', tmp_dir)
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
print("pdb_filename", pdb_filename, protonated_file)
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
# Extract protein.
# out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
out_filename1 = tmp_dir+"/"+pdb_id+'_protein'
# print(pdb_filename, out_filename1+".pdb", chain_ids1)
# extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

#Extact protein atoms
extract_protein(pdb_filename, out_filename1+".pdb")
print(pdb_filename, out_filename1+".pdb")
# exit()
# Step 1: Compute MSMS of surface w/hydrogens, 
probe='0.70'
probe_int=0.70
print("PROBE:", probe)
try:
    print("Running MSMS")
    vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
        protonate=True, probe=probe)
    # import pandas as pd
    # save="data_preparation/01-benchmark_MSMS/" + pdb_id+'.npy'
    # print(save)
    # np.save(save,
    #         {"vertices1":vertices1,
    #          "faces1":faces1, 
    #          "normals1":normals1, 
    #          "names1":names1, 
    #          "areas1":areas1})
    # df = pd.DataFrame.from_dict({"vertices1":vertices1,
    #          "faces1":faces1, 
    #          "normals1":normals1, 
    #          "names1":names1, 
    #          "areas1":areas1})
    # print("PD3")
    # df.to_csv("data_preparation/01-benchmark_MSMS/" + pdb_id+"_"+chain_ids1)
    # print("PD4")
except:
    # set_trace()
    print('error with probe=0.70')
   
    vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
        protonate=True, probe='0.71')
    
A1=np.where([re.search('Fe2',i) for i in names1])[0]
# print(names1[A1])
while len(A1)==0:
    print("Cannot get a surface for cofactor")
    probe_int=probe_int-.01
    vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
    protonate=True, probe=str(probe_int))
    A1=np.where([re.search('Fe2',i) for i in names1])[0]

sys.stdout.flush()
# Step 2: Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)
    

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
regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                          vertices1, names1)  
regular_names = np.concatenate(regular_names)
A=np.where([re.search('Fe2',i) for i in regular_names])[0]
while len(A)==0:
    print("no cofactor in reduced mesh")
    probe_int=probe_int-.01
    v, f, n, n1, a1 = computeMSMS2(out_filename1+".pdb",\
    protonate=True, probe=str(probe_int))
    # means missing vertex for cofactor
    m = pymesh.form_mesh(v, f)
    masif_opts['mesh_res']=masif_opts['mesh_res']-0.01
    print("masif_opts['mesh_res']", masif_opts['mesh_res'])
    regular_mesh = fix_mesh(m, masif_opts['mesh_res'])
    regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                        vertices1, names1) 
    regular_names = np.concatenate(regular_names)
    A=np.where([re.search('Fe2',i) for i in regular_names])[0]

print("len vertices2", len(vertices2), "len(regular_mesh.vertices)", len(regular_mesh.vertices))
sys.stdout.flush()
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
np.save(masif_opts['ply_chain_dir']+pdb_id+'_names.npy',
            {'names1':np.array(regular_names)})

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)
sys.stdout.flush()
if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

iface = np.zeros(len(regular_mesh.vertices))
# we have to change this for any dioxygens within 3.5 angstroms of any residues
print('computeMSMS2', pdb_filename)
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    sys.stdout.flush()
    # Compute the surface of the entire complex and from that compute the interface.
    try:
        v3, f3, _, names3, _ = computeMSMS2(pdb_filename,protonate=True, probe=probe)
    except:
        v3, f3, _, names3, _ = computeMSMS2(pdb_filename,protonate=True, probe='0.71')
    # Regularize the mesh
    
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    # full_regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
    full_regular_names = assignAtomNamesToNewMesh(mesh.vertices, 
                                          v3, names3) 
    full_regular_names=np.concatenate(full_regular_names)  
    regular_names=np.concatenate(regular_names)              
    full_iface = np.zeros(len(full_regular_names))
    full_iface[np.where([re.search('O2I',i) for i in full_regular_names])[0]] = 1
    # vertices from full mesh with gas/everything
    O = np.where([re.search('O2I',i) for i in full_regular_names])[0]
    tO=mesh.vertices[O]
    # vertices from regular mesh of protein/cofactor
    A=np.where([re.search('Fe2',i) for i in regular_names])[0]
    if len(A)==0:
        # means missing vertex for cofactor
        m = pymesh.form_mesh(vertices2, faces2)
        while len(A)==0:
            masif_opts['mesh_res']=masif_opts['mesh_res']-0.1
            print("masif_opts['mesh_res']", masif_opts['mesh_res'])
            regular_mesh = fix_mesh(m, masif_opts['mesh_res'])
            regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                                vertices1, names1) 
            regular_names = np.concatenate(regular_names)
            A=np.where([re.search('Fe2',i) for i in regular_names])[0]
    
        if masif_opts['use_hbond']:
            vertex_hbond = computeCharges(out_filename1, vertices1, names1)
        # Step 3: For each surface residue, assign the hydrophobicity of its amino acid. 
        if masif_opts['use_hphob']:
            vertex_hphobicity = computeHydrophobicity(names1)
        print('len(mesh.vertices)',len(regular_mesh.vertices))
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        print('HBOND assignChargesToNewMesh')
        if masif_opts['use_hbond']:
            vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hbond, masif_opts)

        # regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
        #                                         vertices1, names1)  
        # regular_names=np.concatenate(regular_names) 
        np.save(masif_opts['ply_chain_dir']+pdb_id+'_names.npy',
                    {'names1':np.array(regular_names)})

        if masif_opts['use_hphob']:
            vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                vertex_hphobicity, masif_opts)
        sys.stdout.flush()
        if masif_opts['use_apbs']:
            vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

        iface = np.zeros(len(regular_mesh.vertices))
        A=np.where([re.search('Fe2',i) for i in regular_names])[0]
        print("2###########################################", len(A))
        
    tA=regular_mesh.vertices[A]

    t2=scipy.spatial.distance_matrix(regular_mesh.vertices,tO)
    t3=scipy.spatial.distance_matrix(regular_mesh.vertices,tA)
    iface1=np.where([(i<3).sum() for i in t2])[0] # protein res near gas
    iface2=np.where([(i<6).sum() for i in t3])[0] # protein res near cofactor
    iface3=np.where([(i<2).sum() for i in t2])[0] # protein res VERY near gas

    np.save(masif_opts['ply_chain_dir']+pdb_id+'_fullnames.npy',
            {'names1':np.array(full_regular_names)})
    # Find the vertices that are in the iface.
    v3 = mesh.vertices
    # full_normal = compute_normal(full_regular_mesh.vertices, full_regular_mesh.faces)
    full_normal = compute_normal(mesh.vertices, mesh.faces)

    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    # kdt = KDTree(v3)
    # d, r = kdt.query(regular_mesh.vertices)
    # d = np.square(d) # Square d, because this is how it was in the pyflann version.
    # assert(len(d) == len(regular_mesh.vertices))
    # iface_v = np.where(d >= 2.0)[0]
    # small gases
    # iface_v = np.where(d >= 1.0)[0]
    # iface[iface_v] = 1.0
    iface[iface1] = 1.0
    iface[np.intersect1d(iface3, iface2)] = 2.0

    t_gas_AKG = scipy.spatial.distance_matrix(regular_mesh.vertices[iface1],
                                          regular_mesh.vertices[A])
    dist_near_gas_to_AKG = np.zeros(len(regular_names))
    dist_near_gas_to_AKG[iface1] = t_gas_AKG.min(axis=1)

    # Convert to ply and save.
    print("saving ply", out_filename1+".ply")
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface, names=regular_names, csv=out_filename1+".csv",
                        cofactor_dist=dist_near_gas_to_AKG)
    print("not Saving complex")
    # save_ply(out_filename1+"_complex.ply", mesh.vertices,
    #                     mesh.faces, normals=full_normal, iface=full_iface)

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
                        names=regular_names, csv=out_filename1+".csv")
# if not os.path.exists(masif_opts['ply_chain_dir']):
#     os.makedirs(masif_opts['ply_chain_dir'])
# if not os.path.exists(masif_opts['pdb_chain_dir']):
#     os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
# shutil.copy(out_filename1+"_complex.ply", masif_opts['ply_chain_dir']) 
# shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
# shutil.copy(out_filename1+'.csv', masif_opts['pdb_chain_dir']) 