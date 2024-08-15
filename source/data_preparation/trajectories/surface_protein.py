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
from sklearn.neighbors import KDTree
import re




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



def surface(pdb_id, gas, metal, overwrite=False, mesh_res=1.0):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
    try:
        
        print(masif_opts['ply_chain_dir']+pdb_id+'_protein.ply')
        if not overwrite and os.path.exists(masif_opts['ply_chain_dir']+pdb_id+'_protein.ply'):
            print("file exists for",pdb_id)
            return "file exists for %s" % pdb_id
        
        tmp_dir= masif_opts['tmp_dir']
        out_filename1 = tmp_dir+"/"+pdb_id+'_protein'
        print('tmp_dir', tmp_dir)
        protonated_file = tmp_dir+"/"+pdb_id+".pdb"
        print("pdb_filename", pdb_filename, protonated_file)
        protonate(pdb_filename, protonated_file)
        pdb_filename = protonated_file

       
        
        #Extact protein atoms which includes cofactors and metals
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
            
        except:
            print('error with probe=0.70')
            vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
                protonate=True, probe='0.71')
            
        A1=np.where([re.search(metal,i) for i in names1])[0]
        
        while len(A1)==0:
            print("Cannot get a surface for cofactor")
            probe_int=probe_int-.01
            vertices1, faces1, normals1, names1, areas1 = computeMSMS2(out_filename1+".pdb",\
            protonate=True, probe=str(probe_int))
            A1=np.where([re.search(metal,i) for i in names1])[0]

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
        regular_mesh = fix_mesh(mesh, mesh_res)
        regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                                vertices1, names1)  
        regular_names = np.concatenate(regular_names)
        A=np.where([re.search(metal,i) for i in regular_names])[0]
        counter=1
        while len(A)==0:
            print("no cofactor in reduced mesh %d" % counter, pdb_id)
            probe_int=probe_int-.01
            v, f, n, n1, a1 = computeMSMS2(out_filename1+".pdb",\
            protonate=True, probe=str(probe_int))
            # means missing vertex for cofactor
            m = pymesh.form_mesh(v, f)
            mesh_res=mesh_res-0.01
            print("mesh_res", mesh_res, pdb_id)
            regular_mesh = fix_mesh(m, mesh_res)
            print("length regular_mesh.vertices %d" % counter, len(regular_mesh.vertices), pdb_id)
            regular_names = assignAtomNamesToNewMesh(regular_mesh.vertices, 
                                                vertices1, names1) 
            regular_names = np.concatenate(regular_names)
            A=np.where([re.search(metal,i) for i in regular_names])[0]
            counter += 1
            sys.stdout.flush()

        print(pdb_id, "len vertices2", len(vertices2), "len(regular_mesh.vertices)", len(regular_mesh.vertices))
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
            full_iface[np.where([re.search(gas,i) for i in full_regular_names])[0]] = 1
            # vertices from full mesh with gas/everything
            O = np.where([re.search(gas,i) for i in full_regular_names])[0]
            tO=mesh.vertices[O]
            # vertices from regular mesh of protein/cofactor
            A=np.where([re.search(metal,i) for i in regular_names])[0]
           
                
            tA=regular_mesh.vertices[A]

            t2=scipy.spatial.distance_matrix(regular_mesh.vertices,tO)
            t3=scipy.spatial.distance_matrix(regular_mesh.vertices,tA)
            iface1=np.where([(i<3).sum() for i in t2])[0] # protein res near gas
            iface2=np.where([(i<6).sum() for i in t3])[0] # protein res near cofactor and iface3
            iface2_5=np.where([(i<10).sum() for i in t3])[0] # iface1 and iface2_5
            iface3=np.where([(i<2).sum() for i in t2])[0] # protein res VERY near gas

            np.save(masif_opts['ply_chain_dir']+pdb_id+'_fullnames.npy',
                    {'names1':np.array(full_regular_names)})
            # Find the vertices that are in the iface.
            v3 = mesh.vertices
            # full_normal = compute_normal(full_regular_mesh.vertices, full_regular_mesh.faces)
            full_normal = compute_normal(mesh.vertices, mesh.faces)

        
            iface[iface1] = 1.0
            iface[np.intersect1d(iface1, iface2_5)] = 1.5
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

        shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
        os.remove(out_filename1+'.ply')
        os.remove(out_filename1+'.log')
        os.remove(out_filename1+'.csv')
        os.remove(out_filename1+'.pdb')
        os.remove(tmp_dir+"/"+pdb_id+".pdb")
        return "Success %s" % pdb_id
    except Exception as e:
        print("error", e, pdb_id)
        return "error %s" % pdb_id


if __name__ == "__main__":
    # python get_close_pdbs.py 2 3 O2IF 2
    sys.path.insert(0, "/data/pompei/bw973/Oxygenases/masif/source")
    from default_config.masif_opts import masif_opts
    params = masif_opts['ligand']
    params['training_list']="pdbs_p.txt"
    if not os.path.exists(masif_opts['ply_chain_dir']):
        os.makedirs(masif_opts['ply_chain_dir'])


    # python surface_protein.py N2 Fe2 5    
    if (len(sys.argv)>1):
        gas=sys.argv[1]
        metal=sys.argv[2]
    else:
        gas='O2I'
        metal='Fe2'
    processes = int(sys.argv[3])
    if (len(sys.argv)>5):
        params['training_list']=sys.argv[4]
        overwrite=int(sys.argv[5])
    else:
        params['training_list']=params['training_list']

    pdb_ids = [x.rstrip() for x in open(params['training_list']).readlines()]
    # pdb_ids = pdb_ids[0:10]
    print(pdb_ids)
    print("gas/metal", gas, metal, params['training_list'])

    

    pool = Pool(processes=processes)
    _ = pool.map(partial(surface, gas=gas, metal=metal, overwrite=overwrite, mesh_res=masif_opts['mesh_res']), pdb_ids)
    print(_)