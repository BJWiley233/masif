
# Neural network ligand application specific parameters.
import os
import sys
from multiprocessing import Pool
import subprocess as sp
import itertools
# need to preprocess to only select O2 within 3.5 angstroms of protein
import mdtraj as md
import numpy as np
import scipy
from subprocess import Popen, PIPE
import subprocess as sp
from biopandas.pdb import PandasPdb
from functools import partial

sys.path.insert(0, "/data/pompei/bw973/Oxygenases/masif/source")
from default_config.masif_opts import masif_opts
masif_opts["data_preparation"] = "data_preparation"
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_protein.ply"
masif_opts["ligand"] = {}
masif_opts["ligand"]["assembly_dir"] = "data_preparation/00b-pdbs_assembly/"
masif_opts["ligand"]["assembly_dir2"] = "data_preparation/00c-pdbs_assembly/"
masif_opts["ligand"]["ligand_coords_dir"] = "data_preparation/00c-ligand_coords/"
masif_opts["ligand"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_12A/precomputation/"
masif_opts["ligand"]["max_shape_size"] = 50
masif_opts["ligand"]["feat_mask"] = [1.0] * 5
masif_opts["ligand"]["train_fract"] = 0.9 * 0.8
masif_opts["ligand"]["val_fract"] = 0.1 * 0.8
masif_opts["ligand"]["test_fract"] = 0.2
masif_opts["ligand"]["tfrecords_dir"] = "data_preparation/tfrecords/"
masif_opts["ligand"]["max_distance"] = 5.0
masif_opts["ligand"]["n_classes"] = 7
masif_opts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["ligand"]["costfun"] = "dprime"
masif_opts["ligand"]["model_dir"] = "nn_models/all_feat/"
masif_opts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

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


def cmap_per_residue(traj,sel_list,ref_list,cutoff=3.5,metric='euclidean'):
    if len(ref_list)==0 or len(sel_list)==0:
        print("One of the selections is empty")
        return []
    coord1 = traj.xyz[:,sel_list,:]*10.0 # gas
    coord2 = traj.xyz[:,ref_list,:]*10.0 # protein
    residue_sel = np.array([traj.topology.atom(ind).residue.index for ind in sel_list]) # gas
    residue_ref = np.array([traj.topology.atom(ind).residue.index for ind in ref_list])
    residue_sel_un = np.unique(residue_sel) # gas
    residue_ref_un = np.unique(residue_ref)
    dic_sel={t:c  for c,t in enumerate(residue_sel_un)} # gas
    dic_ref={t:c  for c,t in enumerate(residue_ref_un)}
    histo = np.zeros((len(coord1),len(residue_sel_un),len(residue_ref_un)))
    for c,j in enumerate(coord1):
        axis_sel,axis_ref = np.where(scipy.spatial.distance.cdist(coord1[c],coord2[c],metric=metric) < cutoff)
        real_a= residue_sel[axis_sel]
        real_b= residue_ref[axis_ref]
        elements,counts = np.unique(list(zip(real_a,real_b)),axis=0,return_counts=True)
        for c1,el in enumerate(elements):
            ax1=dic_sel[el[0]];
            ax2=dic_ref[el[1]];
            histo[c,ax1,ax2] = counts[c1];
    return residue_sel_un,residue_ref_un,histo


nfrcond=4
lengths = []
cutoff1 = 2.25
cutoff2 = 7.87
def preprocess(trjnum, simnum=1, gas='O2IF',check=False):
    print(trjnum, simnum, gas)
    cmds = ["echo '14 0' | gmx trjconv -f run%d -s run1.tpr -o an%d.xtc -pbc mol -ur compact -center" % (trjnum,trjnum)]
    if check:
        if not os.path.exists("an%d.xtc" % trjnum):
            print("making an%d.xtc" % trjnum)
            sys.stdout.flush()
            _ = run_commands(cmds, supress=True)
            print('#############################', _)
        else:
            print("an%d.xtc exists" % trjnum)

    else:
        print("making an%d.xtc" % trjnum)
        sys.stdout.flush()
        _ = run_commands(cmds, supress=True)
    t = './an%d.xtc' % (trjnum)
    print("loading centered an%d.xtc" % trjnum)
    sys.stdout.flush()
    traj = md.load(t, top=top)
    time=np.arange(len(traj))*.01
    protein = traj.topology.select('protein')
    gas2 = traj.topology.select('resname %s' % gas)
    AKG = traj.topology.select('resname AKG')
    FE = traj.topology.select('resname Fe2p')

    test = cmap_per_residue(traj,gas2,protein,cutoff=cutoff1)
    test_akg = cmap_per_residue(traj,gas2,FE,cutoff=cutoff2)
    segments_dict = {}
    for i in range(0,50):
        res=test[2][0:,i]
        t=np.zeros(len(traj))
        res_akg=test_akg[2][0:,i]
        
        t[np.concatenate((np.where(res)[0], np.where(res_akg)[0]))] = 1

        segment=[]
        subsegment,subtime,subframe=[],[],[]
        for c,value in enumerate(t):
            if value==0 and len(subsegment)==0:
                subsegment,subtime=[],[]
            elif value==0 and len(subsegment)!=0:
                if len(subsegment) > nfrcond: 
                    segment.append(np.array([subframe,subtime,subsegment]))
                subsegment,subtime,subframe=[],[],[]
            else: 
                subsegment.append(value)
                subtime.append(time[c])
                subframe.append(c)
        if len(subsegment) > nfrcond: segment.append(np.array([subframe,subtime,subsegment]))
        segments_dict[i] = segment

    l=[]
    for k in segments_dict.keys():
        if len(segments_dict[k]) > 0:
            sl = np.concatenate([r[0] for r in segments_dict[k]])
            l.append(np.unique(sl))
    if len(l) > 0:
        frames = np.unique(np.concatenate(l).astype(int))
        fn = frames+1
        print(len(fn), "traj:", trjnum)
        lengths.append(len(fn))
    else:
        print(0)
        lengths.append(0)
        return "No frames traj%d"%trjnum
    if len(frames) < 1:
        return "No frames traj%d"%trjnum 
    np.save('lengths2.npy', lengths)
    trajf=traj[frames]
    print(np.array(frames))
    sys.stdout.flush()

    for i,f in enumerate(frames):
        trajfn = trajf[i]
        l2 = np.array([], dtype=np.int32)
        test2 = cmap_per_residue(trajfn,gas2,protein,cutoff=cutoff1)
        test2_akg = cmap_per_residue(trajfn,gas2,FE,cutoff=cutoff2)
        whr = np.where([j.sum()>0 for j in test2[2][0]])[0]*2
        whr_akg = np.where([j.sum()>0 for j in test2_akg[2][0]])[0]*2
        if len(whr_akg)>1:
            print("whr metal center:",whr_akg, "traj:", trjnum, "frame:", f+1)
        if len(whr)>1:
            print("whr protein:",whr, "traj:", trjnum, "frame:", f+1)
        l2 = np.append(l2, whr)
        l2 = np.append(l2, whr_akg)
        whr2 = np.concatenate([(n,(n+1)) for n in np.unique(l2)])
        atms = np.sort(np.concatenate((gas2[whr2],AKG,FE,protein)))
        sel = trajfn.atom_slice(atms)

        pdb="Sim_%d_traj_%d_frame_%d" % (simnum, trjnum, f+1)+'.pdb'
        sel.save(masif_opts["ligand"]["assembly_dir"]+'/'+pdb)
        
        ppdb_df = PandasPdb().read_pdb(masif_opts["ligand"]["assembly_dir"]+'/'+pdb)
        o2if = ppdb_df.df['ATOM']
        # o2if.loc[o2if.residue_name=='O2Q','atom_name'] = np.concatenate([['OQ1','OQ2','M1']] * (len(o2if.loc[o2if.residue_name=='O2Q','atom_name'])//3))
        ppdb_df._df['ATOM'] = o2if[(o2if['residue_name']!='ACE') & (o2if['residue_name']!='HOH')]
        atom = o2if[~o2if.residue_name.isin(gases_ligands)]
        hetatm = o2if[o2if.residue_name.isin(gases_ligands)]
        ppdb_df._df['ATOM'] = atom
        ppdb_df._df['HETATM'] = hetatm
        ppdb_df._df['HETATM'].loc[:,'record_name']='HETATM'
        ppdb_df.to_pdb(masif_opts["ligand"]["assembly_dir"]+'/'+pdb)

        df_atom=ppdb_df._df['ATOM']
        df=ppdb_df._df['HETATM']
        o2i_uniq = np.unique(df[df.residue_name=='O2Q'].residue_number)
        o2i_uniq = np.sort(o2i_uniq)

        lig_types = []
        akg_rn = np.unique(df_atom[df_atom.residue_name=='AKG'].residue_number)
        lig_types.append(['AKG','res_%d' % akg_rn])
        [lig_types.append([gas, 'res_%d' % i]) for i in o2i_uniq]
        np.save(masif_opts["ligand"]["ligand_coords_dir"]+pdb.split('.')[0]+'_ligtypes.npy', lig_types)

        lig_coords = {}
        lig_coords[akg_rn[0]] = df_atom[df_atom.residue_name=='AKG'][['x_coord', 'y_coord','z_coord']].values
        for val in o2i_uniq:
            lig_coords[val] = df[df.residue_number==val][['x_coord', 'y_coord','z_coord']].values

        np.save(masif_opts["ligand"]["ligand_coords_dir"]+pdb.split('.')[0]+'_ligcoords.npy', lig_coords)
    cmds = ["rm an%d.xtc" % (trjnum)]
    run_commands(cmds, supress=True)
    return "Success traj%d"%trjnum

if __name__ == "__main__":
    # python get_close_pdbs.py 2 3 N2 1
    if not os.path.exists(masif_opts["data_preparation"]):
        os.makedirs(masif_opts["data_preparation"])
    if not os.path.exists(masif_opts["ligand"]["assembly_dir"]):
        os.makedirs(masif_opts["ligand"]["assembly_dir"])
    if not os.path.exists(masif_opts["ligand"]["ligand_coords_dir"]):
        os.makedirs(masif_opts["ligand"]["ligand_coords_dir"])
        
    processes = int(sys.argv[1])
    rngs = int(sys.argv[2])
    rnge = int(sys.argv[3])
    
    gas = sys.argv[4]
    simnum=int(sys.argv[5])
    check=int(sys.argv[6])
    top = 'equil5.gro'
    gases_ligands = ['O2I','O2Q','N2','XE2','CO2','NO','CO']#, 'AKG'
    pool = Pool(processes=processes)
    _ = pool.map(partial(preprocess, simnum=simnum, gas=gas, check=check), range(rngs,rnge))
    print(_)
