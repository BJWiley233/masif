

import torch_geometric.transforms as T
import mdtraj as md
import os
import torch 
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv, knn_graph, Linear
from torch.optim import SGD, Adam, Optimizer
import math
from torch.nn.init import kaiming_uniform_
from torch_geometric.transforms import ToDevice
import sys
import pandas as pd

# python contacts_df.py cat_fit.dcd an1.gro contacts 100 O2QD 4
t = sys.argv[1]
top = sys.argv[2]
traj = md.load(t, top=top)

contacts_folder = sys.argv[3]
contacts_folder = contacts_folder + '/*.npy'
num_mols = sys.argv[4]
# cmds

xyz = torch.tensor(traj.xyz) * 10

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "VS": {"FE": 5, "MG":6}} # all 0's will be padding for gases not within 3.5 angstroms of any protein atom
gas=sys.argv[5]
Sim=sys.argv[6]
metal='FE'
gas2 = traj.topology.select('resname %s' % gas)
residue_ref = np.array([traj.topology.atom(ind).residue.resSeq for ind in gas2])
FE = traj.topology.select('resname Fe2p')
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un
nogas = np.setdiff1d(range(0,traj.xyz.shape[1]),gas2)
rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
anames2=np.array([traj.topology.atom(i).name for i in nogas])
anums = [ele2num[a] if a != 'VS' else ele2num[a][metal] for a in anames ]

rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
rnames2 = np.array([traj.topology.atom(ind).residue for ind in nogas])
pCA = traj.topology.select('protein and name CA or (resname ACE and name C)')
protein_ids = np.array([str(traj.topology.atom(ind).residue) for ind in pCA])
protein_ids.shape
cat = np.where(anames=='VS')[0]


# device = torch.device("cuda")
device = torch.device("cpu")
protein_coords_traj = xyz[:,nogas,:]
rs = int(len(residue_ref)/len(residue_sel_un))
dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)
d=torch.cdist(dioxygen_coords_ave, xyz[:,cat,:])
residue_ref_single = np.unique(residue_ref)

import glob

files = glob.glob(contacts_folder)
files.sort()
print(files)
print([file.split("/")[-1] for file in files])

def ret_data(file):
    print(file)
    # Sim = file.split('/')[8][3:]
    fn = file.split("/")[-1]
    
    gasid = int(fn.split("_")[1])
    Gas_Resid = residue_ref_single[gasid]
    start = int(fn.split("_")[3].split('-')[0])
    end = int(fn.split("_")[3].split('-')[1])
    tin = (np.where(d[start:end,gasid] < 6)[0]*10/1000)[0]
    t_total_frames = (end - start)
    ttotal = (end - start)*10/1000
    co = np.load(file, allow_pickle=True)

    m = d[start:end,gasid].min().item()
    tmin_index = d[start:end,gasid].argmin().item()
    tmin = tmin_index*10/1000
    tout = ttotal - tmin
    tout_frames = t_total_frames - tmin_index
    frame_contacts = np.array([i.sum() >= 1 for i in co])
    # contact_cs = np.unique(np.concatenate([protein_ids[(i > 0)[0]] for i in co[frame_contacts]]), return_counts=True)
    co_in = co[0:tmin_index]
    co_out = co[tmin_index:]

    prop_in = ((co[0:tmin_index,0,:]>=1).sum(axis=0)/(tmin_index))
    prop_out = ((co[tmin_index:,0,:]>=1).sum(axis=0)/(t_total_frames-tmin_index))

    frame_contacts_in = np.array([i.sum() >= 1 for i in co_in])
    contact_in = np.unique(np.concatenate([protein_ids[(i > 0)[0]] for i in co_in[frame_contacts_in]]), return_counts=True)

    frame_contacts_out = np.array([i.sum() >= 1 for i in co_out])
    contact_out = np.unique(np.concatenate([protein_ids[(i > 0)[0]] for i in co_out[frame_contacts_out]]), return_counts=True)
    contact_out

    df_left_in_p = pd.DataFrame({
        'FF': ['O2QD'],
        'molecules': num_mols,
        'Sim': Sim,
        'Gas_Resid': Gas_Resid,
        'time_in': tmin,
        'in_row': 1,
        'time_out': tout,
        'min_dist': m,
        'range': f'{start}-{end}',
        'tot_time': ttotal,
        'prop': 'prop'
    })
    df_right_in_p = pd.DataFrame({
        'residues': protein_ids,
        'counts': prop_in
    })

    df_right_in_p = df_right_in_p[['counts']]
    df_right_in_p = df_right_in_p.T
    df_right_in_p.columns = protein_ids  # Set first row as column names
    df_right_in_p=df_right_in_p.reset_index(drop=True)
    df_in_p = pd.concat([df_left_in_p, df_right_in_p], axis=1)


    df_left_in_c = pd.DataFrame({
        'FF': ['O2QD'],
        'molecules': num_mols,
        'Sim': Sim,
        'Gas_Resid': Gas_Resid,
        'time_in': tmin,
        'in_row': 1,
        'time_out': tout,
        'min_dist': m,
        'range': f'{start}-{end}',
        'tot_time': ttotal,
        'prop': 'counts'
    })
    df_right_in_c = pd.DataFrame({
        'residues': protein_ids,
        'counts': prop_in*tmin_index
    })

    df_right_in_c = df_right_in_c[['counts']]
    df_right_in_c = df_right_in_c.T
    df_right_in_c.columns = protein_ids  # Set first row as column names
    df_right_in_c=df_right_in_c.reset_index(drop=True)
    df_in_c = pd.concat([df_left_in_c, df_right_in_c], axis=1)


    df_left_out_p = pd.DataFrame({
        'FF': ['O2QD'],
        'molecules': num_mols,
        'Sim': Sim,
        'Gas_Resid': Gas_Resid,
        'time_in': tmin,
        'in_row': 0,
        'time_out': tout,
        'min_dist': m,
        'range': f'{start}-{end}',
        'tot_time': ttotal,
        'prop': 'prop'
    })
    df_right_out_p = pd.DataFrame({
        'residues': protein_ids,
        'counts': prop_out
    })

    df_right_out_p = df_right_out_p[['counts']]
    df_right_out_p = df_right_out_p.T
    df_right_out_p.columns = protein_ids  # Set first row as column names
    df_right_out_p=df_right_out_p.reset_index(drop=True)
    df_out_p = pd.concat([df_left_out_p, df_right_out_p], axis=1)


    df_left_out_c = pd.DataFrame({
        'FF': ['O2QD'],
        'molecules': num_mols,
        'Sim': Sim,
        'Gas_Resid': Gas_Resid,
        'time_in': tmin,
        'in_row': 0,
        'time_out': tout,
        'min_dist': m,
        'range': f'{start}-{end}',
        'tot_time': ttotal,
        'prop': 'counts'
    })
    df_right_out_c = pd.DataFrame({
        'residues': protein_ids,
        'counts': prop_out*tout_frames
    })

    df_right_out_c = df_right_out_c[['counts']]
    df_right_out_c = df_right_out_c.T
    df_right_out_c.columns = protein_ids  # Set first row as column names
    df_right_out_c=df_right_out_c.reset_index(drop=True)
    df_out_c = pd.concat([df_left_out_c, df_right_out_c], axis=1)


    final_df = pd.concat([df_in_p, df_in_c, df_out_p, df_out_c])

    return final_df


df_orig = pd.read_csv('contacts.csv')
df = pd.concat([ret_data(file) for file in files])

df.sort_values(
    by=['Gas_Resid', 'in_row', 'range'],
    ascending=[True, False, True],
    inplace=True  # optional: if you want to modify df directly
)
df_orig.sort_values(
    by=['Gas_Resid', 'in_row', 'range'],
    ascending=[True, False, True],
    inplace=True  # optional: if you want to modify df directly
)

if (df_orig[['Gas_Resid','in_row','tot_time','prop']].values == df[['Gas_Resid','in_row','tot_time','prop']].values).all():
    print("GOOOOOD SAME ORDER   ###################")
    df.insert(4, 'PATH', df_orig.PATH.values)
    df.to_csv('contacts_test.csv', index=False)