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
import scipy
import sys

# python contacts.py an1.gro cat_fit.dcd O2IF
top = sys.argv[1]
t = sys.argv[2]
traj = md.load(t, top=top)



# Example distance array (Replace this with your actual data)
def get_ra(distances):
    # distances = np.array()  # Your 1D distance array
    indices = np.arange(len(distances))

    # Step 1: Find indices where distance < 6.0
    below_6_indices = np.where(distances < 6.0)[0]

    # Step 2: Find nearest points before and after where distance > 20.0
    above_20_indices = np.where(distances > 20.0)[0]

    ranges = []

    for idx in below_6_indices:
        # Find the nearest "above 20" before and after
        before_idx = above_20_indices[above_20_indices < idx]
        after_idx = above_20_indices[above_20_indices > idx]

        before = before_idx[-1] if len(before_idx) > 0 else None
        after = after_idx[0] if len(after_idx) > 0 else None

        if before is not None and after is not None:
            ranges.append((before, after))

    # Step 3: Merge overlapping ranges
    merged_ranges = []
    for start, end in sorted(ranges):
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append((start, end))
        else:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))

    # Output merged ranges
    return merged_ranges
    # print("Final Ranges:", merged_ranges)

def cmap_per_residue (traj,sel_list,ref_list,cutoff=3.5,metric='euclidean'):
    if len(ref_list)==0 or len(sel_list)==0:
        print("One of the selections is empty")
        return []
    coord1 = traj.xyz[:,sel_list,:]*10.0 # sel_yeo
    coord2 = traj.xyz[:,ref_list,:]*10.0 # sel_POPC
    residue_sel = np.array([traj.topology.atom(ind).residue.index for ind in sel_list]) # sel_yeo
    residue_ref = np.array([traj.topology.atom(ind).residue.index for ind in ref_list])
    residue_sel_un = np.unique(residue_sel) # sel_yeo
    residue_ref_un = np.unique(residue_ref)
    dic_sel={t:c  for c,t in enumerate(residue_sel_un)} # sel_yeo
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


xyz = torch.tensor(traj.xyz) * 10

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "VS": {"FE": 5, "MG":6}} # all 0's will be padding for gases not within 3.5 angstroms of any protein atom
gas=sys.argv[3]
metal='FE'
gas2 = traj.topology.select('resname %s' % gas)
residue_ref = np.array([traj.topology.atom(ind).residue.resSeq for ind in gas2])
residue_ref_single = np.unique(residue_ref)
FE = traj.topology.select('resname Fe2p')
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un
nogas = np.setdiff1d(range(0,traj.xyz.shape[1]),gas2)
rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
anames2=np.array([traj.topology.atom(i).name for i in nogas])
anames
anums = [ele2num[a] if a != 'VS' else ele2num[a][metal] for a in anames ]

rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
rnames2 = np.array([traj.topology.atom(ind).residue for ind in nogas])

cat = np.where(anames=='VS')[0]
cat

# device = torch.device("cuda")
device = torch.device("cpu")
protein_coords_traj = xyz[:,nogas,:]
rs = int(len(residue_ref)/len(residue_sel_un))


dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)
d=torch.cdist(dioxygen_coords_ave, xyz[:,cat,:])
diox = np.where(d < 6.0)[1]
frames = np.where(d < 6.0)[0]

print(np.unique(diox, return_counts=True))

gases = np.unique(diox)

protein = traj.topology.select('protein')
pCA = traj.topology.select('protein and name CA')
protein_ids = np.array([traj.topology.atom(ind).residue for ind in pCA])


for gasid in gases:
    contacts_array = []
    merged_ranges = get_ra(d[:,gasid])
    indvg = traj.topology.select('resname %s and resSeq %d' % (gas,residue_ref_single[gasid]))
    for r in merged_ranges:

        print(gasid,range(r[0], r[1]+1))
        take=traj.slice(range(r[0], r[1]+1))
        contacts = cmap_per_residue(take, indvg, protein)
        np.save('contacts/gas_%d_range_%d-%d_contacts.npy' % (gasid, r[0], r[1]+1), contacts[2])
