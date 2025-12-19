import torch_geometric.transforms as T
import mdtraj as md
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
import os
import sys

# t = './an%d.xtc' % (trjnum)
t = './cat_fit.dcd'
top = './an%d.gro' % (1)
traj = md.load(t, top=top)

import torch

xyz = torch.tensor(traj.xyz) * 10


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "Fe": 5} # all 0's will be padding for
ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "VS": {"FE": 5, "MG":6}} # all  gases not within 3.5 angstroms of any protein atom
gas='O2QD'
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
anames
anums = [ele2num[a] if a != 'VS' else ele2num[a][metal] for a in anames ]

rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
rnames2 = np.array([traj.topology.atom(ind).residue for ind in nogas])

cat = np.where(anames=='VS')[0]
cat
rs = int(len(residue_ref)/len(residue_sel_un))
device = 'cpu'
protein_coords_traj = xyz[:,nogas,:]#.to(device)

rdict = {}
i=0
for r in np.unique(rnames):
    rdict[r]=i
    i+=1
rnums = [rdict[a] for a in rnames ]
rdict['Gas']=23

types_array_atom = torch.zeros((len(nogas)+len(gas2), (len(ele2num))))
for i, t in enumerate(anums):
    types_array_atom[i,t] = 1.0
# types_array_atom.shape
types_array_atom[-len(gas2):,ele2num['O']] = 1
types_array_atom = types_array_atom.to(device)

types_array_res = torch.zeros((len(nogas)+len(gas2), (len(rdict))))
for i, t in enumerate(rnums):
    types_array_res[i,t] = 1.0
types_array_res.shape
types_array_res[-len(gas2):,rdict['Gas']] = 1
types_array_res = types_array_res.to(device)

types_array = torch.cat([types_array_atom,types_array_res], dim=1).to(device)
types_array.shape
protein_coords_traj = xyz[:,nogas,:].to(device)

import os
import numpy as np

cutoff=int(sys.argv[1])


if os.path.exists('./whr_list%d.pt'%cutoff):
    whr_list=torch.load('./whr_list%d.pt'%cutoff) 
else:
    exit() 

dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)
whr2=torch.where(torch.tensor([len(i)>0 for i in whr_list]))[0]


from torch_geometric.data import Data

def extract_point_cloud(atom_matrix, positions, center):
    """
    Formats the already-cropped atom features and positions into a Data object.
    """
    if positions.shape[0] == 0:
        return None  # Skip empty cubes

    # Center coordinates relative to cube center
    centered_positions = positions - center

    return Data(x=atom_matrix, pos=centered_positions)

# atom_matrix_arr = []
# positions_arr = []
names = []
pos_embedding = []
# tensor = torch.zeros(n, 20, 20, 20, 6)


j=0
# for c in range(40000,60000,1):
for c in whr2:
# for c in range(1,len(traj),1):
# for c in range(0,1005):
    if c % 1000 == 0:  # Save when c is around a multiple of 1000
        print(f"SAVE at c={c}", flush=True)
    mean_gas=xyz[c][gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
    atom_coords = protein_coords_traj[c]
    for i in whr_list[c]:
        name="frame_%d_gas_%d" % (c+1, i+577)
        
        names.append(name)
        center = mean_gas[i]
        if c >= 31600 and c <= 31700:
            print(center, name)
        
        # Half-size of the cube
        half = 5.0

        # Define cube boundaries
        min_corner = center - half
        max_corner = center + half

        # `atom_coords` is your (n_atoms, 3) array of positions
        # e.g., from MDTraj: atom_coords = mol.xyz[0] * 10  # assuming single frame and Angstroms

        # Find indices of atoms inside the cube
        inside_indices = torch.where(
            torch.all((atom_coords >= min_corner) & (atom_coords <= max_corner), axis=1)
        )[0]

        atom_matrix=types_array_atom[inside_indices]
        positions=atom_coords[inside_indices]
        local_pos_normalized = (positions - center)/5  # shape [N, 3]
        
        # atom_matrix_arr.append(atom_matrix)
        # positions_arr.append(positions)
        test = extract_point_cloud(atom_matrix, positions, center)
        test.name = name
        test.node_attr = local_pos_normalized
        pos_embedding.append(test)
        # tensor[j] = test
        # j+=1

print(len(pos_embedding), flush=True)
torch.save(pos_embedding, "./pos_SE3embedding_big_%d.pt" % cutoff)

# for i in {1..20}; do cp embed_posSE3.py ./Sim$i; cd ./Sim$i; python embedSE3.py 10 > pos.log; cd ..; done