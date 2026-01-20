import torch    
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
from e3nn.io import CartesianTensor
import math
import pandas as pd 
import MDAnalysis as mda

device='cpu'
top ='equil5.gro'
t = "fit2.dcd"
# traj = md.load(t, top=top)
u = mda.Universe(top, t)
print("traj loaded", flush=True)
from rdkit import Chem
mol = Chem.MolFromPDBFile("../Sim9/an1_water.pdb", removeHs=False)
Chem.SanitizeMol(mol)

def build_complete_edge_index(N, device):
    idx = torch.arange(N, device=device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")

    mask = i != j  # no self-loops
    edge_index = torch.stack([i[mask], j[mask]], dim=0)
    return edge_index


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

traj = md.load("cat_fit.dcd", top='an1.gro')
xyz = torch.tensor(traj.xyz) * 10

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "Fe": 5, "Mg":6, "Na":7, "Cl":8}
t = "../Sim9/an1_water.pdb"
traj = md.load_frame(t, index=0,top=top)
gas='O2IF'
metal='FE'
gas2 = traj.topology.select('resname %s' % gas)
residue_ref = np.array([traj.topology.atom(ind).residue.resSeq for ind in gas2])
FE = traj.topology.select('resname Fe2p')
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un

print('LEN FE', FE)
residue_sel_un = np.unique(residue_ref) # gas
residue_sel_un
nogas = np.setdiff1d(range(0,traj.xyz.shape[1]),gas2)
rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
anames2=np.array([traj.topology.atom(i).name for i in nogas])
anums = [ele2num[a] if a != 'VS' else ele2num[a][metal] for a in anames]

rnames = np.array([traj.topology.atom(ind).residue.name for ind in nogas])
rindex = np.array([traj.topology.atom(ind).residue.resSeq for ind in nogas])
anames = np.array([traj.topology.atom(ind).element.symbol for ind in nogas])
rnames2 = np.array([traj.topology.atom(ind).residue for ind in nogas])

cat = np.where(anames=='Fe')[0]
print(cat)

device = torch.device("cpu")
# device = torch.device("cpu")
# protein_coords_traj = xyz[:,nogas,:]
rs = int(len(residue_ref)/len(residue_sel_un))
gas_atoms=gas2.reshape(len(residue_sel_un),rs)

dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)
d=torch.cdist(dioxygen_coords_ave, xyz[:,cat,:])
# diox = np.where(d < 6.0)[1]
# frames = np.where(d < 6.0)[0]

types_array_atom = torch.zeros((len(nogas)+len(gas2), (len(ele2num))))
for i, t in enumerate(anums):
    types_array_atom[i,t] = 1.0
# types_array_atom.shape
types_array_atom[-len(gas2):,ele2num['O']] = 1
types_array_atom = types_array_atom.to(device)

path_dict = {
    'P1':'P1', 
    'P_main (PmR)':'PmR', 
    'P_main (mid)':'mid', 
    'P_reverse':'P_reverse', 
    'P_main (PmL)':'PmL',
       'P3':'P3'
}

# DO THIS ONCE
global_bonds = []  # (a1, a2, order)

for bond in mol.GetBonds():
    a1 = bond.GetBeginAtomIdx()
    a2 = bond.GetEndAtomIdx()

    if a1 in gas_atoms or a2 in gas_atoms:
        continue

    bt = bond.GetBondType()
    order = (
        1 if bt == Chem.rdchem.BondType.SINGLE else
        2 if bt == Chem.rdchem.BondType.DOUBLE else
        3 if bt == Chem.rdchem.BondType.AROMATIC else
        1
    )

    global_bonds.append((a1, a2, order))
    global_bonds.append((a2, a1, order))

# gas bonds
for a1, a2 in gas_atoms:
    global_bonds.append((a1, a2, 1))
    global_bonds.append((a2, a1, 1))



predicted_paths = pd.read_csv("/home/bw973/Documents/PHD2/full_contacts4.csv")
df = predicted_paths[(predicted_paths.FF=='O2IF') & (predicted_paths.Sim==4) & (predicted_paths.prop=='counts') & (predicted_paths.molecules==100)]
# df.iloc[:,[0,1,2,3,4,5,6,7,8,254]]

time=0
for i, row in df.iloc[0:,].iterrows():
    gas_resid=row['Gas_Resid']
    gas_idx = np.where(residue_sel_un==gas_resid)[0][0]
    protein_cofactors = np.setdiff1d(np.arange(0,xyz.shape[1]), gas_atoms[gas_idx])
    print(protein_cofactors.shape)
    
    start = int(row['range'].split('-')[0])
    end = int(row['range'].split('-')[1])
    time += (end-start+1)

time


path_data = {
    'P1':[], 
    'P_main (PmR)':[], 
    'P_main (mid)':[], 
    'P_reverse':[], 
    'P_main (PmL)':[],
       'P3':[]
}

# Map from global RDKit atom idx → local 0..N-1
import itertools

# def build_edges_and_attrs(inside_indices):
#     atom_to_local = {int(a): i for i, a in enumerate(inside_indices.tolist())}

#         # --- Extract bonded edges from RDKit ---
#     bonded_edges = []
#     bonded_attrs = []

#     for bond in mol.GetBonds():
#         a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

#         # Only keep if both atoms are in the subset
#         if a1 in atom_to_local and a2 in atom_to_local and a1 not in gas_atoms and a2 not in gas_atoms:
#             i1, i2 = atom_to_local[a1], atom_to_local[a2]

#             bt = bond.GetBondType()
#             if bt == Chem.rdchem.BondType.SINGLE:
#                 order = 1
#             elif bt == Chem.rdchem.BondType.DOUBLE:
#                 order = 2
#             elif bt == Chem.rdchem.BondType.AROMATIC:
#                 order = 3
#             else:
#                 order = 0

#             # undirected edges
#             bonded_edges += [[i1, i2], [i2, i1]]
#             bonded_attrs += [order, order]

#     for (a1,a2) in gas_atoms:
#         if a1 in atom_to_local and a2 in atom_to_local:
#             i1, i2 = atom_to_local[a1], atom_to_local[a2]
#             bonded_edges += [[i1, i2], [i2, i1]]
#             bonded_attrs += [1, 1]

#     # --- Build nonbonded edges among all pairs in subset ---
#     N = len(inside_indices)
#     all_pairs = list(itertools.combinations(range(N), 2))

#     bonded_set = set(tuple(sorted(e)) for e in [(a, b) for a, b in bonded_edges if a < b])

#     nonbonded_edges = []
#     nonbonded_attrs = []

#     for i, j in all_pairs:
#         if (i, j) not in bonded_set:
#             nonbonded_edges += [[i, j], [j, i]]
#             nonbonded_attrs += [0, 0]  # edge_attr = 0 for nonbonded

#     # --- Combine everything ---
#     # edge_index = torch.tensor(bonded_edges + nonbonded_edges, dtype=torch.long).T
#     edges = bonded_edges + nonbonded_edges  # list of [i, j] pairs
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(bonded_attrs + nonbonded_attrs, dtype=torch.long)

#     return edge_index.to(device), edge_attr.unsqueeze(1).to(device)

def build_edges_and_attrs_fast(inside_indices, edge_index_cache):
    device = inside_indices.device
    N = inside_indices.shape[0]

    # global → local
    atom_to_local = {int(a): i for i, a in enumerate(inside_indices.tolist())}

    # edge_index is reused
    edge_index = edge_index_cache

    # initialize all nonbonded = 0
    edge_attr = torch.zeros(
        (edge_index.shape[1], 1),
        dtype=torch.long,
        device=device
    )

    # overwrite bonded edges only
    # map (i,j) → edge position once
    edge_pos = {
        (int(i), int(j)): k
        for k, (i, j) in enumerate(edge_index.t().tolist())
    }

    for a1, a2, order in global_bonds:
        if a1 in atom_to_local and a2 in atom_to_local:
            i = atom_to_local[a1]
            j = atom_to_local[a2]
            edge_attr[edge_pos[(i, j)], 0] = order

    return edge_index, edge_attr


def sinusoidal_embedding(frame_idx: torch.Tensor, dim: int = 16):
    """
    frame_idx: tensor of shape [N] with normalized frame values in [0,1]
    dim: embedding dimension (should be even)
    Returns: tensor of shape [N, dim]
    """
    device = frame_idx.device
    N = frame_idx.size(0)
    pe = torch.zeros(N, dim, device=device)

    position = frame_idx.unsqueeze(1)  # [N, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))  # [dim/2]

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(device)  # [N, dim]


def embed_event_pos(path, gas_resid, gas_idx, in_row, start, end, protein_cofactors,global_to_local,direction):
    pos_embedding = []
    device='cpu'

    frames = torch.arange(start, end, dtype=torch.float32)  # [0,1,...,N-1]
    frame_norm = frames / (end - start)    
    frame_emb = sinusoidal_embedding(frame_norm.to(device), dim=16)
    print("frame_emb.shape", frame_emb.shape, flush=True)
    
    # protein_coords_traj = xyz[:,protein_cofactors,:]

    if direction=='reverse':
        range_ = range(end-1,start-1,-1)
    else:
        range_ = range(start,end)

    for idx, c in enumerate(range_):
        u.trajectory[c]
        coords = torch.tensor(u.atoms.positions.copy())
        mean_gas = coords[gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        # mean_gas=xyz[c][gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        atom_coords = coords[protein_cofactors].to(device)
        # atom_coords = protein_coords_traj[c].to(device)
        
        name="frame_%d_gas_%d_sequence_%d_path_%s_in_%d_pos" % (c, gas_resid, idx, path, in_row)
        
        center = mean_gas[gas_idx]
        dist_FE = torch.norm(center - atom_coords[cat[0],:])
    

        radius = 6.0

        # Compute distances of all atoms to perturbed point
        dists = torch.norm(atom_coords - center, dim=1)

        # Indices of atoms inside sphere
        inside_indices = protein_cofactors[torch.where(dists <= radius)[0]]

        # Ensure inside_indices is always a 1D tensor
        if torch.is_tensor(inside_indices) and inside_indices.ndim == 0:
            inside_indices = inside_indices.unsqueeze(0)

        if isinstance(inside_indices, (int, np.integer)):
            inside_indices = torch.tensor([inside_indices], device=device)

        if inside_indices.numel() == 0:
            dddd = torch.cdist(center.unsqueeze(0), atom_coords)
            print('too far start or end', dddd.min(), flush=True)
            continue

        local_inside_indices = np.array([global_to_local[int(g)] for g in inside_indices])
        # print(local_inside_indices)
        # print(inside_indices)
        
        print("\tpos sequence %d of %d; frame %d" % (idx,(end-start), c), "len indices:", len(local_inside_indices), flush=True)

        atom_matrix=types_array_atom[local_inside_indices]
        positions=atom_coords[local_inside_indices]
        node_directions = positions - atom_coords[cat[0],:]
        node_distances = torch.norm(node_directions, dim=1)
        local_pos_normalized = (positions - center)/6  # shape [N, 3]
        
        # atom_matrix_arr.append(atom_matrix)
        # positions_arr.append(positions)
        test = extract_point_cloud(atom_matrix, positions, center)
        test.name = name
        # test.x=torch.column_stack([test.x, torch.tensor(frame_emb[idx]).repeat(len(test.x))])
        frame_vector = frame_emb[idx]          # [16]
        frame_vector = frame_vector.unsqueeze(0)           # [1, 16]
        frame_vector = frame_vector.expand(len(test.x), -1)  # [N, 16]
        test.x = torch.cat([test.x, frame_vector], dim=1)    # [N, 6 + 16]
        
        N=len(inside_indices)
        edge_index_cache = build_complete_edge_index(N, 'cpu')

        test.edge_index, test.edge_attr = build_edges_and_attrs_fast(inside_indices, edge_index_cache)
        test.center = center
        test.node_attr = local_pos_normalized
        test.node_directions = node_directions
        test.node_distances = node_distances
        test.inside_indices=inside_indices
        test.local_inside_indices=local_inside_indices
        test.distance = dist_FE.expand(len(inside_indices))
        pos_embedding.append(test)
    
    return pos_embedding

def embed_event_neg(path, gas_resid, gas_idx, in_row, start, end, protein_cofactors, global_to_local, direction):
    neg_embedding = []
    device='cpu'
    frames = torch.arange(start, end, dtype=torch.float32)  # [0,1,...,N-1]
    frame_norm = frames / (end - start)    
    frame_emb = sinusoidal_embedding(frame_norm.to(device), dim=16)
    # protein_coords_traj = xyz[:,protein_cofactors,:]

    if direction=='reverse':
        range_ = range(end-1,start-1,-1)
    else:
        range_ = range(start,end)

    for idx, c in enumerate(range_):
        u.trajectory[c]
        coords = torch.tensor(u.atoms.positions.copy())
        mean_gas = coords[gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        atom_coords = coords[protein_cofactors].to(device)   
        # mean_gas=xyz[c][gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1).to(device)
        # atom_coords = protein_coords_traj[c].to(device)
        name="frame_%d_gas_%d_sequence_%d_path_%s_in_%d_neg" % (c, gas_resid, idx, path, in_row)
        center = mean_gas[gas_idx]

        min_dist = 2.0
        max_dist = 4.0
        
        for w in range(4):
            tried=0
            while True:
                tried+=1
                if tried%5000==0:
                    print("tried", tried, flush=True)
                # random unit direction
                direction = torch.randn(3, device=atom_coords.device)
                direction = direction / direction.norm()

                # random step
                if tried < 5000:
                    step = torch.empty(1, device=atom_coords.device).uniform_(0.1, 1.)
                elif tried < 10000:
                    step = torch.empty(1, device=atom_coords.device).uniform_(0.1, 1.)
                elif tried > 25000:
                    break

                # candidate
                perturbed = center + direction * step

                # distances to all atoms
                dists = (atom_coords - perturbed).norm(dim=1)

                if torch.all(dists >= min_dist) and torch.any(dists <= max_dist):
                    break
            
            dist_FE = torch.norm(perturbed - atom_coords[cat[0],:])
            radius = 6.0

            # Compute distances of all atoms to perturbed point
            dists = torch.norm(atom_coords - perturbed, dim=1)
            
            # Indices of atoms inside sphere
            inside_indices = protein_cofactors[torch.where(dists <= radius)[0]]

            # Ensure inside_indices is always a 1D tensor
            if torch.is_tensor(inside_indices) and inside_indices.ndim == 0:
                inside_indices = inside_indices.unsqueeze(0)

            if isinstance(inside_indices, (int, np.integer)):
                inside_indices = torch.tensor([inside_indices], device=device)

            if inside_indices.numel() == 0:
                dddd = torch.cdist(center.unsqueeze(0), atom_coords)
                print('too far start or end', dddd.min(), flush=True)
                continue
            
            local_inside_indices = np.array([global_to_local[int(g)] for g in inside_indices])

            print("\tneg sequence %d of %d; frame %d" % (idx,(end-start), c), "len indices:", len(local_inside_indices), flush=True)

           

            atom_matrix=types_array_atom[local_inside_indices]
            positions=atom_coords[local_inside_indices]
            node_directions = positions - atom_coords[cat[0],:]
            node_distances = torch.norm(node_directions, dim=1)
            local_pos_normalized = (positions - perturbed)/6  # shape [N, 3]
            
            # atom_matrix_arr.append(atom_matrix)
            # positions_arr.append(positions)
            test = extract_point_cloud(atom_matrix, positions, perturbed)
            test.name = name
            # test.x=torch.column_stack([test.x, torch.tensor(frame_emb[idx]).repeat(len(test.x))])
            frame_vector = frame_emb[idx]          # [16]
            frame_vector = frame_vector.unsqueeze(0)           # [1, 16]
            frame_vector = frame_vector.expand(len(test.x), -1)  # [N, 16]
            test.x = torch.cat([test.x, frame_vector], dim=1)    # [N, 6 + 16]
            
            N=len(inside_indices)
            edge_index_cache = build_complete_edge_index(N, 'cpu')
            test.edge_index, test.edge_attr = build_edges_and_attrs_fast(inside_indices, edge_index_cache)
            test.center = perturbed
            test.perturbed_from = center
            test.node_attr = local_pos_normalized
            test.node_directions = node_directions
            test.node_distances = node_distances
            test.inside_indices=inside_indices
            test.local_inside_indices=local_inside_indices
            test.distance = dist_FE.expand(len(inside_indices))
            neg_embedding.append(test)
    
    return neg_embedding


pos_embedding_arr = []
neg_embedding_arr = []
pos_embedding_arr_r = []
neg_embedding_arr_r = []
for j, (i, row) in enumerate(df.iloc[0:,].iterrows()):
    print("ROW:", j, flush=True)
    gas_resid=row['Gas_Resid']
    gas_idx = np.where(residue_sel_un==gas_resid)[0][0]
    start = int(row['range'].split('-')[0])
    end = int(row['range'].split('-')[1])
    path = row['PATH']
    dists = d[start:end+1,gas_idx,0]
    first = torch.where(dists < 6)[0][0]
    protein_cofactors = torch.tensor(np.setdiff1d(np.arange(0,len(u.atoms)), gas_atoms[gas_idx])).to(device)
    global_to_local = {int(g): i for i, g in enumerate(protein_cofactors)}
    print(len(protein_cofactors), gas_resid, row['range'])
    # print(global_to_local)

    if row['in_row']:
        # pos_embedding = embed_event_pos(path, gas_resid, gas_idx, row['in_row'], start-2,  start+first+1, protein_cofactors, global_to_local, 'forward')
        # pos_embedding_arr += [p.cpu() for p in pos_embedding]
        neg_embedding = embed_event_neg(path, gas_resid, gas_idx, row['in_row'], start-2,  start+first+1, protein_cofactors, global_to_local, 'forward')
        neg_embedding_arr += [p.cpu() for p in neg_embedding]

    # else: 
    #     pos_embedding = embed_event_pos(path, gas_resid, gas_idx, row['in_row'], start+first+1, end+1, protein_cofactors, global_to_local, 'reverse')
    #     pos_embedding_arr_r += [p.cpu() for p in pos_embedding]
    #     neg_embedding = embed_event_neg(path, gas_resid, gas_idx, row['in_row'], start+first+1, end+1, protein_cofactors, global_to_local, 'reverse')
    #     neg_embedding_arr_r += [p.cpu() for p in neg_embedding]

# torch.save(pos_embedding_arr, "pos_embedding_inWater.pt")
torch.save(neg_embedding_arr, "posLowPeturb_embedding_inWater.pt")
# torch.save(pos_embedding_arr_r, "pos_embedding_out3.pt")
# torch.save(neg_embedding_arr_r, "neg_embedding_out3.pt")