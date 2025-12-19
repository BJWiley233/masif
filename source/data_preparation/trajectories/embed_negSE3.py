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
import sys
import os

cutoff = int(sys.argv[1])
if os.path.exists('./whr_list%d.pt'%cutoff):
    whr_list=torch.load('./whr_list%d.pt'%cutoff)
else:
    exit() 
whr2=torch.where(torch.tensor([len(i)>0 for i in whr_list]))[0]

# t = './an%d.xtc' % (trjnum)
t = './cat_fit.dcd'
top = './an%d.gro' % (1)
traj = md.load(t, top=top)


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

device = torch.device("cuda")
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



# if os.path.exists('../whr_list%d.npy'%20):
#     whr_list=np.load('../whr_list%d.npy'%20, allow_pickle=True) 
# n=sum([len(i) for i in whr_list])
dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)



def sample_from_range(tensor, low, high, count):
    # Find indices where values are in the desired range
    mask = (tensor >= low) & (tensor < high)
    # print('sum(mask)',sum(mask))
    
    # matching_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    matching_indices = torch.nonzero(mask, as_tuple=False).view(-1)  # Always 1D
    # print('len(matching_indices)',matching_indices)

    if matching_indices.numel() < count:
        # raise ValueError(f"Not enough values in range [{low}, {high}) to sample {count} elements.")
        return None, None

    # Shuffle and select the desired number of random indices
    selected = matching_indices[torch.randperm(len(matching_indices))[:count]]
    values = tensor[selected]
    return selected, values



# def get_neg_centers_vectorized_fast(c, cube_size=10.0, step=1.0, protein_dist_cutoff=3.5, len_get=1, max_cutoff=20):
#     ligand_coords = xyz[c][gas2].reshape(len(residue_sel_un), rs, 3).mean(dim=1).to(device)  # [num_ligands, 3]
#     system_coords = protein_coords_traj[c].to(device)  # [num_atoms, 3]

#     # Bounding box
#     min_corner = torch.floor(system_coords.min(dim=0).values)
#     max_corner = torch.ceil(system_coords.max(dim=0).values)

#     # Grid setup
#     x = torch.arange(min_corner[0], max_corner[0] - cube_size + 1, step, device=device)
#     y = torch.arange(min_corner[1], max_corner[1] - cube_size + 1, step, device=device)
#     z = torch.arange(min_corner[2], max_corner[2] - cube_size + 1, step, device=device)

#     cube_mins = torch.cartesian_prod(x, y, z)               # [num_cubes, 3]
#     cube_centers = cube_mins + (cube_size / 2.0)            # [num_cubes, 3]
#     cube_maxs = cube_mins + cube_size

#     # Check if any ligand is inside the cube
#     in_cube = ((ligand_coords[None, :, :] >= cube_mins[:, None, :]) &
#                (ligand_coords[None, :, :] < cube_maxs[:, None, :])).all(dim=2)  # [num_cubes, num_ligands]
#     cube_mask = ~in_cube.any(dim=1)  # [num_cubes]
#     empty_cube_centers = cube_centers[cube_mask]

#     # --- Vectorized protein distance check ---
#     # Compute distance from all empty_cube_centers to all protein atoms
#     diff = empty_cube_centers[:, None, :] - system_coords[None, :, :]  # [Ncubes, Natoms, 3]
#     dists = torch.norm(diff, dim=2)  # [Ncubes, Natoms]
#     near_protein_mask = (dists < protein_dist_cutoff).any(dim=1)  # [Ncubes]
#     valid_cubes = empty_cube_centers[near_protein_mask]  # [Nvalid, 3]
    


#     if valid_cubes.shape[0] == 0:
#         raise ValueError("No valid cubes found near protein for frame", c)

#     # ---- Sample based on distance to last protein atom ----
#     ref_atom = system_coords[-1]  # [3]
#     dists = torch.norm(valid_cubes - ref_atom, dim=1)  # [Nvalid]

#     if len_get > 1:
#         idx_5_10, val_5_10 = sample_from_range(dists, 3, 6, 1)
#         idx_10_15, val_10_15 = sample_from_range(dists, 6, max_cutoff+1, len_get-1)
#         all_idx = torch.cat([idx_5_10, idx_10_15])
#     else:
#         idx_15_20, val_15_20 = sample_from_range(dists, 4, max_cutoff+1, len_get)
#         all_idx = idx_15_20

    
#     return valid_cubes[all_idx]

def get_neg_centers_vectorized_fast(c, cube_size=10.0, step=1.0,
                                    protein_dist_cutoff=3.5, len_get=1, max_cutoff=20):
    ligand_coords = xyz[c][gas2].reshape(len(residue_sel_un), rs, 3).mean(dim=1).to(device)  # [num_ligands, 3]
    system_coords = protein_coords_traj[c].to(device)  # [num_atoms, 3]

    # Bounding box
    min_corner = torch.floor(system_coords.min(dim=0).values)
    max_corner = torch.ceil(system_coords.max(dim=0).values)

    # Grid setup
    x = torch.arange(min_corner[0], max_corner[0] - cube_size + 1, step, device=device)
    y = torch.arange(min_corner[1], max_corner[1] - cube_size + 1, step, device=device)
    z = torch.arange(min_corner[2], max_corner[2] - cube_size + 1, step, device=device)

    cube_mins = torch.cartesian_prod(x, y, z)               # [num_cubes, 3]
    cube_centers = cube_mins + (cube_size / 2.0)            # [num_cubes, 3]
    cube_maxs = cube_mins + cube_size

    # Exclude cubes containing ligand atoms
    in_cube = ((ligand_coords[None, :, :] >= cube_mins[:, None, :]) &
               (ligand_coords[None, :, :] < cube_maxs[:, None, :])).all(dim=2)  # [num_cubes, num_ligands]
    cube_mask = ~in_cube.any(dim=1)  # [num_cubes]
    empty_cube_centers = cube_centers[cube_mask]

    # --- Distance check ---
    diff = empty_cube_centers[:, None, :] - system_coords[None, :, :]  # [Ncubes, Natoms, 3]
    dists = torch.norm(diff, dim=2)  # [Ncubes, Natoms]

    # 1. Must be within protein_dist_cutoff of any atom
    near_protein_mask = (dists < protein_dist_cutoff).any(dim=1)  # [Ncubes]

    # 2. Closest atom must be within [2.25, 3.5]
    min_dist = dists.min(dim=1).values  # [Ncubes]
    shell_mask = (min_dist >= 2.25) & (min_dist <= 3.25)

    # Combine masks
    final_mask = near_protein_mask & shell_mask
    filtered_cubes = empty_cube_centers[final_mask]  # [Nvalid, 3]
    filtered_dists = min_dist[final_mask] 

    if filtered_cubes.shape[0] == 0:
        raise ValueError("No valid cubes found near protein for frame", c)

    mu = 3.5
    sigma = 0.12  # Smaller = sharper peak; tune as needed
    weights = torch.exp(-0.5 * ((filtered_dists - mu) / sigma) ** 2)
    weights /= weights.sum()  # Normalize to get probabilities

    sample_count = min(5000, len(weights))
    sample_indices = torch.multinomial(weights, sample_count, replacement=False)
    valid_cubes = filtered_cubes[sample_indices]

    # ---- Sample based on distance to reference atom ----
    ref_atom = system_coords[-1]  # [3]
    dists_to_ref = torch.norm(valid_cubes - ref_atom, dim=1)  # [Nvalid]
    # print("##",len(valid_cubes), len(dists_to_ref))

    if len_get > 1:
        # print('Here1',c.item())
        idx_5_10, _ = sample_from_range(dists_to_ref, 3, 6, 1)
        if idx_5_10 is None:
            # print('Here2',c.item())
            idx_10_15, _ = sample_from_range(dists_to_ref, 6, max_cutoff + 1, len_get)
            all_idx = idx_10_15
            if all_idx is None:
                print('Here3',c.item())
                return None
        else:
            # print('Here2 else',c.item(), idx_5_10)
            idx_10_15, _ = sample_from_range(dists_to_ref, 6, max_cutoff + 1, len_get - 1)#
            # print(idx_10_15, dists_to_ref)
            all_idx = torch.cat([idx_5_10, idx_10_15])
    else:
        # print('Here1 else',c.item())
        idx_15_20, _ = sample_from_range(dists_to_ref, 4, max_cutoff + 1, len_get)
        all_idx = idx_15_20
        if all_idx is None:
            # print('Here1 else none',c.item())
            return None


    valid_cubes = valid_cubes.view(-1, 3) 
    # print("#####################",all_idx, valid_cubes.shape)
    
    return valid_cubes[all_idx]




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


neg_embedding = []



j=0
# for c in range(0, len(traj), 3):
# 26036 - > 100K
# for c in range(26035, 79205, 1):
for c in whr2:
# for c in torch.tensor([27438,27439,27440]):
# for c in range(8926, 8930):
    mean_gas = get_neg_centers_vectorized_fast(c, len_get=len(whr_list[c]), max_cutoff=cutoff)
    if mean_gas is None:
        print("None for,", c)
        continue
    print(f"{c}", flush=True)
    j+=1
    if j % 20000 == 0 and j != 0:  # Save when c is around a multiple of 1000
        print(f"SAVE at c={c}", flush=True)
        torch.save(neg_embedding, "neg_SE3embedding_bigClosePos2_%d.pt"%cutoff)
        # break
    atom_coords = protein_coords_traj[c]
    print(mean_gas)

    for i in range(0, len(mean_gas)):
        name="frame_%d_%d_%d_%d" % (c+1,mean_gas[i][0],mean_gas[i][1],mean_gas[i][2])
        print('\t',name)
        center = mean_gas[i]
        
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
        # neg_atom_matrix_arr.append(atom_matrix)
        test = extract_point_cloud(atom_matrix, positions, center)
        test.name = name
        test.node_attr = local_pos_normalized
        neg_embedding.append(test.to('cpu'))
        

# torch.save(neg_atom_matrix_arr,"neg_atom_matrix_arr.pt")
# torch.save(neg_tensor,"neg_positions_arr.pt")
# python embed_negSE3.py 20 > test2.log 2>&1 
torch.save(neg_embedding, "neg_SE3embedding_bigClosePos2_%d.pt"%cutoff)
