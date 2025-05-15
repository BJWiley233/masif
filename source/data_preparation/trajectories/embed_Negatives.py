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

# t = './an%d.xtc' % (trjnum)
t = '../cat_fit.dcd'
top = '../an%d.gro' % (1)
traj = md.load(t, top=top)

import torch

xyz = torch.tensor(traj.xyz) * 10


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "Fe": 5} # all 0's will be padding for
ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "VS": {"FE": 5, "MG":6}} # all  gases not within 3.5 angstroms of any protein atom
gas='O2IF'
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

import os

# if os.path.exists('../whr_list%d.npy'%20):
#     whr_list=np.load('../whr_list%d.npy'%20, allow_pickle=True) 
# n=sum([len(i) for i in whr_list])
dioxygen_coords_ave = xyz[:,gas2,:].reshape(-1,len(residue_sel_un),rs,3).mean(axis=2)

def embed_voxel_atom(atom_matrix, positions, voxel_size=0.5, embedding_dim=3):

    # Set up embedding layer
    num_atom_types = atom_matrix.shape[1]
    atom_embedding = nn.Embedding(num_atom_types, embedding_dim).to(device)

    # Box setup
    box_size = 10.0
    grid_dim = [int(box_size / voxel_size)] * 3
    min_bound = -box_size / 2

    # Initialize voxel grid
    voxel_descriptor_grid = torch.zeros((*grid_dim, embedding_dim), dtype=torch.float32, device=positions.device).to(device)
    voxel_counts = torch.zeros(grid_dim, dtype=torch.float32, device=positions.device).to(device)

    # Map atoms
    atom_type_indices = atom_matrix.argmax(dim=1)  # [N]
    # embedded_features = atom_embedding(atom_type_indices)  # [N, D]
    embedded_features = atom_matrix

    # Calculate grid positions
    grid_pos = torch.floor((positions - min_bound) / voxel_size).to(dtype=torch.long).to(device)  # [N, 3]
    # Precompute max bounds as a tensor
    max_bound = torch.tensor(grid_dim, device=positions.device) - 1

    # Clamp each coordinate separately
    grid_pos = torch.minimum(torch.maximum(grid_pos, torch.zeros_like(grid_pos).to(device)), max_bound.to(device)).to(device)

    # Scatter add features into voxel grid
    for i in range(positions.shape[0]):
        x, y, z = grid_pos[i]
        voxel_descriptor_grid[x, y, z] += embedded_features[i]
        voxel_counts[x, y, z] += 1

    # Normalize per voxel
    voxel_descriptor_grid /= torch.clamp(voxel_counts.unsqueeze(-1), min=1)

    return voxel_descriptor_grid


def sample_from_range(tensor, low, high, count):
    # Find indices where values are in the desired range
    mask = (tensor >= low) & (tensor < high)
    matching_indices = torch.nonzero(mask, as_tuple=False).squeeze()

    if matching_indices.numel() < count:
        raise ValueError(f"Not enough values in range [{low}, {high}) to sample {count} elements.")

    # Shuffle and select the desired number of random indices
    selected = matching_indices[torch.randperm(len(matching_indices))[:count]]
    values = tensor[selected]
    return selected, values



# for c in range(0,100):
def get_neg_centers(c):
# Get coordinates of the ligand atoms
# ligand_coords = xyz[:, ligand_atoms, :]*10  # Shape: (n_frames, n_lig_atoms, 3)
    ligand_coords = xyz[c][gas2].reshape(len(residue_sel_un),rs,3).mean(axis=1)
    ligand_coords = ligand_coords.reshape(-1, 3)  # Flattening to (n_lig_atoms * n_frames, 3)

    # Get the full system coordinates
    # system_coords = mol.xyz.reshape(-1, 3)*10  # Shape: (n_atoms * n_frames, 3)
    system_coords = protein_coords_traj[c]

    # Define the bounding box for the system
    min_corner = torch.floor(system_coords.min(axis=0).values)
    max_corner = torch.ceil(system_coords.max(axis=0).values)

    # Create grid of 10x10x10 Å cubes
    step = 1.0
    cube_size = 10.0

    # Generate sliding cube origins
    x = torch.arange(min_corner[0], max_corner[0] - cube_size + 1, step)
    y = torch.arange(min_corner[1], max_corner[1] - cube_size + 1, step)
    z = torch.arange(min_corner[2], max_corner[2] - cube_size + 1, step)

    empty_cubes = []

    for xi in x:
        for yi in y:
            for zi in z:
                cube_min = torch.tensor([xi, yi, zi])
                cube_max = cube_min + 10

                # Check if any ligand atom is inside this cube
                in_cube = torch.all((ligand_coords >= cube_min) & (ligand_coords < cube_max), dim=1)
                if not torch.any(in_cube):
                    empty_cubes.append(cube_min+5)  # Or store center: cube_min + 5

    from scipy.spatial import cKDTree

    # Build a KD-tree of system (protein + ligand) coordinates
    protein_tree = cKDTree(system_coords.cpu())

    # Find cubes that are near the protein
    valid_cubes = []
    for center in (torch.stack(empty_cubes)):
        # Query if the cube center is within 10 Å of any protein atom
        if protein_tree.query_ball_point(center, r=3.5):
            valid_cubes.append(center)
    valid_cubes = torch.stack(valid_cubes).to(device)

    distances_neg_frame = torch.norm(system_coords[-1] - valid_cubes, dim=1)

    # Perform the selections
    idx_5_10, val_5_10 = sample_from_range(distances_neg_frame, 5, 10, 1)
    idx_10_15, val_10_15 = sample_from_range(distances_neg_frame, 10, 15, 2)
    idx_15_20, val_15_20 = sample_from_range(distances_neg_frame, 15, 20, 3)

    # Combine results
    all_indices = torch.cat([idx_5_10, idx_10_15, idx_15_20])
    all_values = torch.cat([val_5_10, val_10_15, val_15_20])

    # neg_centers.append(valid_cubes[all_indices])
    return valid_cubes[all_indices]


def get_neg_centers_fast(c, cube_size=10.0, step=1.0, protein_dist_cutoff=3.5):
    ligand_coords = xyz[c][gas2].reshape(len(residue_sel_un), rs, 3).mean(axis=1)
    system_coords = protein_coords_traj[c]

    min_corner = torch.floor(system_coords.min(axis=0).values)
    max_corner = torch.ceil(system_coords.max(axis=0).values)

    # Create grid of cube centers
    x = torch.arange(min_corner[0], max_corner[0] - cube_size + 1, step)
    y = torch.arange(min_corner[1], max_corner[1] - cube_size + 1, step)
    z = torch.arange(min_corner[2], max_corner[2] - cube_size + 1, step)

    grid = torch.cartesian_prod(x, y, z).to(device)  # Shape: [N_cubes, 3]
    cube_centers = grid + (cube_size / 2.0)

    # Mask out cubes that intersect the ligand
    lig_min = ligand_coords.min(dim=0).values - cube_size / 2
    lig_max = ligand_coords.max(dim=0).values + cube_size / 2

    # Bounding box overlap filter
    valid_mask = (
        (cube_centers[:, 0] < lig_min[0]) | (cube_centers[:, 0] > lig_max[0]) |
        (cube_centers[:, 1] < lig_min[1]) | (cube_centers[:, 1] > lig_max[1]) |
        (cube_centers[:, 2] < lig_min[2]) | (cube_centers[:, 2] > lig_max[2])
    )

    cube_centers = cube_centers[valid_mask]

    # Use torch.cdist to find cubes within `protein_dist_cutoff` Å of the protein
    dists = torch.cdist(cube_centers, system_coords)
    near_protein = (dists < protein_dist_cutoff).any(dim=1)

    valid_cubes = cube_centers[near_protein]

    # Measure distances to the last gas atom to stratify
    last_gas_atom = system_coords[-1].unsqueeze(0)
    dists_to_gas = torch.norm(valid_cubes - last_gas_atom, dim=1)

    idx_5_10, val_5_10 = sample_from_range(dists_to_gas, 5, 10, 1)
    idx_10_15, val_10_15 = sample_from_range(dists_to_gas, 10, 15, 2)
    idx_15_20, val_15_20 = sample_from_range(dists_to_gas, 15, 20, 3)

    all_indices = torch.cat([idx_5_10, idx_10_15, idx_15_20])
    return valid_cubes[all_indices]

def get_neg_centers_vectorized(c, cube_size=10.0, step=1.0, protein_dist_cutoff=3.5):
    # Ligand and system coords
    ligand_coords = xyz[c][gas2].reshape(len(residue_sel_un), rs, 3).mean(dim=1).to(device)  # [num_ligands, 3]
    system_coords = protein_coords_traj[c]  # [num_atoms, 3]

    # Bounding box
    min_corner = torch.floor(system_coords.min(dim=0).values)
    max_corner = torch.ceil(system_coords.max(dim=0).values)

    # Grid setup
    x = torch.arange(min_corner[0], max_corner[0] - cube_size + 1, step)
    y = torch.arange(min_corner[1], max_corner[1] - cube_size + 1, step)
    z = torch.arange(min_corner[2], max_corner[2] - cube_size + 1, step)

    cube_mins = torch.cartesian_prod(x, y, z).to(device)  # [num_cubes, 3]
    cube_maxs = cube_mins + cube_size
    cube_centers = cube_mins + (cube_size / 2.0)

    # ---- Check if any ligand atom is inside each cube ----
    # Expand for broadcasting
    ligand_coords_exp = ligand_coords.unsqueeze(0)  # [1, num_ligands, 3]
    cube_mins_exp = cube_mins.unsqueeze(1)          # [num_cubes, 1, 3]
    cube_maxs_exp = cube_maxs.unsqueeze(1)

    # Boolean mask for each cube-ligand pair
    in_cube = torch.all((ligand_coords_exp >= cube_mins_exp) & (ligand_coords_exp < cube_maxs_exp), dim=2)  # [num_cubes, num_ligands]
    cube_mask = ~in_cube.any(dim=1)  # [num_cubes] — True = no ligand inside

    empty_cube_centers = cube_centers[cube_mask]

    # ---- Check which empty cubes are near the protein (using cKDTree) ----
    from scipy.spatial import cKDTree
    protein_tree = cKDTree(system_coords.cpu())
    valid_list = []

    for center in empty_cube_centers.cpu():
        if protein_tree.query_ball_point(center.numpy(), r=protein_dist_cutoff):
            valid_list.append(center)

    print(len(valid_list), len(empty_cube_centers))

    if len(valid_list) == 0:
        raise ValueError("No valid cubes found near protein for frame", c)

    valid_cubes = torch.stack(valid_list).to(device)  # [N, 3]

    # ---- Sample based on distance to a ligand (e.g., last one) ----
    # ref_ligand = ligand_coords[-1]
    dists = torch.norm(valid_cubes - system_coords[-1], dim=1)

    idx_5_10, val_5_10 = sample_from_range(dists, 5, 10, 1)
    idx_10_15, val_10_15 = sample_from_range(dists, 10, 15, 2)
    idx_15_20, val_15_20 = sample_from_range(dists, 15, 20, 3)

    all_idx = torch.cat([idx_5_10, idx_10_15, idx_15_20])
    return valid_cubes[all_idx]



# negn = len(traj)*6//3
negn = len(range(0,len(traj), 3))*6
print(negn)

neg_atom_matrix_arr = []
neg_positions_arr = []
neg_names = []
neg_tensor = torch.load("neg_embedding_big.pt")
# neg_tensor = torch.zeros(negn, 20, 20, 20, 6)

j=0
for c in range(0, len(traj), 3):
    mean_gas = get_neg_centers_vectorized(c)
    print(f"{c}", flush=True)
    if c % 1000 < 3:  # Save when c is around a multiple of 1000
        print(f"SAVE at c={c}", flush=True)
        torch.save(neg_tensor, "neg_embedding_big.pt")
        np.save("neg_names_big.npy", neg_names)
    atom_coords = protein_coords_traj[c]
    for i in range(0, len(mean_gas)):
        name="frame_%d_%d_%d_%d" % (c+1,mean_gas[i][0],mean_gas[i][1],mean_gas[i][2])
        neg_names.append(name)
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
        # neg_atom_matrix_arr.append(atom_matrix)
        # neg_positions_arr.append(positions)
        test = embed_voxel_atom(atom_matrix, positions, voxel_size=0.5, embedding_dim=6)
        # pos_embedding.append(test)
        neg_tensor[j] = test
        j+=1
        

# torch.save(neg_atom_matrix_arr,"neg_atom_matrix_arr.pt")
# torch.save(neg_tensor,"neg_positions_arr.pt")
torch.save(neg_tensor, "neg_embedding_big.pt")
np.save("neg_names_big.npy", neg_names)
