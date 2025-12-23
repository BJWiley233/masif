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

device='cuda'
pos_embedding_arr = []
neg_embedding_arr = []

for i in range(2,18,2):
# for i in range(2,16):
    pos_embedding_arr += torch.load("../Sim%d/pos_embedding_in4.pt"%i)
    print(len(pos_embedding_arr), flush=True)

for i in range(2,18,2):
# for i in range(2,16):
    neg_embedding_arr += torch.load("../Sim%d/neg_embedding_in4.pt"%i)
    print(len(neg_embedding_arr), flush=True)


for i in range(2,7):
    pos_embedding_arr += torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2-O2/Bundle/Sim%d/pos_embedding_in4.pt"%i)
    print(len(pos_embedding_arr), flush=True)
for i in range(2,7):
    neg_embedding_arr += torch.load("/data/pompei/bw973/Oxygenases/PHD2/PHD2-O2/Bundle/Sim%d/neg_embedding_in4.pt"%i)
    print(len(neg_embedding_arr), flush=True)

neg_embedding_arr2 = []
for neg in neg_embedding_arr:
    if torch.norm(neg.center - neg.perturbed_from) > 3.:
        neg_embedding_arr2.append(neg)

print(f"Len + {len(pos_embedding_arr)}; Len -  {len(neg_embedding_arr2)}; ratio {len(neg_embedding_arr2)/len(pos_embedding_arr)}", flush=True)


from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

for pos in pos_embedding_arr:
    pos.perturbed_from = pos.center


import random
def shuffle_data_list_(data_list, seed=None):
    if seed is not None:
        rnd = random.Random(seed)  # independent RNG
        rnd.shuffle(data_list)     # shuffle in place
    else:
        random.shuffle(data_list)  # default RNG

class PointCloudDataset(Dataset):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives
        # self.n = min(len(positives), len(negatives))
        # self.total = 2 * self.n
        self.n_pos = len(positives)
        self.n_neg = 2 * self.n_pos  # we expect negatives to be at least 2× positives
        
        # dataset size = pos + neg
        self.total = self.n_pos + self.n_neg

    def __len__(self):
        return self.total

    def __getitem__(self, idx):

        if idx < self.n_pos:
            return self.positives[idx], 1.0
        else:
            neg_idx = idx - self.n_pos
            return self.negatives[neg_idx], 0.0
        
shuffle_data_list_(pos_embedding_arr, seed=42)
shuffle_data_list_(neg_embedding_arr, seed=42)

n = min(len(pos_embedding_arr), len(neg_embedding_arr))

n_pos = len(pos_embedding_arr)
# n_pos=15000
n_pos=int(n_pos)
n_pos=15000 # changed this from 25000 to 15000 and this made epoch 18... (19 w/ 0-index)
n_neg = int(n_pos*2.5)

positives = pos_embedding_arr[:n_pos]
negatives = neg_embedding_arr[:n_neg]

labels = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
full_dataset = PointCloudDataset(positives, negatives)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

indices = torch.arange(n_pos + n_neg)
train_idx, val_idx = next(sss.split(indices, labels))

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



def noise_node_features(
    data: Data,
    node_frac=0.3,
    atom_noise_std=0.5,
    dist_noise_std=0.5,
    dir_noise_std=0.5,
    frame_noise_std=0.25,
):
    N = data.num_nodes
    device = data.x.device

    n_noise = max(1, int(node_frac * N))
    idx = torch.randperm(N, device=device)[:n_noise]

    # -------- atom type one-hot (x[:, :6]) --------
    atom_oh = data.x[idx, :6]
    new_types = torch.randint(0, 6, (n_noise,), device=device)
    atom_oh.zero_()
    atom_oh[torch.arange(n_noise), new_types] = 1.0
    data.x[idx, :6] = atom_oh

    # -------- frame embedding (x[:, 6:]) --------
    data.x[idx, 6:] += frame_noise_std * torch.randn_like(data.x[idx, 6:])

    # -------- node distances (scalar) --------
    if hasattr(data, "node_distances"):
        data.node_distances[idx] += dist_noise_std * torch.randn_like(
            data.node_distances[idx]
        )
        data.node_distances.clamp_(min=0.0)

    # -------- node directions (3-vector, renormalize) --------
    if hasattr(data, "node_directions"):
        noisy_dir = (
            data.node_directions[idx]
            + dir_noise_std * torch.randn_like(data.node_directions[idx])
        )
        data.node_directions[idx] = F.normalize(noisy_dir, dim=1)

    # -------- node_attr (first 3-vector + derived scalar) --------
    data.node_attr[idx, :3] += atom_noise_std * torch.randn_like(
        data.node_attr[idx, :3]
    )

    # recompute distance scalar if you use it downstream
    # distance = ||node_attr[:3]||
    # (optional – only if stored explicitly)
    return data


def noise_edge_attrs(
    data: Data,
    edge_frac=0.3,
):
    if data.edge_attr is None:
        return data

    E = data.edge_attr.size(0)
    device = data.edge_attr.device

    n_noise = max(1, int(edge_frac * E))
    idx = torch.randperm(E, device=device)[:n_noise]

    # Only change bonded edges (edge_attr > 0)
    bonded = data.edge_attr[idx].squeeze(-1) > 0
    bonded_idx = idx[bonded]

    if bonded_idx.numel() == 0:
        return data

    # bond types {1,2,3}
    new_bonds = torch.randint(
        0, 4, (bonded_idx.numel(),), device=device
    )#.float()

    data.edge_attr[bonded_idx, 0] = new_bonds
    return data

import copy

class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_prob=0.1):
        self.dataset = dataset
        self.noise_prob = noise_prob

        n = len(dataset)
        n_noisy = max(1, int(noise_prob * n))
        self.noisy_indices = set(
            torch.randperm(n)[:n_noisy].tolist()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # --- unpack ---
        if isinstance(sample, tuple):
            data, *rest = sample
        else:
            data = sample
            rest = []

        # no noise
        if idx not in self.noisy_indices:
            return sample

        # clone only Data
        data = data.clone()

        data = noise_node_features(data)
        data = noise_edge_attrs(data)

        # --- repack ---
        if rest:
            return (data, *rest)
        return data


noisy_train_dataset = NoisyDataset(train_dataset, noise_prob=0.5)
noisy_train_loader = DataLoader(
    noisy_train_dataset,
    batch_size=4,
    shuffle=True
)
# train_loader = noisy_train_loader


# orig_model = torch.load("...pt")
from e3nn.io import CartesianTensor
from torch_geometric.loader import DataLoader
from copy import deepcopy
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork, NetworkForAGraphWithAttributes

model = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2_50_O2IF/Bundle/Sim4/predict/models_10_Both/output_ep_4_bs_24_lr_0.0003_opt_adamw_inw_xavier_neigh_45_nodes_85_mul_30_lay_3_lmax_2.pt')

irreps_node_attr = Irreps("17x0e + 1x1o")
# 2. Build new model with 7 scalar features
# model2 = NetworkForAGraphWithAttributes(
#     # irreps_node_input=Irreps("1x0e + 1x1o"),      # updated input
#     irreps_node_input=Irreps("6x0e"),
#     irreps_node_attr=irreps_node_attr,     # keep the same
#     irreps_edge_attr=model.irreps_edge_attr,     # keep the same
#     irreps_node_output=model.irreps_node_output,
#     max_radius=6.0,
#     num_neighbors=45,
#     num_nodes=85,
#     mul=40,
#     layers=3,
#     lmax=2,
#     pool_nodes=True,
# )


model2=torch.load("../Sim5/model17_FE2dis_inr3.pt")
model2 = model2.to('cuda')

optimizer=torch.optim.NAdam(model2.parameters(), lr=0.001)
for epoch in range(18,20):
    model2.train()
    total_train = 0
    correct_train = 0
    pos_total_train = 0
    pos_correct_train = 0
    neg_total_train = 0
    neg_correct_train = 0

    for batch_idx, (data_list, labels) in enumerate(train_loader):
        # If batch_size > 1, data_list will be a list of Data objects
        # Batch them for PyG model input:
        # print(labels.sum())
        if isinstance(data_list, list):
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(data_list)
        else:
            batch = data_list  # if batch_size=1, it might already be a Data object

        batch = batch.to('cuda')
        labels = labels.to('cuda')

        num_pos = (labels == 1).sum()
        num_neg = (labels == 0).sum()

        # Avoid division by zero
        if num_pos > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=labels.device)
        else:
            pos_weight = torch.tensor([1.0], device=labels.device)  # neutral weight

        
        distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

        # 2. Convert Cartesian vectors to irreps vector
        x = CartesianTensor("i")
        vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

        # 3. Concatenate scalar + vector as node_attr tensor
        atom_type_onehot = batch.x[:, 0:6]
        frame_emb = batch.x[:, 6:] 
        # node_attr = torch.cat([distance, vector_irrep, frame_emb], dim=1)
        node_attr = torch.cat([
            distance,            # 1 scalar (0e)
            # atom_type_onehot,    # 6 scalars (0e)
            frame_emb,           # k scalars (0e)
            vector_irrep,        # 3-vector (1o)
        ], dim=1)
        # node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)
        # node_input = torch.cat([
        #     # batch.distance.unsqueeze(-1), # 1 scalar (0e) distance of center of graph to iron
        #     batch.node_distances.unsqueeze(-1),    # 1 scalar (0e) distance to iron for each atom
        #     F.normalize(batch.node_directions, p=2, dim=1),        # 3-vector (1o) direction to iron for each atom
        # ], dim=1)

        data = {
            "batch": batch.batch,
            # "x": batch.x[:,0:6], # atom type
            # "frame_emb": frame,
            "x": atom_type_onehot,
            "node_attr": node_attr, 
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "pos": batch.pos,  # if needed in preprocess
        }

        optimizer.zero_grad()
        # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
        outputs = model2(data)
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(outputs.squeeze(-1), labels)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer.step()

        # Predictions
        preds = (torch.sigmoid(outputs.squeeze(-1)) > 0.5)

        # Convert labels to bool
        labels_bool = labels.bool()

        # ---- metrics ----
        correct_train += (preds == labels_bool).sum().item()
        total_train += labels.size(0)

        # positive (label=1)
        pos_mask = labels_bool
        pos_total_train += pos_mask.sum().item()
        # pos_correct = ((preds & pos_mask)).sum().item()
        pos_correct_train += ((preds & pos_mask)).sum().item()
        if num_pos > 0:
            pos_acc = pos_correct_train / pos_total_train
        else:
            pos_acc = float("nan")

        # negative (label=0)
        neg_mask = ~labels_bool
        neg_total_train += neg_mask.sum().item()
        neg_correct_train += ((~preds & neg_mask)).sum().item()
        if neg_total_train > 0:
            neg_acc = neg_correct_train / neg_total_train
        else:
            neg_acc = float("nan")

        # print every 20 batches
        if (batch_idx + 1) % 20 == 0:
            train_acc = correct_train / total_train
            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1} | "
                f"Acc: {train_acc:.4f} | "
                f"PosAcc: {pos_acc:.4f} | "
                f"NegAcc: {neg_acc:.4f}",
                flush=True
            )
        torch.cuda.empty_cache()
    train_accuracy = correct_train / total_train
    pos_acc_train = pos_correct_train / pos_total_train if pos_total_train > 0 else 0
    neg_acc_train = neg_correct_train / neg_total_train if neg_total_train > 0 else 0
    
    print('\ttrain_accuracy:', train_accuracy, flush=True)
    print('\tpos train_accuracy:', pos_acc_train, flush=True)
    print('\tneg train_accuracy:', neg_acc_train, flush=True)

    with torch.no_grad():
        model2.eval()
        total_train = 0
        correct_train = 0
        pos_total_train = 0
        pos_correct_train = 0

        neg_total_train = 0
        neg_correct_train = 0

        
        for batch_idx, (data_list, labels) in enumerate(val_loader):
            # print(labels.sum())
            # If batch_size > 1, data_list will be a list of Data objects
            # Batch them for PyG model input:
            if isinstance(data_list, list):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(data_list)
            else:
                batch = data_list  # if batch_size=1, it might already be a Data object

            batch = batch.to('cuda')
            labels = labels.to('cuda')

            num_pos = (labels == 1).sum()
            num_neg = (labels == 0).sum()
               
            distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

            # 2. Convert Cartesian vectors to irreps vector
            x = CartesianTensor("i")
            vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

            # 3. Concatenate scalar + vector as node_attr tensor
            atom_type_onehot = batch.x[:, 0:6]
            frame_emb = batch.x[:, 6:] 
            # node_attr = torch.cat([distance, vector_irrep, frame_emb], dim=1)
            node_attr = torch.cat([
                distance,            # 1 scalar (0e)
                # atom_type_onehot,    # 6 scalars (0e)
                frame_emb,           # k scalars (0e)
                vector_irrep,        # 3-vector (1o)
            ], dim=1)
            # node_input = torch.ones((batch.num_nodes, 1), device=batch.x.device)
            # node_input = torch.cat([
            #     # batch.distance.unsqueeze(-1), # 1 scalar (0e) distance of center of graph to iron
            #     batch.node_distances.unsqueeze(-1),    # 1 scalar (0e) distance to iron for each atom
            #     F.normalize(batch.node_directions, p=2, dim=1),        # 3-vector (1o) direction to iron for each atom
            # ], dim=1)

            data = {
                "batch": batch.batch,
                # "x": batch.x[:,0:6], # atom type
                # "frame_emb": frame,
                "x": atom_type_onehot,
                "node_attr": node_attr, 
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "pos": batch.pos,  # if needed in preprocess
            }
           
            # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
            outputs = model2(data)

            preds = (torch.sigmoid(outputs.squeeze(-1)) > 0.5)

            # Convert labels to bool
            labels_bool = labels.bool()

            # ---- metrics ----
            correct_train += (preds == labels_bool).sum().item()
            total_train += labels.size(0)

            # positive (label=1)
            
            pos_mask = labels_bool
            pos_total_train += pos_mask.sum().item()
            # pos_correct = ((preds & pos_mask)).sum().item()
            pos_correct_train += ((preds & pos_mask)).sum().item()
            if num_pos > 0:
                pos_acc = pos_correct_train / pos_total_train
            else:
                pos_acc = float("nan")

            # negative (label=0)
            neg_mask = ~labels_bool
            neg_total_train += neg_mask.sum().item()
            neg_correct_train += ((~preds & neg_mask)).sum().item()
            if neg_total_train > 0:
                neg_acc = neg_correct_train / neg_total_train
            else:
                neg_acc = float("nan")

            # print every 20 batches
            if (batch_idx + 1) % 20 == 0:
                train_acc = correct_train / total_train
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1} | "
                    f"Val Acc: {train_acc:.4f} | "
                    f"Val PosAcc: {pos_acc:.4f} | "
                    f"Val NegAcc: {neg_acc:.4f}",
                    flush=True
                )
            torch.cuda.empty_cache()
        train_accuracy = correct_train / total_train
        pos_acc_val = pos_correct_train / pos_total_train if pos_total_train > 0 else 0
        neg_acc_val = neg_correct_train / neg_total_train if neg_total_train > 0 else 0
        torch.save(model2, "../Sim5/model%d_FE2dis_inrTESTING2.pt"%epoch)
        print('\tval_accuracy:', train_accuracy, flush=True)
        print('\tpos val_accuracy:', pos_acc_val, flush=True)
        print('\tneg val_accuracy:', neg_acc_val, flush=True)
        # if pos_acc_train > 0.8 and (pos_acc_train - pos_acc_val) > 0.2:
        #     print("Overfit positives so adding noise to positives and negatives")
        #     train_loader = noisy_train_loader
        # elif neg_acc_train > 0.9 and (neg_acc_train - neg_acc_val) > 0.1:
        #     print("Overfit negatives so adding noise to positives and negatives")
        #     train_loader = noisy_train_loader
    
