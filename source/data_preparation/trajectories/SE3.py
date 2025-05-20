import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb
import logging
from importlib import reload
import pandas as pd
import numpy as np
from itertools import islice
import sys
import gc
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F



# import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Dropout,
    Linear as Lin,
    LeakyReLU,PReLU,ELU,
    Tanh,
    ReLU,
    BatchNorm1d as BN,
)
device = torch.device("cuda")




from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset






class PointCloudDataset(Dataset):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives
        self.n = min(len(positives), len(negatives))
        self.total = 2 * self.n

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.n:
            return self.positives[idx], 1.0
        else:
            return self.negatives[idx - self.n], 0.0

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork, NetworkForAGraphWithAttributes

from e3nn.o3 import Irreps
from torch_geometric.nn import radius_graph


 

from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader  # Correct loader for PyG Data

def make_loader(batch_size=32, test=False, aug_prob=0.2):
    
    # Load embeddings
    # positives = torch.load('pos_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)
    # negatives = torch.load('neg_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)

    positives1 = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2_O2QD/Bundle/Sim18/pos_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)
    print('len(positives1)', len(positives1))
    negatives1 = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2_O2QD/Bundle/Sim18/neg_SE3embedding_big2.pt')  # (N, 20, 20, 20, 6)
    print('len(negatives1)', len(negatives1))

    positives2 = torch.load('pos_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)
    print('len(positives2)', len(positives2))
    negatives2 = torch.load('neg_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)
    print('len(negatives2)', len(negatives2))

    positives3 = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2-O2/Bundle/Sim8/pos_SE3embedding_big.pt')  # (N, 20, 20, 20, 6)
    print('len(positives3)', len(positives3))
    negatives3 = torch.load('/data/pompei/bw973/Oxygenases/PHD2/PHD2-O2/Bundle/Sim8/neg_SE3embedding_big2.pt')  # (N, 20, 20, 20, 6)
    print('len(negatives3)', len(negatives3))

    positives = (positives1 + positives2 + positives3)[1::3]
    negatives = (negatives1 + negatives2 + negatives3)[1::3]

    print("positives size:", len(positives))
    print("negatives size:", len(negatives))

    # Labels
    pos_labels = torch.ones(len(positives), dtype=torch.float32)
    neg_labels = torch.zeros(len(negatives), dtype=torch.float32)

    # Equalize dataset size by trimming the larger set
    n = min(len(positives), len(negatives))
    # n=60000
    positives = positives[:n]
    negatives = negatives[:n]
    # Create full dataset (memory-safe)
    full_dataset = PointCloudDataset(positives, negatives)

    # Generate labels just for splitting
    labels = torch.cat([torch.ones(n), torch.zeros(n)])

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(torch.arange(2 * n), labels))

    # train_dataset = AugmentedVoxelDataset(full_dataset, train_idx, aug_prob)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    sys.stdout.flush()

    return train_loader, val_loader


from e3nn.io import CartesianTensor

def get_optimizer(name, model, lr, weight_decay, momentum):
    # name="adagrad"
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def make(config, test=False):
    # Make the data
    print("################# 'batch_size'", config['batch_size'], test)
    # train_loader, val_loader = make_loader(batch_size=config['batch_size'], test=test)
    if config['batch_size']==1:
        bn = False
    else:
        bn = True
    # Make the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)
    # model = VoxelCNN(input_channels=6, f1=config['f1'], output_size=1, init_weights=config['init_weights'], act=config['act'], ns=config['ns'],
    # dropout=config['dropout'])
    # model = SE3Classifier(hidden_irreps=config['hidden_irreps'])
    irreps_node_input = Irreps("6x0e")  # example: 6 scalar features per node
    # irreps_node_attr = Irreps("1x0e + 3x1o")       # scalar distance from center + 3D local position
    irreps_node_attr = Irreps("1x0e + 1x1o")
    irreps_edge_attr = Irreps("1x0e")  # scalar + vector edge attributes
    irreps_node_output = Irreps("1x0e")  
    num_neighbors = config['num_neighbors']
    num_nodes = config['num_nodes']
    model = NetworkForAGraphWithAttributes(
        irreps_node_input=irreps_node_input,
        irreps_node_attr=irreps_node_attr,
        irreps_edge_attr=irreps_edge_attr,
        irreps_node_output=irreps_node_output,
        max_radius=5.0,
        num_neighbors=num_neighbors,
        num_nodes=num_nodes,
        mul=config['mul'],
        layers=config['layers'],
        lmax=config['lmax'],
        pool_nodes=True,
    )

    model = model.to(device)
    
    # optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    print('config["optimizer"]', config["optimizer"])
    optimizer = get_optimizer(name=config["optimizer"], 
                    model=model, 
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay'], 
                    momentum=config['momentum'])
    # print('optimizer', optimizer)
    train_loader, val_loader = make_loader(batch_size=config['batch_size'], test=test)
    # model.train(True)
    # Make the loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
   
    return model, train_loader, val_loader, criterion, optimizer




def main():
    start = time.time()
    test=True
    print("RUN_BRIAN:")
    run = wandb.init()  # Initialize a run
    run.mark_preempting()
    # print('keys', wandb.config.keys())
    # note that we define values from `wandb.config`
    # instead of defining hard values
    # act1 = wandb.config.act
    # if act1 == 'ReLU':
    #     act = nn.ReLU
    # elif act1 == 'ELU':
    #     act = nn.ELU
    # elif act1 == 'LeakyReLU':
    #     act = nn.LeakyReLU
    # else:
    #     act = nn.PReLU

    init_weights = wandb.config.init_weights
    bias_init = wandb.config.bias_init

    lr = wandb.config.learning_rate
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    weight_decay = wandb.config.weight_decay
    opt = wandb.config.optimizer
    neigh = wandb.config.num_neighbors
    nodes = wandb.config.num_nodes
    mul = wandb.config.mul
    lay = wandb.config.layers
    lmax = wandb.config.lmax 
    
    c = {
        "init_weights": init_weights,
        "bias_init": bias_init,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size, 
        "weight_decay": weight_decay, 
        "optimizer": opt,
        "momentum": wandb.config.momentum,
        "num_neighbors": neigh,
        "num_nodes": nodes
        "mul": mul,
        "layers": lay,  
        "lmax": lmax       
    }
    pt='models/output_ep_{}_bs_{}_lr_{}_opt_{}_inw_{}_neigh_{}_nodes_{}_mul_{}_lay_{}_lmax_{}.pt'.format(epochs,batch_size,lr,opt,init_weights,neigh,nodes,mul,lay,lmax)
    print(pt)
    
    # make the model, data, and optimization problem
    
    model, train_loader, val_loader, criterion, optimizer = make(c, test=test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print('optimizer', optimizer)

    early_stop_counter = 0
    early_stop_patience = 2
    val_acc_threshold = 0.5
    tolerance = 0.02  # +/- around 0.5
    best_val_acc = 0

    required_total_improvement = 0.02  # 2%
    early_stop_counter2 = 0
    early_stop_patience2 = 3
    best_val_accuracy = None
    reset_threshold = 0.005  # 1%

    

    for epoch in range(c["epochs"]):
        model.train()
        total_train = 0
        correct_train = 0
        

        for batch_idx, (data_list, labels) in enumerate(train_loader):
            # If batch_size > 1, data_list will be a list of Data objects
            # Batch them for PyG model input:
            if isinstance(data_list, list):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(data_list)
            else:
                batch = data_list  # if batch_size=1, it might already be a Data object

            batch = batch.to(device)
            labels = labels.to(device)

            
            distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

            # 2. Convert Cartesian vectors to irreps vector
            x = CartesianTensor("i")
            vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

            # 3. Concatenate scalar + vector as node_attr tensor
            node_attr = torch.cat([distance, vector_irrep], dim=1)  # [N, 4]

            # node_input, node_attr, edge_index, edge_attr = prepare_inputs(batch, max_radius=5.0)
            # x is one_hot_atom_types
            data = {
                "batch": batch.batch,
                "x": batch.x,
                # "node_attr": torch.ones(N, 1, device=device),
                "node_attr": node_attr,
                # "edge_attr": edge_attr,
                "pos": batch.pos,  # if needed in preprocess
            }


            optimizer.zero_grad()
            # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
            outputs = model(data)

            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.sigmoid(outputs.squeeze(-1)) > 0.5
            correct_train += (preds == labels.bool()).sum().item()
            total_train += labels.size(0)
            if (batch_idx + 1) % 100 == 0:
                train_acc = correct_train / total_train
                # print((preds == labels.bool()).sum().item())
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Train Accuracy: {train_acc:.4f}", flush=True)

        train_accuracy = correct_train / total_train
        

        # Validation
        model.eval()
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for batch_idx, (data_list, labels) in enumerate(val_loader):
                if isinstance(data_list, list):
                    from torch_geometric.data import Batch
                    batch = Batch.from_data_list(data_list)
                else:
                    batch = data_list  # if batch_size=1, it might already be a Data object

                batch = batch.to(device)
                labels = labels.to(device)

                distance = torch.norm(batch.node_attr[:, :3], dim=1, keepdim=True)  # [N, 1]

                # 2. Convert Cartesian vectors to irreps vector
                x = CartesianTensor("i")
                vector_irrep = x.from_cartesian(batch.node_attr[:, :3])  # [N, 3]

                # 3. Concatenate scalar + vector as node_attr tensor
                node_attr = torch.cat([distance, vector_irrep], dim=1)  # [N, 4]

                # node_input, node_attr, edge_index, edge_attr = prepare_inputs(batch, max_radius=5.0)
                # x is one_hot_atom_types
                data = {
                    "batch": batch.batch,
                    "x": batch.x,
                    # "node_attr": torch.ones(N, 1, device=device),
                    "node_attr": node_attr,
                    # "edge_attr": edge_attr,
                    "pos": batch.pos,  # if needed in preprocess
                }
                # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()
                outputs = model(data)
                    
                # outputs = model(node_input, node_attr, edge_index, edge_attr).squeeze()

                preds = torch.sigmoid(outputs.squeeze(-1)) > 0.5

                correct_val += (preds == labels.bool()).sum().item()
                total_val += labels.size(0)

                if (batch_idx + 1) % 100 == 0:
                    val_acc = correct_val / total_val
                    print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Val Accuracy: {val_acc:.4f}", flush=True)

            # Debug output for final batch
            print('Batch size:', labels.size(0))
            probs = torch.sigmoid(outputs)
            print("Mean output:", probs.mean().item(), probs)
            print("Std output:", probs.std().item())
            print("Predictions:", (probs > 0.5).int().unique(return_counts=True))
            print()

            val_accuracy = correct_val / total_val
        print(f"Epoch {epoch+1} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if (val_accuracy) > best_val_acc:
            best_val_acc = val_accuracy
            print('\nSAVING with val_accuracy {} in epoch {}\n'.format(val_accuracy, epoch))
            torch.save(model, pt)

        # -- Early stopping based on val accuracy near 0.5 --
        print("ABS", abs(val_accuracy - val_acc_threshold), tolerance)
        if abs(val_accuracy - val_acc_threshold) <= tolerance:
            early_stop_counter += 1
            print("early_stop_counter", early_stop_counter)
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_counter} epochs with val acc ~0.5\n")
                break
        else:
            print("WHY", abs(val_accuracy - val_acc_threshold), tolerance)
            early_stop_counter = 0  # reset if val acc improved


        if best_val_accuracy is None:
            best_val_accuracy = val_accuracy
            print(f"Initial val_accuracy: {val_accuracy:.4f}")
        else:
            improvement = val_accuracy - best_val_accuracy

            if improvement >= reset_threshold:
                print(f"Improved val_acc by {improvement:.4f} (>= {reset_threshold}) — resetting counter.")
                best_val_accuracy = val_accuracy
                early_stop_counter2 = 0
            elif val_accuracy < best_val_accuracy + required_total_improvement:
                early_stop_counter2 += 1
                print(f"No sufficient improvement. Early stop counter: {early_stop_counter2}")
                if early_stop_counter2 >= early_stop_patience2:
                    print(f"\nEarly stopping triggered: val_acc did not improve by {required_total_improvement:.2%} in {early_stop_patience2} epochs\n")
                    break
            else:
                print(f"Val accuracy is OK but not enough to reset counter (improvement: {improvement:.4f})")


        # W&B logging
        wandb.log({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_loss': loss.item(),  # Last batch loss
        })

    print("Finished training in {:.2f} seconds.".format(time.time() - start))

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    

    run.finish()


# sweep_id = wandb.sweep(sweep=sweep_configuration, project="un")

# wandb.agent(sweep_id, function=main, count=20)
if __name__ == '__main__':
    # wandb sweep -e university_of_bath -p diox_prob sweepSE3.yaml
    # wandb agent -e university_of_bath -p diox_prob x7u6bgh3 > trainSE3.log 2>&1

    main()
