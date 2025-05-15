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


class VoxelCNN(nn.Module):
    def __init__(self, input_channels, f1=32, output_size=1, init_weights='normal', act=nn.ReLU, ns=0.1, dropout=0.3):
        super(VoxelCNN, self).__init__()
        self.act = act or partial(nn.LeakyReLU, negative_slope=ns)
        self.dropout = dropout

        print("CNN F1 is {}".format(f1))
        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, f1, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1),
            act() if callable(act) else act,
            nn.Dropout3d(self.dropout),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(f1, f1 * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1 * 2),
            act() if callable(act) else act,
            nn.Dropout3d(self.dropout),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(f1 * 2, f1, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1),
            act() if callable(act) else act,
            # nn.Dropout3d(self.dropout),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(f1 , f1, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1),
            act() if callable(act) else act,
            # nn.Dropout3d(self.dropout),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(f1, f1 // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1 // 2),
            act() if callable(act) else act,
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            # nn.Dropout(self.dropout),  # Dropout before final FC layer
            nn.Linear(f1 // 2, output_size)
        )

        if init_weights == 'normal':
            # self.apply(self.init_weights2)
            pass
        elif init_weights == 'kaiming':
            self.apply(self.init_weights)
        elif init_weights == 'xavier':
            self.apply(self.init_weights_xavier)

    def init_weights(self,m):
        # Apply Xavier uniform initialization to Conv3d layers
        for m in self.conv:
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # Apply Xavier uniform to weights
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)  # Initialize bias to 0
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def init_weights_xavier(self,m):
        # Apply Xavier uniform initialization to Conv3d layers
        for m in self.conv:
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)  # Apply Xavier uniform to weights
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)  # Initialize bias to 0
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def init_weights2(self,m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            print(m)
            init.normal_(m.weight, mean=0.0, std=0.01)  # Normal distribution initialization
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias to zero

    def forward(self, x):
        x = self.conv(x)
        # print('x.shape',x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

def augment_voxel_grid(voxel_grid, p=0.3):
    
    if torch.rand(1).item() < p:
        # print("AUGMENT", p)
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[0])  # Flip X
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[1])  # Flip Y
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[2])  # Flip Z
    return voxel_grid


class AugmentedVoxelDataset(Dataset):
    def __init__(self, X, y, aug_prob=0.3):
        self.X = X
        self.y = y
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        voxel = self.X[idx]
        label = self.y[idx]
        voxel = augment_voxel_grid(voxel, p=self.aug_prob)
        return voxel, label

def make_loader(batch_size=32, test=False, aug_prob=0.2):
    
    # Load embeddings
    positives = torch.load('pos_embedding.pt')  # (N, 20, 20, 20, 6)
    negatives = torch.load('neg_embedding.pt')  # (N, 20, 20, 20, 6)

    print("positives size:", len(positives))
    print("negatives size:", len(negatives))

    # Labels
    pos_labels = torch.ones(len(positives), dtype=torch.float32)
    neg_labels = torch.zeros(len(negatives), dtype=torch.float32)

    # Equalize dataset size by trimming the larger set
    n = min(len(positives), len(negatives))
    positives = positives[:n]
    negatives = negatives[:n]
    pos_labels = pos_labels[:n]
    neg_labels = neg_labels[:n]

    # Combine
    X = torch.cat([positives, negatives], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, val_idx in sss.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

    # train_dataset = TensorDataset(X_train, y_train)
    # val_dataset = TensorDataset(X_val, y_val)
    train_dataset = AugmentedVoxelDataset(X_train, y_train, aug_prob=aug_prob)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    sys.stdout.flush()

    return train_loader, val_loader


def make(config, test=False):
    # Make the data
    print("################# 'batch_size'", config['batch_size'], test)
    train_loader, val_loader = make_loader(batch_size=config['batch_size'], test=test)
    print('####################', config['act'])
    if config['batch_size']==1:
        bn = False
    else:
        bn = True
    # Make the model
    model = VoxelCNN(input_channels=6, f1=config['f1'], output_size=1, init_weights=config['init_weights'], act=config['act'], ns=config['ns'],
    dropout=config['dropout'])
    
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
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
    act1 = wandb.config.act
    if act1 == 'ReLU':
        act = nn.ReLU
    elif act1 == 'ELU':
        act = nn.ELU
    elif act1 == 'LeakyReLU':
        act = nn.LeakyReLU
    else:
        act = nn.PReLU

    init_weights = wandb.config.init_weights
    f1 = wandb.config.f1
    bias_init = wandb.config.bias_init
    ns = wandb.config.ns

    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    weight_decay = wandb.config.weight_decay
    
    c = {
        "act": act,
        "init_weights": init_weights,
        "f1": f1,
        "bias_init": bias_init,
        "ns": ns,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size, 
        "dropout": wandb.config.dropout,
        "weight_decay": weight_decay,       
    }
    
    # make the model, data, and optimization problem
    
    model, train_loader, val_loader, criterion, optimizer = make(c, test=test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    early_stop_counter = 0
    early_stop_patience = 2
    val_acc_threshold = 0.5
    tolerance = 0.02  # +/- around 0.5

    for epoch in range(c["epochs"]):
        model.train()
        total_train = 0
        correct_train = 0
        

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == y.bool()).sum().item()
            total_train += y.size(0)

            if (batch_idx + 1) % 100 == 0:
                train_acc = correct_train / total_train
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Train Accuracy: {train_acc:.4f}", flush=True)

        train_accuracy = correct_train / total_train

        # Validation
        model.eval()
        total_val = 0
        correct_val = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 4, 1, 2, 3)
                outputs = model(x).squeeze()
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == y.bool()).sum().item()
                total_val += y.size(0)

                if (batch_idx + 1) % 100 == 0:
                    val_acc = correct_val / total_val
                    print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Val Accuracy: {val_acc:.4f}", flush=True)

            print('x.shape', x.shape)
            logits = model(x)  # Use a val batch
            probs = torch.sigmoid(logits)
            print("Mean output:", probs.mean().item(), probs)
            print("Std output:", probs.std().item())
            print("Predictions:", (probs > 0.5).int().unique(return_counts=True))
            print()
        val_accuracy = correct_val / total_val
        print(f"Epoch {epoch+1} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}")

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
    #wandb sweep -e university_of_bath -p diox_prob sweep.yaml
    #wandb agent -e university_of_bath -p diox_prob q98v0jij > train.log 2>&1

    main()
