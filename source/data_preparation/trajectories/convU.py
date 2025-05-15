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

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.ReLU, dropout=0.0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            act() if callable(act) else act,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            act() if callable(act) else act,
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3DClassifier(nn.Module):
    def __init__(self, in_channels=6, features=[32, 64, 128, 256], init_weights='normal', act=nn.ReLU, ns=0.1, dropout=0.3):
        super(UNet3DClassifier, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.act = act or partial(nn.LeakyReLU, negative_slope=ns)
        self.dropout = dropout
        #[0.3, 0.2, 0.1, 0.05, 0.02, 0.05, 0.1, 0.2, 0.3, 0.1]
        # Create layer-wise dropout values if a single float is given
        if isinstance(dropout, float):
            # Dropout decays from high to low over the depth
            self.dropout_enc = [dropout * (0.5 ** i) for i in range(len(features))]
            self.dropout_dec = list(reversed(self.dropout_enc))
            self.dropout_bottleneck = dropout * (0.5 ** len(features))
            self.dropout_fc = dropout * 0.25
        elif isinstance(dropout, (list, tuple)):
            self.dropout_enc = dropout[:len(features)] #[0.3, 0.2, 0.1, 0.05]
            self.dropout_dec = dropout[-len(features)-1:-1] #[0.05, 0.1, 0.2, 0.3]
            self.dropout_bottleneck = dropout[len(features)] # 0.02
            self.dropout_fc = dropout[-1] #, 0.1
        else:
            raise ValueError("Dropout must be float or list/tuple of floats.")

        prev_channels = in_channels
        for idx, feature in enumerate(features):
            self.encoder_layers.append(DoubleConv(prev_channels, feature, act=act, dropout=self.dropout_enc[idx]))
            prev_channels = feature

        self.bottleneck = DoubleConv(prev_channels, prev_channels * 2, act=act, dropout=self.dropout_bottleneck)
        bottleneck_channels = prev_channels * 2

        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for idx, feature in enumerate(reversed(features)):
            self.upconvs.append(nn.ConvTranspose3d(bottleneck_channels, feature, kernel_size=2, stride=2))
            self.decoder_layers.append(DoubleConv(feature * 2, feature, act=act, dropout=self.dropout_dec[idx]))
            bottleneck_channels = feature

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_fc),
            nn.Linear(bottleneck_channels, 1)
        )

        if init_weights == 'normal':
            # self.apply(self.init_weights2)
            pass
        elif init_weights == 'kaiming':
            self.apply(self.init_weights_kaiming)
        elif init_weights == 'xavier':
            self.apply(self.init_weights_xavier)


    def init_weights_kaiming(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_weights_xavier(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder_layers:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]

            # Resize if mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_layers[idx](x)

        x = self.global_pool(x)  # [B, C, 1, 1, 1]
        return self.fc(x)        # [B, 1]



from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


def get_gaussian_kernel3d(kernel_size=3, sigma=1.0, device='cpu'):
    """Create a 3D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    grid = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(grid[0]**2 + grid[1]**2 + grid[2]**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.to(device)

def gaussian_blur_3d(voxel_grid, kernel_size=3, sigma=1.0):
    # Input shape: [D, H, W, C] (single voxel sample)
    voxel_grid = voxel_grid.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, D, H, W]

    B, C, D, H, W = voxel_grid.shape

    # Build Gaussian kernel
    kernel = get_gaussian_kernel3d(kernel_size, sigma, device=voxel_grid.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kD, kH, kW]
    kernel = kernel.repeat(C, 1, 1, 1, 1)      # [C, 1, kD, kH, kW]

    padding = kernel_size // 2
    blurred = F.conv3d(voxel_grid, kernel, padding=padding, groups=C)

    blurred = blurred.squeeze(0).permute(1, 2, 3, 0)  # [D, H, W, C]
    return blurred


def augment_voxel_grid(voxel_grid, p=0.3):
    if torch.rand(1).item() < p:
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[0])  # Flip X
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[1])  # Flip Y
        if torch.rand(1).item() > 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[2])  # Flip Z
        if torch.rand(1).item() > 0.5:
            voxel_grid = gaussian_blur_3d(voxel_grid)
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

def make_loader(batch_size=32, test=False, aug_prob=0.1):
    
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
    
    print('####################', config['act'])
    if config['batch_size']==1:
        bn = False
    else:
        bn = True
    # Make the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE", device)
    model = UNet3DClassifier(in_channels=6, init_weights=config['init_weights'], act=config['act'], ns=config['ns'], dropout=config['dropout'], features=config['features'])
    model = model.to(device)
    
    # optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    optimizer = get_optimizer(name=config["optimizer"], 
                    model=model, 
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay'], 
                    momentum=config['momentum'])

    # Make the loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loader, val_loader = make_loader(batch_size=config['batch_size'], test=test)
   
    return model, train_loader, val_loader, criterion, optimizer


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
    # f1 = wandb.config.f1
    bias_init = wandb.config.bias_init
    ns = wandb.config.ns

    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    weight_decay = wandb.config.weight_decay
    
    c = {
        "act": act,
        "init_weights": init_weights,
        "features": wandb.config.features,
        "bias_init": bias_init,
        "ns": ns,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size, 
        "dropout": wandb.config.dropout,
        "weight_decay": weight_decay, 
        "optimizer": wandb.config.optimizer,
        "momentum": wandb.config.momentum,       
    }
    
    # make the model, data, and optimization problem
    
    model, train_loader, val_loader, criterion, optimizer = make(c, test=test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    early_stop_counter = 0
    early_stop_patience = 1
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

            # print('x.shape', x.shape)
            logits = model(x)  # Use a val batch
            probs = torch.sigmoid(logits)
            print("Mean output:", probs.mean().item(), probs)
            print("y:", y)
            print("Std output:", probs.std().item())
            print("Predictions:", (probs > 0.5).int().unique(return_counts=True))
            print()
        print("Linear weight std:", model.fc[2].weight.std().item())
        val_accuracy = correct_val / total_val
        print(f"Epoch {epoch+1} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # -- Early stopping based on val accuracy near 0.5 --
        print("ABS", abs(val_accuracy - val_acc_threshold), tolerance)
        if abs(val_accuracy - val_acc_threshold) <= tolerance:
            early_stop_counter += 1
            print("early_stop_counter += 1", early_stop_counter)
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_counter} epochs with val acc ~0.5\n")
                break
        else:
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
    #wandb sweep -e university_of_bath -p diox_prob sweep2.yaml
    #wandb agent -e university_of_bath -p diox_prob zl39yygx > train4.log 2>&1

    main()
