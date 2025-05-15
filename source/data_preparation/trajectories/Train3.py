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

def MLP(channels, batch_norm=True, act=None, ns=0.1):
    """Multi-layer perceptron, with configurable activation and batch normalization."""
    if act is None:
        # Default to a callable instance of LeakyReLU
        act = partial(nn.LeakyReLU, negative_slope=ns)
    
    def get_activation():
        # Handle both callable classes and instances
        if isinstance(act, partial):
            return act()  # Call the partial to create the instance
        elif isinstance(act, type):  # Class
            return act()
        elif isinstance(act, nn.Module):  # Instance
            return act
        else:
            raise TypeError("Unsupported activation type: {}".format(type(act)))

    return Seq(
        *[
            Seq(
                Lin(channels[i - 1], channels[i]),
                BN(channels[i]) if batch_norm else nn.Identity(),
                get_activation(),
            )
            for i in range(1, len(channels))
        ]
    )

class VoxelCNN(nn.Module):
    def __init__(self, input_channels, f1=32, output_size=20, init_weights='normal', act=nn.ReLU, ns=0.1):
        super(VoxelCNN, self).__init__()
        self.act = act or partial(nn.LeakyReLU, negative_slope=ns)
        print("CNN F1 is {}".format(f1))
        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, f1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(f1),
            act() if callable(act) else act,
            nn.MaxPool3d(kernel_size=2),  # Halves the spatial size

            
            nn.Conv3d(f1, f1*2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(f1*2),
            act() if callable(act) else act,
            nn.MaxPool3d(kernel_size=2),  # Halves the spatial size
        
            nn.Conv3d(f1*2, f1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(f1),
            act() if callable(act) else act,
            nn.MaxPool3d(kernel_size=2),  # Halves the spatial size
            
            nn.Conv3d(f1, f1 // 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(f1 // 2),
            act() if callable(act) else act,
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Ensures fixed size output
        )
        self.fc = nn.Linear(f1 // 2, output_size)

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


class My_Layer(nn.Module):
    def __init__(self, in_channels=33, out_channels=3, act=nn.LeakyReLU, transform_1_o=64, transform_4_i=128, p=False, nneigh=50, 
                 CNN_layer_in=9, CNN_layer_out=16, init_weights='normal', cnn_act=nn.LeakyReLU,
                 f1=32, ns=0.1, middle1 = [64,128], middle2=[32, 32, 32], bias_init=False, batch_norm=True):
        super(My_Layer, self).__init__()
      
        self.act = act if callable(act) else lambda: act
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transform_1_o = transform_1_o
        self.transform_4_i = transform_4_i
        self.p=p
        self.nneigh = nneigh
        self.transform_1 = MLP([self.in_channels,transform_1_o] + middle1 + [transform_4_i], batch_norm=batch_norm,act=self.act, ns=ns)
        self.transform_2 = MLP([transform_4_i]+middle2+[16], batch_norm=batch_norm, act=act, ns=ns)
        self.cnn_mesh = VoxelCNN(input_channels=CNN_layer_in, f1=f1, output_size=CNN_layer_out, init_weights=init_weights, act=cnn_act, ns=ns)
        self.bias_init = bias_init
        self.transform_5 = nn.Linear(16+CNN_layer_out, out_channels)

        # Pooling layer to aggregate across the 50 points
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces dimension to (N, 20, 1)

        # self.apply(self.init_weights2)
        if init_weights == 'normal':
            self.apply(self.init_weights2)
        elif init_weights == 'kaiming':
            self.apply(self.init_weights)
        elif init_weights == 'xavier':
            self.apply(self.init_weights_xavier)

   
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_weights_xavier(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_weights2(self,m):
        if isinstance(m, nn.Linear):# or isinstance(m, nn.Conv3d):
            print(m)
            init.normal_(m.weight, mean=0.0, std=0.01)  # Normal distribution initialization
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias to zero
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, cnn_x):
        batch_size = x.size(0)
        x = x.view(-1, self.in_channels)             # Reshape to [N * nneigh, in_channels]
        x = self.transform_1(x)  # [N * nneigh, in_channels] -> (N * nneigh, transform_1_o]
        x = x.view(batch_size, self.nneigh, self.transform_4_i)  # Reshape back to [N, nneigh, transform_1_o]
    
        # Pool across the 50 points to get a summary of shape [N, 20, 1]
        x = x.permute(0, 2, 1)          # [N, nneigh, transform_1_o] -> [N, transform_1_o, nneigh] for pooling
        x = self.pool(x)   # [N, transform_1_o, nneigh] -> [N, transform_1_o, 1]
        x = x.squeeze(-1)   # [N, transform_1_o, 1] -> [N, transform_1_o]

        # x = F.relu(self.transform_4(x)) # [N, 16]
        x = self.transform_2(x)
        # x = self.transform_4(x)
        cnn_out = self.cnn_mesh(cnn_x) # [N, CNN_layer_out=16]
        # print(cnn_out)
        x = torch.cat([x, cnn_out], dim=-1)
        x = self.transform_5(x)
        return x

# act=nn.LeakyReLU
nneig=50
# layer_ = My_Layer(in_channels=33-24, out_channels=3, act=act, transform_1_o=128, transform_4_i=256, 
#                   p=False, nneigh=nneig, CNN_layer_in=9, CNN_layer_out=16, init_weights='kaiming', cnn_act=act).to(device)


# extend pyg Dataset
from torch_geometric.data import Dataset
class ProteinLigandDataset(Dataset):
    def __init__(self, graphs=None, transform=None):
        super().__init__()
        self.graphs = graphs
        self.transform = transform
    def len(self):
        return len(self.graphs)
    def get(self, idx):
        return self.graphs[idx]


from torch.utils.data import Dataset, DataLoader
# extend torch dataset
class MyDataset(Dataset):
    def __init__(self, data_paths, batch_size, transform=None):
        self.data_paths = data_paths
        self.batch_size = batch_size  # Mini-batch size for graphs
        self.transform = transform

    def __len__(self):
        # Return the number of files (not the number of graphs within them)
        return len(self.data_paths)

    def __getitem__(self, index):
        print(f"Fetching data from {self.data_paths[index]}")
        data_time = time.time()
        sys.stdout.flush()
        Sim = self.data_paths[index].split('/')[0]
        ds = self.data_paths[index].split('/')[2]
        # Load graphs from a single .pt file (list of `torch_geometric.data.Data` objects)
        graphs = torch.load(self.data_paths[index])
        
        if self.transform:
            graphs = [self.transform(graph) for graph in graphs]
        
        # Extract features from graphs, need a distance to cat site maybe so we can filterbased on distance to metal
        # as in X = [d.x for d in graphs if torch.norm(d.y) < 2 and torch.norm(d.y) != 0 and d.cat_dist <= 20]
        X = [d.x for d in graphs if torch.norm(d.y) < 2 and torch.norm(d.y) != 0]
        voxel_descriptor_grids = [d.voxel_descriptor_grid for d in graphs  if torch.norm(d.y) < 2 and torch.norm(d.y) != 0]
        y = [d.y for d in graphs if torch.norm(d.y) < 2 and torch.norm(d.y) != 0]
        names = np.array([d.name for d in graphs if torch.norm(d.y) < 2 and torch.norm(d.y) != 0])
        zeros = [torch.norm(i) == 0 for i in y]

        shuffler = torch.randperm(len(X))
        X=torch.stack(X)[shuffler]
        voxel_descriptor_grids=torch.stack(voxel_descriptor_grids)[shuffler]
        y=torch.stack(y)[shuffler]
        names=np.array(names)[shuffler.numpy()]


        # print("Zeros: ", sum(zeros))
        # print(f"Returning {len(X)} graphs")
        print(f"# of samples {len(X)}")
        print("Data loading time {}".format(time.time()-data_time))
        return X, voxel_descriptor_grids, y, names, Sim, ds


def collate_fn(batch, batch_size=32):
    """
    This function will combine a list of items into mini-batches.
    We will divide the list of graphs into mini-batches based on the specified batch size.
    """
    all_X = []
    all_voxel_descriptor_grids = []
    all_y = []
    all_names = []
    
    # Iterate over each file (each item in the batch)
    for X, voxel_descriptor_grids, y, names, Sim, ds in batch:
        # print(names)
        # Create mini-batches within each file
        num_batches = len(X) // batch_size  # Get number of mini-batches
        remainder = len(X) % batch_size  # If there's a remainder, we handle it as a smaller batch

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            # all_X.append(torch.stack(X[start_idx:end_idx]))
            # all_voxel_descriptor_grids.append(torch.stack(voxel_descriptor_grids[start_idx:end_idx]))
            # all_y.append(torch.stack(y[start_idx:end_idx]))
            all_X.append(X[start_idx:end_idx])
            all_voxel_descriptor_grids.append(voxel_descriptor_grids[start_idx:end_idx])
            all_y.append(y[start_idx:end_idx])
            # print(names[start_idx:end_idx])
            all_names.append(names[start_idx:end_idx])

        # Handle any remainder if it's smaller than the batch size
        if remainder > 0:
            # all_X.append(torch.stack(X[-remainder:]))
            # all_voxel_descriptor_grids.append(torch.stack(voxel_descriptor_grids[-remainder:]))
            # all_y.append(torch.stack(y[-remainder:]))
            all_X.append(X[-remainder:])
            all_voxel_descriptor_grids.append(voxel_descriptor_grids[-remainder:])
            all_y.append(y[-remainder:])
            all_names.append(names[-remainder:])
    
    return all_X, all_voxel_descriptor_grids, all_y, all_names, Sim, ds


#############
# data_paths = []
# with open('out_minor.txt', 'r') as f:
#     for line in f.readlines():
#         data_paths.append(line.strip())

# from functools import partial
# batch_size = 10  # Set your desired batch size here
# dataset = MyDataset(data_paths=data_paths[-40:-10], batch_size=10)
# loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_fn,batch_size=20))

# Example usage
# for X_batch, voxel_descriptor_grids_batch, y_batchs, names_batch, Sim, ds in loader:
#     print(f"Mini-batch size: {len(y_batchs)} graphs", "in Sim {}, ds {}".format(Sim,ds))
#     print("normc", torch.concat([torch.norm(i,dim=1) for i in y_batchs]).mean())
    # break

##############


def custom_loss(output, target):
    '''
    if any movement in x or y or z is 0.0 in the target you cannot divide, so make 0.0001
    '''
    target[target==0.0] = 0.0001
    mse_loss_per_sample = torch.norm((output - target) ** 2, dim=1)/torch.norm(target, dim=1) * 10
    # Compute MSE for each sample individually
    individual_losses = torch.mean((output - target) ** 2, dim=1)  # MSE along feature dimensions
    individual_losses2 = torch.mean(abs((output - target) ** 2/target), dim=1)*10
    # Return the sum or another operation (e.g., mean) across samples
    return torch.sum(individual_losses), mse_loss_per_sample  # Or keep as individual_losses for further customization


def custom_loss2(output, target):
    '''
    if any movement in x or y or z is 0.0 in the target you cannot divide, so make 0.0001
    '''
    target[target==0.0] = 0.0001
    # Compute MSE for each sample individually
    individual_losses = torch.mean((output - target) ** 2, dim=1)  # MSE along feature dimensions
    mse_loss_per_sample = torch.norm((output - target) ** 2, dim=1)/torch.norm(target, dim=1) * 10

    # Compute the sign mismatch penalty (1 if mismatch, 0 if match)
    sign_mismatch = (torch.sign(output) != torch.sign(target)).float()  # Element-wise sign mismatch
    sign_mismatch_penalty = sign_mismatch.sum(dim=1)  # Sum mismatches across feature dimensions
    # Compute the L1 regularization penalty
    penalty = 1.5 * torch.abs(((target - output)/target))
    l1_penalty = penalty.sum(dim=1)  # Sum L1 penalties across feature dimensions
    # Combine all penalties and individual losses
    individual_losses = individual_losses + 4*sign_mismatch_penalty + l1_penalty
    # Return the sum of all sample losses (or use mean for averaging across batch)
    return torch.sum(individual_losses), mse_loss_per_sample


def custom_loss_with_sign_penalty(output, target, sign_penalty_weight=5, lambda_penalty = 10):
    mse_loss = nn.MSELoss()(output, target)
    mse_loss_per_sample = torch.norm((output - target) ** 2, dim=1)/torch.norm(target, dim=1) * 10

    # Compute sign mismatch penalty
    sign_mismatch = (torch.sign(output) != torch.sign(target)).float()  # 1 where signs mismatch, 0 otherwise
    sign_penalty = sign_mismatch.mean()  # Average penalty across all elements

    # Add an additional penalty term
    # lambda_penalty = 0.1  # Regularization strength
    penalty = lambda_penalty * torch.sum(torch.abs(target - output)) 
    penalty_per_sample = lambda_penalty * torch.abs((target - output)/target)
    # Combine MSE and sign penalty
    total_loss = mse_loss + sign_penalty_weight * sign_penalty + penalty

    total_loss_per_sample = (
        mse_loss_per_sample
        + sign_penalty_weight * sign_mismatch
        + penalty_per_sample
    )
    return total_loss, total_loss_per_sample


def MSE_Loss(output, target):
    mse_loss_per_sample = torch.norm((output - target) ** 2, dim=1)/torch.norm(target, dim=1) * 10
    mse_loss = nn.MSELoss()(output, target)

    return mse_loss, mse_loss_per_sample


neigh=50
atom=False
# 55987677425


def make_loader2(batch_size=32, test=False):
    data_paths = []
    with open('smaller/train_too_many.txt', 'r') as f:
        for line in f.readlines():
            data_paths.append(line.strip())
    if test:
        data_paths=data_paths[0:]
        print("UGHHHHHHHH", data_paths)
        sys.stdout.flush()
        
    
    dataset = MyDataset(data_paths=data_paths, batch_size=batch_size)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, 
                        collate_fn=partial(collate_fn,batch_size=batch_size))
    print("LEN loader", len(loader))
    sys.stdout.flush()
    return loader

def make_val_loader(batch_size=32, test=False):
    data_paths = []
    with open('smaller/val_too_many.txt', 'r') as f:
        for line in f.readlines():
            data_paths.append(line.strip())
    if test:
        data_paths=data_paths[0:]
        print("UGHHHHHHHH", data_paths)
        sys.stdout.flush()
        
    
    dataset = MyDataset(data_paths=data_paths, batch_size=batch_size)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, 
                        collate_fn=partial(collate_fn,batch_size=batch_size))
    print("LEN VAL loader", len(loader))
    sys.stdout.flush()
    return loader

def make(config, test=False):
    # Make the data
    print("################# 'batch_size'", config['batch_size'], test)
    train_loader = make_loader2(batch_size=config['batch_size'], test=test)
    print('####################', config['act'])
    if config['batch_size']==1:
        bn = False
    else:
        bn = True
    # Make the model
    if atom:
        model = My_Layer(in_channels=33-24, out_channels=3, act=config['act'], transform_1_o=config['transform_1_o'], transform_4_i=config['transform_4_i'], 
                         p=False, nneigh=nneig, CNN_layer_in=3, CNN_layer_out=config['CNN_layer_out'], init_weights=config['init_weights'], cnn_act=config['act'], f1=config['f1'],
                         ns=config['ns'], batch_norm=bn).to(device)
    else:
        model = My_Layer(in_channels=33, out_channels=3, act=config['act'], transform_1_o=config['transform_1_o'], transform_4_i=config['transform_4_i'], 
                         p=False, nneigh=nneig, CNN_layer_in=3, CNN_layer_out=config['CNN_layer_out'], init_weights=config['init_weights'], cnn_act=config['act'], f1=config['f1'],
                         ns=config['ns'], batch_norm=bn).to(device)
    
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    # model.train(True)
    # Make the loss and optimizer
    criterion = config['criterion']
   
    return model, train_loader, criterion, optimizer


def train_one_epoch_file(epoch_number, optimizer, X_batchs, voxel_descriptor_grids_batch, y_batchs, names_batch, 
                         Sim, ds, criterion, model, status=10, status2=100, dataset=1):
    # wandb.watch(model, criterion, log="all", log_freq=10)
    
    running_loss = 0.
    running_loss2 = 0.
    good_dict = {}
    bad_dict = {}
    total_mse_loss = 0
    c=0
    total_samples=sum([len(i) for i in X_batchs])
    df_lists = []
    df_l_bad = []
    if len(X_batchs) < status:
        status = 1
    
    for i in range(len(X_batchs)):
        # Every data instance is an input + label pair
        inputs1=X_batchs[i].reshape(-1,50,33).to(device)
        # if len(inputs1) == 1:
        #     continue
        inputs2=voxel_descriptor_grids_batch[i].reshape(-1,40,40,40,3).to(torch.float32).to(device)
        inputs2 = inputs2.permute(0,4, 1, 2, 3)
        labels = y_batchs[i].reshape(-1,3).to(device)
        names = names_batch[i]
        frames = np.array([i.split('_')[1] for i in names])
        gases = np.array([i.split('_')[3] for i in names])
        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        if len(inputs1)==1:
            inputs1 = inputs1.unsqueeze(0)
            inputs2 = inputs2.unsqueeze(0)
            continue
        # Make predictions for this batch
        # print("Inputs sizes {}, {}".format(inputs1.shape,inputs2.shape))
        outputs = model(inputs1, inputs2)

        # Compute the loss and its gradients
        loss, il = criterion(outputs, labels)
        loss.backward() # back prop on the custom loss
        mse_loss = nn.MSELoss()(outputs, labels) # report the mse loss

        abs_diff = torch.abs(labels - outputs)  # Element-wise absolute difference
        average_norm = torch.norm((labels+outputs)/2, p=2, dim=1) 
        normalized_diff = abs_diff / (average_norm.unsqueeze(1) + 1e-8)

        det=torch.norm(normalized_diff, dim=1).detach().cpu()
        good = np.where(det < 0.15)[0]
        bad = np.where(det > 1.5)[0]
        # if criterion == MSE_Loss:
        #     bad = np.where(det > 60)[0]
        # elif criterion == custom_loss2 or criterion == custom_loss:
        #     bad = np.where(det > 20)[0]
        good_f = names[good]
        good_pred = outputs[good]
        good_targ = labels[good]
        bad_f = names[bad]
        bad_pred = outputs[bad]
        bad_targ = labels[bad]
        
        # Adjust learning weights
        optimizer.step()
        running_loss2 += mse_loss.item()
        running_loss += mse_loss.item()
        if (i) % status == (status-1):
            last_loss = running_loss / status # loss per batch
            print('         batch {} loss: {}'.format(i + 1, last_loss))
            sys.stdout.flush()
            running_loss = 0.

        if len(good_f) > 0:
            df = pd.DataFrame(
                {
                    'Sim': np.repeat(Sim,len(good)),
                    'dataset': np.repeat(ds,len(good)),
                    'drame': frames[good],
                    'gase': gases[good],
                    'predictx': good_pred.detach().cpu()[:,0],
                    'predicty': good_pred.detach().cpu()[:,1],
                    'predictz': good_pred.detach().cpu()[:,2],
                    'targetx': good_targ.cpu()[:,0],
                    'targety': good_targ.cpu()[:,1],
                    'targetz': good_targ.cpu()[:,2]
                    
                }
            )
            df_lists.append(df)
            # good_list.append(good_f)
            for i, k in enumerate(good_f):
               good_dict[k] = [good_pred[i], good_targ[i]]  
        if len(bad_f) > 0:
            df = pd.DataFrame(
                {
                    'Sim': np.repeat(Sim,len(bad)),
                    'dataset': np.repeat(ds,len(bad)),
                    'drame': frames[bad],
                    'gase': gases[bad],
                    'predictx': bad_pred.detach().cpu()[:,0],
                    'predicty': bad_pred.detach().cpu()[:,1],
                    'predictz': bad_pred.detach().cpu()[:,2],
                    'targetx': bad_targ.cpu()[:,0],
                    'targety': bad_targ.cpu()[:,1],
                    'targetz': bad_targ.cpu()[:,2]
                    
                }
            )
            df_l_bad.append(df)
            for i, k in enumerate(bad_f):
               bad_dict[k] = [bad_pred[i], bad_targ[i]]
    
    avg_mse_loss = running_loss2/len(X_batchs)

    logging.info('    dataset {}, epoch {}, running_loss {}:'.format(dataset,epoch_number,running_loss))
    logging.info('           ouputs {}'.format(outputs[-4:]))
    logging.info('           labels {}'.format(labels[-4:]))
    if len(df_lists) > 0:
        df_lists=pd.concat(df_lists)
    if len(df_l_bad) > 0:
        df_l_bad=pd.concat(df_l_bad)

    return avg_mse_loss, total_samples, good_dict, bad_dict, df_lists, df_l_bad, running_loss



def val_one_epoch_file(epoch_number, X_batchs, voxel_descriptor_grids_batch, y_batchs, names_batch, 
                         Sim, ds, criterion, model, status=10, status2=100, dataset=1):
    # wandb.watch(model, criterion, log="all", log_freq=10)
    
    running_loss = 0.
    running_loss2 = 0.
    good_dict = {}
    bad_dict = {}
    total_mse_loss = 0
    c=0
    total_samples=sum([len(i) for i in X_batchs])
    df_lists = []
    df_l_bad = []
    if len(X_batchs) < status:
        status = 1
    
    for i in range(len(X_batchs)):
        # Every data instance is an input + label pair
        inputs1=X_batchs[i].reshape(-1,50,33).to(device)
        # if len(inputs1) == 1:
        #     continue
        inputs2=voxel_descriptor_grids_batch[i].reshape(-1,40,40,40,3).to(torch.float32).to(device)
        inputs2 = inputs2.permute(0,4, 1, 2, 3)
        labels = y_batchs[i].reshape(-1,3).to(device)
        names = names_batch[i]
        frames = np.array([i.split('_')[1] for i in names])
        gases = np.array([i.split('_')[3] for i in names])
        

        # Zero your gradients for every batch!
        # optimizer.zero_grad()

        if len(inputs1)==1:
            inputs1 = inputs1.unsqueeze(0)
            inputs2 = inputs2.unsqueeze(0)
            continue
        # Make predictions for this batch
        # print("Inputs sizes {}, {}".format(inputs1.shape,inputs2.shape))
        outputs = model(inputs1, inputs2)

        # Compute the loss and its gradients
        loss, il = criterion(outputs, labels)
        # loss.backward() # back prop on the custom loss
        mse_loss = nn.MSELoss()(outputs, labels) # report the mse loss

        abs_diff = torch.abs(labels - outputs)  # Element-wise absolute difference
        average_norm = torch.norm((labels+outputs)/2, p=2, dim=1) 
        normalized_diff = abs_diff / (average_norm.unsqueeze(1) + 1e-8)

        det=torch.norm(normalized_diff, dim=1).detach().cpu()
        good = np.where(det < 0.15)[0]
        bad = np.where(det > 1.5)[0]
        # if criterion == MSE_Loss:
        #     bad = np.where(det > 60)[0]
        # elif criterion == custom_loss2 or criterion == custom_loss:
        #     bad = np.where(det > 20)[0]
        good_f = names[good]
        good_pred = outputs[good]
        good_targ = labels[good]
        bad_f = names[bad]
        bad_pred = outputs[bad]
        bad_targ = labels[bad]
        
        # Adjust learning weights
        # optimizer.step()
        running_loss2 += mse_loss.item()
        running_loss += mse_loss.item()
        if (i) % status == (status-1):
            last_loss = running_loss / status # loss per batch
            print('         batch {} loss: {}'.format(i + 1, last_loss))
            sys.stdout.flush()
            running_loss = 0.

        if len(good_f) > 0:
            df = pd.DataFrame(
                {
                    'Sim': np.repeat(Sim,len(good)),
                    'dataset': np.repeat(ds,len(good)),
                    'drame': frames[good],
                    'gase': gases[good],
                    'predictx': good_pred.detach().cpu()[:,0],
                    'predicty': good_pred.detach().cpu()[:,1],
                    'predictz': good_pred.detach().cpu()[:,2],
                    'targetx': good_targ.cpu()[:,0],
                    'targety': good_targ.cpu()[:,1],
                    'targetz': good_targ.cpu()[:,2]
                    
                }
            )
            df_lists.append(df)
    
            for i, k in enumerate(good_f):
               good_dict[k] = [good_pred[i], good_targ[i]]  
        if len(bad_f) > 0:
            df = pd.DataFrame(
                {
                    'Sim': np.repeat(Sim,len(bad)),
                    'dataset': np.repeat(ds,len(bad)),
                    'drame': frames[bad],
                    'gase': gases[bad],
                    'predictx': bad_pred.detach().cpu()[:,0],
                    'predicty': bad_pred.detach().cpu()[:,1],
                    'predictz': bad_pred.detach().cpu()[:,2],
                    'targetx': bad_targ.cpu()[:,0],
                    'targety': bad_targ.cpu()[:,1],
                    'targetz': bad_targ.cpu()[:,2]
                    
                }
            )
            df_l_bad.append(df)
            for i, k in enumerate(bad_f):
               bad_dict[k] = [bad_pred[i], bad_targ[i]]
    
    avg_mse_loss = running_loss2/len(X_batchs)

    logging.info('    val dataset {}, epoch {}, running_loss {}:'.format(dataset,epoch_number,running_loss))
    logging.info('          val ouputs {}'.format(outputs[-4:]))
    logging.info('          val labels {}'.format(labels[-4:]))
    if len(df_lists) > 0:
        df_lists=pd.concat(df_lists)
    if len(df_l_bad) > 0:
        df_l_bad=pd.concat(df_l_bad)

    return avg_mse_loss, total_samples, good_dict, bad_dict, df_lists, df_l_bad, running_loss


def key_of_min(d):
  return min(d, key = d.get)


sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "val_avg_epoch_loss"},
    "parameters": {
        "batch_size": {"values": [16, 24, 32, 64]},
        "epochs": {"values": [10,15,20]},
        # "epochs": {"values": [5, 10, 15,20, 25, 40]},
        "learning_rate": {"values": [0.01,0.005, 0.001, 0.002, 0.0001, 0.0003, 0.0005]},
        # "act": {'values': [nn.ReLU, nn.ELU, nn.LeakyReLU, nn.PReLU]},
        "act": {'values': ['ReLU', 'ELU', 'LeakyReLU', 'PReLU']},
        "transform_1_o": {"values": [32, 64, 128]},
        "transform_4_i": {"values": [128, 256]},
        "middle1": {"values": [
            [64,64],
            [32,64],
            [64,128],
            [64,64,64],
            [64,64,64,64],
            [64,64,64,128],
            [32,32,32],
            [32,32,64,64],
        ]},
        "middle2": {"values": [
            # [64,64,64],
            # [64,128,64],
            [128,128,64],
            [128,64,64,64],
            [128,128,128],
            [64,64,64,64],
            [64,64,64,64,64],
            [256,256,256],
            [128,64,32],
            [512,512,512],
            [128,256,64]
        ]},
        "init_weights": {"values": ['kaiming', 'xavier', 'normal']},
        "f1": {"values": [32, 64, 128, 256, 512]},
        "bias_init": {"values": [True]},
        "ns": {"values": [0.1, 0.2, 0.4]},
        "criterion": {"values": ['MSELoss','custom_loss2','custom_loss']},
        "CNN_layer_out": {"values": [64 ,32, 16, 10, 8]},
    },
}




from tqdm.auto import tqdm

import logging
from importlib import reload
# import wandb

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
  
    transform_1_o = wandb.config.transform_1_o
    transform_4_i = wandb.config.transform_4_i
    middle1 = wandb.config.middle1
    middle2 = wandb.config.middle2
    init_weights = wandb.config.init_weights
    f1 = wandb.config.f1
    bias_init = wandb.config.bias_init
    ns = wandb.config.ns
    
    criterion1 = wandb.config.criterion
    if criterion1 == 'MSELoss':
        criterion = MSE_Loss
    elif criterion1 == 'custom_loss':
        criterion = custom_loss
    elif criterion1 == 'custom_loss2':
        criterion = custom_loss2
    else:
        criterion1 = custom_loss_with_sign_penalty

    learning_rate = wandb.config.learning_rate
    CNN_layer_out = wandb.config.CNN_layer_out
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    
    c = {
        "act": act,
        "transform_1_o": transform_1_o,
        "middle1": middle1,
        "middle2": middle2,
        "transform_4_i":transform_4_i,
        "init_weights": init_weights,
        "f1": f1,
        "bias_init": bias_init,
        "ns": ns,
        "criterion": criterion,
        "learning_rate": learning_rate,
        "CNN_layer_out": CNN_layer_out,
        "epochs": epochs,
        "batch_size": batch_size,       
    }
    print('MLP:', [transform_1_o] + middle1 + [transform_4_i] + middle2)
    sys.stdout.flush()
    good=[]
    bad=[]
    # make the model, data, and optimization problem
    model, loader, criterion, optimizer = make(c, test=test)
    val_loader = make_val_loader()
    print("BRIAN!!", len(loader), len(val_loader))
    # status=int(len(loader)/40)
    # status2=int(len(loader)/40*4)
    ldir='test'
    mdir='test_models_good'

    s14=str(c['transform_1_o'])+'-'+str(c['transform_4_i'])
    m1='..'.join([str(i) for i in c['middle1']])
    m2='..'.join([str(i) for i in c['middle2']])
    lr=round(learning_rate,5)
    filename='{}/output_ep_{}_bs_{}_ns_{}_lr_{}_crt_{}_14_{}_inw_{}_m1_{}_m2_{}_a_{}_f1_{}.log'.format(ldir,epochs,batch_size,ns,lr,criterion1,s14,init_weights,m1,m2,act1,f1)
    print("FILENAME",filename)
    reload(logging)
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        filemode='w'
    )
    logging.info('c: {}'.format(c))

    total_loss = 0
    # dataset = 1
    total_samples = 0
    dataset_loss = {}
    dataset_gb = {}
    all_good = 0
    all_bad = 0
    best_avg_loss = torch.inf
    good_dfs = []
    bad_dfs = []

    pt='{}/output_ep_{}_bs_{}_ns_{}_lr_{}_crt_{}_14_{}_inw_{}_m1_{}_m2_{}_a_{}_f1_{}.pt'.format(mdir,epochs,batch_size,ns,lr,criterion1,s14,init_weights,m1,m2,act1,f1)
    best_val_avg_epoch_loss = torch.inf
        
    for epoch in tqdm(range(epochs)):
        e_time = time.time()
        
        model.train(True)
        avg_loss_array = []
        epoch_loss = 0
        dataset = 1
        for X_batch, voxel_descriptor_grids_batch, y_batch, names_batch, Sim, ds in loader:
            print(f"Mini-batch train size: {len(X_batch)} graphs")
            sys.stdout.flush()           
        
            avg_loss, l, good_dict, bad_dict, good_df, bad_df, epoch_ds_loss = train_one_epoch_file(epoch, optimizer, X_batch,voxel_descriptor_grids_batch, y_batch, names_batch, Sim, ds, criterion, model, dataset=dataset)
            avg_loss_array.append(avg_loss)

            print('    avg_loss for dataset {}, epoch {}, length {}, is {}'.format(dataset, epoch, l, avg_loss))
            sys.stdout.flush()
            
            logging.info('avg_loss for dataset {}, epoch {}, length {}, is {}'.format(dataset, epoch, l, avg_loss))
            if epoch==0:
                dataset_loss[dataset] = {'len': 0, 'epoch_losses': []}
                dataset_gb[dataset] = {'good': [], 'bad': []}
                total_samples += l
                normc = torch.concat([torch.norm(i,dim=1) for i in y_batch]).mean()
                print("mean distance travelled", normc)
                sys.stdout.flush()
                dataset_loss[dataset]['len'] = l
            good_dict['Sim'] = Sim
            bad_dict['Sim'] = Sim
            
            epoch_loss = np.mean(avg_loss_array) # is an array of [(means of mini-batches), ..., for each dataset]
            dataset_loss[dataset]['epoch_losses'].append(epoch_loss)
            dataset_gb[dataset]['good'].append(good_dict)
            all_good += len(good_dict)
            dataset_gb[dataset]['bad'].append(bad_dict)
            all_bad += len(bad_dict)
            dataset += 1
        
        model.eval()
        dataset = 1
        with torch.no_grad():
            val_avg_loss_array = []
            for X_batch, voxel_descriptor_grids_batch, y_batch, names_batch, Sim, ds in val_loader:
                print(f"Mini-batch val size: {len(X_batch)} graphs")
                sys.stdout.flush()
                avg_loss, l, good_dict, bad_dict, good_df, bad_df, epoch_ds_loss = val_one_epoch_file(epoch, X_batch,voxel_descriptor_grids_batch, y_batch, names_batch, Sim, ds, criterion, model, dataset=dataset)

                if epoch==0:
                    normc = torch.concat([torch.norm(i,dim=1) for i in y_batch]).mean()
                    print("mean distance travelled", normc)

                if len(good_df) > 0:
                    good_dfs.append(good_df)
                if len(bad_df) > 0:
                    bad_dfs.append(bad_df)

                val_avg_loss_array.append(avg_loss)
                print('    avg_loss for dataset {}, epoch {}, length {}, is {}'.format(dataset, epoch, l, avg_loss))
                dataset += 1
        val_epoch_loss = np.mean(val_avg_loss_array)
 
        print("        Epoch {}: average train loss {}".format(epoch,epoch_loss))
        print("        Epoch {}: average val loss {}".format(epoch,val_epoch_loss))
        sys.stdout.flush()
        # print(_dict)
        
        # total_loss_train += epoch_loss
        # total_val_loss += val_epoch_loss
        
    
        _dict={
                "train_avg_epoch_loss": epoch_loss,
                "epoch": epoch,
                "val_avg_epoch_loss": val_epoch_loss,
                # "train_mse": train_mse,
            }
        wandb.log(
                _dict
            )
        
        if (val_epoch_loss) < best_val_avg_epoch_loss:
            best_val_avg_epoch_loss = val_epoch_loss
            print('\nSAVING with val_avg_loss {} in epoch {}\n'.format(val_epoch_loss, epoch))
            
            logging.info('\nSAVING with val_avg_loss {} in epoch {}\n'.format(val_epoch_loss, epoch))
            torch.save(model, pt)
        print("Epoch {} time {}".format(epoch ,time.time()-e_time))

    good_dfs = pd.concat(good_dfs)
    good_dfs.to_csv('{}/output_ep_{}_bs_{}_ns_{}_lr_{}_crt_{}_14_{}_inw_{}_m1_{}_m2_{}_a_{}_f1_{}_good.csv'.format(ldir,epochs,batch_size,ns,lr,criterion1,s14,init_weights,m1,m2,act1,f1), index=False)
    bad_dfs = pd.concat(bad_dfs)
    bad_dfs.to_csv('{}/output_ep_{}_bs_{}_ns_{}_lr_{}_crt_{}_14_{}_inw_{}_m1_{}_m2_{}_a_{}_f1_{}_bad.csv'.format(ldir,epochs,batch_size,ns,lr,criterion1,s14,init_weights,m1,m2,act1,f1), index=False)


    print("TOTAL time {}".format(time.time()-start))

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    

    run.finish()


# sweep_id = wandb.sweep(sweep=sweep_configuration, project="un")

# wandb.agent(sweep_id, function=main, count=20)
if __name__ == '__main__':
    #wandb sweep -e university_of_bath -p small_project sweep.yaml
    #wandb agent -e university_of_bath -p small_project ex445p67 > smaller/train12.log 2>&1

    main()
