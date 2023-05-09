import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv,ChebConv
import numpy as np
from utils.utils import MergeLayer
from torch_geometric.nn.inits import glorot, zeros
from torch.nn.utils import weight_norm
from math import log

class GlobalEmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout,use_memory,window=2,tcn_kernel_size=2,tcn_layers=3):
        super(GlobalEmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.use_memory = use_memory
        self.device = device
        self.window = window
        self.tcn_kernel_size=tcn_kernel_size
        self.tcn_layers=tcn_layers
        
        self.gcn_tcn = GCN_TCN(input_size=self.embedding_dimension, 
                              output_size=self.embedding_dimension, 
                              #num_channels=[self.embedding_dimension for _ in range(int(log(self.window,2))+1)],
                              num_channels=[self.embedding_dimension for _ in range(self.tcn_layers)],
                              kernel_size=self.tcn_kernel_size, 
                              dropout=self.dropout,
                              device=self.device,
                              num_nodes=self.node_features.shape[0],
                              n_layers=self.n_layers,
                              window=self.window)

    def compute_global_embedding(self, memory, source_nodes, timestamps, edge_idxs, global_edge_index, n_layers):
        """
        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(timestamps_torch)

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            node_global_embedding = self.gcn_tcn(memory, global_edge_index)
            

            node_global_embedding = node_global_embedding[source_nodes, :]

            
            return node_global_embedding

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):   
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, edge_index)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Cut out the extra paddings
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        param n_inputs: int, number of input channels
        param n_outputs: int, number of output channels
        param kernel_size: int, size of kernel
        param stride: int, stride
        param dilation: int, the dilation of each TCN layer
        param padding: int, size of padding
        param dropout: float, dropout ratio
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        param x: size of (Batch, input_channel, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        param num_inputs: int, number of input channels
        param num_channels: list, number of hidden_channels of each TCN layer
        param kernel_size: int, size of kernel
        param dropout: float, dropout ratio
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        param x: size of (Batch, input_channel, seq_len)
        return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)
    
class GCN_TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, device, num_nodes, n_layers=1, window=1):
        super(GCN_TCN, self).__init__()
        self.device = device
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

        self.gcn = GCN(in_channels=input_size,
                            hidden_channels=output_size,
                            out_channels=output_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            use_bn=True)
        
        self.window = window
        self.h_window = torch.zeros(num_nodes,output_size,1).to(self.device)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        if self.window==1:
             return x
        self.h_window = torch.cat([self.h_window, x.unsqueeze(-1)], dim=-1)
        self.h_window = self.h_window[:, :, -self.window-1:-1].detach()

        y = self.tcn(self.h_window)

        return self.linear(y[:, :, -1])