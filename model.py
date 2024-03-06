import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from edge_pool import EdgePooling
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'



class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # XJTU
        self.conv1 = ChebConv(1025, 512, 2)
        self.pool1 = EdgePooling(in_channels=1025)
        self.conv2 = ChebConv(512, 512, 2)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)

        self.lin1 = torch.nn.Linear(1024, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 4)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(256)

    def forward(self, data):
        x, edge_index, batch, A = data.x, data.edge_index, data.batch, data.A

        edge_index1, edge_index2, batch, edge_score = self.pool1(x=x, edge_index=edge_index, batch=batch)
        edge_index1 = edge_index1.to(args.device)
        edge_index2 = edge_index2.to(args.device)

        x1 = F.relu(self.bn1(self.conv1(x, edge_index1)))
        x1 = F.relu(self.bn2(self.conv2(x1, edge_index1)))
        x1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.bn3(self.conv1(x, edge_index2)))
        x2 = F.relu(self.bn4(self.conv2(x2, edge_index2)))
        x2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        # Class-wise output
        x1 = F.relu(self.bn5(self.lin1(x1)))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.bn6(self.lin2(x1)))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        feature1 = x1
        x1 = F.relu(self.lin3(x1))

        x2 = F.relu(self.bn5(self.lin1(x2)))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.bn6(self.lin2(x2)))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        feature2 = x2
        x2 = F.relu(self.lin3(x2))

        output = F.softmax(0.5 * (x1 + x2), dim=-1)
        output1 = F.softmax(x1, dim=-1)
        output2 = F.softmax(x2, dim=-1)

        feature = torch.cat([feature1, feature2], dim=1)

        return output1, output2, output, feature
