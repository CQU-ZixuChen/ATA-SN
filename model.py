import torch
import numpy as np
from thop import profile
import scipy.io as scio
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from edge_pool import EdgePooling
import torch.nn.functional as F
import argparse
from Dataset import MyGraphDataset

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.conv1 = ChebConv(1025, 512, 2)
        self.pool = EdgePooling(in_channels=1025)
        self.conv2 = ChebConv(512, 512, 2)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)

        self.lin1 = torch.nn.Linear(1024, 512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.lin3 = torch.nn.Linear(256, 5)
        

    def forward(self, data):
        x, edge_index, batch, A = data.x, data.edge_index, data.batch, data.A
        edge_index = edge_index.to(args.device)
        new_edge_index = edge_index_f

        x = F.relu(self.bn1(self.conv1(x, new_edge_index)))
        x = F.relu(self.bn2(self.conv2(x, new_edge_index)))

        # Class-wise output
        x = F.relu(self.bn3(self.lin1(x)))
        x = F.relu(self.bn4(self.lin2(x)))
        output = F.log_softmax(self.lin3(x), dim=-1)

        return output


def main():
    model = Net(args).to(args.device)
    dataset = MyGraphDataset('Name of your graph dataset')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for i, data in enumerate(zip(train_loader)):
        data1 = data[0].to(args.device)
        # print(data1.y)
        output = model(data1)


if __name__ == '__main__':
     main()




