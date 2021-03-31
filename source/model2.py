import numpy as np
import torch
from utils import to_sparse
from torch import sparse
class Graph_conv(torch.nn.Module):
    def __init__(self,input_dim,out_dim,feature):
        super(Graph_conv, self).__init__()
        self.feature=feature
        self.weight=torch.nn.parameter.Parameter(torch.randn(input_dim,out_dim))
        init_range=np.sqrt(6/(input_dim+out_dim))
        torch.nn.init.uniform_(self.weight,-init_range,init_range)
        self.drop=torch.nn.Dropout(0.5)

    def forward(self,features,adj):
        """

        :param features:node features,shape(node_num,input_dim)
        :param adj: matrix of adjacent,there this uses weight matrix
        :return:
        """
        if not self.feature:
            metrics=torch.sparse.mm(adj.float(),features.float())
        else:
            metrics=adj.float()
        features=torch.mm(metrics,self.weight)
        features=self.drop(features)
        return features
class Model(torch.nn.Module):
    def __init__(self,layer_nums,layer_dims,node_num,label_num):
        super(Model, self).__init__()
        assert layer_nums==len(layer_dims)-1
        # self.init_embedding=torch.nn.Parameter(torch.randn(node_num,layer_dims[0]))
        # torch.nn.init.xavier_uniform_(self.init_embedding, gain=1)
        layers=[]
        for i in range(layer_nums):
            if i == 0:
                feature=True
            else:
                feature=False
            layers.append(Graph_conv(layer_dims[i],layer_dims[i+1],feature))
        self.layers=torch.nn.ModuleList(layers)
        self.linear=torch.nn.Linear(layer_dims[-1],label_num)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax(-1)
    def forward(self,A_,feature):
        #因为A_一直保持不变，因此在model开始计算避免重复计算

        # feature=self.init_embedding
        for layer in self.layers:
            feature=self.relu(feature)
            feature=layer(feature,A_)
        # feature=self.layers[1](self.relu(self.layers[0](feature,A_)),A_)
        # feature=self.linear(feature)
        return feature



