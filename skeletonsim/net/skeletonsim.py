import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class

class SkeletonSim(nn.Module):


    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, 
                 mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_hidden = self.encoder.fc.weight.shape[1]
                dim_out = self.encoder.fc.weight.shape[0]
                self.encoder.fc = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  self.encoder.fc,
                                                  nn.BatchNorm1d(dim_out))

                self.predictor = nn.Sequential(nn.Linear(dim_out, dim_out//2),
                                                        nn.BatchNorm1d(dim_out//2),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(dim_out//2, dim_out))                                 


    def forward(self, im_1, im_2 = None, im_3 = None):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        """

        if not self.pretrain:
            return self.encoder(im_1)

        # compute online features
        z1 = self.encoder(im_1)
        z2 = self.encoder(im_2)
        z3 = self.encoder(im_3)     

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)       
        p3 = self.predictor(z3)
        
        return p1, p2, p3, z1, z2, z3
        