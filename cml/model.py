import torch
import torch.nn as nn
import torch.nn.functional as F


class CML(nn.Module):
    def __init__(self, user_size=7947, item_size=25975, embed_dim=20):
        super(CML, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(user_size, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_size, embed_dim, padding_idx=0)

    
    def forward(self, data):
        #batch, user 
        self.user = self.user_embedding(data[:,  0])
        self.pos = self.item_embedding(data[:, 1])
        self.neg = self.item_embedding(data[:, 3])
        # これでいいのか?
        return torch.cat([self.user, self.pos, self.neg], axis=1).reshape(-1, self.embed_dim, 3)
    

class CMLLoss(nn.Module):
    def __init__(self):
        super(CMLLoss, self).__init__()
        self.pos_distance =  torch.nn.MSELoss()
        self.neg_distance =  torch.nn.MSELoss()
    
    def forward(self, x):
        loss = self.emebdding_loss(x)
        return loss
    
    def emebdding_loss(self, x):
        mse =  torch.nn.MSELoss()
        pos_distance = mse(x[:, :, 0], x[:, :, 1])
        neg_distance = mse(x[:, :, 1], x[:, :, 2])
        closet_neg = torch.min(neg_distance)
        margin = 10
        loss = pos_distance - closet_neg + margin
        loss[loss < 0] = 0
        return loss

