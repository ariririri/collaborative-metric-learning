from cml.model import CML, CMLLoss
from cml.dataloader import UserItemDataset
from cml import config

import numpy as np
import os 
from torch import nn, optim
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

def train():
    data_set = UserItemDataset()
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True)
    model = CML()
    model.train()
    criterion = CMLLoss()
    optimizer = optim.Adam(model.parameters(),
        lr=0.0002, betas=(0.5, 0.999))
    
    iteration = 0

    for data in dataloader:
        model.zero_grad()
        output = model(data)
        loss = criterion(output)
        loss.backward()
        optimizer.step()

        iteration += 1

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    # pytorchだといろいろ改善余地がある?
    model_for_saving = model
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


if __name__ == "__main__":
    train()