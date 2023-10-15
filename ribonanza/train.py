from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from dataset import RibonanzaDataset
from model import RNAPredictor, RNALoss
import tqdm as tqdm
from params import *

train_iter = RibonanzaDataset(file=TRAIN_FILE)
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

model = RNAPredictor(d_model=192, n_heads=6)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = RNALoss()

def train_epoch(train_dataloader: DataLoader, model: RNAPredictor):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        print(inputs, labels)
        preds = model(inputs)

        loss = loss_fn(preds, labels)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

        losses += loss.item()
    
    return losses / len(train_dataloader)

def evaluate(val_dataloader: DataLoader, model: RNAPredictor):
    model.eval()
    losses = 0


    for inputs, labels in tqdm(val_dataloader):
        print(inputs, labels)
        preds = model(inputs)

        loss = loss_fn(preds, labels)

        losses += loss.item()
    
    return losses / len(val_dataloader)

for epoch in range(EPOCH):
    continue
