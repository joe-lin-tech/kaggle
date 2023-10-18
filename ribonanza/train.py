from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
import torch
import torch.nn as nn
from dataset import RibonanzaDataset, LengthMatchSampler
from model import RNAPredictor, RNALoss
from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer
import wandb
import random
import os
from params import *

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

wandb.init(
    # set the wandb project where this run will be logged
    project="ribonanza",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED
    }
)

model = RNAPredictor(d_model=192, n_heads=6)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
loss_fn = RNALoss()

def train_epoch(train_dataloader: DataLoader, model: RNAPredictor, optimizer, scheduler):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        # print(inputs, labels)
        preds = model(inputs)

        loss = loss_fn(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.zero_grad()
        optimizer.step()

        losses += loss.item()

        if ((i + 1) % LOG_INTERVAL == 0) or (i + 1 == len(train_dataloader)):
            wandb.log({ "loss": loss.item() })
        
        scheduler.step(epoch + i / len(train_dataloader))
    
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


for fold in range(N_FOLDS):
    train_iter = RibonanzaDataset(file=TRAIN_FILE, fold=fold)
    # train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)
    train_sampler = LengthMatchSampler(RandomSampler(train_iter), batch_size=BATCH_SIZE, drop_last=True)
    train_dataloader = DataLoader(train_iter, batch_sampler=train_sampler)

    val_iter = RibonanzaDataset(file=TRAIN_FILE, fold=fold, split='val')
    val_sampler = LengthMatchSampler(SequentialSampler(val_iter), batch_size=BATCH_SIZE, drop_last=True)
    val_dataloader = DataLoader(val_iter, batch_sampler=val_sampler)

    wandb.watch(model, log_freq=LOG_INTERVAL)

    for epoch in range(EPOCHS):
        start_time = timer()
        train_loss = train_epoch(train_dataloader, model, optimizer, scheduler)
        end_time = timer()
        val_loss = evaluate(val_dataloader, model)

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))
        torch.save({
            'fold': fold,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_FOLDER + f'/ribonanza_fold{fold}_epoch{epoch}.pt')