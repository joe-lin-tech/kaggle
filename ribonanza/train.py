from torch.utils.data import RandomSampler, SequentialSampler
import torch
from dataset import RibonanzaDataset, LengthMatchSampler
from model import RNAPredictor, RNALoss
import numpy as np
from fastai.vision.all import *
from fastai.callback.wandb import *
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

model = RNAPredictor(d_model=192, n_heads=8)
model.to(DEVICE)
loss_fn = RNALoss()

for fold in range(N_FOLDS):
    train_iter = RibonanzaDataset(file=TRAIN_FILE, fold=fold)
    train_len = RibonanzaDataset(file=TRAIN_FILE, fold=fold, mask_only=True)
    random_sampler = RandomSampler(train_len)
    train_sampler = LengthMatchSampler(random_sampler, batch_size=BATCH_SIZE, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_iter, batch_sampler=train_sampler)

    val_iter = RibonanzaDataset(file=TRAIN_FILE, fold=fold, split='val')
    val_len = RibonanzaDataset(file=TRAIN_FILE, fold=fold, split='val', mask_only=True)
    sequential_sampler = SequentialSampler(val_len)
    val_sampler = LengthMatchSampler(sequential_sampler, batch_size=BATCH_SIZE, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_iter, batch_sampler=val_sampler)

    learner = Learner(DataLoaders(train_dataloader, val_dataloader, device=DEVICE), model, loss_func=loss_fn,
                        cbs=[GradientClip(3.0), WandbCallback()]).to_fp16()

    for epoch in range(EPOCHS):
        learner.fit_one_cycle(1, lr_max=5e-4, wd=0.05, pct_start=0.02)
        torch.save(learner.model.state_dict(), CHECKPOINT_FOLDER + f'/ribonanza_fold{fold}_epoch{epoch}.pt')