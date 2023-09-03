import torch
import torchvision
import torch.nn as nn
from dataset import RSNADataset
from model import TraumaDetector, CombinedLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import KFold
from grad import plot_gradient
from tqdm import tqdm
from timeit import default_timer as timer
from params import *

torch.manual_seed(SEED)

data = pd.read_csv(CSV_FILE)

sss = KFold(n_splits=20, shuffle=True, random_state=SEED)
splits = sss.split(data)

writer = SummaryWriter()

def train_epoch(train_dataloader, model, optimizer, scheduler):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs = inputs.to(DEVICE).float()
        labels = labels.to(DEVICE)

        out = model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(out, labels)
        loss.backward()
        # plot_gradient(model.named_parameters())
        optimizer.step()

        losses += loss.item()
        # writer.add_scalar("Loss/step", loss, i)
    scheduler.step()
    # scheduler.step(losses / len(train_iter))

    return losses / len(train_dataloader)

def evaluate(val_dataloader, model):
    model.eval()
    losses = 0

    for inputs, labels in tqdm(val_dataloader):
        inputs = inputs.to(DEVICE).float()
        labels = labels.to(DEVICE)

        out = model(inputs)
        loss = loss_fn(out, labels)
        losses += loss.item()
    
    return losses / len(val_dataloader)


for i, (train_idx, val_idx) in enumerate(splits):
    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    train_iter = RSNADataset(split=train_data, root_dir=ROOT_DIR, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=7.5)
    ]), input_type='jpeg')
    train_sampler = WeightedRandomSampler(train_iter.weights, len(train_iter.weights))
    # train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, sampler=train_sampler)
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_iter = RSNADataset(split=val_data, root_dir=ROOT_DIR, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224))
    ]), mode='val', input_type='jpeg')
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True)

    model = TraumaDetector()
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    loss_fn = CombinedLoss()

    for epoch in range(1, EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(train_dataloader, model, optimizer, scheduler)
        end_time = timer()
        val_loss = evaluate(val_dataloader, model)
        writer.add_scalars("Loss/epoch", { 'train': train_loss, 'val': val_loss }, epoch)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))
        torch.save({
            'split': i,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_FOLDER + f'/rsna_split{i}_epoch{epoch}.pt')
