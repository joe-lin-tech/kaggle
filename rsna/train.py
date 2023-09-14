#!/usr/bin/env python3

import torch
import torchvision
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from dataset import RSNADataset, get_mean_std
from model import TraumaDetector, CombinedLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from grad import log_grad_cam
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from argparse import Namespace
import itertools
from SAM_Med2D.segment_anything import sam_model_registry
from SAM_Med2D.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from tqdm import tqdm
from timeit import default_timer as timer
from params import *

torch.manual_seed(SEED)

wandb.init(
    # set the wandb project where this run will be logged
    project="rsna-abdominal-trauma",
    
    # track hyperparameters and run metadata
    config={
        # "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED
    }
)

parser = ArgumentParser(prog='train.py')
parser.add_argument('-c', '--checkpoint', action='store_true')
args = parser.parse_args()

model = sam_model_registry["vit_b"](Namespace(image_size=256, encoder_adapter=True, sam_checkpoint=MASK_MODEL)).to(MASK_DEVICE)
mask_generator = SamAutomaticMaskGenerator(model, pred_iou_thresh=0.5, stability_score_thresh=0.8)

data = pd.read_csv(CSV_FILE)

# bowel_healthy = data['bowel_healthy'].mean(axis=0)
# bowel_injury = data['bowel_injury'].mean(axis=0) * 2
# bowel = bowel_healthy + bowel_injury
# print(bowel_healthy / bowel, bowel_injury / bowel)
# extravasation_healthy = data['extravasation_healthy'].mean(axis=0)
# extravasation_injury = data['extravasation_injury'].mean(axis=0) * 6
# extravasation = extravasation_healthy + extravasation_injury
# print(extravasation_healthy / extravasation, extravasation_injury / extravasation)

sss = KFold(n_splits=5, shuffle=True, random_state=SEED)
splits = sss.split(data)

def train_epoch(train_dataloader, model, optimizer, scheduler):
    model.train()
    losses = 0

    for i, batch in enumerate(tqdm(train_dataloader)):
        scans = batch['scans'].to(DEVICE).float()
        masked_scans = batch['masked_scans'].to(DEVICE).float()
        labels = batch['labels'].to(DEVICE)

        with torch.cuda.amp.autocast():
            out = model(scans, masked_scans)
            loss = loss_fn(out, labels)

        scaler.scale(loss).backward()

        if ((i + 1) % ACCUM_ITER == 0) or (i + 1 == len(train_dataloader)):
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if i % LOG_INTERVAL == 0:
            size = MASK_DEPTH
            raw = [wandb.Image(scans[0, c, :, :]) for c in range(size // 2, N_CHANNELS, size)]
            masked = [wandb.Image(masked_scans[0, c, :, :]) for c in range(size // 2, N_CHANNELS, size)]
            # cam = log_grad_cam(model=model, target_layers=model.mask_encoder, input_tensor=scans)
            wandb.log({ "raw": raw, "masked": masked, "loss": loss.item() })

        losses += loss.item()
    scheduler.step()
    print(scheduler.get_last_lr())
    # scheduler.step(losses / len(train_iter))

    return losses / len(train_dataloader)

def evaluate(val_dataloader, model):
    model.eval()
    losses = 0

    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            scans = batch['scans'].to(DEVICE).float()
            masked_scans = batch['masked_scans'].to(DEVICE).float()
            labels = batch['labels'].to(DEVICE)

            out = model(scans, masked_scans)
            loss = loss_fn(out, labels)
        
        losses += loss.item()
    
    return losses / len(val_dataloader)


for i, (train_idx, val_idx) in enumerate(splits):
    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    train_iter = RSNADataset(split=train_data, root_dir=ROOT_DIR, mask_generator=mask_generator,
                             transform=dict(
                                 preprocess=torchvision.transforms.Compose([
                                     torchvision.transforms.Resize((512, 512), antialias=True),
                                    #  torchvision.transforms.Normalize(mean=40.5436, std=64.4406)
                                 ]),
                                 random=torchvision.transforms.Compose([
                                     torchvision.transforms.RandomHorizontalFlip(),
                                     torchvision.transforms.RandomVerticalFlip(),
                                     torchvision.transforms.RandomResizedCrop((512, 512), antialias=True)
                                 ])), mode='train')
    # train_sampler = WeightedRandomSampler(train_iter.weights, len(train_iter.weights))
    # train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=N_WORKERS)

    val_iter = RSNADataset(split=val_data, root_dir=ROOT_DIR, mask_generator=mask_generator,
                           transform=dict(
                               preprocess=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((512, 512), antialias=True),
                                #    torchvision.transforms.Normalize(mean=40.5436, std=64.4406)
                               ])), mode='val')
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    # print(get_mean_std(train_dataloader, val_dataloader))

    model = TraumaDetector()
    if args.checkpoint:
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    wandb.watch(model, log='all', log_freq=LOG_INTERVAL)
    # cam = GradCAM(model=model, target_layers=[model.out], use_cuda=True)

    model_lr = [
        { 'params': itertools.chain(*[
            layer.self_attention.parameters() for layer in model.mask_encoder.backbone.encoder.layers
        ]), 'lr': MASK_ENCODER_LR },
        { 'params': model.mask_encoder.fcn.parameters(), 'lr': MASK_FCN_LR },
        { 'params': model.backbone[-1].parameters(), 'lr': BACKBONE_LR },
        { 'params': model.head.parameters(), 'lr': HEAD_LR },
        { 'params': itertools.chain(*[
            model.out.parameters(),
            model.out_bowel.parameters(),
            model.out_extravasation.parameters(),
            model.out_kidney.parameters(),
            model.out_liver.parameters(),
            model.out_spleen.parameters()
        ]), 'lr': OUT_LR }
    ]
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-2)
    optimizer = torch.optim.Adam(model_lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model_lr, momentum=0.9, weight_decay=1e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    # scheduler = None

    loss_fn = CombinedLoss()
    scaler = GradScaler()

    start, end = 1, EPOCHS + 1
    if args.checkpoint:
        start, end = checkpoint['epoch'] + 1, checkpoint['epoch'] + EPOCHS + 1
    for epoch in range(start, end):
        start_time = timer()
        train_loss = train_epoch(train_dataloader, model, optimizer, scheduler)
        end_time = timer()
        val_loss = evaluate(val_dataloader, model)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))
        torch.save({
            'split': i,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_FOLDER + f'/rsna_split{i}_epoch{epoch}.pt')
