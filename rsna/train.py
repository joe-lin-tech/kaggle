import torch
import torchvision
import torch.nn as nn
from dataset import RSNADataset
from model import TraumaDetector, CombinedLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timeit import default_timer as timer
from params import *

torch.manual_seed(SEED)

train_iter = RSNADataset(csv_file=CSV_FILE, root_dir=ROOT_DIR, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512))
]), input_type='jpeg')
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

val_iter = RSNADataset(csv_file=CSV_FILE, root_dir=ROOT_DIR, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512))
]), mode='val', input_type='jpeg')
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True)

model = TraumaDetector()
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = None # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)

loss_fn = CombinedLoss()

writer = SummaryWriter()

def train_epoch(model, optimizer, scheduler):
    model.train()
    losses = 0

    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs = inputs.to(DEVICE).float()
        labels = labels.to(DEVICE)

        out = model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        # writer.add_scalar("Loss/step", loss, i)
    # scheduler.step()

    return losses / len(train_iter)

def evaluate(model):
    model.eval()
    losses = 0

    for inputs, labels in tqdm(val_dataloader):
        inputs = inputs.to(DEVICE).float()
        labels = labels.to(DEVICE)

        out = model(inputs)
        loss = loss_fn(out, labels)
        losses += loss.item()
    
    return losses / len(val_iter)

for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, scheduler)
    end_time = timer()
    val_loss = evaluate(model)
    writer.add_scalars("Loss/epoch", { 'train': train_loss, 'val': val_loss }, epoch)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, CHECKPOINT_FOLDER + f'/rsna_epoch{epoch}.pt')