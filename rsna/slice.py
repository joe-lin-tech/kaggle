import torch
import torchvision
from torch.cuda.amp.grad_scaler import GradScaler
from dataset import SlicePredictionDataset
from model import TraumaDetector, CombinedLoss
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from timeit import default_timer as timer
from params import *

torch.manual_seed(SEED)

wandb.init(
    # set the wandb project where this run will be logged
    project="slice-predictor",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "eta_min": ETA_MIN,
        "epochs": EPOCHS,
        "seed": SEED
    }
)

def train_epoch(train_dataloader: DataLoader, model: TraumaDetector, optimizer, scheduler):
    model.train()
    losses = 0

    for i, batch in enumerate(tqdm(train_dataloader)):
        images = batch['inputs'].to(DEVICE).float()
        labels = batch['labels'].to(DEVICE)

        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out, labels)

        scaler.scale(loss).backward()

        if ((i + 1) % ACCUM_ITER == 0) or (i + 1 == len(train_dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if ((i + 1) % LOG_INTERVAL == 0) or (i + 1 == len(train_dataloader)):
            wandb.log({ "loss": loss.item() })

        losses += loss.item()
    scheduler.step()
    wandb.log({ "current_lr": scheduler.get_last_lr()[0] })
    print(scheduler.get_last_lr())

    return losses / len(train_dataloader)

def evaluate(val_dataloader: DataLoader, model: TraumaDetector):
    model.eval()
    losses = 0

    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            scans = batch['scans'].to(DEVICE).float()
            labels = batch['labels'].to(DEVICE)

            out = model(scans)
            loss = loss_fn(out, labels)
        
        losses += loss.item()
    
    return losses / len(val_dataloader)


train_iter = SlicePredictionDataset(root_dir=ROOT_DIR, transform=torchvision.transforms.Compose([
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.RandomVerticalFlip(),
                                    torchvision.transforms.RandomRotation(30)
                                ]), mode='train')
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=N_WORKERS)

val_iter = SlicePredictionDataset(root_dir=ROOT_DIR, mode='val')
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

model = TraumaDetector()
if FROM_CHECKPOINT:
    checkpoint = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
wandb.watch(model, log_freq=LOG_INTERVAL)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
if FROM_CHECKPOINT:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN)
if FROM_CHECKPOINT:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

loss_fn = CombinedLoss()
scaler = GradScaler()

start, end = 1, EPOCHS + 1
if FROM_CHECKPOINT:
    start, end = checkpoint['epoch'] + 1, checkpoint['epoch'] + EPOCHS + 1
for epoch in range(start, end):
    start_time = timer()
    train_loss = train_epoch(train_dataloader, model, optimizer, scheduler)
    end_time = timer()
    val_loss = evaluate(val_dataloader, model)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, SLICE_CHECKPOINT_FOLDER + f'/rsna_epoch{epoch}.pt')
