import torch
import torchvision
from model import TraumaDetector
from datasets import RSNADataset
from params import *

model = TraumaDetector()
print(model)
model = model.to(DEVICE)

model.load_state_dict(torch.load('models/rsna.pt'))

test_iter = RSNADataset(csv_file=TEST_FILE, root_dir=TEST_DIR, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512))])

print(model)
