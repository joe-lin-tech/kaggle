import torch
from model import TraumaDetector
from params import *

model = TraumaDetector()
print(model)
model = model.to(DEVICE)

model.load_state_dict(torch.load('models/rsna.pt'))

print(model)