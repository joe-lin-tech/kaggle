import torch
from model import TraumaDetector
from params import *

model = TraumaDetector()
model.load_state_dict(torch.load('models/rsna.pt', map_location=DEVICE)['model_state_dict'])

model_scripted = torch.jit.script(model)
model_scripted.save('models/rsna_jit.pt')