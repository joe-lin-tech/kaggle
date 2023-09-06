import torch
import torchvision
import torch.nn.functional as F
from model import TraumaDetector
import numpy as np
from PIL import Image
import pydicom as dicom
import csv
import os
from params import *

model = TraumaDetector()
model = model.to(DEVICE)

final = torch.load('models/rsna.pt', map_location=DEVICE)

model.load_state_dict(final['model_state_dict'])
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224))
])

fieldnames = [
    'patient_id', 'bowel_healthy', 'bowel_injury',
    'extravasation_healthy', 'extravasation_injury',
    'kidney_healthy', 'kidney_low', 'kidney_high',
    'liver_healthy', 'liver_low', 'liver_high',
    'spleen_healthy', 'spleen_low', 'spleen_high'
]

with open('submission.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

def predict(model, batch_id, batch_input):
    batch_input = torch.concat(batch_input)
    batch_input = batch_input.to(DEVICE)

    out = model(batch_input)
    # bowel = torch.sigmoid(out['bowel'])
    # extravasation = torch.sigmoid(out['extravasation'])
    kidney = F.softmax(out['kidney'], dim=-1)
    liver = F.softmax(out['liver'], dim=-1)
    spleen = F.softmax(out['spleen'], dim=-1)

    with open('submission.csv', 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for i in range(len(batch_id)):
            result = {
                'patient_id': batch_id[i],
                'bowel_healthy': 0.9601370289629398, # 1 - bowel[i].item(),
                'bowel_injury': 0.03986297103706011, # bowel[i].item(),
                'extravasation_healthy': 0.7106341933928141, # 1 - extravasation[i].item(),
                'extravasation_injury': 0.2893658066071859, # extravasation[i].item(),
                'kidney_healthy': kidney[i][0].item(),
                'kidney_low': kidney[i][1].item(),
                'kidney_high': kidney[i][2].item(),
                'liver_healthy': liver[i][0].item(),
                'liver_low': liver[i][1].item(),
                'liver_high': liver[i][2].item(),
                'spleen_healthy': spleen[i][0].item(),
                'spleen_low': spleen[i][1].item(),
                'spleen_high': spleen[i][2].item()
            }
            writer.writerow(result)

batch_id = []
batch_input = []
for f in os.scandir('../../train_images_mini'):
    if f.is_dir():
        path = f.path
        images = []
        for root, dirs, _ in os.walk(path):
            for dirname in dirs:
                scan = []
                files = os.listdir(os.path.join(root, dirname))
                channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                for filename in [files[int(c)] for c in channels]:
                    img = Image.open(os.path.join(root, dirname, filename))
                    img = transform(img)
                    scan.append(img)
                images.append(torch.concat(scan))
        input = images[0].unsqueeze(0) # fix sample selection
        batch_id.append(f.name)
        batch_input.append(input)
        if len(batch_id) == BATCH_SIZE:
            predict(model, batch_id, batch_input)
            batch_id.clear()
            batch_input.clear()

if len(batch_id) > 0:
    predict(model, batch_id, batch_input)
    batch_id.clear()
    batch_input.clear()
