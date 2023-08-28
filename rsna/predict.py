import torch
import torchvision
import torch.nn.functional as F
from model import TraumaDetector, CombinedLoss
import numpy as np
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
    torchvision.transforms.Resize((512, 512))
])

with open('submission.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'patient_id', 'bowel_healthy', 'bowel_injury',
        'extravasation_healthy', 'extravasation_injury',
        'kidney_healthy', 'kidney_low', 'kidney_high',
        'liver_healthy', 'liver_low', 'liver_high',
        'spleen_healthy', 'spleen_low', 'spleen_high'
    ])
    writer.writeheader()

def predict(model, id, input):
    input = input.to(DEVICE)

    out = model(input)
    bowel = F.sigmoid(out['bowel'])[0]
    extravasation = F.sigmoid(out['extravasation'])[0]
    kidney = F.softmax(out['kidney'])[0]
    liver = F.softmax(out['liver'])[0]
    spleen = F.softmax(out['spleen'])[0]

    result = {
        'patient_id': id,
        'bowel_healthy': bowel.item(),
        'bowel_injury': 1 - bowel.item(),
        'extravasation_healthy': extravasation.item(),
        'extravasation_injury': 1 - extravasation.item(),
        'kidney_healthy': kidney[0].item(),
        'kidney_low': kidney[1].item(),
        'kidney_high': kidney[2].item(),
        'liver_healthy': liver[0].item(),
        'liver_low': liver[1].item(),
        'liver_high': liver[2].item(),
        'spleen_healthy': spleen[0].item(),
        'spleen_low': spleen[1].item(),
        'spleen_high': spleen[2].item()
    }

    with open('submission.csv', 'a') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writerow(result)

for f in os.scandir('/Volumes/SSD/rsna/train_images'):
    if f.is_dir():
        path = f.path
        images = []
        for root, dirs, _ in os.walk(path):
            for dirname in dirs:
                scan = []
                files = os.listdir(os.path.join(root, dirname))
                channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                for filename in [files[int(c)] for c in channels]:
                    dcm = dicom.dcmread(os.path.join(root, dirname, filename))
                    if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
                        center, width = int(dcm.WindowCenter), int(dcm.WindowWidth)
                        low = center - width / 2
                        high = center + width / 2    
                        image = (dcm.pixel_array * dcm.RescaleSlope) + dcm.RescaleIntercept
                        image = np.clip(image, low, high)

                        image = (image / np.max(image) * 255).astype(np.float32)
                        image = transform(image)
                        scan.append(image)
                images.append(torch.concat(scan))
        input = images[0].unsqueeze(0) # fix sample selection
        predict(model, f.name, input)