import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from model import TraumaDetector
import numpy as np
from PIL import Image
import pydicom as dicom
import csv
import os
from params import *
import dicomsdl

model = TraumaDetector()
model = model.to(DEVICE)

final = torch.load('models/rsna.pt', map_location=DEVICE)

model.load_state_dict(final['model_state_dict'])
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512), antialias=True),
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

def predict(model, batch_id, batch_input, batch_masked_input):
    batch_input = torch.concat(batch_input)
    batch_input = batch_input.to(DEVICE)
    batch_masked_input = torch.concat(batch_masked_input)
    batch_masked_input = batch_masked_input.to(DEVICE)

    out = model(batch_input, batch_masked_input)
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
batch_masked_input = []
for f in os.scandir('../../'):
    if f.is_dir():
        path = f.path
        images = []
        for root, dirs, _ in os.walk(path):
            for dirname in dirs:
                scan = []
                files = os.listdir(os.path.join(root, dirname))
                channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                for filename in [files[int(c)] for c in channels]:
                    dcm = dicomsdl.open(os.path.join(root, dirname, filename))
                    info = dcm.getPixelDataInfo()
                    pixel_array = np.empty((info['Rows'], info['Cols']), dtype=info['dtype'])
                    dcm.copyFrameData(0, pixel_array)
                
                    if dcm.PixelRepresentation == 1:
                        bit_shift = dcm.BitsAllocated - dcm.BitsStored
                        pixel_array = (pixel_array << bit_shift).astype(pixel_array.dtype) >> bit_shift
                        
                    if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
                        pixel_array = (pixel_array.astype(np.float32) * dcm.RescaleSlope) + dcm.RescaleIntercept
                        center, width = int(dcm.WindowCenter), int(dcm.WindowWidth)
                        low = center - 0.5 - (width - 1) // 2
                        high = center - 0.5 + (width - 1) // 2

                        image = np.empty_like(pixel_array, dtype=np.uint8)
                        dicomsdl.util.convert_to_uint8(pixel_array, image, low, high)
                    
                    if dcm.PhotometricInterpretation == "MONOCHROME1":
                        image = 255 - image

                    scan.append(image)
                images.append(np.stack(scan))
        input = images[0] # fix sample selection

        input = transform(torch.tensor(input).float())
        masked_input = resize(input.clone(), (256, 256))

        size = MASK_DEPTH # 12
        if f.name + '.npz' in os.listdir(os.path.join(MASK_FOLDER, 'train')):
            masks = np.load(os.path.join(MASK_FOLDER, 'train', f.name + '.npz'))
        else:
            masks = np.load(os.path.join(MASK_FOLDER, 'val', f.name + '.npz'))
        
        for i in range(size // 2, N_CHANNELS, size):
            masked_input[0, i - (size // 2):i + (size // 2), :, :] *= masks[str(i)]

        batch_id.append(f.name)
        batch_input.append(input)
        batch_masked_input.append(masked_input)
        if len(batch_id) == BATCH_SIZE:
            predict(model, batch_id, batch_input, batch_masked_input)
            batch_id.clear()
            batch_input.clear()
            batch_masked_input.clear()

if len(batch_id) > 0:
    predict(model, batch_id, batch_input, batch_masked_input)
    batch_id.clear()
    batch_input.clear()
    batch_masked_input.clear()
