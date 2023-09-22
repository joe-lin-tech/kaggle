import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from model import TraumaDetector
import numpy as np
import pandas as pd
import csv
import os
import nibabel as nib
from utils import pad_scan, scale_scan, preprocess_scan
from natsort import natsorted
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

def predict(model, batch_id, batch_input):
    batch_input = torch.stack(batch_input)
    batch_input = batch_input.to(DEVICE)
    # batch_masked_input = torch.stack(batch_masked_input)
    # batch_masked_input = batch_masked_input.to(DEVICE)

    # out = model(batch_input, batch_masked_input)
    out = model(batch_input)
    kidney, liver, spleen = out
    # bowel = torch.sigmoid(out['bowel'])
    # extravasation = torch.sigmoid(out['extravasation'])
    kidney = F.softmax(kidney, dim=-1)
    liver = F.softmax(liver, dim=-1)
    spleen = F.softmax(spleen, dim=-1)

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

patient_df = pd.read_csv(CSV_FILE)
for i in range(len(patient_df)):
    path = os.path.join(ROOT_DIR, str(patient_df.iloc[i].patient_id))
    images = []
    for root, dirs, _ in os.walk(path):
        for dirname in dirs:
            scan = []
            files = natsorted(os.listdir(os.path.join(root, dirname)))
            # slices = np.linspace(SIDE_CHANNELS, len(files) - 1 - SIDE_CHANNELS, N_SLICES)
            # slices = np.linspace(len(files) // 4, 3 * len(files) // 4, N_SLICES)
            mask_nifti = nib.load(os.path.join(MASK_FOLDER, str(patient_df.iloc[i].patient_id), dirname + '.nii.gz'))
            mask_nifti = np.transpose(mask_nifti.get_fdata(), (2, 1, 0))[:, ::-1, :]
            indices = np.argwhere(np.isin(mask_nifti, ORGAN_IDS))[:, 0]
            # min_index, max_index = 0, len(files)
            # if len(indices > 0):
            #     min_index, max_index = np.min(indices), np.max(indices)
            dcm_start = dicomsdl.open(os.path.join(root, dirname, files[0]))
            dcm_end = dicomsdl.open(os.path.join(root, dirname, files[-1]))
            dx, dy = dcm_start.PixelSpacing
            dz = np.abs((dcm_end.ImagePositionPatient[2] - dcm_start.ImagePositionPatient[2]) / len(files))
            # if dcm_start.ImagePositionPatient[2] > dcm_end.ImagePositionPatient[2]:
            #     mask = np.transpose(mask_nifti.get_fdata(), (2, 1, 0))[::-1, ::-1, :]
            # else:
            #     mask = np.transpose(mask_nifti.get_fdata(), (2, 1, 0))[:, ::-1, :]
            # channels = []
            # for s in slices:
            #     channels += [int(s) - 1, int(s), int(s) + 1]
            # for i, filename in enumerate([files[c] for c in channels]):
            save_file = str(patient_df.iloc[i].patient_id) + '_' + dirname + '.npy'
            if save_file in os.listdir(TEMP_DIR):
                scan = np.load(os.path.join(TEMP_DIR, save_file))
            else:
                for filename in files:
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
                scan = np.stack(scan)
                np.save(os.path.join(TEMP_DIR, save_file), scan)
            if dcm_end.ImagePositionPatient[2] > dcm_start.ImagePositionPatient[2]:
                scan = scan[::-1]
            scan = torch.tensor(scan.copy()).float()
            scan = pad_scan(scan)
            prob = torch.zeros(len(files))
            prob[indices] = 1
            scan, prob = scale_scan(scan, (dz, dy, dx), prob)
            scan = preprocess_scan(scan, prob)
            images.append(scan)
            break
    input = images[0] # fix sample selection

    input = transform(torch.tensor(input).float())

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
