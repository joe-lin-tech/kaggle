import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import dicomsdl
import os
import pandas as pd
from tqdm import tqdm

MASK_FOLDER = 'masks/total_seg'

ORGAN_IDS = [1, 2, 3, 5, 55]

# slice_index = 0

def find_indices(image_data):
    for i in range(image_data.shape[-1]):
        if any([o in np.unique(image_data[:, :, i]) for o in ORGAN_IDS]):
            lower_index = i
            break

    for i in range(image_data.shape[-1] - 1, -1, -1):
        if any([o in np.unique(image_data[:, :, i]) for o in ORGAN_IDS]):
            upper_index = i
            break
    
    return lower_index, upper_index

lower_indices = []
upper_indices = []
for root, _, files in os.walk(MASK_FOLDER):
    for file in tqdm(files):
        nifti_image = nib.load(os.path.join(root, file))
        image_data = nifti_image.get_fdata()
        lower_index, upper_index = find_indices(image_data)
        lower_indices.append(lower_index)
        upper_indices.append(upper_index)
print(min(lower_indices), max(upper_indices))


# files = natsorted(os.listdir(DICOM_FOLDER))
# dcm = dicomsdl.open(os.path.join(DICOM_FOLDER, files[-1 - slice_index]))
# info = dcm.getPixelDataInfo()
# pixel_array = np.empty((info['Rows'], info['Cols']), dtype=info['dtype'])
# dcm.copyFrameData(0, pixel_array)

# if dcm.PixelRepresentation == 1:
#     bit_shift = dcm.BitsAllocated - dcm.BitsStored
#     pixel_array = (pixel_array << bit_shift).astype(pixel_array.dtype) >> bit_shift
    
# if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
#     pixel_array = (pixel_array.astype(np.float32) * dcm.RescaleSlope) + dcm.RescaleIntercept
#     center, width = int(dcm.WindowCenter), int(dcm.WindowWidth)
#     low = center - 0.5 - (width - 1) // 2
#     high = center - 0.5 + (width - 1) // 2

#     image = np.empty_like(pixel_array, dtype=np.uint8)
#     dicomsdl.util.convert_to_uint8(pixel_array, image, low, high)

# if dcm.PhotometricInterpretation == "MONOCHROME1":
#     image = 255 - image

# fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.imshow(image, cmap='bone')
# ax2.imshow(np.transpose(image_data[:, :, slice_index], (1, 0))[::-1, :])
# plt.show()