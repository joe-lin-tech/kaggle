import nibabel as nib
import matplotlib.pyplot as plt

nifti_image = nib.load('21057.nii.gz')

image_data = nifti_image.get_fdata()

# Choose a slice index (for example, the middle slice along the z-axis)
slice_index = image_data.shape[-1] // 2

# Plot the selected slice
plt.imshow(image_data[:, :, slice_index], cmap='viridis')
plt.title("NIfTI Image Slice")
plt.colorbar()
plt.show()