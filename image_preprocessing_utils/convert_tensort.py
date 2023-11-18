import torch
import glob
from PIL import Image
import numpy as np
import os
# Set the path to the directory containing the PNG images
image_dir = 'png_to_tensor_set/ivrnn2_10/1'

# Get a list of PNG files in the directory
png_files = sorted(glob.glob(image_dir + '/*.png'))


# Replace backslashes with forward slashes
png_files = [path.replace('\\', '/') for path in png_files]
#print(png_files)
# Define a list to store individual frames
frames = []

# Loop through the PNG files and convert each to a PyTorch tensor
for png_file in png_files:
    print(png_file)
    new_size = new_size = (64, 64)
    image = np.array(Image.open(png_file).convert("RGB").resize(new_size, Image.LANCZOS))
        # Check the number of channels in the image
    if image.shape[-1] != 3:  # Grayscale image
        # Expand dimensions to add the third channel
        print('added channel')
        image = np.expand_dims(image, axis=2)  # Shape: (64, 64, 1)

        # Duplicate the channel along the third dimension
        image = np.repeat(image, 3, axis=2)  # Shape: (64, 64, 3)

        # Convert back to PIL Image
        image = Image.fromarray(image)


    print(np.array(image).shape)
    # Assuming you want to convert the image to a tensor and normalize it
    frame_tensor = torch.tensor(
        np.array(image), dtype=torch.uint8
    ).permute(2, 0, 1)
    frames.append(frame_tensor)

# Stack the list of frame tensors along a new dimension (time dimension)
video_tensor = torch.stack(frames, dim=1)
print(video_tensor.shape)

# Set the path for saving the PyTorch tensor as a ".pt" file
output_file = 'rvd_video_generated_1.pt'

# Save the PyTorch tensor in ".pt" format
torch.save(video_tensor, output_file)