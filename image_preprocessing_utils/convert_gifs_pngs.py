from PIL import Image
import os
# Input folder
input_folder = './svg_gif/output_gifs'

# Create the output parent folder if it doesn't exist
output_parent_folder = './svg_gif/output_folders'
os.makedirs(output_parent_folder, exist_ok=True)

# Load the GIF file
#gif_path = './gif/input_folder/val_sample_499_500.gif'
#gif = Image.open(gif_path)

for filename in os.listdir(input_folder):
    if filename.endswith('.gif'):
        gif_path = os.path.join(input_folder, filename)
        gif = Image.open(gif_path)

        output_folder = os.path.join(output_parent_folder, f'{filename[:-4]}_frame')
        os.makedirs(output_folder, exist_ok=True)
        # Iterate through each frame and save as PNG
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)  # Move to the current frame
            frame = gif.copy()  # Create a copy of the current frame
            individual_image_path = os.path.join(output_folder, f'frame_{frame_index:03d}.png' )
            frame.save(individual_image_path)

print("Conversion complete.")