from PIL import Image
import os

# Define the dimensions of the grid
num_rows = 5
num_columns = 6

# Input stacked GIF file
input_gif_path = './svg_gif/Input_folder/sample_149.gif'

# Create the output folder if it doesn't exist
output_folder = './svg_gif/output_gifs'
os.makedirs(output_folder, exist_ok=True)

# Load the stacked GIF
stacked_gif = Image.open(input_gif_path)

# Calculate cell dimensions
cell_width = stacked_gif.width // num_columns
cell_height = stacked_gif.height // num_rows

# Iterate through each cell in the grid
for row in range(num_rows):
    for col in range(num_columns):
        # Calculate pixel coordinates for the current cell
        left = col * cell_width
        upper = row * cell_height
        right = left + cell_width
        lower = upper + cell_height

        # Extract pixels for the current cell
        cell_gif_frames = []
        for frame_index in range(stacked_gif.n_frames):
            stacked_gif.seek(frame_index)
            cell_frame = stacked_gif.crop((left, upper, right, lower))
            cell_gif_frames.append(cell_frame.copy())  # Copy frame to preserve content

        # Create an output GIF for the current cell
        output_gif_path = os.path.join(output_folder, f'output_cell_{row}_{col}.gif')
        cell_gif_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=cell_gif_frames[1:],  # Append remaining frames
            loop=0,  # No looping
            duration=stacked_gif.info['duration']  # Use original frame duration
        )

print("Separation complete.")