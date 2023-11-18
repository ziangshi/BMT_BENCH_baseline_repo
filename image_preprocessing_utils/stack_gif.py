from PIL import Image
import os

# Define the dimensions of the grid
num_rows = 2
num_columns = 7

# Input folder
input_folder = './gif/input_folder'

# Create the output parent folder if it doesn't exist
output_parent_folder = './gif/output_folders'
os.makedirs(output_parent_folder, exist_ok=True)

# Iterate through the GIF files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.gif'):
        gif_path = os.path.join(input_folder, filename)
        gif = Image.open(gif_path)

        # Iterate through each frame of the GIF
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            stacked_image = gif.copy()

            # Calculate the dimensions of each individual image
            image_width = stacked_image.width // num_columns
            image_height = stacked_image.height // num_rows

            # Create a unique output folder for each GIF frame
            output_folder = os.path.join(output_parent_folder, f'{filename[:-4]}_frame_{frame_index:03d}')
            os.makedirs(output_folder, exist_ok=True)

            # Iterate through the rows and columns to extract each image
            for row in range(num_rows):
                for col in range(num_columns):
                    # Calculate the region to extract
                    left = col * image_width
                    upper = row * image_height
                    right = left + image_width
                    lower = upper + image_height

                    # Extract the individual image
                    individual_image = stacked_image.crop((left, upper, right, lower))

                    # Save the individual image with a unique filename
                    image_index = row * num_columns + col
                    individual_image_path = os.path.join(output_folder, f'individual_image_{image_index:02d}.png')
                    individual_image.save(individual_image_path)

print("Extraction complete.")