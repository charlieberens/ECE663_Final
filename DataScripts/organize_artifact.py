import os
import pandas as pd
import shutil

# Define the paths
artifact_folder = '../img2img-turbo/my_data/ArtiFact'
sorted_images_folder = '../img2img-turbo/my_data/sorted_images'
real_folder = os.path.join(sorted_images_folder, 'real_images')
fake_folder = os.path.join(sorted_images_folder, 'fake_images')

# Create the sorted_images, real_images, and fake_images folders if they don't exist
os.makedirs(sorted_images_folder, exist_ok=True)
os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

# Iterate through all subfolders in the ArtiFact directory
for subfolder_name in os.listdir(artifact_folder):
    subfolder_path = os.path.join(artifact_folder, subfolder_name)

    # Only process if it's a directory
    if os.path.isdir(subfolder_path):
        # Path to metadata.csv
        metadata_path = os.path.join(subfolder_path, 'metadata.csv')

        # Ensure metadata.csv exists
        if os.path.exists(metadata_path):
            print(f"Processing {metadata_path}...")

            # Read the metadata.csv
            df = pd.read_csv(metadata_path)

            # Ensure 'filename' and 'target' columns are present
            if 'filename' in df.columns and 'target' in df.columns:
                # Check if all targets are the same
                all_real = all(df['target'] == 0)
                all_fake = all(df['target'] != 0)

                if all_real:
                    # Copy all images to the real folder
                    print(f"All images in {subfolder_name} are real. Copying entire folder...")
                    for root, dirs, files in os.walk(subfolder_path):
                        for file in files:
                            # Ensure we're only copying images
                            if file.endswith(('.jpg', '.jpeg', '.png')):
                                src_path = os.path.join(root, file)
                                dest_path = os.path.join(real_folder, f"{subfolder_name}_{file}")
                                shutil.copy(src_path, dest_path)

                elif all_fake:
                    # Copy all images to the fake folder
                    print(f"All images in {subfolder_name} are fake. Copying entire folder...")
                    for root, dirs, files in os.walk(subfolder_path):
                        for file in files:
                            # Ensure we're only copying images
                            if file.endswith(('.jpg', '.jpeg', '.png')):
                                src_path = os.path.join(root, file)
                                dest_path = os.path.join(fake_folder, f"{subfolder_name}_{file}")
                                shutil.copy(src_path, dest_path)

                else:
                    # Mixed labels, process each image individually
                    print(f"Mixed labels in {subfolder_name}. Processing each image individually...")
                    for idx, row in df.iterrows():
                        filename = row['filename']
                        target = row['target']

                        # Walk through all subdirectories to find the image
                        image_path = None
                        for root, dirs, files in os.walk(subfolder_path):
                            if filename in files:
                                image_path = os.path.join(root, filename)
                                break

                        # Check if the image was found
                        if image_path and os.path.exists(image_path):
                            # Determine the destination folder based on the target value
                            if target == 0:
                                dest_folder = real_folder
                            else:
                                dest_folder = fake_folder

                            # Create a unique filename to prevent overwriting
                            new_filename = f"{subfolder_name}_{filename}"
                            dest_path = os.path.join(dest_folder, new_filename)

                            # Copy the image to the corresponding folder
                            shutil.copy(image_path, dest_path)
                        else:
                            print(f"Image file {filename} not found in {subfolder_path} or its subdirectories.")
            else:
                print(f"'filename' or 'target' columns are missing in {metadata_path}.")
        else:
            print(f"metadata.csv not found in {subfolder_path}.")

print("Image sorting completed.")
