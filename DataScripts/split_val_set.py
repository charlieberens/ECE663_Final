import os
import random
import shutil
import argparse


def split_val_set(directory, val_spec):
    # Create paths for train and validation sets
    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # List all images in the directory
    images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Shuffle the images randomly to ensure randomness in selection
    random.shuffle(images)

    # Determine the number of images to move to the validation set
    if 0 < val_spec <= 1:
        val_count = int(len(images) * val_spec)
    elif val_spec >= 1:
        val_count = min(int(val_spec), len(images))
    else:
        raise ValueError("The validation specification must be either a proportion (0 < val_spec <= 1) or a number of images (val_spec >= 1)")

    # Move images to the train and validation sets
    val_images = images[:val_count]
    train_images = images[val_count:]

    for img in train_images:
        shutil.move(os.path.join(directory, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.move(os.path.join(directory, img), os.path.join(val_dir, img))

    print(f'Moved {len(train_images)} images to the train set and {len(val_images)} images to the validation set.')


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split a directory of images into a train and validation set.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the train set of images.")
    parser.add_argument("val_spec", type=float, help="The proportion or number of images to move to the validation set (0 < val_spec <= 1 for proportion, val_spec >= 1 for specific number).")

    # Parse arguments
    args = parser.parse_args()

    # Run the split function
    split_val_set(args.directory, args.val_spec)
