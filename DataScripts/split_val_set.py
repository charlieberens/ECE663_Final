import os
import random
import shutil
import argparse

def split_val_set(input_dir, output_dir, val_spec, postfix):
    # Create paths for train and validation sets in the output directory with postfix
    train_dir = os.path.join(output_dir, f'train{postfix}')
    val_dir = os.path.join(output_dir, f'test{postfix}')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # List all images in the input directory (filter by common image extensions)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

    # Shuffle the images randomly to ensure randomness in selection
    random.shuffle(images)

    # Determine the number of images to copy to the validation set
    if 0 < val_spec <= 1:
        val_count = int(len(images) * val_spec)
    elif val_spec >= 1:
        val_count = min(int(val_spec), len(images))
    else:
        raise ValueError("The validation specification must be either a proportion (0 < val_spec <= 1) or a number of images (val_spec >= 1)")

    # Split images into train and validation sets
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Copy images to the train and validation sets in the output directory
    for img in train_images:
        shutil.copy(os.path.join(input_dir, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.copy(os.path.join(input_dir, img), os.path.join(val_dir, img))

    print(f'Copied {len(train_images)} images to the train{postfix} set and {len(val_images)} images to the test{postfix} set.')

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split a directory of images into a train and validation set with optional postfix.")
    parser.add_argument("input_directory", type=str, help="The path to the input directory containing the set of images.")
    parser.add_argument("output_directory", type=str, help="The path to the output directory where the new structure will be created.")
    parser.add_argument("val_spec", type=float, help="The proportion or number of images to copy to the validation set (0 < val_spec <= 1 for proportion, val_spec >= 1 for specific number).")
    parser.add_argument("--postfix", type=str, default="", help="Optional postfix to add to the train and test directories.")

    # Parse arguments
    args = parser.parse_args()

    # Run the split function
    split_val_set(args.input_directory, args.output_directory, args.val_spec, args.postfix)
