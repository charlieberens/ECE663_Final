import os
import pandas as pd
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

def hash_from_path(path):
    """
    Generate a hash from a given path.
    """
    return hashlib.md5(path.encode()).hexdigest()[:8]  # Shorten hash to 8 characters for brevity

def load_metadata(base_path, saved_metadata_path=None):
    """
    Load metadata from all model directories or load from a saved file.
    Returns a pandas DataFrame containing metadata from all models.
    """
    if saved_metadata_path and os.path.exists(saved_metadata_path):
        print(f"Loading metadata from {saved_metadata_path}")
        return pd.read_pickle(saved_metadata_path)

    # Load metadata from scratch if no saved file is available
    metadata_list = []
    for root, _, files in os.walk(base_path):
        if 'metadata.csv' in files:
            metadata_path = os.path.join(root, 'metadata.csv')
            try:
                metadata = pd.read_csv(metadata_path)
                if not metadata.empty:
                    metadata['model'] = root.split(os.sep)[-1]  # Add model information from the directory name
                    metadata['full_path'] = root + os.sep + metadata['image_path']  # Add full path to metadata
                    metadata['hashed_filename'] = metadata['full_path'].apply(lambda x: os.path.basename(x).split('.')[0] + '_' + hash_from_path(x) + '.jpg')
                    metadata_list.append(metadata)
                    print(f"Loaded metadata from: {metadata_path}")
                else:
                    print(f"Warning: Metadata file is empty at: {metadata_path}")
            except Exception as e:
                print(f"Error loading {metadata_path}: {e}")

    if metadata_list:
        combined_metadata = pd.concat(metadata_list, ignore_index=True)
        if saved_metadata_path:
            combined_metadata.to_pickle(saved_metadata_path)
            print(f"Metadata saved to {saved_metadata_path}")
        print(f"Total metadata entries loaded: {len(combined_metadata)}")
        print(f"Columns in metadata: {combined_metadata.columns.tolist()}")
        return combined_metadata
    else:
        print(f"No metadata files found in {base_path} or files are empty. Please check the base path.")
        return pd.DataFrame()

def list_available_parameters(df):
    """
    List all available parameters in the dataset, including models and categories.
    """
    if not df.empty:
        models = df['model'].unique().tolist() if 'model' in df.columns else []
        categories = df['category'].unique().tolist() if 'category' in df.columns else []
        targets = df['target'].unique().tolist() if 'target' in df.columns else []

        print("\nAvailable Parameters:")
        print(f"Models: {models}")
        print(f"Categories: {categories}")
        print(f"Targets: {targets}")
        
        return {'models': models, 'categories': categories, 'targets': targets}
    else:
        print("Metadata is empty. Cannot list available parameters.")
        return {}

def query_images(df, include={}, exclude={}):
    """
    Filter the metadata DataFrame based on include and exclude parameters.
    """
    query_df = df.copy()

    # Handle include filters
    for key, value in include.items():
        if key not in query_df.columns:
            print(f"Warning: Column '{key}' not found in metadata. Skipping include filter for '{key}'.")
            continue
        if key == 'model':
            # Parse model and optional subdirectory
            for model_value in value:
                model_parts = model_value.split('{')
                model_name = model_parts[0]
                if len(model_parts) > 1:
                    subdirectory = model_parts[1].rstrip('}')
                    query_df = query_df[query_df['model'].str.contains(model_name) & query_df['image_path'].str.contains(subdirectory)]
                else:
                    query_df = query_df[query_df['model'].str.contains(model_name)]
        else:
            query_df = query_df[query_df[key].isin(value)]
        print(f"After include filter '{key}': {len(query_df)} entries")

    # Handle exclude filters
    for key, value in exclude.items():
        if key not in query_df.columns:
            print(f"Warning: Column '{key}' not found in metadata. Skipping exclude filter for '{key}'.")
            continue
        if key == 'model':
            # Parse model and optional subdirectory
            for model_value in value:
                model_parts = model_value.split('{')
                model_name = model_parts[0]
                if len(model_parts) > 1:
                    subdirectory = model_parts[1].rstrip('}')
                    query_df = query_df[~(query_df['model'].str.contains(model_name) & query_df['image_path'].str.contains(subdirectory))]
                else:
                    query_df = query_df[~query_df['model'].str.contains(model_name)]
        else:
            query_df = query_df[~query_df[key].isin(value)]
        print(f"After exclude filter '{key}': {len(query_df)} entries")

    return query_df

def select_images(df, num_images=None, sampling_method='random', distribution_params=None):
    """
    Select a subset of images from the filtered DataFrame with dynamic distribution-based sampling.
    """
    if num_images is None or num_images > len(df):
        print(f"Requested number of images ({num_images}) exceeds available ({len(df)}). Returning all available images.")
        return df

    cumulative_selected_df = pd.DataFrame()
    selected_indices = set()
    remaining_images_needed = num_images
    remaining_df = df.copy()

    if sampling_method == 'random':
        cumulative_selected_df = df.sample(n=num_images)
        print(f"Randomly selected {num_images} images.")
        return cumulative_selected_df

    elif sampling_method == 'distribution' and distribution_params:
        # Dynamic distribution-based sampling
        while remaining_images_needed > 0 and not remaining_df.empty:
            groups = remaining_df.groupby(distribution_params)
            num_groups = len(groups)

            if num_groups == 0:
                print("No more groups available for distribution-based sampling.")
                break

            group_size = max(1, remaining_images_needed // num_groups)
            round_selected_list = []

            # Sample from each group up to the group size
            for _, group in groups:
                sample_size = min(len(group), group_size)
                sampled_group = group.sample(sample_size)
                round_selected_list.append(sampled_group)
                selected_indices.update(sampled_group.index)

            round_selected_df = pd.concat(round_selected_list, ignore_index=True)
            cumulative_selected_df = pd.concat([cumulative_selected_df, round_selected_df], ignore_index=True)

            # Update remaining images needed and remaining_df
            remaining_images_needed -= len(round_selected_df)
            remaining_df = remaining_df.loc[~remaining_df.index.isin(selected_indices)]
            remaining_df = remaining_df.groupby(distribution_params).filter(lambda x: len(x) > 0)

            print(f"Remaining images needed: {remaining_images_needed}, continuing distribution sampling...")

        print(f"Selected {len(cumulative_selected_df)} images using distribution-based sampling across '{distribution_params}'.")
        return cumulative_selected_df
    else:
        raise ValueError("Invalid sampling method or missing distribution parameters")

def summary_of_images(df, stage=""):
    """
    Provide a detailed summary of images.
    """
    if df.empty:
        print(f"No images available at stage: {stage}.")
        return
    
    summary = {
        'Total Images': len(df),
        'Models': df['model'].value_counts().to_dict() if 'model' in df.columns else 'N/A',
        'Categories': df['category'].value_counts().to_dict() if 'category' in df.columns else 'N/A',
        'Targets': df['target'].value_counts().to_dict() if 'target' in df.columns else 'N/A'
    }

    print(f"\nSummary of Images at stage: {stage}")
    for key, value in summary.items():
        print(f"{key}: {value}")

def copy_images(df, output_dir):
    """
    Copy images to the specified output directory while maintaining metadata.
    The copied images are renamed to be randomly sorted alphanumerically, with only the appended hash.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Shuffle the DataFrame to randomize the order
    df = df.sample(frac=1).reset_index(drop=True)

    metadata = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # Loop through each row in the shuffled DataFrame
        for i, row in df.iterrows():
            model_name = row['model']
            src = os.path.join(base_path, model_name, row['image_path'])
            # Extract only the text after the last underscore in the hashed filename
            hash_part = row['hashed_filename'].rsplit('_', 1)[-1]
            # Create the new filename with a zero-padded sequence number and appended hash
            new_filename = f"image{str(i).zfill(7)}_{hash_part}"
            dest = os.path.join(output_dir, new_filename)

            # Update the metadata with the new filename
            row['new_filename'] = new_filename
            metadata.append(row)
            
            futures.append(executor.submit(shutil.copy2, src, dest))

        # Process futures and log errors if any
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                errors.append(str(e))

    # Write updated metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

    # Log errors if any occurred during copying
    if errors:
        with open(os.path.join(output_dir, 'errors.log'), 'w') as f:
            f.write('\n'.join(errors))

# Parameters
# base_path = '../ArtiFact'  # The base directory where the original dataset is located
# output_directory = '../DataSets/cats_fake10000'  # The directory where the new, filtered dataset should be saved
# saved_metadata_path = '../ArtiFact/SavedMetaData.pkl'  # Path to a saved metadata file for faster loading. If it doesn't exist, the script will reindex the metadata, which can be time-consuming.
# include_params = {'category': ['cat'], 'target': [1, 2, 3, 4, 5, 6]}  # Specifies which categories, targets, and/or models to include in the new dataset.
# exclude_params = {'model': ['pro_gan']}  # Specifies which categories, targets, and/or models to exclude from the new dataset.
# num_images = 10000  # The total number of images to select for the new dataset.
# sampling_method = 'distribution'  # Specifies how to sample the images. Options are 'random' or 'distribution'. 'distribution' will aim for an even distribution across the specified parameters in distribution_params.
# distribution_params = ['model']  # Defines the parameters to distribute the sampling across.

base_path = '../ArtiFact'  # The base directory where the original dataset is located
output_directory = '../DataSets/cats_real10000'  # The directory where the new, filtered dataset should be saved
saved_metadata_path = '../ArtiFact/SavedMetaData.pkl'  # Path to a saved metadata file for faster loading. If it doesn't exist, the script will reindex the metadata, which can be time-consuming.
include_params = {'category': ['cat', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075'], 'target': [0]}  # Specifies which categories, targets, and/or models to include in the new dataset.
exclude_params = {}  # Specifies which categories, targets, and/or models to exclude from the new dataset.
num_images = 10000  # The total number of images to select for the new dataset.
sampling_method = 'distribution'  # Specifies how to sample the images. Options are 'random' or 'distribution'. 'distribution' will aim for an even distribution across the specified parameters in distribution_params.
distribution_params = ['model']  # Defines the parameters to distribute the sampling across.

# Execution
metadata_df = load_metadata(base_path, saved_metadata_path=saved_metadata_path)
if not metadata_df.empty:
    print("\nQuerying images based on include/exclude filters...")
    filtered_df = query_images(metadata_df, include=include_params, exclude=exclude_params)
    print(f"Total images after querying: {len(filtered_df)}")

    summary_of_images(filtered_df, stage="Before Selection")

    print("\nSelecting images based on sampling method...")
    selected_df = select_images(filtered_df, num_images=num_images, sampling_method=sampling_method, distribution_params=distribution_params)
    print(f"Total images selected: {len(selected_df)}")

    summary_of_images(selected_df, stage="After Selection")

    print("\nCopying images to output directory...")
    copy_images(selected_df, output_directory)
else:
    print("Metadata loading failed. Please check the dataset path and structure.")
