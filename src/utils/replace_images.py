import os
import shutil

def replace_images_in_subdirectories(source_folder, target_folder):
    """
    Replaces images in subdirectories of a target folder with images 
    from a source folder based on matching filenames.

    Args:
        source_folder (str): The path to the folder containing the replacement images.
        target_folder (str): The path to the main folder containing subdirectories 
                             with images to be replaced.
    """
    
    # 1. Input Validation
    if not os.path.isdir(source_folder):
        print(f"Error: Source folder not found at '{source_folder}'")
        return

    if not os.path.isdir(target_folder):
        print(f"Error: Target folder not found at '{target_folder}'")
        return

    print(f"--- Starting Image Replacement Process ---")
    print(f"Source: {source_folder}")
    print(f"Target: {target_folder}")
    print("-" * 40)
    
    # 2. Get a set of all image names available in the source folder
    source_images = set(os.listdir(source_folder))
    
    replacements_made = 0
    images_checked = 0

    # 3. Walk through all directories and files in the target folder
    for root, dirs, files in os.walk(target_folder):
        # Skip the root target_folder itself, only process subdirectories
        if root == target_folder:
            continue
            
        print(f"Checking subdirectory: {os.path.basename(root)}")

        # 4. Iterate over files in the current subdirectory
        for filename in files:
            images_checked += 1
            
            # Check if the file is an image (you might need to adjust extensions)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                
                # Check if an image with the exact same name exists in the source folder
                if filename in source_images:
                    
                    # Full paths for the source and target files
                    source_file_path = os.path.join(source_folder, filename)
                    target_file_path = os.path.join(root, filename)
                    
                    # 5. Perform the replacement
                    try:
                        shutil.copy2(source_file_path, target_file_path)
                        print(f"  ✅ Replaced: {filename} in {os.path.basename(root)}")
                        replacements_made += 1
                    except Exception as e:
                        print(f"  ❌ Error replacing {filename}: {e}")
                
    print("-" * 40)
    print(f"Finished process.")
    print(f"Total files checked in subdirectories: {images_checked}")
    print(f"Total replacements made: **{replacements_made}**")


# --- Configuration ---
# You MUST update these two variables with your actual folder paths
SOURCE_IMAGES_FOLDER = '/scratch/users/sonnet/ddi'
TARGET_MAIN_FOLDER = '/home/groups/roxanad/sonnet/vcr/processed_images' 
# --- End Configuration ---


# Run the script
if __name__ == "__main__":
    replace_images_in_subdirectories(SOURCE_IMAGES_FOLDER, TARGET_MAIN_FOLDER)