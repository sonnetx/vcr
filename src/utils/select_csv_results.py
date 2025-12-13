import shutil
from pathlib import Path
import os

def aggregate_concept_csvs(parent_dir_path: str, output_path: str):
    """
    Iterates through subdirectories starting with 'OpenFlamingo', copies the 
    specified concept CSV files from 'analysis_outputs', and renames them 
    to include the parent directory name, placing them in 'results_csv_only'.

    Args:
        parent_dir_path: The path to the directory containing the 
                         'OpenFlamingo*' subdirectories.
    """
    
    # --- Configuration ---
    # The parent directory where the script will start searching. 
    # Change '.' to the specific path if it's not the current directory.
    parent_path = Path(parent_dir_path).resolve()
    output_path = Path(output_path)
    
    # The specific files we are looking to copy.
    target_files = [
        "top_20_negative_concepts.csv",
        "top_20_positive_concepts.csv"
    ]
    
    # The subdirectory containing the target files.
    source_sub_dir = "analysis_outputs"

    print(f"Starting file aggregation from: {parent_path}")
    print(f"Targeting output directory: {output_path}")
    
    # 1. Create the final output directory if it doesn't exist
    try:
        output_path.mkdir(exist_ok=True)
        print(f"Successfully ensured output directory exists: {output_path.name}")
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    # 2. Iterate through subdirectories matching "OpenFlamingo*"
    # We use parent_path.glob() to search only the immediate children.
    found_dirs = list(parent_path.glob("OpenFlamingo*"))
    
    if not found_dirs:
        print("\nNo subdirectories starting with 'OpenFlamingo' found. Exiting.")
        return

    files_copied = 0

    # 3. Process each found directory
    for flamingo_dir in found_dirs:
        if not flamingo_dir.is_dir():
            continue # Skip files that might match the pattern
            
        print(f"\nProcessing directory: {flamingo_dir.name}")
        
        # 4. Loop through the two target files
        for target_filename in target_files:
            # Construct the full source path: e.g., OpenFlamingo-X/analysis_outputs/file.csv
            source_file_path = flamingo_dir / source_sub_dir / target_filename
            
            if source_file_path.is_file():
                # 5. Construct the new, unique filename
                # Example: OpenFlamingo-3B-Instruct_..._prob_top_20_negative_concepts.csv
                new_filename = f"{flamingo_dir.name}_{target_filename}"
                destination_path = output_path / new_filename
                
                try:
                    # 6. Copy the file (shutil.copy2 preserves metadata)
                    shutil.copy2(source_file_path, destination_path)
                    print(f"  -> Copied {target_filename} to {destination_path.name}")
                    files_copied += 1
                except Exception as e:
                    print(f"  -> ERROR: Could not copy {source_file_path}: {e}")
            else:
                print(f"  -> WARNING: File not found at: {source_file_path}")

    print(f"\n--- Process Complete ---")
    print(f"Total files copied: {files_copied}")
    print(f"All aggregated files are in: {output_path}")


if __name__ == "__main__":
    parent_directory_to_scan = "/home/groups/roxanad/sonnet/vcr/scripts" 
    aggregate_concept_csvs(parent_directory_to_scan, "/home/groups/roxanad/sonnet/vcr/results_csv_only")
    parent_directory_to_scan = "/home/groups/roxanad/sonnet/vcr/scripts/contrastive10k" 
    aggregate_concept_csvs(parent_directory_to_scan, "/home/groups/roxanad/sonnet/vcr/results_csv_only")