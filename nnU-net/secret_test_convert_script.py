import os
import shutil
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import argparse


def make_out_dirs(dataset_id: int, task_name="SingleDataset"):
    """Create output directory for a single dataset"""
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"
    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_images_dir = out_dir / "imagesTs"  # All data goes to imagesTs for inference

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    return out_dir, out_images_dir


def copy_files(src_data_folder: Path, images_dir: Path):
    """Recursively copy .nii.gz files from patient subfolders."""
    num_cases = 0

    print(f"Checking files in: {src_data_folder}")

    if not src_data_folder.exists():
        print(f"ERROR: Input folder {src_data_folder} does not exist!")
        return 0

    # Search recursively for .nii.gz files
    nii_files = list(src_data_folder.rglob("*.nii.gz"))
    if not nii_files:
        print(f"WARNING: No .nii.gz files found in {src_data_folder}")

    for file in sorted(nii_files):
        if "_gt" not in file.name and "_4d" not in file.name:  # Exclude ground truth & 4D data
            new_filename = images_dir / f"{file.stem}_0000.nii.gz"
            shutil.copy(file, new_filename)
            print(f"Copied: {file} -> {new_filename}")
            num_cases += 1
        else:
            print(f"Skipping (ground truth or 4D data): {file}")

    return num_cases


def convert_single_dataset(src_data_folder: str, dataset_id=99):
    """Convert a single dataset into nnUNet format with recursive search."""
    out_dir, images_dir = make_out_dirs(dataset_id=dataset_id)
    num_cases = copy_files(Path(src_data_folder), images_dir)

    if num_cases == 0:
        print("ERROR: No files were copied! Check the dataset format.")
        return

    generate_dataset_json(
        str(out_dir),
        channel_names={0: "Image"},
        labels={"background": 0},  # No segmentation labels needed for inference
        file_ending=".nii.gz",
        num_training_cases=num_cases,
    )
    print(f"Dataset JSON created at {out_dir}/dataset.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing the dataset (with subdirectories for patients).",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=99, help="nnU-Net Dataset ID, default: 99"
    )
    args = parser.parse_args()

    print("Converting dataset for nnUNet inference...")
    convert_single_dataset(args.input_folder, args.dataset_id)
    print("Conversion complete!")
