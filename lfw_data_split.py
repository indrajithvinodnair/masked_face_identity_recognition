import os
import shutil
import random
from tqdm import tqdm
from config import MASKED_AND_UNMASKED_MERGED_PATH, UNMASKED_DATASET_SPLIT_PATH,MASKED_DATASET_SPLIT_PATH,UNMASKED_LFW_PATH
import argparse
import sys

def clean_output_directory(output_root):
    """
    Cleans the output directory by removing all its contents.
    """
    if os.path.exists(output_root):
        print(f"\n🧹 Cleaning output directory: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    print("✅ Output directory cleaned and ready.")


def split_by_identity(input_path, output_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    identities = [name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]
    random.shuffle(identities)

    n_total = len(identities)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]

    for split, id_list in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
        split_path = os.path.join(output_root, split)
        os.makedirs(split_path, exist_ok=True)
        print(f"\nCopying {split.upper()} set:")
        for identity in tqdm(id_list):
            src_path = os.path.join(input_path, identity)
            dst_path = os.path.join(split_path, identity)
            shutil.copytree(src_path, dst_path)

    print("\n✅ Dataset split completed successfully!")

# === USAGE EXAMPLE ===

parser = argparse.ArgumentParser(description="Split LFW dataset by identity.")
parser.add_argument(
    "--datasetType",
    choices=["masked", "unmasked"],
    help="Type of dataset to split: 'masked' or 'unmasked'"
)
args = parser.parse_args()
datasetType = args.datasetType.lower()
if datasetType == "masked":
    clean_output_directory(MASKED_DATASET_SPLIT_PATH)
    split_by_identity(MASKED_AND_UNMASKED_MERGED_PATH, MASKED_DATASET_SPLIT_PATH)
elif datasetType == "unmasked":
    clean_output_directory(UNMASKED_DATASET_SPLIT_PATH)
    split_by_identity(UNMASKED_LFW_PATH, UNMASKED_DATASET_SPLIT_PATH)
