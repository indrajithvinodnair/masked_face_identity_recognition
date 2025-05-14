import os
import shutil
import random
from tqdm import tqdm
from config import INPUT_MERGED_DATASET, OUTPUT_SPLIT_PATH

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

# Clean the output directory before splitting
clean_output_directory(OUTPUT_SPLIT_PATH)

split_by_identity(INPUT_MERGED_DATASET, OUTPUT_SPLIT_PATH)
