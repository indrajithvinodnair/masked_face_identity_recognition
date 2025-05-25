import os
import shutil
from tqdm import tqdm
from config import UNMASKED_LFW_PATH, MASKED_LFW_PATH, MASKED_AND_UNMASKED_MERGED_PATH

def merge_lfw_datasets(original_path, masked_path, output_path):
    if os.path.exists(output_path):
        print(f"Cleaning old output directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    identities = os.listdir(original_path)
    for identity in tqdm(identities, desc="Merging identities"):
        orig_identity_path = os.path.join(original_path, identity)
        masked_identity_path = os.path.join(masked_path, identity)
        
        if not os.path.isdir(orig_identity_path):
            continue
        
        output_identity_path = os.path.join(output_path, identity)
        os.makedirs(output_identity_path, exist_ok=True)

        # Copy original (unmasked) images
        for img_file in os.listdir(orig_identity_path):
            src_file = os.path.join(orig_identity_path, img_file)
            dst_file = os.path.join(output_identity_path, img_file)
            shutil.copy2(src_file, dst_file)

        # Copy masked images (if they exist for this identity)
        if os.path.exists(masked_identity_path):
            for img_file in os.listdir(masked_identity_path):
                src_file = os.path.join(masked_identity_path, img_file)
                dst_file = os.path.join(output_identity_path, img_file)
                shutil.copy2(src_file, dst_file)

    print(f"\n✅ Merging completed successfully! Merged data saved at: {output_path}")

# === USAGE EXAMPLE ===

merge_lfw_datasets(UNMASKED_LFW_PATH, MASKED_LFW_PATH, MASKED_AND_UNMASKED_MERGED_PATH)
