import os
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
from torchvision import transforms


def extract_embedding(model, image_path: str, transform, device: torch.device) -> np.ndarray:
    """
    Loads and preprocesses an image, then returns the embedding from the model.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return emb.cpu().numpy().flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D NumPy vectors.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_masked_unmasked_pairs(
    root: str,
    mask_keywords: List[str]
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Constructs positive and negative image pairs for masked vs. unmasked evaluation.

    Args:
        root: path to validation/test set where subfolders are identities
        mask_keywords: list of substrings (e.g. 'n95', 'surgical') that identify masked images

    Returns:
        - List of (masked_path, unmasked_path) pairs
        - Corresponding list of labels: 1 for positive, 0 for negative
    """
    people = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    pairs = []
    labels = []

    for person in people:
        person_dir = os.path.join(root, person)
        files = os.listdir(person_dir)

        masked = [f for f in files if any(kw in f.lower() for kw in mask_keywords)]
        unmasked = [f for f in files if all(kw not in f.lower() for kw in mask_keywords)]

        if not masked or not unmasked:
            continue

        # Positive pair
        pairs.append((os.path.join(person_dir, masked[0]),
                      os.path.join(person_dir, unmasked[0])))
        labels.append(1)

        # Negative pair: masked of this vs unmasked of different identity
        for other in people:
            if other == person:
                continue
            other_dir = os.path.join(root, other)
            other_files = os.listdir(other_dir)
            other_unmasked = [f for f in other_files if all(kw not in f.lower() for kw in mask_keywords)]
            if not other_unmasked:
                continue
            pairs.append((os.path.join(person_dir, masked[0]),
                          os.path.join(other_dir, other_unmasked[0])))
            labels.append(0)
            break

    return pairs, labels
