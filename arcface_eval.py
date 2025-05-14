import os
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm
import itertools
from PIL import Image
import json
from config import BASELINE_RESULTS_PATH, TEST_ROOT

# ---------- Argument Parsing ---------- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True, help="e.g., LFW_only or LFW_plus_masked")
parser.add_argument("--epochs", type=int, required=True, help="e.g., 30, 50, or 100")
args = parser.parse_args()

dataset_type = args.dataset_type
epochs = args.epochs

base_path = os.path.join(BASELINE_RESULTS_PATH, dataset_type, f"epoch_{epochs}")
model_path = os.path.join(base_path, "model_full.pth")
assert os.path.exists(model_path), f"Model not found at {model_path}"

# ---------- Device ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Indicate the device being used

# ---------- Load Model ---------- #
model = torch.load(model_path, map_location=device)
model.eval()

# ---------- Transform ---------- #
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------- Generate Pairs ---------- #
test_root = TEST_ROOT
all_people = os.listdir(test_root)
pairs = []
labels = []

for person in tqdm(all_people, desc="Creating positive pairs"):
    person_dir = os.path.join(test_root, person)
    images = os.listdir(person_dir)
    for img1, img2 in itertools.combinations(images, 2):
        pairs.append((os.path.join(person_dir, img1), os.path.join(person_dir, img2)))
        labels.append(1)

for _ in range(len(pairs)):
    p1, p2 = np.random.choice(all_people, 2, replace=False)
    img1 = np.random.choice(os.listdir(os.path.join(test_root, p1)))
    img2 = np.random.choice(os.listdir(os.path.join(test_root, p2)))
    pairs.append((os.path.join(test_root, p1, img1), os.path.join(test_root, p2, img2)))
    labels.append(0)

# ---------- Embedding Extraction ---------- #
def extract_embedding(model, image, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy().flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Evaluation ---------- #
scores = []
for img1_path, img2_path in tqdm(pairs, desc="Evaluating pairs"):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    emb1 = extract_embedding(model, img1, transform, device)
    emb2 = extract_embedding(model, img2, transform, device)
    sim = cosine_similarity(emb1, emb2)
    scores.append(sim)

threshold = 0.5
preds = [1 if s > threshold else 0 for s in scores]

acc = accuracy_score(labels, preds)
roc_auc = roc_auc_score(labels, scores)
fpr, tpr, _ = roc_curve(labels, scores)
tpr_at_fpr001 = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else 0

# ---------- Save Metrics ---------- #
metrics = {
    "dataset_type": dataset_type,
    "epochs": epochs,
    "accuracy": round(acc, 4),
    "roc_auc": round(roc_auc, 4),
    "tpr_at_fpr_0.01": round(tpr_at_fpr001, 4)
}

with open(os.path.join(base_path, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ---------- Output ---------- #
print("\n✅ Evaluation Complete")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"TPR @ FPR=0.01: {tpr_at_fpr001:.4f}")
