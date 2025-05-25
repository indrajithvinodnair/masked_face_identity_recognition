# masked_vs_unmasked_eval.py
import os
import argparse
import torch
import json
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm
from utils import extract_embedding, cosine_similarity, make_masked_unmasked_pairs
from config import  mask_keywords


# ---------- Args ---------- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--test_root", type=str, required=True)
args = parser.parse_args()

# ---------- Paths ---------- #
experiment_path = f"baseline_results/{args.dataset_type}/epoch_{args.epochs}"
model_path = os.path.join(experiment_path, "model_full.pth")
metrics_path = os.path.join(experiment_path, "masked_vs_unmasked_metrics.json")

# ---------- Setup ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------- Evaluation ---------- #
pairs, labels = make_masked_unmasked_pairs(args.test_root, mask_keywords)

scores = []
for m_path, u_path in tqdm(pairs, desc="Evaluating masked ↔ unmasked"):
    emb_m = extract_embedding(model, m_path, transform, device)
    emb_u = extract_embedding(model, u_path, transform, device)
    scores.append(cosine_similarity(emb_m, emb_u))

# ---------- Metrics ---------- #
threshold = 0.5
preds = [1 if s > threshold else 0 for s in scores]
acc = accuracy_score(labels, preds)
roc_auc = roc_auc_score(labels, scores)
fpr, tpr, _ = roc_curve(labels, scores)
tpr_at_fpr001 = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else 0

metrics = {
    "masked_vs_unmasked_only": True,
    "dataset": args.dataset_type,
    "epochs": args.epochs,
    "accuracy": round(acc, 4),
    "roc_auc": round(roc_auc, 4),
    "tpr@fpr=0.01": round(tpr_at_fpr001, 4)
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Masked ↔ Unmasked Evaluation Complete")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"TPR @ FPR=0.01: {tpr_at_fpr001:.4f}")
