# train.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime
from config import BASELINE_RESULTS_PATH, mask_keywords
from utils import extract_embedding, cosine_similarity, make_masked_unmasked_pairs


# -------- Args -------- #
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()


# -------- ArcFace -------- #
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = nn.functional.one_hot(label, num_classes=self.weight.shape[0]).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


# -------- Transforms -------- #
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------- Data -------- #
train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
num_classes = len(train_dataset.classes)

val_root = os.path.join(args.data_dir, "val")
val_pairs, val_labels = make_masked_unmasked_pairs(val_root, mask_keywords)

# -------- Model Setup -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.resnet50(pretrained=True)
in_features = backbone.fc.in_features
backbone.fc = nn.Identity()

embedding_size = 512
fc = nn.Linear(in_features, embedding_size)
model = nn.Sequential(backbone, fc).to(device)

arcface = ArcMarginProduct(embedding_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=1e-3)

# -------- Logging -------- #
experiment_name = f"{args.dataset_type}/epoch_{args.epochs}"
base_dir = os.path.join(BASELINE_RESULTS_PATH, experiment_name)
os.makedirs(base_dir, exist_ok=True)

epoch_losses = []
epoch_accuracies = []
start_time = datetime.now()

# -------- Training -------- #
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        embeddings = model(imgs)
        logits = arcface(embeddings, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)

    # Verification Accuracy (masked ↔ unmasked)
    model.eval()
    scores = [cosine_similarity(
        extract_embedding(model, m_path, transform, device),
        extract_embedding(model, u_path, transform, device))
        for m_path, u_path in val_pairs
    ]
    threshold = 0.5
    preds = [1 if s > threshold else 0 for s in scores]

    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    acc = accuracy_score(val_labels, preds)
    roc_auc = roc_auc_score(val_labels, scores)
    fpr, tpr, _ = roc_curve(val_labels, scores)
    tpr_at_fpr001 = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else 0

    epoch_accuracies.append(acc)
    print(f"Epoch [{epoch+1}] Loss: {epoch_loss:.4f} | Verif Acc: {acc:.4f} | AUC: {roc_auc:.4f}")

# -------- Save -------- #
torch.save(model, os.path.join(base_dir, "model_full.pth"))
end_time = datetime.now()
time_per_epoch = (end_time - start_time).total_seconds() / args.epochs

metrics = {
    "dataset": args.dataset_type,
    "epochs": args.epochs,
    "last_verif_accuracy": epoch_accuracies[-1],
    "last_roc_auc": roc_auc,
    "time_per_epoch_sec": time_per_epoch,
}
with open(os.path.join(base_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

plt.figure()
plt.plot(range(1, args.epochs + 1), epoch_losses, label="Loss")
plt.plot(range(1, args.epochs + 1), epoch_accuracies, label="Verif Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.savefig(os.path.join(base_dir, "training_curves.png"))
plt.close()
