import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime
from config import DATASET_ROOT, BASELINE_RESULTS_PATH


# -------- Command-Line Args -------- #
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
parser.add_argument('--dataset_type', type=str, required=True, help='Dataset identifier (e.g., LFW_only)')
parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
args = parser.parse_args()

# -------- ArcFace Loss -------- #
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
        one_hot = torch.nn.functional.one_hot(label, num_classes=self.weight.shape[0]).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        print("Class count:", num_classes)
        print("Example label from train/val:", labels[0].item())
        print("train_dataset.class_to_idx")
        print("val_dataset.class_to_idx")


        return logits

# -------- Data Prep -------- #
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
num_classes = len(train_dataset.classes)

# -------- Model -------- #
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance
else:
    device = torch.device("cpu")
    device_name = "CPU"

# Ensure the correct device is being used
if device.type == "cuda":
    torch.cuda.set_device(0)  # Set to the first CUDA device

print(f"Using device: {device} ({device_name})")  # Indicate the device being used
backbone = models.resnet50(pretrained=True)
in_features = backbone.fc.in_features
backbone.fc = nn.Identity()

embedding_size = 512
fc = nn.Linear(in_features, embedding_size)
model = nn.Sequential(backbone, fc).to(device)

arcface = ArcMarginProduct(embedding_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=1e-3)

# -------- Logging Setup -------- #
experiment_name = f"{args.dataset_type}/epoch_{args.epochs}"
base_dir = os.path.join(BASELINE_RESULTS_PATH, experiment_name)
os.makedirs(base_dir, exist_ok=True)

epoch_losses = []
epoch_accuracies = []
start_time = datetime.now()

# -------- Training Loop -------- #
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

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings = model(imgs)
            logits = arcface(embeddings, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(f"Preds: {preds[:5].tolist()}, Labels: {labels[:5].tolist()}")


    val_accuracy = correct / total
    epoch_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_accuracy:.4f}")

# -------- Save Outputs -------- #
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
time_per_epoch = total_time / args.epochs

torch.save(model, os.path.join(base_dir, "model_full.pth"))


# Save metrics as JSON
metrics = {
    "dataset": args.dataset_type,
    "epochs": args.epochs,
    "final_val_accuracy": epoch_accuracies[-1],
    "time_per_epoch_sec": time_per_epoch,
    "notes": "ArcFace + ResNet50"
}
with open(os.path.join(base_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Plot curves
plt.figure(figsize=(8, 4))
plt.plot(range(1, args.epochs + 1), epoch_losses, label='Loss')
plt.plot(range(1, args.epochs + 1), epoch_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title(f"Training on {args.dataset_type}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "training_curves.png"))
plt.close()
