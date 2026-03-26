"""
CIFAR-10 classifier (PyTorch)

Simple CNN trained from scratch on CIFAR-10.
Uses some basic augmentation + dropout + batch norm.
Tracks train/val metrics and saves best checkpoint.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# --------------------------
# config
# --------------------------
batch_size = 64
lr = 1e-3
epochs = 15
weight_decay = 1e-4
checkpoint_path = "best_cifar_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

print("device:", device)


# --------------------------
# data
# --------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("train:", len(train_data), "test:", len(test_data))


# --------------------------
# model
# --------------------------
class SmallCifarNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SmallCifarNet().to(device)
print("params:", sum(p.numel() for p in model.parameters()))


# --------------------------
# training setup
# --------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# --------------------------
# helpers
# --------------------------
def train_one_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = out.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    return total_loss / len(train_loader), 100 * correct / total


def evaluate():
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

    return total_loss / len(test_loader), 100 * correct / total


def unnormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


# --------------------------
# training loop
# --------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val = 0

print("\n--- training ---")

for e in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"{e:2d} | {train_loss:.3f} {train_acc:.2f}% | {val_loss:.3f} {val_acc:.2f}%")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), checkpoint_path)

print("best val:", best_val)


# --------------------------
# nicer plots
# --------------------------
def plot_curves():
    ep = np.arange(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(ep, history["train_loss"], marker="o", linewidth=2, label="train")
    ax1.plot(ep, history["val_loss"], marker="o", linewidth=2, label="val")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(ep, history["train_acc"], marker="o", linewidth=2, label="train")
    ax2.plot(ep, history["val_acc"], marker="o", linewidth=2, label="val")
    ax2.set_title("accuracy")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.show()


# --------------------------
# predictions
# --------------------------
def show_preds(n=8):
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    idxs = np.random.choice(len(test_data), n, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, idx in enumerate(idxs):
        img, label = test_data[idx]

        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
            pred = out.argmax(dim=1).item()

        axes[i].imshow(unnormalize(img))
        axes[i].axis("off")
        axes[i].set_title(f"{classes[pred]} / {classes[label]}", fontsize=10)

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=200)
    plt.show()


plot_curves()
show_preds()
