"""
Step 10 – Multitask regression (valence, energy, danceability)
Runs a single model with a multitask MSE loss over the three affective axes.
Includes loss curves and test Spearman per task.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from lstm import LSTMBackbone


# ------------------------------------------------------------
# Configuration (reuses globals when present)
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = globals().get("BATCH_SIZE", 8)
MAX_LENGTH = globals().get("MAX_LENGTH", 150)
LR = globals().get("LR", 1e-4)
EPOCHS = globals().get("EPOCHS", 40)
MODEL_CHOICE = "lstm"  # options: "lstm", "cnn", "ast"

PARENT_DATA_DIR = globals().get(
    "PARENT_DATA_DIR",
    "../input/patreco3-multitask-affective-music/data/"
)
MULTITASK_ROOT = os.path.join(PARENT_DATA_DIR, "multitask_dataset")
LABELS_PATH = os.path.join(MULTITASK_ROOT, "train_labels.txt")

TASK_NAMES = ["valence", "energy", "danceability"]
TASK_WEIGHTS = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)  # adjust if needed


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class MultiTaskAffectiveDataset(torch.utils.data.Dataset):
    def __init__(self, labels_csv, data_root, max_length=MAX_LENGTH):
        df = pd.read_csv(labels_csv)
        self.ids = df["Id"].astype(str).tolist()
        self.labels = df[["valence", "energy", "danceability"]].values.astype(np.float32)
        self.data_root = data_root
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        track_id = self.ids[idx]
        y = self.labels[idx]
        spec = np.load(os.path.join(self.data_root, "train", f"{track_id}.fused.full.npy"))
        x = spec[:128, :].T  # mel only, shape (T, F)
        length = min(x.shape[0], self.max_length)
        if x.shape[0] > self.max_length:
            x = x[: self.max_length]
        else:
            pad = np.zeros((self.max_length - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([x, pad])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), torch.tensor(length, dtype=torch.long)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_backbone(name, input_shape):
    if name == "lstm":
        return LSTMBackbone(input_shape[1], rnn_size=128, num_layers=2, bidirectional=True)
    if name == "cnn":
        if "CNNBackbone" not in globals():
            raise RuntimeError("CNNBackbone not found. Run Step 7.1 first.")
        return CNNBackbone(
            input_dims=input_shape,
            in_channels=1,
            filters=[32, 64, 128, 256],
            feature_size=1024,
        )
    if name == "ast":
        if "ASTBackbone" not in globals():
            raise RuntimeError("ASTBackbone not found. Run Step 7.2 first.")
        return ASTBackbone(
            input_fdim=input_shape[1],
            input_tdim=input_shape[0],
            imagenet_pretrain=True,
            model_size="base384",
            feature_size=1024,
        )
    raise ValueError("MODEL_CHOICE must be one of ['lstm', 'cnn', 'ast'].")


class MultiTaskRegressor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 3)
        self.criterion = nn.MSELoss()

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        preds = self.output_layer(feats)
        losses = [self.criterion(preds[:, i], targets[:, i]) for i in range(3)]
        loss = torch.sum(torch.stack(losses) * TASK_WEIGHTS)
        return loss, preds


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_loaders():
    ds = MultiTaskAffectiveDataset(LABELS_PATH, MULTITASK_ROOT, max_length=MAX_LENGTH)
    indices = np.arange(len(ds))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, shuffle=True)

    def loader(idxs):
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(idxs))

    train_loader, val_loader, test_loader = loader(train_idx), loader(val_idx), loader(test_idx)
    xb, _, _ = next(iter(train_loader))
    return train_loader, val_loader, test_loader, xb[0].shape  # (T, F)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_t, all_p = 0.0, [], []
    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(DEVICE), yb.to(DEVICE), lengths.to(DEVICE)
        if is_train:
            optimizer.zero_grad()
        loss, preds = model(xb, yb, lengths)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        all_t.append(yb.cpu()); all_p.append(preds.detach().cpu())
    avg_loss = total_loss / len(loader)
    targets = torch.cat(all_t).numpy()
    preds = torch.cat(all_p).numpy()
    return avg_loss, targets, preds


def evaluate(model, loader):
    loss, targets, preds = run_epoch(model, loader, optimizer=None)
    mse = [mean_squared_error(targets[:, i], preds[:, i]) for i in range(3)]
    corr = [spearmanr(targets[:, i], preds[:, i]).correlation for i in range(3)]
    corr = [0.0 if np.isnan(c) else c for c in corr]
    return loss, mse, corr


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
train_loader, val_loader, test_loader, input_shape = build_loaders()
backbone = build_backbone(MODEL_CHOICE, input_shape)
model = MultiTaskRegressor(backbone).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_state, best_val, history = None, np.inf, {"train": [], "val": []}
for epoch in range(EPOCHS):
    train_loss, _, _ = run_epoch(model, train_loader, optimizer)
    val_loss, _, _ = evaluate(model, val_loader)
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    if val_loss < best_val:
        best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"[{epoch+1:03d}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

if best_state is not None:
    model.load_state_dict(best_state)

# ------------------------------------------------------------
# Test evaluation
# ------------------------------------------------------------
test_loss, test_mse, test_corr = evaluate(model, test_loader)
print(f"\nTest loss={test_loss:.4f}")
for i, task in enumerate(TASK_NAMES):
    print(f"{task}: MSE={test_mse[i]:.4f}, Spearman={test_corr[i]:.4f}")

# ------------------------------------------------------------
# Plots: loss curves + Spearman bar
# ------------------------------------------------------------
epochs_axis = np.arange(1, len(history["train"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs_axis, history["train"], label="Train loss")
axes[0].plot(epochs_axis, history["val"], label="Val loss")
axes[0].set_title("Multitask loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].bar(TASK_NAMES, test_corr, color=["steelblue", "orange", "seagreen"])
axes[1].set_title("Test Spearman per task")
axes[1].set_ylim(0, max(test_corr) * 1.2 + 1e-6)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Markdown summary (for notebook)
# ------------------------------------------------------------
markdown_explainer = """
- **Step 10 – Multitask regression (valence, energy, danceability)**
  - Shared backbone (default: LSTM; swap to CNN/AST by setting MODEL_CHOICE) with a 3-output head.
  - Custom loss = sum of per-task MSE (optional weights to balance scales).
  - Train on the multitask dataset (mel-only, padded to MAX_LENGTH) with a 70/15/15 split; early-stop on validation loss.
  - Evaluate on the held-out test split and report MSE + Spearman for each task.
  - Plots: training/validation loss curves and a bar chart of test Spearman per task.
"""
