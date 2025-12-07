"""
Step 9 – Transfer Learning
Two notebook-ready code cells:
1) Train a chosen source classifier on fma_genre_spectrograms and save the best checkpoint.
2) Fine-tune that backbone on a regression target (valence/energy/danceability) and evaluate.
"""

# =========================================
# Cell 1 — train source classifier (Q b, c)
# =========================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import SpectrogramDataset, CLASS_MAPPING, torch_train_val_split
from lstm import LSTMBackbone

# If the helper Classifier from earlier steps is not in scope, define a minimal version here.
if "Classifier" not in globals():
    class Classifier(nn.Module):
        def __init__(self, num_classes, backbone):
            super().__init__()
            self.backbone = backbone
            self.is_lstm = isinstance(self.backbone, LSTMBackbone)
            self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x, targets, lengths):
            feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
            logits = self.output_layer(feats)
            loss = self.criterion(logits, targets)
            return loss, logits


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LENGTH = 150
LR = 1e-4
EPOCHS = 40
MODEL_CHOICE = "lstm"  # choose among: "lstm", "cnn", "ast"

MEL_ROOT = "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/"
NUM_CATEGORIES = len(CLASS_MAPPING)


def build_classification_loaders():
    dataset = SpectrogramDataset(
        MEL_ROOT,
        class_mapping=CLASS_MAPPING,
        train=True,
        feat_type="mel",
        max_length=MAX_LENGTH,
    )
    train_loader, val_loader = torch_train_val_split(
        dataset, batch_train=BATCH_SIZE, batch_eval=BATCH_SIZE, val_size=0.2
    )
    xb, yb, lengths = next(iter(train_loader))
    input_shape = xb[0].shape  # (T, F)
    return train_loader, val_loader, input_shape


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


def evaluate_classification(model, loader, device=DEVICE):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb, lengths in loader:
            loss, logits = model(
                xb.float().to(device),
                yb.to(device),
                lengths.to(device),
            )
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds.cpu() == yb).sum().item()
            total += len(yb)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def train_source_model():
    train_loader, val_loader, input_shape = build_classification_loaders()
    backbone = build_backbone(MODEL_CHOICE, input_shape)
    model = Classifier(NUM_CATEGORIES, backbone).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    checkpoint_path = f"transfer_source_{MODEL_CHOICE}.pth"

    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        for xb, yb, lengths in train_loader:
            loss, logits = model(
                xb.float().to(DEVICE),
                yb.to(DEVICE),
                lengths.to(DEVICE),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validate
        val_loss, val_acc = evaluate_classification(model, val_loader, device=DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict()
            torch.save(best_state, checkpoint_path)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[{epoch+1:03d}/{EPOCHS}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best checkpoint: epoch {best_epoch}, val_acc={best_acc:.4f}")
    return model, checkpoint_path, history


source_model, checkpoint_path, clf_history = train_source_model()

# Plot training curves (loss + val accuracy)
epochs_axis = np.arange(1, len(clf_history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs_axis, clf_history["train_loss"], label="Train loss")
axes[0].plot(epochs_axis, clf_history["val_loss"], label="Val loss")
axes[0].set_title("Classification loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_axis, clf_history["val_acc"], label="Val accuracy", color="green")
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1)
axes[1].legend()
plt.tight_layout()
plt.show()


# =========================================
# Cell 2 — fine-tune on regression (Q d)
# =========================================

import copy
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


class Regressor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        loss = self.criterion(out.float(), targets.float())
        return loss, out


REGRESSION_LABELS = {"valence": 1, "energy": 2, "danceability": 3}
REG_TARGET = "valence"  # choose which affective axis to fine-tune on
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4
MULTITASK_ROOT = "../input/patreco3-multitask-affective-music/data/multitask_dataset"


def build_regression_loaders(target_name, batch_size=BATCH_SIZE, seed=42):
    label_idx = REGRESSION_LABELS[target_name]
    dataset = SpectrogramDataset(
        MULTITASK_ROOT,
        class_mapping=None,
        train=True,
        feat_type="mel",
        max_length=MAX_LENGTH,
        regression_label_index=label_idx,
    )
    indices = np.arange(len(dataset))
    train_idx, temp_idx = np.split(
        np.random.permutation(indices),
        [int(0.7 * len(indices))],
    )
    val_split = int(0.5 * len(temp_idx))
    val_idx, test_idx = temp_idx[:val_split], temp_idx[val_split:]

    def loader(idxs):
        return DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(idxs))

    train_loader = loader(train_idx)
    val_loader = loader(val_idx)
    test_loader = loader(test_idx)

    xb, _, _ = next(iter(train_loader))
    input_shape = xb[0].shape  # (T, F)
    return train_loader, val_loader, test_loader, input_shape


def load_backbone_weights(backbone, checkpoint_path):
    """Load only backbone.* weights from a saved Classifier checkpoint."""
    state = torch.load(checkpoint_path, map_location=DEVICE)
    backbone_state = {
        k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print("Missing keys when loading backbone:", missing)
    if unexpected:
        print("Unexpected keys when loading backbone:", unexpected)
    return backbone


def train_regressor(model, train_loader, val_loader, optimizer, epochs=FINE_TUNE_EPOCHS, device=DEVICE):
    best_state, best_val = None, np.inf
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        # train
        model.train()
        running = 0.0
        for xb, yb, lengths in train_loader:
            loss, _ = model(xb.float().to(device), yb.float().unsqueeze(1).to(device), lengths.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        # val
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb, lengths in val_loader:
                loss, _ = model(xb.float().to(device), yb.float().unsqueeze(1).to(device), lengths.to(device))
                val_running += loss.item()
        val_loss = val_running / len(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 2 == 0:
            print(f"[{epoch+1:02d}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_regressor(model, loader, device=DEVICE):
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for xb, yb, lengths in loader:
            _, preds = model(xb.float().to(device), yb.float().unsqueeze(1).to(device), lengths.to(device))
            all_t.append(yb)
            all_p.append(preds.cpu().squeeze(-1))
    targets = torch.cat(all_t).numpy()
    preds = torch.cat(all_p).numpy()
    mse = mean_squared_error(targets, preds)
    corr = spearmanr(targets, preds).correlation
    corr = 0.0 if np.isnan(corr) else corr
    return mse, corr


# Data for fine-tuning
train_reg_loader, val_reg_loader, test_reg_loader, reg_input_shape = build_regression_loaders(REG_TARGET)

# Rebuild backbone and load source weights
transfer_backbone = build_backbone(MODEL_CHOICE, reg_input_shape)
transfer_backbone = load_backbone_weights(transfer_backbone, checkpoint_path)
transfer_model = Regressor(transfer_backbone).to(DEVICE)
transfer_optimizer = torch.optim.Adam(transfer_model.parameters(), lr=FINE_TUNE_LR)

# Fine-tune
reg_history = train_regressor(
    transfer_model,
    train_reg_loader,
    val_reg_loader,
    transfer_optimizer,
    epochs=FINE_TUNE_EPOCHS,
    device=DEVICE,
)

# Evaluate on held-out test split
test_mse, test_spearman = evaluate_regressor(transfer_model, test_reg_loader, device=DEVICE)
print(f"Fine-tuned on {REG_TARGET}: test MSE={test_mse:.4f}, Spearman={test_spearman:.4f}")

# Visualize fine-tuning curves and test metrics
epochs_axis = np.arange(1, len(reg_history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs_axis, reg_history["train_loss"], label="Train loss")
axes[0].plot(epochs_axis, reg_history["val_loss"], label="Val loss")
axes[0].set_title(f"Regression loss ({REG_TARGET})")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].bar(["Test MSE", "Test Spearman"], [test_mse, test_spearman], color=["steelblue", "orange"])
axes[1].set_title("Fine-tuning results")
axes[1].set_ylim(0, max(test_mse, test_spearman) * 1.2 + 1e-6)
plt.tight_layout()
plt.show()
