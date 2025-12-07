"""
Step 8 – regression for valence / energy / danceability on the multitask dataset.
Assumes that Steps 5, 7.1 and 7.2 have already run, so backbones (LSTMBackbone,
CNNBackbone, ASTBackbone) and common globals (DEVICE, BATCH_SIZE, MAX_LENGTH, etc.)
are available in the current notebook.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from dataset import SpectrogramDataset
from lstm import LSTMBackbone


# Regressor wrapper provided by the instructor (slightly guarded for checkpoints)
class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        super(Regressor, self).__init__()
        self.backbone = backbone
        if load_from_checkpoint is not None:
            # load_backbone_from_checkpoint is available in the helper modules
            from modules import load_backbone_from_checkpoint

            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        loss = self.criterion(out.float(), targets.float())
        return loss, out


# ------------------------------------------------------------------
# Configuration (reuses values from previous steps when present)
# ------------------------------------------------------------------
DEVICE = globals().get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = globals().get("BATCH_SIZE", 8)
MAX_LENGTH = globals().get("MAX_LENGTH", 150)
LR = globals().get("LR", 1e-4)
EPOCHS = globals().get("EPOCHS", 40)

CNN_IN_CHANNELS = globals().get("CNN_IN_CHANNELS", 1)
CNN_FILTERS = globals().get("CNN_FILTERS", [32, 64, 128, 256])
CNN_FEATURE_SIZE = globals().get("CNN_FEATURE_SIZE", 1024)
CNN_LR = globals().get("CNN_LR", 3e-4)
CNN_WEIGHT_DECAY = globals().get("CNN_WEIGHT_DECAY", 1e-4)

AST_FEATURE_SIZE = globals().get("AST_FEATURE_SIZE", 1024)
AST_LR = globals().get("AST_LR", 1e-4)

PARENT_DATA_DIR = globals().get(
    "PARENT_DATA_DIR", "../input/patreco3-multitask-affective-music/data/"
)
MULTITASK_ROOT = os.path.join(PARENT_DATA_DIR, "multitask_dataset")

REGRESSION_LABELS = {
    "valence": 1,
    "energy": 2,
    "danceability": 3,
}

# We keep mel-only features to stay consistent with Steps 5/7 inputs
REG_FEAT_TYPE = "mel"


def build_regression_loaders(target_name, batch_size=BATCH_SIZE, seed=42):
    """
    Create train/val/test loaders (70/15/15) for a single regression target.
    """
    label_idx = REGRESSION_LABELS[target_name]
    dataset = SpectrogramDataset(
        MULTITASK_ROOT,
        class_mapping=None,
        train=True,
        feat_type=REG_FEAT_TYPE,
        max_length=MAX_LENGTH,
        regression_label_index=label_idx,
    )

    indices = np.arange(len(dataset))
    # First split: train vs temp (val+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, shuffle=True
    )
    # Second split: val vs test (equal share → 15% each overall)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, shuffle=True
    )

    def loader(idxs):
        sampler = SubsetRandomSampler(idxs)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    train_loader = loader(train_idx)
    val_loader = loader(val_idx)
    test_loader = loader(test_idx)

    x_b, _, _ = next(iter(train_loader))
    input_shape = x_b[0].shape  # (T, F)
    return train_loader, val_loader, test_loader, input_shape


def run_epoch(model, loader, optimizer=None, device=DEVICE):
    """
    Shared train/val/test loop. Returns average loss and numpy targets/preds.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_targets, all_preds = [], []

    for xb, yb, lengths in loader:
        xb = xb.float().to(device)
        yb = yb.float().unsqueeze(1).to(device)
        lengths = lengths.to(device)

        if is_train:
            optimizer.zero_grad()

        loss, preds = model(xb, yb, lengths)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_targets.append(yb.detach().cpu())
        all_preds.append(preds.detach().cpu())

    avg_loss = total_loss / len(loader)
    targets_np = torch.cat(all_targets).squeeze(-1).numpy()
    preds_np = torch.cat(all_preds).squeeze(-1).numpy()
    return avg_loss, targets_np, preds_np


def train_regressor(model, train_loader, val_loader, optimizer, epochs=EPOCHS, device=DEVICE):
    """
    Train with early stopping on validation MSE; restores best weights.
    """
    best_state, best_val = None, np.inf
    patience, wait = 5, 0

    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, device=device)
        val_loss, _, _ = run_epoch(model, val_loader, optimizer=None, device=device)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[{epoch+1:03d}/{epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1} (val_loss={val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def evaluate_regressor(model, loader, device=DEVICE):
    """
    Run full evaluation and compute MSE + Spearman correlation.
    """
    loss, targets_np, preds_np = run_epoch(model, loader, optimizer=None, device=device)
    mse = mean_squared_error(targets_np, preds_np)
    corr = spearmanr(targets_np, preds_np).correlation
    corr = 0.0 if np.isnan(corr) else corr
    return {"loss": loss, "mse": mse, "spearman": corr}


# ------------------------------------------------------------------
# Main loop over targets and models
# ------------------------------------------------------------------
all_results = {}
model_names = ["lstm", "cnn", "ast"]

for target_name in REGRESSION_LABELS:
    print(f"\n================ Regression for {target_name} ================")
    train_loader, val_loader, test_loader, input_shape = build_regression_loaders(
        target_name
    )

    if "CNNBackbone" not in globals():
        raise RuntimeError("CNNBackbone not found. Run the Step 7.1 cell first.")
    if "ASTBackbone" not in globals():
        raise RuntimeError("ASTBackbone not found. Run the Step 7.2 cell first.")

    ast_input_tdim = input_shape[0]
    ast_input_fdim = input_shape[1]

    # Build backbones exactly as in Steps 5/7
    lstm_backbone = LSTMBackbone(
        input_shape[1], rnn_size=128, num_layers=2, bidirectional=True
    )
    cnn_backbone = CNNBackbone(
        input_dims=input_shape,
        in_channels=CNN_IN_CHANNELS,
        filters=CNN_FILTERS,
        feature_size=CNN_FEATURE_SIZE,
    )
    ast_backbone = ASTBackbone(
        input_fdim=ast_input_fdim,
        input_tdim=ast_input_tdim,
        imagenet_pretrain=True,
        model_size="base384",
        feature_size=AST_FEATURE_SIZE,
    )

    models = {
        "lstm": Regressor(lstm_backbone).to(DEVICE),
        "cnn": Regressor(cnn_backbone).to(DEVICE),
        "ast": Regressor(ast_backbone).to(DEVICE),
    }
    optimizers = {
        "lstm": torch.optim.Adam(models["lstm"].parameters(), lr=LR),
        "cnn": torch.optim.Adam(
            models["cnn"].parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
        ),
        "ast": torch.optim.Adam(models["ast"].parameters(), lr=AST_LR),
    }

    all_results[target_name] = {}
    for name in model_names:
        print(f"\n>>> Training {name.upper()} regressor for {target_name}")
        val_loss = train_regressor(
            models[name],
            train_loader,
            val_loader,
            optimizers[name],
            epochs=EPOCHS,
            device=DEVICE,
        )
        test_metrics = evaluate_regressor(models[name], test_loader, device=DEVICE)
        all_results[target_name][name] = {
            "val_loss": val_loss,
            "test_mse": test_metrics["mse"],
            "test_spearman": test_metrics["spearman"],
        }
        print(
            f"{target_name} – {name.upper()} | "
            f"test MSE: {test_metrics['mse']:.4f} | "
            f"Spearman: {test_metrics['spearman']:.4f}"
        )


# ------------------------------------------------------------------
# Aggregate metrics and plot grouped bars
# ------------------------------------------------------------------
mean_spearman = {
    name: np.mean([all_results[t][name]["test_spearman"] for t in REGRESSION_LABELS])
    for name in model_names
}
print("\nMean Spearman correlation across targets:")
for name, score in mean_spearman.items():
    print(f"  {name.upper()}: {score:.4f}")

axes = list(REGRESSION_LABELS.keys())
x = np.arange(len(axes))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, name in enumerate(model_names):
    offsets = x + (i - 1) * width
    vals = [all_results[a][name]["test_spearman"] for a in axes]
    ax.bar(offsets, vals, width=width, label=name.upper())

ax.set_xticks(x)
ax.set_xticklabels([a.title() for a in axes])
ax.set_ylabel("Spearman correlation (test)")
ax.set_title("Step 8: Spearman per axis and model")
ax.legend()
plt.tight_layout()
plt.show()
