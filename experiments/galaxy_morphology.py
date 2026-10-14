"""First CNN experiment on Galaxy10 SDSS: three broad morphology classes.

Dataset: https://zenodo.org/records/10844811/files/Galaxy10.h5
Run from astronomy_visual: python -m experiments.galaxy_morphology
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("data/Galaxy10.h5")
OUT = Path("reports/galaxy_morphology")
SEED = 42
NAMES = ["Smooth", "Edge-on", "Spiral"]
# Galaxy10 SDSS source labels; classes 0 and 5 are intentionally excluded.
GROUPS = [(1, 2, 3), (4, 6), (7, 8, 9)]


def load_balanced(path=DATA, per_class=1200):
    rng = np.random.default_rng(SEED)
    with h5py.File(path, "r") as f:
        labels = np.asarray(f["ans"]).reshape(-1)
        original_indices = []
        for source_labels in GROUPS:
            pool = np.flatnonzero(np.isin(labels, source_labels))
            if len(pool) < per_class:
                raise ValueError(f"Not enough objects for {source_labels}: {len(pool)}")
            original_indices.extend(rng.choice(pool, per_class, replace=False))
        original_indices = np.sort(np.array(original_indices, dtype=int))
        images = np.asarray(f["images"][original_indices], dtype=np.uint8)
        selected_labels = labels[original_indices]

    y = np.full(len(original_indices), -1, dtype=np.int64)
    for cls, source_labels in enumerate(GROUPS):
        y[np.isin(selected_labels, source_labels)] = cls
    assert np.all(y >= 0)
    return original_indices, images, y


def colour_features(images):
    """Six simple features: channel means and standard deviations."""
    pixels = images.astype(np.float32) / 255
    return np.column_stack([pixels.mean(axis=(1, 2)), pixels.std(axis=(1, 2))])

class AstroNNCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.1),

            nn.Linear(16 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 3),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.layers(x)



class AstroNNCNN(nn.Module):
    """PyTorch adaptation of astroNN Galaxy10CNN/Cifar10CNN layer layout.

    Source: henrysky/astroNN, src/astroNN/models/misc_models.py.
    Changed: 3 outputs (instead of 10). Uses the existing PyTorch trainer,
    not astroNN's original optimizer schedule or Dense MaxNorm constraint.
    Expects 69x69 images, as in Galaxy10 SDSS.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Flatten(), nn.Dropout(0.1),
            nn.Linear(16 * 17 * 17, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, len(NAMES)),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)


def predict(model, loader, device):
    model.eval()
    result = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device).float() / 255)
            result.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(result)


def save_examples(images, labels, out_dir=OUT):
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for cls, row in enumerate(axes):
        for ax, i in zip(row, np.flatnonzero(labels == cls)[:4]):
            ax.imshow(images[i])
            ax.set_title(NAMES[cls], fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "examples.png", dpi=160)
    plt.close(fig)


def save_plots(images, truth, predicted, out_dir=OUT):
    ConfusionMatrixDisplay.from_predictions(
        truth, predicted, display_labels=NAMES, normalize="true",
        values_format=".2f", cmap="Blues",
    )
    plt.title("CNN: held-out confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=160)
    plt.close()

    wrong = np.flatnonzero(truth != predicted)[:9]
    if len(wrong):
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, wrong):
            ax.imshow(images[i])
            ax.set_title(f"True: {NAMES[truth[i]]}\nCNN: {NAMES[predicted[i]]}", fontsize=9)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / "mistakes.png", dpi=160)
        plt.close(fig)


def main(per_class=1200, epochs=6, augment=False):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out_dir = OUT.with_name("galaxy_morphology_augmented") if augment else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f"Download Galaxy10 SDSS to {DATA} first")

    source_ids, images, y = load_balanced(DATA, per_class)
    save_examples(images, y, out_dir)
    indices = np.arange(len(y))
    trainval, test = train_test_split(indices, test_size=0.20, stratify=y, random_state=SEED)
    train, valid = train_test_split(
        trainval, test_size=0.20, stratify=y[trainval], random_state=SEED,
    )
    print(f"Galaxy10: {len(y):,} selected ({per_class} per class)")
    print(f"train={len(train):,}, validation={len(valid):,}, test={len(test):,}")
    print(f"Rotation/flip augmentation: {augment}")

    features = colour_features(images)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    baseline.fit(features[train], y[train])
    baseline_pred = baseline.predict(features[test])

    # Keep uint8 pixels in memory. Convert each minibatch to float in predict/train.
    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    labels = torch.from_numpy(y)

    def loader(which, shuffle=False):
        return DataLoader(TensorDataset(x[which], labels[which]), batch_size=64, shuffle=shuffle)

    train_loader = loader(train, shuffle=True)
    valid_loader = loader(valid)
    test_loader = loader(test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, target in train_loader:
            batch = batch.to(device).float() / 255
            if augment:
                batch = augment_batch(batch)
            target = target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        val_pred = predict(model, valid_loader, device).argmax(axis=1)
        val_f1 = f1_score(y[valid], val_pred, average="macro", zero_division=0)
        print(f"epoch={epoch} train_loss={total_loss / len(train):.4f} val_macro_F1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_weights)
    probabilities = predict(model, test_loader, device)
    cnn_pred = probabilities.argmax(axis=1)
    rows = []
    for name, predicted in [("Colour baseline", baseline_pred), ("CNN", cnn_pred)]:
        acc = accuracy_score(y[test], predicted)
        f1 = f1_score(y[test], predicted, average="macro", zero_division=0)
        rows.append({"model": name, "accuracy": acc, "macro_F1": f1})
    scores = pd.DataFrame(rows)
    print("\n[Held-out test]")
    print(scores.to_string(index=False, float_format=lambda z: f"{z:.3f}"))
    print("\n[Per-class recall]")
    for k, name in enumerate(NAMES):
        mask = y[test] == k
        print(f"{name:10s} {np.mean(cnn_pred[mask] == k):.3f}  n={mask.sum()}")

    scores.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame({
        "source_index": source_ids[test], "true": y[test], "predicted": cnn_pred,
        "confidence": probabilities.max(axis=1),
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    save_plots(images[test], y[test], cnn_pred, out_dir)
    print(f"Saved metrics and plots to {out_dir}/")

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random rotations and flips, independently for each image."""
    turns = torch.randint(4, (x.size(0),), device=x.device)
    out = torch.empty_like(x)

    for k in range(4):
        mask = turns == k
        if mask.any():
            out[mask] = torch.rot90(x[mask], k, dims=(-2, -1))

    for dim in (-2, -1):
        mask = torch.rand(x.size(0), device=x.device) < 0.5
        out[mask] = torch.flip(out[mask], dims=(dim,))

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--augment", action="store_true",
                        help="Apply random rotations and flips to training images")
    args = parser.parse_args()
    main(per_class=args.per_class, epochs=args.epochs, augment=args.augment)
