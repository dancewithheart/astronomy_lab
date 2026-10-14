"""Compare a few CNN training configurations on the SAME Galaxy10 split.

Run: python -m experiments.galaxy_morphology_tune
Only validation scores determine the winner; test is evaluated once afterward.
NOTE: this project's test split has already been inspected in earlier runs,
so the resulting test score is exploratory, not a fresh independent holdout.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.galaxy_morphology import (
    DATA, NAMES, SEED, SmallCNN, augment_batch, load_balanced, predict, save_plots,
)

OUT = Path("reports/galaxy_morphology_tuning")
CONFIGS = [
    # name, augmentation, learning rate, weight decay
    ("plain",          False, 1e-3, 0.0),
    ("augmented",       True,  1e-3, 0.0),
    ("augmented_slow",  True,  3e-4, 0.0),
    ("augmented_decay", True,  1e-3, 1e-4),
]


def run_one(name, augmented, lr, weight_decay, x, y, train, valid,
            device, epochs, patience):
    # Identical weight initialization and minibatch order for fair comparisons.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(x[train], torch.from_numpy(y[train])),
        batch_size=64, shuffle=True, generator=generator,
    )
    valid_loader = DataLoader(
        TensorDataset(x[valid], torch.from_numpy(y[valid])), batch_size=64,
    )

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_epoch, best_weights = -1.0, 0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, target in train_loader:
            batch = batch.to(device).float() / 255
            if augmented:
                batch = augment_batch(batch)
            target = target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        predicted = predict(model, valid_loader, device).argmax(axis=1)
        val_f1 = f1_score(y[valid], predicted, average="macro", zero_division=0)
        row = {"config": name, "epoch": epoch,
               "train_loss": total_loss / len(train), "val_macro_F1": val_f1}
        history.append(row)
        print(f"  epoch={epoch:2d} train_loss={row['train_loss']:.4f} "
              f"val_macro_F1={val_f1:.3f}", flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_weights = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  early stopping: no improvement for {patience} epochs")
            break

    summary = {"config": name, "augment": augmented, "lr": lr,
               "weight_decay": weight_decay, "best_epoch": best_epoch,
               "best_val_macro_F1": best_f1}
    return summary, history, best_weights


def main(per_class=1200, epochs=16, patience=6):
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}: download Galaxy10.h5 first")
    OUT.mkdir(parents=True, exist_ok=True)
    source_ids, images, y = load_balanced(DATA, per_class)
    ids = np.arange(len(y))
    trainval, test = train_test_split(ids, test_size=0.20,
                                      stratify=y, random_state=SEED)
    train, valid = train_test_split(trainval, test_size=0.20,
                                    stratify=y[trainval], random_state=SEED)
    print(f"Galaxy10: train={len(train)}, validation={len(valid)}, test={len(test)}")
    print("Selection metric: BEST validation macro-F1 (test unseen during selection)")

    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {device}")

    summaries, histories = [], []
    winning_weights, winning_score, winner = None, -1.0, ""
    for name, augmented, lr, wd in CONFIGS:
        print(f"\n[{name}] augment={augmented}, lr={lr}, weight_decay={wd}", flush=True)
        summary, history, weights = run_one(
            name, augmented, lr, wd, x, y, train, valid,
            device, epochs, patience,
        )
        summaries.append(summary)
        histories.extend(history)
        if summary["best_val_macro_F1"] > winning_score:
            winning_score = summary["best_val_macro_F1"]
            winning_weights, winner = weights, name

    leaderboard = pd.DataFrame(summaries).sort_values("best_val_macro_F1", ascending=False)
    pd.DataFrame(histories).to_csv(OUT / "history.csv", index=False)
    leaderboard.to_csv(OUT / "validation.csv", index=False)
    print("\n[Validation leaderboard — used to select model]")
    print(leaderboard.to_string(index=False, formatters={
        "lr": lambda z: f"{z:.0e}",
        "weight_decay": lambda z: f"{z:.0e}",
        "best_val_macro_F1": lambda z: f"{z:.3f}",
    }))

    # Evaluate on the test split only AFTER choosing the configuration.
    model = SmallCNN().to(device)
    model.load_state_dict(winning_weights)
    test_loader = DataLoader(
        TensorDataset(x[test], torch.from_numpy(y[test])), batch_size=64,
    )
    probs = predict(model, test_loader, device)
    pred = probs.argmax(axis=1)
    test_metrics = {"selected_config": winner,
                    "test_accuracy": accuracy_score(y[test], pred),
                    "test_macro_F1": f1_score(y[test], pred, average="macro")}
    for cls, name in enumerate(NAMES):
        mask = y[test] == cls
        test_metrics[f"recall_{name.lower().replace('-', '_')}"] = (pred[mask] == cls).mean()
    pd.DataFrame([test_metrics]).to_csv(OUT / "selected_test_metrics.csv", index=False)
    pd.DataFrame({"source_index": source_ids[test],
                  "true": y[test], "predicted": pred,
                  "confidence": probs.max(axis=1)}).to_csv(OUT / "test_predictions.csv", index=False)
    save_plots(images[test], y[test], pred, OUT)
    print("\n[Selected configuration; test metrics are exploratory]")
    for key, value in test_metrics.items():
        print(f"  {key}: {value if isinstance(value, str) else f'{value:.3f}'}")
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    main(args.per_class, args.epochs, args.patience)
