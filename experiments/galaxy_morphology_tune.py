"""Compare a few CNN training configurations on the SAME Galaxy10 split.

Run: python -m experiments.galaxy_morphology_tune
Only validation scores determine the winner; test is evaluated once afterward.
NOTE: this project's test split has already been inspected in earlier runs,
so the resulting test score is exploratory, not a fresh independent holdout.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.galaxy_morphology import (
    DATA, NAMES, SEED, SmallCNN, AstroNNCNN, augment_batch, load_balanced, predict, save_plots,
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
            device, epochs, patience, seed=SEED, model_factory=SmallCNN):
    # Same initialization and minibatch order for paired configurations.
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x[train], torch.from_numpy(y[train])),
        batch_size=64, shuffle=True, generator=generator,
    )
    valid_loader = DataLoader(
        TensorDataset(x[valid], torch.from_numpy(y[valid])), batch_size=64,
    )

    model = model_factory().to(device)
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
        row = {"seed": seed, "config": name, "epoch": epoch,
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

    summary = {"seed": seed, "config": name, "augment": augmented, "lr": lr,
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


def seed_study(per_class=1200, epochs=30, seeds=(42, 43, 44)):
    """Paired comparison across random seeds; does not inspect test labels/results."""
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}: download Galaxy10.h5 first")
    if epochs < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Epochs must be positive and seeds must be unique")

    out = Path("reports/galaxy_morphology_seed_study")
    out.mkdir(parents=True, exist_ok=True)
    _, images, y = load_balanced(DATA, per_class)
    ids = np.arange(len(y))
    trainval, test = train_test_split(
        ids, test_size=0.20, stratify=y, random_state=SEED,
    )
    train, valid = train_test_split(
        trainval, test_size=0.20, stratify=y[trainval], random_state=SEED,
    )
    print(f"Galaxy10: train={len(train)}, validation={len(valid)}, "
          f"test={len(test)} (not evaluated)", flush=True)
    print(f"Seeds: {list(seeds)}; epochs: {epochs}; "
          "no early stopping; validation only", flush=True)

    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {device}", flush=True)

    summaries, histories = [], []
    for seed in seeds:
        for name, augment in (("plain", False), ("augmented", True)):
            print(f"\n[seed={seed} / {name}]", flush=True)
            result, history, _ = run_one(
                name, augment, 1e-3, 0.0, x, y, train, valid,
                device, epochs, patience=epochs + 1, seed=seed,
            )
            result["last_val_macro_F1"] = history[-1]["val_macro_F1"]
            summaries.append(result)
            histories.extend(history)

    runs = pd.DataFrame(summaries)
    curves = pd.DataFrame(histories)
    runs.to_csv(out / "runs.csv", index=False)
    curves.to_csv(out / "history.csv", index=False)

    paired = runs.pivot(index="seed", columns="config", values="best_val_macro_F1")
    paired["aug_minus_plain"] = paired["augmented"] - paired["plain"]
    paired.to_csv(out / "paired_by_seed.csv")

    summary = (runs.groupby("config")["best_val_macro_F1"]
               .agg(["mean", "std", "min", "max"]).round(4))
    summary.to_csv(out / "summary.csv")
    print("\n[Best validation macro-F1 by seed]")
    print(paired.round(4).to_string())
    print("\n[Across-seed summary; validation only]")
    print(summary.to_string())
    print(f"\nMean paired difference (augmented - plain): "
          f"{paired['aug_minus_plain'].mean():+.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("plain", "augmented"):
        group = curves[curves["config"] == name]
        by_epoch = group.groupby("epoch")["val_macro_F1"]
        centre = by_epoch.mean()
        low, high = by_epoch.min(), by_epoch.max()
        line, = ax.plot(centre.index, centre.values, label=name)
        ax.fill_between(centre.index, low.values, high.values,
                        color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Epoch", ylabel="Validation macro-F1",
           title="Galaxy10: mean across seeds (band = min–max)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=160)
    plt.close(fig)
    print(f"Saved to {out}/ (test set untouched)", flush=True)



def architecture_study(per_class=1200, epochs=25):
    """Compare two architectures; one seed, same augmentation and split.

    Validation only: intentionally does not evaluate the previously reused test set.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}: download Galaxy10.h5 first")

    out = Path("reports/galaxy_morphology_architecture")
    out.mkdir(parents=True, exist_ok=True)
    _, images, y = load_balanced(DATA, per_class)
    ids = np.arange(len(y))
    trainval, test = train_test_split(ids, test_size=0.20,
                                      stratify=y, random_state=SEED)
    train, valid = train_test_split(trainval, test_size=0.20,
                                    stratify=y[trainval], random_state=SEED)
    print(f"Galaxy10: train={len(train)}, validation={len(valid)}, "
          f"test={len(test)} (not evaluated)")
    print(f"Single seed={SEED}; augmented=True; epochs={epochs}; "
          "lr=0.001; weight_decay=0; no early stopping")

    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {device}")

    summaries, histories = [], []
    for name, factory in (("SmallCNN", SmallCNN), ("astroNN-layout", AstroNNCNN)):
        params = sum(p.numel() for p in factory().parameters())
        print(f"\n[{name}] parameters={params:,}", flush=True)
        result, history, _ = run_one(
            name, True, 1e-3, 0.0, x, y, train, valid,
            device, epochs, patience=epochs + 1, seed=SEED,
            model_factory=factory,
        )
        result["parameters"] = params
        summaries.append(result)
        histories.extend(history)

    scores = pd.DataFrame(summaries).sort_values("best_val_macro_F1", ascending=False)
    history = pd.DataFrame(histories)
    scores.to_csv(out / "validation.csv", index=False)
    history.to_csv(out / "history.csv", index=False)
    print("\n[Architecture comparison — validation only]")
    print(scores[["config", "parameters", "best_epoch", "best_val_macro_F1"]]
          .to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in history.groupby("config", sort=False):
        ax.plot(group["epoch"], group["val_macro_F1"], label=name)
    ax.set(xlabel="Epoch", ylabel="Validation macro-F1",
           title="Galaxy10: architecture comparison (same split, one seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=160)
    plt.close(fig)
    print(f"Saved to {out}/ (test set untouched)", flush=True)


def final_test(per_class=1200):
    """Evaluate the validation-selected astroNN configuration exactly once.

    No checkpoint was saved by the original architecture study, so reproduce
    its selected epoch using the same seed, split and training configuration.
    """
    out = Path("reports/galaxy_morphology_architecture")
    scores_path = out / "validation.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Run --architecture-study first: {scores_path} missing")
    if (out / "test_metrics.csv").exists():
        raise FileExistsError(
            f"{out / 'test_metrics.csv'} already exists; do not repeatedly tune on test data"
        )

    winner = pd.read_csv(scores_path).sort_values(
        "best_val_macro_F1", ascending=False
    ).iloc[0]
    if winner["config"] != "astroNN-layout":
        raise ValueError(f"Expected selected astroNN-layout, got {winner['config']}")
    selected_epoch = int(winner["best_epoch"])
    seed = int(winner["seed"])
    expected_f1 = float(winner["best_val_macro_F1"])
    print(f"[Frozen selection] astroNN, seed={seed}, "
          f"best_epoch={selected_epoch}, val_macro_F1={expected_f1:.4f}")
    print("Previous study saved no checkpoint; retraining to selected epoch.")

    source_ids, images, y = load_balanced(DATA, per_class)
    ids = np.arange(len(y))
    trainval, test = train_test_split(
        ids, test_size=0.20, stratify=y, random_state=SEED,
    )
    train, valid = train_test_split(
        trainval, test_size=0.20, stratify=y[trainval], random_state=SEED,
    )
    print(f"Galaxy10: train={len(train)}, validation={len(valid)}, test={len(test)}")
    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    result, _, weights = run_one(
        "astroNN-layout", bool(winner["augment"]),
        float(winner["lr"]), float(winner["weight_decay"]),
        x, y, train, valid, device, epochs=selected_epoch,
        patience=selected_epoch + 1, seed=seed, model_factory=AstroNNCNN,
    )
    print(f"[Reproduced validation] best_epoch={result['best_epoch']}, "
          f"macro_F1={result['best_val_macro_F1']:.4f}")
    if (result["best_epoch"] != selected_epoch or
            abs(result["best_val_macro_F1"] - expected_f1) > 0.01):
        raise RuntimeError(
            "Retraining did not reproduce the selected epoch/validation score. "
            "Do not evaluate test yet; inspect environment and random seeds."
        )

    model = AstroNNCNN().to(device)
    model.load_state_dict(weights)  # best validation epoch, NOT last epoch 25
    test_loader = DataLoader(
        TensorDataset(x[test], torch.from_numpy(y[test])), batch_size=64,
    )
    probs = predict(model, test_loader, device)
    pred = probs.argmax(axis=1)
    metrics = {
        "selected_config": "astroNN-layout",
        "selected_epoch": selected_epoch,
        "test_accuracy": accuracy_score(y[test], pred),
        "test_macro_F1": f1_score(y[test], pred, average="macro", zero_division=0),
    }
    for cls, name in enumerate(NAMES):
        mask = y[test] == cls
        metrics[f"recall_{name.lower().replace('-', '_')}"] = (pred[mask] == cls).mean()

    # Write artifacts only AFTER the reproduction check and the single evaluation.
    torch.save({"state_dict": weights, "seed": seed,
                "epoch": selected_epoch, "validation_macro_f1": expected_f1},
               out / "astronn_best.pt")
    pd.DataFrame([metrics]).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame({
        "source_index": source_ids[test], "true": y[test],
        "predicted": pred, "confidence": probs.max(axis=1),
    }).to_csv(out / "test_predictions.csv", index=False)
    save_plots(images[test], y[test], pred, out)

    print(f"\n[Final test — previously inspected test split, n={len(test)}]")
    for key, value in metrics.items():
        printed = value if isinstance(value, str) or key == "selected_epoch" else f"{value:.4f}"
        print(f"  {key}: {printed}")
    print(f"Saved checkpoint, test metrics, predictions and plots to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--seed-study", action="store_true",
                       help="Compare plain/augmented for every seed; validation only")
    modes.add_argument("--architecture-study", action="store_true",
                       help="Compare SmallCNN with astroNN layout; one seed, validation only")
    modes.add_argument("--final-test", action="store_true",
                       help="Retrain selected astroNN epoch and evaluate test once")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.final_test:
        final_test(args.per_class)
    elif args.architecture_study:
        architecture_study(args.per_class, args.epochs)
    elif args.seed_study:
        seed_study(args.per_class, args.epochs, args.seeds)
    else:
        main(args.per_class, args.epochs, args.patience)
