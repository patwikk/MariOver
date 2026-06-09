import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Split a Mario Maker dataset into train/val/test sets.")
    parser.add_argument("--json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--val_pct", type=float, default=0.1)
    parser.add_argument("--test_pct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if abs(args.train_pct + args.val_pct + args.test_pct - 1.0) > 1e-6:
        raise ValueError("Train/val/test percentages must sum to 1.0")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(args.train_pct * n)
    n_val = int(args.val_pct * n)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    base, ext = os.path.splitext(args.json)
    splits = {
        f"{base}-train{ext}": train_data,
        f"{base}-validate{ext}": val_data,
        f"{base}-test{ext}": test_data,
    }

    for path, split in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"Saved {len(split)} samples -> {path}")


if __name__ == "__main__":
    main()
