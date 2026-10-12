"""Behavioural cloning: teach a ResNet18 to predict the steering a human chose.

    python train/train_bc.py data/20261012_141500 data/20261012_150000
    python train/train_bc.py data/<session> --limit 200 --epochs 2    # quick smoke test

Each frame is cropped (top 35% removed), resized to 160x120 and fed to a
pretrained ResNet18 whose last layer predicts one number: the steering.
Outputs go to model_log/bc/: best.pt, loss_curves.png, metrics.json.
"""
import argparse
import csv
import json
import os
import random

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIZE = (160, 120)   # (width, height) the network sees
SHIFT_X, SHIFT_Y = 8, 6  # max random shift in pixels (augmentation)


def preprocess(bgr, crop):
    """Camera image (BGR, 240x320) -> RGB uint8 120x160. car/drive_model.py does exactly the same."""
    img = bgr[int(bgr.shape[0] * crop):]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_session(session, crop, limit):
    """Read labels.csv + images of one session, in time order. Returns (images, steering)."""
    rows = []
    with open(os.path.join(session, "labels.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["timestamp"]), r["filename"], float(r["steering"])))
    rows.sort()
    if limit:
        rows = rows[:limit]
    images, steering = [], []
    for _, name, s in rows:
        bgr = cv2.imread(os.path.join(session, name))
        if bgr is None:  # file missing/corrupt: skip it
            continue
        images.append(preprocess(bgr, crop))
        steering.append(s)
    return np.array(images, dtype=np.uint8), np.array(steering, dtype=np.float32)


def split_by_time_blocks(n):
    """Train/val split for ONE session. Frames 0.1 s apart look almost identical, so a random
    split would leak. Instead, cut the session into contiguous blocks (~10 s) and give every
    5th block to validation. Returns a boolean array: True = validation."""
    block = max(5, min(100, n // 10))
    block_id = np.arange(n) // block
    is_val = (block_id % 5) == 4
    if not is_val.any():          # very short session: use the last block
        is_val = block_id == block_id.max()
    return is_val


class Frames(Dataset):
    def __init__(self, images, steering, train, flip):
        self.images, self.steering, self.train, self.flip = images, steering, train, flip
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        y = float(self.steering[i])
        if self.train:
            # brightness / contrast
            b, c = random.uniform(0.7, 1.3), random.uniform(0.7, 1.3)
            m = x.mean()
            x = ((x - m) * c + m) * b
            x = x.clamp(0, 1)
            # small random shift (pad by repeating edge pixels, then crop back)
            dx, dy = random.randint(-SHIFT_X, SHIFT_X), random.randint(-SHIFT_Y, SHIFT_Y)
            x = F.pad(x[None], (SHIFT_X, SHIFT_X, SHIFT_Y, SHIFT_Y), mode="replicate")[0]
            x = x[:, SHIFT_Y + dy:SHIFT_Y + dy + SIZE[1], SHIFT_X + dx:SHIFT_X + dx + SIZE[0]]
            # mirror image = mirror steering
            if self.flip and random.random() < 0.5:
                x, y = x.flip(-1), -y
        x = (x - self.mean) / self.std
        return x, torch.tensor([y], dtype=torch.float32)


def make_model(pretrained):
    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
    except AttributeError:  # old torchvision without the weights API
        model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)  # one output: steering
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over loader. Trains if an optimizer is given. Returns (mean loss, mean abs error)."""
    model.train(optimizer is not None)
    total_loss, total_abs, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            total_abs += (pred - y).abs().sum().item()
            n += len(x)
    return total_loss / n, total_abs / n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sessions", nargs="+", help="one or more session folders")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--crop", type=float, default=0.35, help="fraction of the image top to remove")
    p.add_argument("--flip", action="store_true", help="augment with horizontal flips (steering negated)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (skips the ImageNet download)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N frames of each session (smoke test)")
    p.add_argument("--out", default=os.path.join(ROOT, "model_log", "bc"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---- load and split ----
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for s in args.sessions:
        imgs, steer = load_session(s, args.crop, args.limit)
        is_val = split_by_time_blocks(len(imgs))
        print("%s: %d frames (%d train / %d val)" % (s, len(imgs), (~is_val).sum(), is_val.sum()))
        tr_x.append(imgs[~is_val]); tr_y.append(steer[~is_val])
        va_x.append(imgs[is_val]); va_y.append(steer[is_val])
    tr_x, tr_y = np.concatenate(tr_x), np.concatenate(tr_y)
    va_x, va_y = np.concatenate(va_x), np.concatenate(va_y)
    if len(va_x) == 0 or len(tr_x) < 2:
        raise SystemExit("Not enough frames to make a train/val split.")

    baseline = float(np.abs(va_y - tr_y.mean()).mean())
    print("baseline val MAE (always predict the train mean %+.3f): %.4f" % (tr_y.mean(), baseline))

    bs = min(args.batch_size, len(tr_x))
    train_loader = DataLoader(Frames(tr_x, tr_y, True, args.flip), batch_size=bs, shuffle=True,
                              drop_last=len(tr_x) > bs)  # BatchNorm needs more than 1 sample
    val_loader = DataLoader(Frames(va_x, va_y, False, False), batch_size=bs)

    model = make_model(not args.no_pretrained).to(device)
    loss_fn = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- train ----
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        tl, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        vl, vmae = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        history["train_loss"].append(tl); history["val_loss"].append(vl); history["val_mae"].append(vmae)
        flag = ""
        if vmae < best_mae:
            best_mae = vmae
            flag = "  <- best, saved"
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "crop": args.crop, "size": SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "val_mae": vmae, "epoch": epoch,
            }, os.path.join(args.out, "best.pt"),
                _use_new_zipfile_serialization=False)  # old PyTorch on the Jetson can read this
        print("epoch %2d/%d  train loss %.4f  val loss %.4f  val MAE %.4f%s" % (
            epoch, args.epochs, tl, vl, vmae, flag))

    # ---- save curves + metrics ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(history["train_loss"], label="train")
    a1.plot(history["val_loss"], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel(args.loss.upper() + " loss"); a1.legend()
    a2.plot(history["val_mae"], label="val MAE")
    a2.axhline(baseline, color="gray", linestyle="--", label="predict-the-mean baseline")
    a2.set_xlabel("epoch"); a2.set_ylabel("val MAE (steering units)"); a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curves.png"), dpi=120)
    metrics = {"best_val_mae": best_mae, "baseline_val_mae": baseline, "n_train": int(len(tr_x)),
               "n_val": int(len(va_x)), "args": vars(args), "history": history}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("best val MAE %.4f (baseline %.4f). Saved to %s" % (best_mae, baseline, args.out))


if __name__ == "__main__":
    main()
