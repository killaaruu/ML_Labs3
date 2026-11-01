"""Модель, обучение и визуализация для классификации изображений CIFAR-10.

Используется в лабораторной 1 (без аугментации) и лабораторной 2 (с аугментацией).
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import DATA, make_amp

CIFAR_DIR = DATA / "cifar10"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

BASE_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CifarCNN(nn.Module):
    """VGG-подобная сверточная сеть: 3 блока conv-conv-pool + классификатор."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64),     # 32 -> 16
            conv_block(64, 128),   # 16 -> 8
            conv_block(128, 256),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loaders(train_transform, batch_size: int = 128, workers: int = 2):
    if not CIFAR_DIR.exists():
        raise SystemExit("Датасет не найден. Запустите: python data/prepare_cifar10.py")
    train_ds = datasets.ImageFolder(CIFAR_DIR / "train", transform=train_transform)
    test_ds = datasets.ImageFolder(CIFAR_DIR / "test", transform=BASE_TRANSFORM)
    kw = dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        train_ds.classes,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        p = out.argmax(1)
        correct += (p == y).sum().item()
        n += y.numel()
        preds.append(p.cpu())
        targets.append(y.cpu())
    return total_loss / n, correct / n, torch.cat(preds), torch.cat(targets)


def train(model, train_loader, test_loader, device, epochs: int, lr: float = 0.05, amp: bool = True):
    model.to(device)
    autocast, scaler = make_amp(device, amp)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                out = model(x)
                loss = loss_fn(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * y.numel()
            run_correct += (out.argmax(1) == y).sum().item()
            n += y.numel()

        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        history["train_loss"].append(run_loss / n)
        history["train_acc"].append(run_correct / n)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"Эпоха {epoch:2d}/{epochs}: "
            f"train loss {run_loss / n:.3f} acc {run_correct / n:.3f} | "
            f"test loss {test_loss:.3f} acc {test_acc:.3f}  ({time.time() - t0:.0f} c)"
        )
    return history


# ---------------------------------------------------------------- визуализация

def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_history(history: dict, path: Path, title: str = "") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_title("Функция потерь")
    ax1.set_xlabel("Эпоха")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_title("Точность")
    ax2.set_xlabel("Эпоха")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, dataset, classes, device, path: Path, n: int = 24, seed: int = 0) -> None:
    """Сетка случайных тестовых изображений: зеленым — верные предсказания, красным — ошибки."""
    model.eval()
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n]
    xs = torch.stack([dataset[i][0] for i in idx])
    ys = [dataset[i][1] for i in idx]
    probs = model(xs.to(device)).softmax(1).cpu()
    conf, preds = probs.max(1)

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    for ax, x, y, p, c in zip(axes.flat, xs, ys, preds, conf):
        ax.imshow(denormalize(x))
        ok = p.item() == y
        ax.set_title(
            f"{classes[p]} ({c:.0%})\nистина: {classes[y]}",
            fontsize=8,
            color="green" if ok else "red",
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(preds, targets, classes, path: Path) -> None:
    k = len(classes)
    cm = torch.zeros(k, k, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    cm_norm = cm.float() / cm.sum(1, keepdim=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=45, ha="right")
    ax.set_yticks(range(k), classes)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    for i in range(k):
        for j in range(k):
            v = cm_norm[i, j].item()
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def per_class_accuracy(preds, targets, classes) -> dict:
    return {c: (preds[targets == i] == i).float().mean().item() for i, c in enumerate(classes)}


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")