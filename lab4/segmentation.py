"""Лабораторная работа 4: семантическая сегментация изображений.

Датасет — Oxford-IIIT Pet (https://paperswithcode.com/dataset/oxford-iiit-pets):
фотографии кошек и собак с попиксельной разметкой на 3 класса
(животное / фон / граница). Модель — U-Net, обучаемая с нуля.

Запуск:  python lab4/segmentation.py --epochs 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF

from common import DATA, get_device, make_amp, seed_everything

OUT = Path(__file__).resolve().parent / "results"
PET_ROOT = DATA / "pets"
CLASSES = ["животное", "фон", "граница"]
COLORS = np.array([[230, 80, 60], [40, 40, 40], [250, 220, 80]], dtype=np.uint8)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ данные

class PetSegmentation(Dataset):
    """Читает изображения и маски (trimaps) с диска; маска 1/2/3 -> классы 0/1/2."""

    def __init__(self, split: str, size: int, augment: bool):
        base = OxfordIIITPet(PET_ROOT, split=split, target_types="segmentation", download=True)
        self.images = base._images
        self.masks = base._segs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        mask = Image.open(self.masks[i])
        if self.augment:
            # случайный масштаб/кадрирование и отражение — одинаково для картинки и маски
            w, h = img.size
            scale = np.random.uniform(0.7, 1.0)
            cw, ch = int(w * scale), int(h * scale)
            left, top = np.random.randint(0, w - cw + 1), np.random.randint(0, h - ch + 1)
            img = TF.crop(img, top, left, ch, cw)
            mask = TF.crop(mask, top, left, ch, cw)
            if np.random.rand() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            img = TF.adjust_brightness(img, np.random.uniform(0.8, 1.2))
        img = TF.resize(img, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = torch.from_numpy(np.array(mask, dtype=np.int64)) - 1
        return x, y


# ------------------------------------------------------------------ модель

class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net: энкодер (4 понижения) — боттлнек — декодер с skip-соединениями."""

    def __init__(self, n_classes=3, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList()
        c_in = 3
        for c in ch[:-1]:
            self.downs.append(DoubleConv(c_in, c))
            c_in = c
        self.bottleneck = DoubleConv(ch[-2], ch[-1])
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.dec.append(DoubleConv(c * 2, c))
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def dice_loss(logits, target, eps=1.0):
    probs = logits.softmax(1)
    onehot = F.one_hot(target, logits.size(1)).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


# ------------------------------------------------------------------ метрики

@torch.no_grad()
def evaluate(model, loader, device, n_classes=3):
    model.eval()
    inter = torch.zeros(n_classes)
    union = torch.zeros(n_classes)
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for c in range(n_classes):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
    iou = (inter / union.clamp(min=1)).tolist()
    return {
        "loss": loss_sum / len(loader.dataset),
        "pixel_acc": correct / total,
        "iou": dict(zip(CLASSES, iou)),
        "miou": sum(iou) / n_classes,
    }


# ------------------------------------------------------------------ визуализация

def colorize(mask: np.ndarray) -> np.ndarray:
    return COLORS[mask]


def denorm(x):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def plot_examples(model, dataset, device, path, n=6, seed=0):
    model.eval()
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n]
    fig, axes = plt.subplots(n, 4, figsize=(11, n * 2.7))
    for row, i in enumerate(idx):
        x, y = dataset[i]
        pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        img = denorm(x)
        overlay = 0.55 * img + 0.45 * (pred == 0)[..., None] * np.array([1.0, 0.2, 0.1])
        panels = [img, colorize(y.numpy()), colorize(pred), overlay.clip(0, 1)]
        titles = ["изображение", "разметка", "предсказание", "наложение"]
        for ax, im, t in zip(axes[row], panels, titles):
            ax.imshow(im)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k] / 255) for k in range(3)]
    fig.legend(handles, CLASSES, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(history, path):
    e = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(e, history["train_loss"], label="train")
    ax1.plot(e, history["test_loss"], label="test")
    ax1.set_title("Функция потерь (CE)")
    ax1.legend()
    ax2.plot(e, history["miou"], label="mIoU")
    ax2.plot(e, history["pixel_acc"], label="pixel accuracy")
    ax2.set_title("Метрики на тестовой подвыборке")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xlabel("Эпоха")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--base", type=int, default=16, help="число каналов первого уровня U-Net")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=2, help="процессов загрузки данных (0 — без них)")
    p.add_argument("--no-amp", action="store_true", help="выключить смешанную точность на CUDA")
    args = p.parse_args()

    seed_everything()
    device = get_device()
    OUT.mkdir(exist_ok=True)
    print(f"Устройство: {device}")

    train_ds = PetSegmentation("trainval", args.size, augment=True)
    test_ds = PetSegmentation("test", args.size, augment=False)
    kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    # после каждой эпохи проверяем на каждой 7-й тестовой картинке (~520 шт.), в конце — на всём тесте
    val_loader = DataLoader(Subset(test_ds, range(0, len(test_ds), 7)), shuffle=False, **kw)
    print(f"Обучающих: {len(train_ds)}, тестовых: {len(test_ds)}")

    model = UNet(base=args.base).to(device)
    print(f"Параметров: {sum(q.numel() for q in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    autocast, scaler = make_amp(device, not args.no_amp)
    history = {"train_loss": [], "test_loss": [], "pixel_acc": [], "miou": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y) + dice_loss(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run += F.cross_entropy(logits.detach().float(), y).item() * y.size(0)
            n += y.size(0)
        m = evaluate(model, val_loader, device)
        history["train_loss"].append(run / n)
        history["test_loss"].append(m["loss"])
        history["pixel_acc"].append(m["pixel_acc"])
        history["miou"].append(m["miou"])
        print(f"Эпоха {epoch:2d}/{args.epochs}: train CE {run / n:.3f} | test CE {m['loss']:.3f}, "
              f"pixel acc {m['pixel_acc']:.3f}, mIoU {m['miou']:.3f}  ({time.time() - t0:.0f} c)")

    torch.save(model.state_dict(), OUT / "model.pt")
    final = evaluate(model, test_loader, device)
    (OUT / "metrics.json").write_text(json.dumps({**final, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(history, OUT / "history.png")
    plot_examples(model, test_ds, device, OUT / "examples.png")

    print(f"\nPixel accuracy: {final['pixel_acc']:.4f}, mIoU: {final['miou']:.4f}")
    for c, v in final["iou"].items():
        print(f"  IoU {c:9s} {v:.3f}")
    print(f"Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()