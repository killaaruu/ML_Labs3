"""Лабораторная работа 2: аугментация данных для задачи из лабораторной 1.

Та же модель и тот же режим обучения, что и в лаб. 1, но обучающая выборка
проходит через случайные преобразования. Скрипт сравнивает результат с лаб. 1.
Запуск:  python lab2/train.py --epochs 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from common import get_device, seed_everything
from lab1.classifier import (
    CIFAR_DIR,
    MEAN,
    STD,
    CifarCNN,
    denormalize,
    evaluate,
    make_loaders,
    per_class_accuracy,
    plot_confusion,
    plot_history,
    plot_predictions,
    save_json,
    train,
)

OUT = Path(__file__).resolve().parent / "results"
LAB1_METRICS = Path(__file__).resolve().parents[1] / "lab1" / "results" / "metrics.json"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # сдвиги
    transforms.RandomHorizontalFlip(),                              # зеркальное отражение
    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.5),  # цвет/яркость
    transforms.RandomRotation(10),                                  # небольшие повороты
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),           # вырезание фрагмента (Cutout)
])


def plot_augmentations(path: Path, n_images: int = 6, n_aug: int = 7) -> None:
    """Каждая строка: оригинал и несколько случайных аугментаций того же изображения."""
    raw = datasets.ImageFolder(CIFAR_DIR / "train")
    g = torch.Generator().manual_seed(1)
    idx = torch.randperm(len(raw), generator=g)[:n_images]
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    fig, axes = plt.subplots(n_images, n_aug + 1, figsize=((n_aug + 1) * 1.4, n_images * 1.5))
    for row, i in enumerate(idx):
        img, label = raw[i]
        axes[row, 0].imshow(denormalize(to_tensor(img)))
        axes[row, 0].set_title(f"оригинал\n{raw.classes[label]}", fontsize=8)
        for col in range(1, n_aug + 1):
            axes[row, col].imshow(denormalize(TRAIN_TRANSFORM(img)))
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Примеры аугментации")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(base: dict, aug: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for hist, name in ((base, "без аугментации"), (aug, "с аугментацией")):
        e = range(1, len(hist["test_acc"]) + 1)
        ax1.plot(e, hist["train_acc"], "--", label=f"train, {name}")
        ax1.plot(e, hist["test_acc"], label=f"test, {name}")
        ax2.plot(e, hist["train_loss"], "--", label=f"train, {name}")
        ax2.plot(e, hist["test_loss"], label=f"test, {name}")
    ax1.set_title("Точность")
    ax2.set_title("Функция потерь")
    for ax in (ax1, ax2):
        ax.set_xlabel("Эпоха")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2, help="процессов загрузки данных (0 — без них)")
    parser.add_argument("--no-amp", action="store_true", help="выключить смешанную точность на CUDA")
    args = parser.parse_args()

    seed_everything()
    device = get_device()
    OUT.mkdir(exist_ok=True)
    print(f"Устройство: {device}")

    plot_augmentations(OUT / "augmentations.png")

    train_loader, test_loader, classes = make_loaders(TRAIN_TRANSFORM, args.batch_size, args.workers)
    model = CifarCNN(len(classes))
    history = train(model, train_loader, test_loader, device, args.epochs, args.lr, amp=not args.no_amp)
    torch.save(model.state_dict(), OUT / "model.pt")

    _, test_acc, preds, targets = evaluate(model, test_loader, device)
    per_class = per_class_accuracy(preds, targets, classes)
    save_json({"test_acc": test_acc, "per_class_acc": per_class, "history": history}, OUT / "metrics.json")

    plot_history(history, OUT / "history.png", "Лаб. 2 — с аугментацией")
    plot_predictions(model, test_loader.dataset, classes, device, OUT / "predictions.png")
    plot_confusion(preds, targets, classes, OUT / "confusion_matrix.png")

    print(f"\nИтоговая точность на тесте: {test_acc:.4f}")
    if LAB1_METRICS.exists():
        base = json.loads(LAB1_METRICS.read_text(encoding="utf-8"))
        plot_comparison(base["history"], history, OUT / "comparison.png")
        print(f"Лаб. 1 (без аугментации): {base['test_acc']:.4f}  ->  прирост {test_acc - base['test_acc']:+.4f}")
        print(f"{'класс':12s} {'без аугм.':>10s} {'с аугм.':>10s}")
        for c in classes:
            print(f"{c:12s} {base['per_class_acc'][c]:10.3f} {per_class[c]:10.3f}")
    else:
        print("Для сравнения сначала запустите lab1/train.py")
    print(f"Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()