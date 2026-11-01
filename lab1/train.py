"""Лабораторная работа 1: классификация изображений CIFAR-10 сверточной сетью (PyTorch).

Датасет загружается с диска (data/cifar10, формат ImageFolder).
Запуск:  python lab1/train.py --epochs 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common import get_device, seed_everything
from lab1.classifier import (
    BASE_TRANSFORM,
    CifarCNN,
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

    # Без аугментации: только перевод в тензор и нормализация
    train_loader, test_loader, classes = make_loaders(BASE_TRANSFORM, args.batch_size, args.workers)
    model = CifarCNN(len(classes))
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, train_loader, test_loader, device, args.epochs, args.lr, amp=not args.no_amp)
    torch.save(model.state_dict(), OUT / "model.pt")

    _, test_acc, preds, targets = evaluate(model, test_loader, device)
    per_class = per_class_accuracy(preds, targets, classes)
    save_json({"test_acc": test_acc, "per_class_acc": per_class, "history": history}, OUT / "metrics.json")

    plot_history(history, OUT / "history.png", "Лаб. 1 — без аугментации")
    plot_predictions(model, test_loader.dataset, classes, device, OUT / "predictions.png")
    plot_confusion(preds, targets, classes, OUT / "confusion_matrix.png")

    print(f"\nИтоговая точность на тесте: {test_acc:.4f}")
    for c, a in per_class.items():
        print(f"  {c:12s} {a:.3f}")
    print(f"Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()