"""Классификация одной картинки готовой моделью из лабораторной 1 или 2."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torchvision import transforms

from common import get_device
from lab1.classifier import CifarCNN, MEAN, STD

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    # На Windows стандартная консоль часто не умеет печатать русские буквы.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Распознать картинку моделью CIFAR-10")
    parser.add_argument("image", help="путь к картинке: jpg/png и т.п.")
    parser.add_argument("--lab", type=int, choices=(1, 2), default=1, help="какую обученную модель использовать")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = root / f"lab{args.lab}" / "results" / "model.pt"
    if not model_path.exists():
        raise SystemExit(f"Не найден файл модели: {model_path}")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Не найден файл картинки: {image_path}")

    device = get_device()
    model = CifarCNN(len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    prep = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    image = Image.open(image_path).convert("RGB")
    x = prep(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = model(x).softmax(1)[0]
    values, indices = probs.topk(3)
    print(f"Модель: лабораторная {args.lab} ({model_path})")
    print(f"Картинка: {image_path}")
    print("Три наиболее вероятных класса:")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"  {CLASSES[index]:12s} {value:.1%}")
    print("\nВажно: это CIFAR-10 (самолёт, машина, птица, кошка, олень, собака, лягушка, лошадь, корабль, грузовик),")
    print("модель рассчитана на маленькие изображения 32×32 и может ошибаться на произвольных фото.")


if __name__ == "__main__":
    main()
