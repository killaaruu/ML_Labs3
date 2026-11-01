"""Сегментировать одну картинку готовой U-Net из лабораторной 4.

Сохраняет картинку с цветной маской поверх оригинала в lab4/results.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from common import get_device
from lab4.segmentation import COLORS, MEAN, STD, UNet


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Выделить животное на фото готовой U-Net")
    parser.add_argument("image", help="путь к исходной картинке")
    parser.add_argument("--output", help="куда записать результат (по умолчанию lab4/results/my_prediction.png)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = root / "lab4" / "results" / "model.pt"
    if not model_path.exists():
        raise SystemExit(f"Не найден файл модели: {model_path}")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Не найден файл картинки: {image_path}")
    output_path = Path(args.output).expanduser().resolve() if args.output else root / "lab4" / "results" / "my_prediction.png"

    device = get_device()
    model = UNet(base=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    original = Image.open(image_path).convert("RGB")
    x = TF.resize(original, [128, 128])
    x = TF.normalize(TF.to_tensor(x), MEAN, STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        mask = model(x).argmax(1)[0].cpu().numpy()

    # Маску увеличить до размера исходника; nearest сохраняет чёткие границы классов.
    mask_img = Image.fromarray(mask.astype(np.uint8)).resize(original.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_img)
    overlay = np.asarray(original).astype(np.float32)
    color_mask = COLORS[mask].astype(np.float32)
    is_pet = mask == 0
    overlay[is_pet] = 0.55 * overlay[is_pet] + 0.45 * color_mask[is_pet]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(output_path)
    print(f"Результат сохранён: {output_path}")
    print("Красно-оранжевым отмечено предсказанное животное.")


if __name__ == "__main__":
    main()
