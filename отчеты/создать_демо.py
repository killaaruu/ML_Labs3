"""Создаёт демонстрационные MP4 и скриншоты реального инференса CIFAR-10."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import DATA, get_device, make_amp, seed_everything  # noqa: E402
from lab1.classifier import BASE_TRANSFORM, CifarCNN  # noqa: E402

OUT = ROOT / "отчеты" / "медиа"
WIDTH, HEIGHT, FPS = 1280, 720, 8
BG = "#0b1220"
PANEL = "#141e30"
MUTED = "#a8b4c8"
WHITE = "#f4f7fb"
BLUE = "#60a5fa"
GREEN = "#34d399"
RED = "#fb7185"
YELLOW = "#fbbf24"
CLASS_RU = {
    "airplane": "самолёт", "automobile": "автомобиль", "bird": "птица",
    "cat": "кошка", "deer": "олень", "dog": "собака", "frog": "лягушка",
    "horse": "лошадь", "ship": "корабль", "truck": "грузовик",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


FONTS = {size: font(size, size in (30, 42, 54)) for size in (18, 20, 22, 26, 30, 36, 42, 54)}


def text(draw: ImageDraw.ImageDraw, xy, value: str, size: int = 22, fill=WHITE, bold=False):
    f = FONTS[size] if size in FONTS and (not bold or size in (30, 42, 54)) else font(size, bold)
    draw.text(xy, value, font=f, fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box, fill=PANEL, outline=None, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def blank_frame(lab_title: str, model_name: str, step_title: str, sample_i: int, total: int):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    text(d, (54, 35), "СИИ · ДЕМОНСТРАЦИЯ КЛАССИФИКАЦИИ", 20, BLUE, True)
    text(d, (54, 72), lab_title, 42, WHITE, True)
    text(d, (56, 128), f"CIFAR-10  ·  {model_name}  ·  RTX 3060 Laptop GPU / CUDA", 20, MUTED)
    rounded(d, (54, 174, 1226, 223), fill="#18263b")
    text(d, (76, 187), step_title, 22, YELLOW, True)
    text(d, (1090, 189), f"Кадр {sample_i}/{total}", 18, MUTED)
    rounded(d, (54, 246, 548, 660), fill=PANEL)
    rounded(d, (578, 246, 1226, 660), fill=PANEL)
    # Индикатор прогресса видео.
    d.rounded_rectangle((54, 680, 1226, 691), radius=5, fill="#263247")
    progress = max(0, min(1, (sample_i - 1) / max(total, 1)))
    d.rounded_rectangle((54, 680, 54 + int(1172 * progress), 691), radius=5, fill=BLUE)
    text(d, (54, 698), "Тестовые PNG читаются с диска; предсказание вычисляется сохранённой моделью.", 18, MUTED)
    return frame, d


def draw_image(frame: Image.Image, image: Image.Image | None):
    d = ImageDraw.Draw(frame)
    if image is None:
        rounded(d, (116, 281, 486, 651), fill="#0b1220", outline="#34445e")
        text(d, (192, 412), "ожидание PNG…", 22, MUTED)
    else:
        # nearest-neighbor намеренно показывает исходное разрешение CIFAR-10.
        large = image.resize((360, 360), Image.Resampling.NEAREST)
        frame.paste(large, (121, 286))
        d = ImageDraw.Draw(frame)
        rounded(d, (116, 281, 486, 651), fill=None, outline="#34445e", radius=12)
    return frame


def result_panel(frame: Image.Image, *, path: Path, image: Image.Image | None,
                 stage: str, device_name: str, true_name: str | None = None,
                 pred_name: str | None = None, confidence: float | None = None,
                 probs: list[tuple[str, float]] | None = None):
    d = ImageDraw.Draw(frame)
    x, y = 610, 276
    text(d, (x, y), "ФАЙЛ ИЗОБРАЖЕНИЯ", 18, MUTED, True)
    text(d, (x, y + 31), f"data/cifar10/test/{path.parent.name}/{path.name}", 18, WHITE)
    text(d, (x, y + 63), "ПОДГОТОВКА", 18, MUTED, True)
    text(d, (x, y + 91), "PNG → RGB → тензор [1, 3, 32, 32]", 20, WHITE)
    text(d, (x, y + 127), "УСТРОЙСТВО", 18, MUTED, True)
    text(d, (x, y + 156), device_name, 20, GREEN)
    if stage != "result":
        messages = {
            "load": ("Читаем PNG с диска", "Пиксели ещё не переданы модели."),
            "prepare": ("Изображение загружено", "Нормализуем каналы и формируем тензор."),
            "infer": ("Выполняется предсказание", "CNN рассчитывает вероятности 10 классов."),
        }
        title, subtitle = messages[stage]
        rounded(d, (606, 525, 1196, 632), fill="#1b2c43", outline="#315175")
        text(d, (632, 545), title, 26, BLUE, True)
        text(d, (632, 589), subtitle, 18, MUTED)
        return frame

    assert pred_name is not None and true_name is not None and confidence is not None
    correct = pred_name == true_name
    rounded(d, (606, 556, 1196, 650), fill="#102d2b" if correct else "#35212b",
            outline=GREEN if correct else RED)
    text(d, (632, 563), "ПРЕДСКАЗАНО", 18, MUTED, True)
    text(d, (632, 585), f"{CLASS_RU[pred_name].upper()}  ·  {confidence:.1%}", 30,
         GREEN if correct else RED, True)
    text(d, (632, 624), f"Правильный класс: {CLASS_RU[true_name]}", 18, WHITE)
    text(d, (1040, 624), "верно ✓" if correct else "ошибка", 18,
         GREEN if correct else RED, True)
    # Top-3 bars show the model's actual softmax scores.
    if probs:
        text(d, (610, 462), "Наиболее вероятные классы", 18, MUTED)
        for j, (name, probability) in enumerate(probs[:3]):
            yy = 482 + j * 21
            d.rounded_rectangle((848, yy + 3, 1098, yy + 13), radius=5, fill="#2a374b")
            d.rounded_rectangle((848, yy + 3, 848 + int(250 * probability), yy + 13), radius=5, fill=BLUE)
            text(d, (610, yy - 2), f"{CLASS_RU[name]}  {probability:.1%}", 18, WHITE)
    return frame


def append(writer, frame: Image.Image, copies: int = 1):
    arr = np.asarray(frame.convert("RGB"))
    for _ in range(copies):
        writer.append_data(arr)


def make_intro(title: str, subtitle: str):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    text(d, (76, 94), "ЛАБОРАТОРНАЯ РАБОТА · CIFAR-10", 22, BLUE, True)
    text(d, (76, 176), title, 54, WHITE, True)
    text(d, (80, 266), subtitle, 26, MUTED)
    rounded(d, (80, 366, 1200, 534), fill=PANEL, outline="#315175")
    text(d, (118, 399), "1  Загрузка PNG с диска", 26, WHITE)
    text(d, (118, 444), "2  Подготовка изображения", 26, WHITE)
    text(d, (118, 489), "3  Предсказание CNN на CUDA", 26, WHITE)
    text(d, (80, 595), "Далее показаны реальные тестовые изображения и ответы обученной модели.", 22, MUTED)
    return frame


def make_outro(title: str):
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    text(d, (80, 105), "ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА", 22, BLUE, True)
    text(d, (80, 186), title, 48, WHITE, True)
    text(d, (82, 285), "Показаны результаты инференса сохранённой модели", 28, MUTED)
    text(d, (82, 344), "на тестовых изображениях CIFAR-10, прочитанных с диска.", 28, MUTED)
    return frame


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна; согласно требованиям проекта, демо остановлено.")
    seed_everything(2026)
    device = get_device()
    device_name = torch.cuda.get_device_name(0)
    base = datasets.ImageFolder(DATA / "cifar10" / "test")
    classes = base.classes
    rng = random.Random(2026)
    paths, labels = [], []
    for class_index, class_name in enumerate(classes):
        candidates = [i for i, (_, label) in enumerate(base.samples) if label == class_index]
        selected = rng.choice(candidates)
        paths.append(Path(base.samples[selected][0]))
        labels.append(class_name)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "скриншоты").mkdir(exist_ok=True)
    autocast, _ = make_amp(device)
    summary = {}

    for lab, title, subtitle, checkpoint in (
        (1, "Классификация без аугментации", "Исходная CNN · 15 эпох · 88,77% точности на тесте",
         ROOT / "lab1" / "results" / "model.pt"),
        (2, "Классификация с аугментацией", "Та же CNN с аугментациями · 15 эпох · 89,50% точности на тесте",
         ROOT / "lab2" / "results" / "model.pt"),
    ):
        model = CifarCNN(len(classes)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        video_path = OUT / f"лаба{lab}_загрузка-и-распознавание.mp4"
        records = []
        screenshot_dir = OUT / "скриншоты" / f"лаба{lab}"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        chosen_screens = {0, 4, 9}
        with imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=16) as writer:
            intro = make_intro(title, subtitle)
            append(writer, intro, FPS * 2)
            for i, (path, true_name) in enumerate(zip(paths, labels), start=1):
                # Сначала кадр «читаем файл», затем выполняется фактическое чтение PNG с диска.
                frame, _ = blank_frame(title, f"Лаба {lab}", "ШАГ 1/3 · ЗАГРУЗКА С ДИСКА", i, len(paths))
                frame = draw_image(frame, None)
                result_panel(frame, path=path, image=None, stage="load", device_name=device_name)
                append(writer, frame, 6)
                if i == 1:
                    frame.save(screenshot_dir / "01_чтение_файла.png")

                image = Image.open(path).convert("RGB")
                frame, _ = blank_frame(title, f"Лаба {lab}", "ШАГ 2/3 · ПОДГОТОВКА ТЕНЗОРА", i, len(paths))
                frame = draw_image(frame, image)
                result_panel(frame, path=path, image=image, stage="prepare", device_name=device_name)
                append(writer, frame, 6)
                if i == 1:
                    frame.save(screenshot_dir / "02_изображение_загружено.png")

                tensor = BASE_TRANSFORM(image).unsqueeze(0).to(device)
                frame, _ = blank_frame(title, f"Лаба {lab}", "ШАГ 3/3 · ПРЕДСКАЗАНИЕ НА CUDA", i, len(paths))
                frame = draw_image(frame, image)
                result_panel(frame, path=path, image=image, stage="infer", device_name=device_name)
                append(writer, frame, 6)

                with torch.inference_mode(), autocast():
                    logits = model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
                order = sorted(range(len(classes)), key=lambda k: probabilities[k], reverse=True)
                pred_name = classes[order[0]]
                confidence = probabilities[order[0]]
                top = [(classes[k], probabilities[k]) for k in order[:3]]
                frame, _ = blank_frame(title, f"Лаба {lab}", "ГОТОВО · РЕЗУЛЬТАТ МОДЕЛИ", i, len(paths))
                frame = draw_image(frame, image)
                result_panel(frame, path=path, image=image, stage="result", device_name=device_name,
                             true_name=true_name, pred_name=pred_name, confidence=confidence, probs=top)
                append(writer, frame, FPS * 2)
                if i == 1:
                    frame.save(screenshot_dir / "03_предсказание.png")
                if i in chosen_screens:
                    frame.save(screenshot_dir / f"04_результат_{i:02d}_{true_name}.png")
                record = {"file": str(path.relative_to(ROOT)), "true": true_name,
                          "predicted": pred_name, "confidence": confidence,
                          "top3": [{"class": name, "probability": prob} for name, prob in top]}
                records.append(record)
                print(f"Лаба {lab}: {path.name} -> {pred_name} ({confidence:.1%}), true={true_name}")
            append(writer, make_outro(title), FPS * 2)

        summary[f"lab{lab}"] = {"model": str(checkpoint.relative_to(ROOT)), "video": str(video_path.relative_to(ROOT)),
                                "device": device_name, "fps": FPS, "width": WIDTH, "height": HEIGHT,
                                "samples": records}
        print(f"Видео сохранено: {video_path}")

    (OUT / "результаты_инференса.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Скриншоты и протокол сохранены в {OUT}")


if __name__ == "__main__":
    main()
