"""Скачивает CIFAR-10 (зеркало на Hugging Face, parquet) и раскладывает его на диск PNG-файлами:

    data/cifar10/train/<класс>/<id>.png
    data/cifar10/test/<класс>/<id>.png

После этого лабораторные 1 и 2 читают датасет с диска через ImageFolder.
"""
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "cifar10"
URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def export(split: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    parquet = RAW / f"cifar10_{split}.parquet"
    if not parquet.exists():
        print(f"Скачиваю {split}...")
        urllib.request.urlretrieve(URL.format(split=split), parquet)
    table = pq.read_table(parquet).to_pydict()
    for cls in CLASSES:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(tqdm(zip(table["img"], table["label"]), total=len(table["label"]), desc=split)):
        # изображения в parquet уже закодированы в PNG — пишем байты как есть
        (OUT / split / CLASSES[label] / f"{i:05d}.png").write_bytes(img["bytes"])


if __name__ == "__main__":
    if (OUT / "test").exists():
        print(f"Уже подготовлено: {OUT}")
    else:
        export("train")
        export("test")
        print(f"Готово: {OUT}")