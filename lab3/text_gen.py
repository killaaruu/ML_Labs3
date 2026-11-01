"""Лабораторная работа 3: генерация текста.

Посимвольная языковая модель-трансформер (мини-GPT) на PyTorch.
Датасет по умолчанию — Tiny Shakespeare (~1 МБ текста), можно указать свой .txt:
    python lab3/text_gen.py                         # обучение + генерация
    python lab3/text_gen.py --data мой_текст.txt    # другой корпус
    python lab3/text_gen.py --generate --prompt "ROMEO:" --temperature 0.8
"""
import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, get_device, make_amp, seed_everything

OUT = Path(__file__).resolve().parent / "results"
DEFAULT_DATA = DATA / "text" / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ------------------------------------------------------------------ данные

class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def __len__(self):
        return len(self.chars)


def load_text(path: Path) -> str:
    if not path.exists():
        if path != DEFAULT_DATA:
            raise SystemExit(f"Файл не найден: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю датасет в {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------------ модель

class Block(nn.Module):
    """Блок трансформера: маскированное самовнимание + MLP, pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, d_model=256, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, idx, targets=None):
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self.mask[:t, :t]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ------------------------------------------------------------------ обучение

@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(args, device):
    text = load_text(Path(args.data))
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"Символов в корпусе: {len(text):,}, размер словаря: {len(tok)}")

    model = CharGPT(len(tok), args.block_size, args.d_model, args.layers, args.heads).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    warmup = 100

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    autocast, scaler = make_amp(device, not args.no_amp)
    history = {"step": [], "train": [], "val": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args.block_size, args.batch_size, device)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"шаг {step:5d}: train {losses['train']:.3f}, val {losses['val']:.3f}  ({time.time() - t0:.0f} c)")
        if step == args.steps:
            break
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        with autocast():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # перед обрезкой градиентов вернуть их масштаб
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    torch.save(
        {"model": model.state_dict(), "chars": tok.chars, "config": vars(args)},
        OUT / "model.pt",
    )
    plot_history(history, OUT / "loss.png")
    return model, tok


def load_model(device):
    ckpt = torch.load(OUT / "model.pt", map_location=device)
    cfg = ckpt["config"]
    tok = CharTokenizer(ckpt["chars"])
    model = CharGPT(len(tok), cfg["block_size"], cfg["d_model"], cfg["layers"], cfg["heads"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tok


def generate_text(model, tok, prompt, length, temperature, top_k, device):
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, length, temperature, top_k)
    return tok.decode(out[0].tolist())


def plot_history(history, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["step"], history["train"], label="train")
    ax.plot(history["step"], history["val"], label="val")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Cross-entropy (нат/символ)")
    ax.set_title("Обучение посимвольной языковой модели")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--generate", action="store_true", help="только генерация обученной моделью")
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--length", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="выключить смешанную точность на CUDA")
    args = p.parse_args()

    seed_everything()
    device = get_device()
    OUT.mkdir(exist_ok=True)
    print(f"Устройство: {device}")

    if args.generate:
        model, tok = load_model(device)
        print(generate_text(model, tok, args.prompt, args.length, args.temperature, args.top_k, device))
        return

    model, tok = train(args, device)

    # Примеры генерации при разных температурах
    report = []
    for temp in (0.5, 0.8, 1.0, 1.3):
        text = generate_text(model, tok, args.prompt, args.length, temp, None, device)
        report.append(f"===== температура {temp} =====\n{text}\n")
    (OUT / "samples.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + report[1])
    print(f"Результаты сохранены в {OUT}")


if __name__ == "__main__":
    main()