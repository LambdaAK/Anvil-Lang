"""Write a sentence by hand: the diffusion model of handwriting.anvil draws every character, they are
set on a line the way handwriting is (small letters small, g/j/p/q/y below the line, one slant and
one style for the whole sentence), and the letter network reads the result back. A character the
letter network cannot make out (about one in three) is drawn again, at most three times.

    python3 examples/handwriting/write.py "Hello world"
    python3 examples/handwriting/write.py "The quick brown fox jumps over the lazy dog" --seed 7
    bin/handwrite "..."                        (the same)

Options: --seed N (another hand), --style S (0 … 1: how much the characters share their noise, so
how alike they look; default 0.5), --guidance G (how plainly each character is the one asked for;
default 2), --cpu (draw on the CPU instead of the GPU), --open (open the picture).

Letters, digits, spaces and . , ! ? ' - : are written (EMNIST has no punctuation: those are drawn
with a pen of the same width). The picture goes to examples/data/handwriting/handwriting.png, and
handwriting.gif shows the characters being drawn out of noise.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
ROOT = os.path.dirname(EXAMPLES)
DATA = os.path.join(EXAMPLES, "data", "handwriting")
CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PUNCT = ".,!?'-:"
N = 64                              # characters drawn at a time (draw.anvil is compiled for this many)
T = 200                             # noise levels (as in handwriting_model.anvil)

H = 64                              # the height of a capital, in pixels
TALL = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt")
DESCENDERS = set("gpqy")
PAPER = np.array([250, 248, 241], np.float32)
INK = np.array([22, 32, 78], np.float32)


def size_of(ch: str):
    """(height, how far below the line it reaches), in units of a capital's height."""
    if ch in TALL:
        return 1.0, 0.0
    if ch == "i":
        return 0.82, 0.0
    if ch == "j":
        return 1.08, 0.3
    if ch in DESCENDERS:
        return 0.9, 0.32
    return 0.6, 0.0


def draw(chars: list[str], rng, style: float, guidance: float, metal: bool, fresh: float = 0.0):
    """The model's drawings of the characters ([n, 28, 28] ink in 0 … 1) and its snapshots of
    them being made ([FRAMES, n, 28, 28])."""
    os.makedirs(DATA, exist_ok=True)
    shared = rng.standard_normal((T + 1, 1, 1, 28, 28)).astype(np.float32)
    drawn, frames = [], []
    for i in range(0, len(chars), N):
        part = chars[i:i + N]
        labels = np.zeros(N, np.int32)
        labels[:len(part)] = [CLASSES.index(c) for c in part]
        own = rng.standard_normal((T + 1, N, 1, 28, 28)).astype(np.float32)
        noise = np.sqrt(style) * shared + np.sqrt(1 - style) * own
        np.save(os.path.join(DATA, "labels.npy"), labels)
        np.save(os.path.join(DATA, "noise.npy"), noise.astype(np.float32))
        np.save(os.path.join(DATA, "guidance.npy"), np.array([guidance, fresh], np.float32))
        cmd = [os.path.join(ROOT, "bin", "anvil"), "run"] + (["--metal"] if metal else []) + [os.path.join(HERE, "draw.anvil")]
        if subprocess.run(cmd).returncode != 0:
            sys.exit("drawing failed")
        d = np.load(os.path.join(DATA, "drawn.npy"))[:len(part), 0]
        f = np.load(os.path.join(DATA, "frames.npy"))[:, :len(part), 0]
        drawn.append(np.clip((d + 1) / 2, 0, 1))
        frames.append(np.clip((f + 1) / 2, 0, 1))
    return np.concatenate(drawn), np.concatenate(frames, axis=1)


def clean(a: np.ndarray, threshold=0.25) -> np.ndarray:
    """A drawing without stray specks: the connected parts of its ink smaller than 4% of all of it are
    removed (the dot of an i or a j is larger than that)."""
    on = a > threshold
    seen = np.zeros_like(on)
    parts = []
    for y, x in zip(*np.nonzero(on)):
        if seen[y, x]:
            continue
        stack, part = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            part.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < on.shape[0] and 0 <= nx < on.shape[1] and on[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        parts.append(part)
    total = sum(len(p) for p in parts)
    keep = np.zeros_like(on)
    for part in parts:
        if len(part) >= max(3, 0.04 * total):
            for y, x in part:
                keep[y, x] = True
    grown = keep.copy()                                        # and the soft edge around what is kept
    grown[1:] |= keep[:-1]
    grown[:-1] |= keep[1:]
    grown[:, 1:] |= keep[:, :-1]
    grown[:, :-1] |= keep[:, 1:]
    return a * grown


def crop_box(a: np.ndarray, threshold=0.25):
    rows, cols = np.where(a > threshold)
    if len(rows) == 0:
        return 10, 18, 10, 18
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def layout(text: str, aspects: list, rng, width: int = 1400):
    """Where each character goes, as lines of {ch, x, dy, h, depth, w}: its height from its kind
    (size_of), its width from the drawing's own shape (aspects: width / height of each drawn
    character, in order), with a little wobble. Words wrap at `width`."""
    lines, line, x, k = [], [], 0.0, 0
    for word in text.split(" "):
        placed = []
        for c in word:
            if c in PUNCT:
                placed.append({"ch": c, "h": 0.0, "depth": 0.0, "w": 0.18 * H})
                continue
            h, depth = size_of(c)
            jitter = 1 + 0.04 * rng.standard_normal()
            placed.append({"ch": c, "h": h * H * jitter, "depth": depth * H * jitter,
                           "w": h * H * jitter * aspects[k]})
            k += 1
        need = sum(p["w"] for p in placed) + 0.1 * H * len(placed)
        if line and x + need > width:
            lines.append(line)
            line, x = [], 0.0
        for p in placed:
            p["x"] = x
            p["dy"] = 0.025 * H * rng.standard_normal()
            line.append(p)
            x += p["w"] + 0.09 * H + 0.025 * H * rng.standard_normal()
        x += 0.45 * H
    if line:
        lines.append(line)
    return lines


def resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize((max(w, 1), max(h, 1)), Image.LANCZOS), np.float32) / 255


def pen(canvas: np.ndarray, ch: str, x: float, base: float, width: float):
    """Punctuation EMNIST does not have, drawn with a pen as wide as the strokes of the letters."""
    from PIL import Image, ImageDraw
    img = Image.fromarray((canvas * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    r, w = width * 0.6, int(round(width))
    dot = lambda cx, cy: d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    if ch in ".!?:":
        dot(x + r, base - r)
    if ch == ":":
        dot(x + r, base - 0.45 * H)
    if ch == ",":
        d.line([(x + r, base - r), (x - 0.04 * H, base + 0.18 * H)], fill=255, width=w)
    if ch == "!":
        d.line([(x + r, base - H), (x + r, base - 0.3 * H)], fill=255, width=w)
    if ch == "'":
        d.line([(x + r, base - H), (x + r - 0.03 * H, base - 0.75 * H)], fill=255, width=w)
    if ch == "-":
        d.line([(x - 0.05 * H, base - 0.32 * H), (x + 0.25 * H, base - 0.32 * H)], fill=255, width=w)
    if ch == "?":
        d.arc([x - 0.05 * H, base - H, x + 0.35 * H, base - 0.55 * H], 180, 90, fill=255, width=w)
        d.line([(x + 0.15 * H, base - 0.55 * H), (x + 0.15 * H, base - 0.3 * H)], fill=255, width=w)
        dot(x + 0.15 * H, base - r)
    return np.asarray(img, np.float32) / 255


def compose(lines, glyphs: dict, slant: float, pen_width: float) -> np.ndarray:
    """The page: ink in 0 … 1. glyphs[i]: the cropped drawing of the i-th character (in reading order)."""
    line_h = int(1.9 * H)
    width = int(max(p["x"] for line in lines for p in line) + 2 * H)
    page = np.zeros((line_h * len(lines) + H // 2, width), np.float32)
    k = 0
    for row, line in enumerate(lines):
        base = row * line_h + int(1.25 * H)
        for p in line:
            if p["ch"] in PUNCT:
                page = np.maximum(page, pen(page, p["ch"], p["x"] + H * 0.3, base + p["dy"], pen_width))
                continue
            g = glyphs[k]
            k += 1
            h = int(round(p["h"]))
            w = int(round(p["w"]))
            img = resize(g, w, h)
            y1 = int(round(base + p["depth"] + p["dy"]))
            x0 = int(round(p["x"] + H * 0.3))
            y0 = max(y1 - h, 0)
            img = img[img.shape[0] - (y1 - y0):]
            page[y0:y1, x0:x0 + img.shape[1]] = np.maximum(page[y0:y1, x0:x0 + img.shape[1]], img[:, :page.shape[1] - x0])
    page = np.clip((page - 0.18) / 0.42, 0, 1)                   # crisper strokes: drawn at 28×28,
    page = page * page * (3 - 2 * page)                          # they are blurred when scaled up
    if slant:                                                    # one slant for the whole page
        from PIL import Image
        img = Image.fromarray((page * 255).astype(np.uint8))
        img = img.transform(img.size, Image.AFFINE, (1, slant, -slant * page.shape[0] / 2, 0, 1, 0), Image.BICUBIC)
        page = np.asarray(img, np.float32) / 255
    return page


def colored(page: np.ndarray):
    from PIL import Image
    rgb = PAPER[None, None] * (1 - page[..., None]) + INK[None, None] * page[..., None]
    return Image.fromarray(rgb.astype(np.uint8))


def terminal(page: np.ndarray, cols: int):
    """The page in quarter blocks, `cols` characters wide."""
    blocks = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"
    h = max(2, int(page.shape[0] * 2 * cols / page.shape[1] / 2.2) // 2 * 2)
    small = resize(page, 2 * cols, h) > 0.3
    for r in range(0, h, 2):
        print("  " + "".join(blocks[small[r, 2 * c] + 2 * small[r, 2 * c + 1] + 4 * small[r + 1, 2 * c]
                                    + 8 * small[r + 1, 2 * c + 1]] for c in range(cols)).rstrip())


LOOKALIKE = ["0Oo", "1Il", "5Ss", "2Zz", "9gq", "6b"]       # drawn alike in EMNIST


def letter_reader():
    sys.path.insert(0, os.path.join(EXAMPLES, "draw"))
    from text_reader import TextReader
    return TextReader()


def legible(reader, ch: str, d: np.ndarray) -> bool:
    """Whether the letter network sees the character asked for (in either case, or a character drawn
    just like it) with some confidence, and the drawing is not a field of noise."""
    y0, y1, x0, x1 = crop_box(d)
    if (d[y0:y1, x0:x1] > 0.25).mean() > 0.6 and (y1 - y0) * (x1 - x0) > 300:
        return False
    p = reader.fn(d.astype(np.float32))[0]
    same = {ch, ch.lower(), ch.upper()} | {c for group in LOOKALIKE if ch in group for c in group}
    return sum(float(p[CLASSES.index(c)]) for c in same) >= 0.3


def read_back(reader, drawn: np.ndarray, text: str) -> str:
    """The letter network (letters.anvil) reads every drawing; the text reader's dictionary turns
    each word's probabilities into a word."""
    probs = [reader.fn(d.astype(np.float32))[0] for d in drawn]

    class Read:
        def __init__(self, p):
            self.probs, self.full, self.small = p, 0.0, False

    out, k, start = [], 0, True
    for word in text.split(" "):
        letters = [c for c in word if c not in PUNCT]
        if letters:
            _, read, _ = reader.decide([Read(probs[k + i]) for i in range(len(letters))], start)
            k += len(letters)
            it = iter(read)
            word = "".join(next(it) if c not in PUNCT else c for c in word)
        out.append(word)
        start = word.endswith(".")
    return " ".join(out)


def main():
    args, opts, k = [], {}, 1
    while k < len(sys.argv):
        a = sys.argv[k]
        if a in ("--seed", "--style", "--guidance"):
            opts[a[2:]] = sys.argv[k + 1]
            k += 2
        elif a in ("--cpu", "--open"):
            opts[a[2:]] = True
            k += 1
        else:
            args.append(a)
            k += 1
    text = " ".join(" ".join(args).split())
    if not text:
        sys.exit(__doc__)
    unknown = sorted({c for c in text if c not in CLASSES + PUNCT + " "})
    if unknown:
        print(f"(left out: {' '.join(unknown)})")
        text = "".join(c for c in text if c not in unknown)
    seed = int(opts.get("seed", np.random.SeedSequence().entropy % 100000))
    rng = np.random.default_rng(seed)
    chars = [c for c in text if c in CLASSES]

    style, guidance = float(opts.get("style", 0.5)), float(opts.get("guidance", 2.0))
    metal = not opts.get("cpu") and sys.platform == "darwin"
    drawn, frames = draw(chars, rng, style, guidance, metal)
    reader = letter_reader()
    redrawn = 0
    for _ in range(3):                  # a character the letter network cannot make out is drawn again
        bad = [i for i, (c, d) in enumerate(zip(chars, drawn)) if not legible(reader, c, d)]
        if not bad:
            break
        again, again_frames = draw([chars[i] for i in bad], rng, style, guidance, metal)
        for j, i in enumerate(bad):
            drawn[i], frames[:, i] = again[j], again_frames[:, j]
        redrawn += len(bad)
    drawn = np.stack([clean(d) for d in drawn])
    boxes = [crop_box(d) for d in drawn]
    glyphs = [d[y0:y1, x0:x1] for d, (y0, y1, x0, x1) in zip(drawn, boxes)]
    pen_width = max(2.0, float(np.mean([d.sum() for d in drawn])) / 135 * 0.11 * H)
    lines = layout(text, [(x1 - x0) / max(y1 - y0, 1) for y0, y1, x0, x1 in boxes], rng)
    slant = float(np.clip(0.12 + 0.08 * rng.standard_normal(), -0.05, 0.3))
    page = compose(lines, glyphs, slant, pen_width)
    out_png = os.path.join(DATA, "handwriting.png")
    colored(page).save(out_png)

    pictures = []                                               # the characters being drawn, in place
    for f in range(frames.shape[0]):
        g = [fr[y0:y1, x0:x1] for fr, (y0, y1, x0, x1) in zip(frames[f], boxes)]
        pictures.append(colored(compose(lines, g, slant, pen_width)))
    pictures += [colored(page)] * 6
    out_gif = os.path.join(DATA, "handwriting.gif")
    pictures[0].save(out_gif, save_all=True, append_images=pictures[1:], duration=110, loop=0)

    print()
    terminal(page, min(shutil.get_terminal_size((100, 20)).columns - 4, 120))
    read = read_back(reader, drawn, text)
    same = sum(a.lower() == b.lower() for a, b in zip(read, text)) if len(read) == len(text) else 0
    print(f"\n  wrote:     {text}")
    print(f"  read back: {read}   ({100 * same / max(len(text), 1):.0f}% of the characters)")
    if redrawn:
        print(f"  ({redrawn} drawing{'s' if redrawn > 1 else ''} the letter network could not make out, drawn again)")
    print(f"\n  {out_png}\n  {out_gif}   (the characters being drawn)\n  seed {seed}")
    if opts.get("open"):
        subprocess.run(["open", out_png])


if __name__ == "__main__":
    main()
