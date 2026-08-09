"""Draw digits or write words in the browser; networks trained in Anvil read them as you draw.

    bin/draw                 (or: python3 examples/draw/server.py [--port 8765] [--no-browser])
    bin/draw --text          the page for handwritten English (text_reader.py)

The first time, it trains the network (examples/digits.anvil, about a minute) and saves it to
examples/digits.weights. Then it serves the drawing page on http://127.0.0.1:8765 and opens it.
The text page needs examples/letters.weights, from examples/letters.anvil (EMNIST, about two
minutes); --text trains it first if EMNIST is in examples/data/emnist.

The page sends the strokes you draw (lists of points). Here they are drawn again the way MNIST's
digits look: scaled to fit a 20×20 box, with MNIST's stroke width whatever size you drew at, and
centered by their center of mass in a 28×28 image. The network itself runs as native code compiled
by Anvil (anvil.function), loading the weights that the Anvil program saved.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXAMPLES = os.path.join(ROOT, "examples")
WEIGHTS = os.path.join(EXAMPLES, "digits.weights")
MODEL = os.path.join(EXAMPLES, "digits_model.anvil")
LETTERS_WEIGHTS = os.path.join(EXAMPLES, "letters.weights")
EMNIST = os.path.join(EXAMPLES, "data", "emnist", "emnist-byclass-train-images-idx3-ubyte.gz")
sys.path.insert(0, ROOT)

STROKE_RADIUS = 1.3        # in pixels of the 28×28 image (MNIST strokes are 2–3 pixels wide)
SUPERSAMPLE = 4            # strokes are drawn at 4× and averaged down, for smooth edges


# ----------------------------------------------------------------------------- strokes -> MNIST
def render(strokes, box: float = 20.0, radius: float = STROKE_RADIUS, by_mass: bool = True) -> np.ndarray | None:
    """Strokes (lists of [x, y] points, in any units) as a 28×28 image in [0, 1], like MNIST: the
    drawing scaled to fit box×box pixels, then moved so its center of mass is at the center
    (by_mass=False: its bounding box centered, as EMNIST does)."""
    pts = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return None
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    size = max(hi - lo)
    scale = box / size if size > 0 else 1.0
    n = 28 * SUPERSAMPLE
    # pixel centers of the supersampled image, in units of the 28×28 image
    c = (np.arange(n) + 0.5) / SUPERSAMPLE
    gx, gy = np.meshgrid(c, c)
    ink = np.zeros((n, n), dtype=bool)
    offset = 14 - (lo + hi) / 2 * scale                    # the box's center at the image's center
    for p in pts:
        q = p * scale + offset
        if len(q) > 2:                                     # a mouse gives far more points than 28 pixels need:
            along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])     # keep one
            keep = np.unique(np.append(np.searchsorted(along, np.arange(0, along[-1], 0.5)), len(q) - 1))
            q = q[keep]                                    # every half pixel along the stroke
        a = q[:-1] if len(q) > 1 else q
        b = q[1:] if len(q) > 1 else q
        for (ax, ay), (bx, by) in zip(a, b):               # distance from each pixel to the segment
            # only the pixels near the segment
            x0, x1 = max(int((min(ax, bx) - radius) * SUPERSAMPLE), 0), min(int((max(ax, bx) + radius) * SUPERSAMPLE) + 2, n)
            y0, y1 = max(int((min(ay, by) - radius) * SUPERSAMPLE), 0), min(int((max(ay, by) + radius) * SUPERSAMPLE) + 2, n)
            if x0 >= x1 or y0 >= y1:
                continue
            px, py = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / L, 0, 1) if L > 0 else 0.0
            d2 = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2
            ink[y0:y1, x0:x1] |= d2 <= radius ** 2
    img = ink.reshape(28, SUPERSAMPLE, 28, SUPERSAMPLE).mean(axis=(1, 3))
    return center(img) if by_mass else img


def center(img: np.ndarray) -> np.ndarray:
    """Move the image (whole pixels) so that its center of mass is at the middle, as MNIST does."""
    total = img.sum()
    if total == 0:
        return img
    ys, xs = np.indices(img.shape)
    cy, cx = (ys * img).sum() / total, (xs * img).sum() / total
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    out = np.zeros_like(img)
    src = img[max(0, -dy):28 - max(0, dy), max(0, -dx):28 - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


# ----------------------------------------------------------------------------- the network
class Reader:
    """The network from digits_model.anvil, compiled by Anvil, with the weights digits.anvil saved."""

    def __init__(self, weights: str = WEIGHTS):
        import anvil
        source = (f'use "{MODEL}"\n'
                  f"net = DigitNet()\n"
                  f'found = load(net, "{weights}")\n'
                  f"fn read(image: [28, 28]):\n"
                  f"    p = softmax(net(image.reshape(1, 1, 28, 28)))\n"
                  f"    return p[0], found\n")
        self.fn = anvil.function(source, name="read")
        probs, found = self.fn(np.zeros((28, 28), np.float32))       # compiles, and checks the file
        if found != 1:
            raise RuntimeError(f"cannot load {weights}: run `bin/anvil run examples/digits.anvil` to make it")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        probs, _ = self.fn(img.astype(np.float32))
        return probs


def train(program="digits.anvil", weights=WEIGHTS, how="about a minute"):
    print(f"training the network first (examples/{program}, {how} on an idle machine)…", flush=True)
    r = subprocess.run([os.path.join(ROOT, "bin", "anvil"), "run", os.path.join(EXAMPLES, program)], cwd=EXAMPLES)
    if r.returncode != 0 or not os.path.exists(weights):
        sys.exit("training failed")


def text_reader():
    """The handwriting reader for the text page, or why there isn't one."""
    if not os.path.exists(LETTERS_WEIGHTS):
        how = ("run `bin/draw --text` (it trains the letter network on EMNIST first, about two minutes)"
               if os.path.exists(EMNIST) else
               "it needs the letter network: download EMNIST into examples/data/emnist "
               "(see examples/letters.anvil), then run `bin/draw --text`")
        return None, how
    from text_reader import TextReader
    return TextReader(LETTERS_WEIGHTS), None


def text_json(result, ms):
    """What the text page shows: the text, and each word's characters as the network saw them."""
    from text_reader import CLASSES
    lines = []
    for words in result["lines"]:
        line = []
        for w in words:
            chars = []
            for c in w.chars:
                top = np.argsort(c.probs)[::-1][:3]
                chars.append({"box": [round(float(v), 1) for v in c.box], "text": c.text,
                              "top": [[CLASSES[k], round(float(c.probs[k]), 4)] for k in top],
                              "image": [round(float(v), 2) for v in c.image.reshape(-1)]})
            line.append({"text": w.text, "raw": w.raw, "marks": w.marks, "chars": chars})
        lines.append(line)
    return {"text": result["text"], "lines": lines, "ms": round(ms, 1)}


# ----------------------------------------------------------------------------- the web page
class Handler(http.server.BaseHTTPRequestHandler):
    reader: Reader = None
    text: object = None                                      # the TextReader, made when first asked for
    text_missing: str = None
    lock = threading.Lock()                                  # one request at a time uses the networks

    def log_message(self, fmt, *args):
        pass

    def send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pages = {"/": "index.html", "/index.html": "index.html", "/text": "text.html", "/text.html": "text.html"}
        if self.path in pages:
            with open(os.path.join(HERE, pages[self.path]), "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/read", "/read_text"):
            self.send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            strokes = json.loads(self.rfile.read(min(n, 16 << 20)))["strokes"]
            t = time.perf_counter()
            with self.lock:
                out = self.read_text(strokes, t) if self.path == "/read_text" else self.read_digit(strokes, t)
            self.send(200, json.dumps(out).encode(), "application/json")
        except Exception as e:                               # a bad request: say so, keep serving
            self.send(400, json.dumps({"error": str(e)}).encode(), "application/json")


    def read_digit(self, strokes, t):
        img = render(strokes)
        if img is None:
            return {"empty": True}
        probs = self.reader(img)
        return {"probs": [round(float(p), 5) for p in probs], "digit": int(np.argmax(probs)),
                "image": [round(float(v), 3) for v in img.reshape(-1)],
                "ms": round((time.perf_counter() - t) * 1000, 2)}

    def read_text(self, strokes, t):
        if Handler.text is None and Handler.text_missing is None:
            Handler.text, Handler.text_missing = text_reader()
        if Handler.text is None:
            return {"missing": Handler.text_missing}
        if not any(len(s) for s in strokes):
            return {"empty": True}
        result = Handler.text.read(strokes)
        return text_json(result, (time.perf_counter() - t) * 1000)


def main():
    ap = argparse.ArgumentParser(description="Draw digits or write words; networks trained in Anvil read them.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--text", action="store_true", help="open the page for handwritten English")
    args = ap.parse_args()
    if not os.path.exists(WEIGHTS):
        train()
    if args.text and not os.path.exists(LETTERS_WEIGHTS) and os.path.exists(EMNIST):
        train("letters.anvil", LETTERS_WEIGHTS, "about two minutes")
    print("compiling the networks…", flush=True)
    Handler.reader = Reader()
    if args.text or os.path.exists(LETTERS_WEIGHTS):
        Handler.text, Handler.text_missing = text_reader()
    server = None
    for port in range(args.port, args.port + 20):          # a busy port (another bin/draw?): the next one
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit(f"ports {args.port}–{args.port + 19} are all in use: try `bin/draw --port 9000`")
    if port != args.port:
        print(f"(port {args.port} is in use, so this one uses {port})")
    url = f"http://127.0.0.1:{port}/" + ("text" if args.text else "")
    print(f"draw a digit at http://127.0.0.1:{port}/, write words at http://127.0.0.1:{port}/text"
          f"   (ctrl-C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
