"""bin/draw: strokes become MNIST-like images, and a network trained in Anvil reads them."""
import json
import os
import subprocess
import sys
import threading
import urllib.request

import numpy as np
import pytest

from util import ROOT

sys.path.insert(0, os.path.join(ROOT, "examples", "draw"))
import sample_strokes  # noqa: E402
import server  # noqa: E402


def test_render_is_like_mnist():
    img = server.render(sample_strokes.DIGITS[7])
    assert img.shape == (28, 28) and img.min() >= 0 and img.max() <= 1
    ys, xs = np.nonzero(img > 0.2)
    assert max(ys.max() - ys.min(), xs.max() - xs.min()) <= 23          # fits the 20×20 box (plus the pen)
    total = img.sum()
    cy = (np.indices(img.shape)[0] * img).sum() / total
    cx = (np.indices(img.shape)[1] * img).sum() / total
    assert abs(cy - 13.5) <= 0.5 and abs(cx - 13.5) <= 0.5               # centered by mass
    assert server.render([]) is None and server.render([[]]) is None


def test_size_and_position_do_not_matter():
    """A digit drawn small in a corner, or large in the middle, gives the same image."""
    s = sample_strokes.DIGITS[4]
    small = [[[x * 0.3 + 5, y * 0.3 + 2] for x, y in stroke] for stroke in s]
    large = [[[x * 4 + 100, y * 4 + 60] for x, y in stroke] for stroke in s]
    assert np.abs(server.render(small) - server.render(large)).max() < 1e-9
    dot = server.render([[[3, 3]]])                                      # a single click: a dot
    assert dot.sum() > 0


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    """The network, trained for one epoch (~10 s) into a temporary file."""
    weights = str(tmp_path_factory.mktemp("draw") / "digits.weights")
    r = subprocess.run([os.path.join(ROOT, "bin", "anvil"), "run", os.path.join(ROOT, "examples", "digits.anvil"),
                        "--set", "EPOCHS=1", "--set", f"SAVE={weights}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "test accuracy" in r.stdout and os.path.exists(weights)
    acc = float(r.stdout.split("test accuracy")[1].split("%")[0])
    assert acc > 97.0, r.stdout
    return server.Reader(weights)


def test_reads_drawn_digits(reader):
    """Mouse-style drawings of every digit, in 20 wobbly, stretched, slanted, small and large
    versions each."""
    right = 0
    for d, strokes in sample_strokes.DIGITS.items():
        for v in sample_strokes.variants(strokes, 20, d):
            right += int(np.argmax(reader(server.render(v))) == d)
    assert right >= 180, f"{right} of 200"


def test_missing_weights_are_reported(tmp_path):
    with pytest.raises(RuntimeError, match="cannot load"):
        server.Reader(str(tmp_path / "nothing.weights"))


def test_http(reader):
    server.Handler.reader = reader
    httpd = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<canvas" in page and "/read" in page

        def read(body: bytes):
            req = urllib.request.Request(base + "/read", data=body, headers={"Content-Type": "application/json"})
            try:
                return json.load(urllib.request.urlopen(req))
            except urllib.error.HTTPError as e:
                return json.load(e)
        out = read(json.dumps({"strokes": sample_strokes.DIGITS[0]}).encode())
        assert out["digit"] == 0 and len(out["probs"]) == 10 and len(out["image"]) == 784
        assert abs(sum(out["probs"]) - 1) < 1e-3
        assert read(b'{"strokes": []}') == {"empty": True}
        assert "error" in read(b"not json")
    finally:
        httpd.shutdown()
