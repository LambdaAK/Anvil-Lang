"""`anvil export`: an Anvil function as a C library and header, with what it reads built in."""
import os
import subprocess

import numpy as np
import pytest

from util import ROOT  # noqa: F401

from anvil.diagnostics import AnvilError
from anvil.export import export

SRC = """
const SCALE = 2.0
param W: [3, 2] ~ normal(0, 1)
b = [0.5, -0.5]
fn layer(x: [4, 3]):
    y = relu(x @ W + b) * SCALE
    return y, sum(x)
"""

APP = r"""
#include <stdio.h>
#include "layer_lib.h"
int main(void) {
    float x[LAYER_LIB_LAYER_X_SIZE], y[LAYER_LIB_LAYER_OUT0_SIZE], s[LAYER_LIB_LAYER_OUT1_SIZE];
    for (int i = 0; i < 12; i++) x[i] = (i % 5) - 2.0f;
    layer_lib_layer(x, y, s);
    for (int i = 0; i < 8; i++) printf("%.6f ", y[i]);
    printf("%.6f\n", s[0]);
    return 0;
}
"""


def test_export_and_call_from_c(tmp_path):
    src = tmp_path / "layer.anvil"
    src.write_text(SRC)
    h, lib = export(str(src), str(tmp_path), name="layer_lib")
    header = open(h).read()
    assert "void layer_lib_layer(const float *x, float *out0, float *out1);" in header
    assert "#define LAYER_LIB_LAYER_X_SIZE 12" in header
    (tmp_path / "app.c").write_text(APP)
    exe = str(tmp_path / "app")
    r = subprocess.run(["cc", "app.c", "-I.", "-L.", "-llayer_lib", "-Wl,-rpath,.", "-o", exe],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = np.array([float(v) for v in subprocess.run([exe], capture_output=True, text=True, cwd=tmp_path).stdout.split()])
    # the same function through anvil.function (which bakes the same random W)
    import anvil
    f = anvil.function(str(src))
    x = (np.arange(12) % 5 - 2.0).reshape(4, 3).astype(np.float32)
    y, s = f(x)
    np.testing.assert_allclose(got[:8], y.reshape(-1), rtol=1e-5)
    assert got[8] == s == x.sum()


def test_export_needs_concrete_shapes(tmp_path):
    src = tmp_path / "poly.anvil"
    src.write_text("fn f(x: [n, 3]) = x * 2\n")
    with pytest.raises(AnvilError, match="needs a size"):
        export(str(src), str(tmp_path))
    src.write_text("fn f(x) = x * 2\n")
    with pytest.raises(AnvilError, match="give `x` a type"):
        export(str(src), str(tmp_path))
    src.write_text("x = 1\n")
    with pytest.raises(AnvilError, match="no function"):
        export(str(src), str(tmp_path))


def test_export_with_constants_set(tmp_path):
    src = tmp_path / "scale.anvil"
    src.write_text("const N = 3\nconst K = 2.0\nfn f(x: [N]) = x * K\n")
    h, lib = export(str(src), str(tmp_path), consts={"N": 5, "K": 10.0})
    assert "#define SCALE_F_X_SIZE 5" in open(h).read()
