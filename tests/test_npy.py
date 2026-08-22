"""NumPy files: `npy("w.npy")` reads arrays saved by NumPy or PyTorch; `save_npy` writes them."""
import os

import numpy as np
import pytest

from util import check_native, compile_text, run_native, run_text

from anvil.diagnostics import AnvilError

ARRAYS = {
    "f32": np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32),
    "f64": np.arange(6, dtype=np.float64).reshape(2, 3) / 8,
    "f16": np.array([-2.5, 0.125, 1000], dtype=np.float16),
    "i64": np.array([3, -1, 7, 2 ** 20], dtype=np.int64),
    "i32": np.array([[1, -2], [3, 4]], dtype=np.int32),
    "u8": np.array([0, 200, 255], dtype=np.uint8),
    "i8": np.array([-5, 100], dtype=np.int8),
    "i16": np.array([-300, 300], dtype=np.int16),
    "u16": np.array([65535], dtype=np.uint16),
    "bool": np.array([True, False, True]),
}


def write_all(d):
    for name, a in ARRAYS.items():
        np.save(os.path.join(d, f"{name}.npy"), a)
    lines = [f"{n} = npy(\"{n}.npy\")" for n in ARRAYS] + [f"print({n})" for n in ARRAYS]
    return "\n".join(lines) + "\n"


def test_every_element_type(tmp_path):
    src = write_all(tmp_path)
    path = str(tmp_path / "prog.anvil")
    out = check_native(src, path=path)                 # native == interpreter
    c = compile_text(src, path=path)
    for name, a in ARRAYS.items():
        b = c.elab.globals.vars[name].val.buf
        assert b.shape == a.shape
        assert b.dtype == ("f32" if a.dtype.kind == "f" else "i32")
    assert "1000.0000" in out and "65535" in out and "-300" in out


def test_round_trip_with_numpy(tmp_path):
    w = np.random.default_rng(1).standard_normal((5, 3)).astype(np.float32)
    np.save(tmp_path / "w.npy", w)
    src = 'w = npy("w.npy")\ny = w @ ones(3, 2) * 2\nsave_npy(y, "y.npy")\nsave_npy(i32(sum(w > 0)), "n.npy")\n'
    out, err, code, _ = run_native(src, path=str(tmp_path / "p.anvil"))
    assert code == 0, err
    np.testing.assert_allclose(np.load(tmp_path / "y.npy"), w @ np.ones((3, 2)) * 2, rtol=1e-6)
    n = np.load(tmp_path / "n.npy")
    assert n.dtype == np.int32 and n.shape == () and n == (w > 0).sum()
    os.remove(tmp_path / "y.npy")
    run_text(src, path=str(tmp_path / "p.anvil"))       # the interpreter writes the same file
    np.testing.assert_allclose(np.load(tmp_path / "y.npy"), w @ np.ones((3, 2)) * 2, rtol=1e-6)


def test_bad_files(tmp_path):
    np.save(tmp_path / "be.npy", np.arange(3, dtype=">f4"))
    np.save(tmp_path / "c.npy", np.zeros(2, dtype=np.complex64))
    np.save(tmp_path / "f.npy", np.asfortranarray(np.zeros((2, 3), np.float32)))
    (tmp_path / "x.npy").write_bytes(b"not numpy")
    for name, msg in [("be", "cannot read"), ("c", "cannot read"), ("f", "Fortran"), ("x", "not a .npy"),
                      ("missing", "not found")]:
        with pytest.raises(AnvilError, match=msg):
            compile_text(f'a = npy("{name}.npy")\nprint(a)\n', path=str(tmp_path / "p.anvil"))
