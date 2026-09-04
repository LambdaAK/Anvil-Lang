"""%%anvil cells in Jupyter: function cells define Python functions, other cells run as programs."""
import io

import numpy as np
import pytest

from util import ROOT

from anvil.jupyter import run_cell


def test_function_cells_define_python_functions():
    ns = {}
    out = io.StringIO()
    run_cell("", "fn rows(x: [n, d]) = softmax(x)\nfn double(x) = 2 * x\n", ns, out)
    assert out.getvalue() == "defined rows, double\n"
    np.testing.assert_allclose(ns["double"](np.arange(3.0)), [0, 2, 4])
    np.testing.assert_allclose(ns["rows"](np.zeros((2, 4))), np.full((2, 4), 0.25))
    run_cell("--set K=3", "const K = 1\nfn scale(x) = x * K\n", ns, out)
    np.testing.assert_allclose(ns["scale"](np.ones(2)), [3, 3])


def test_program_cells_run():
    for opts in ("", "--interp"):
        out = io.StringIO()
        run_cell(opts, 'x = [1.0, 2.0, 3.0]\nfor i in range(2):\n    print("step {i}: {sum(x * i)}")\n', {}, out)
        assert out.getvalue() == "step 0: 0.0000\nstep 1: 6.0000\n", (opts, out.getvalue())
    out = io.StringIO()
    run_cell("--check", "x = [1.0, -1.0]\ny = log(x)\nprint(y)\n", {}, out)
    assert "nan in" in out.getvalue() and "made by `log`" in out.getvalue()
    out = io.StringIO()
    run_cell("", "y = [1.0, 2.0] @ [[1.0], [2.0], [3.0]]\n", {}, out)
    assert "cannot multiply [2] by [3, 1]" in out.getvalue()


def test_a_real_notebook():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(f"import sys; sys.path.insert(0, {ROOT!r})\n%load_ext anvil"),
        nbformat.v4.new_code_cell("%%anvil\nfn double(x) = 2 * x"),
        nbformat.v4.new_code_cell("import numpy as np\nprint(double(np.arange(3.0)))"),
        nbformat.v4.new_code_cell('%%anvil\nprint("hello from Anvil {sum([1.0, 2.0])}")'),
    ]
    try:
        nbclient.NotebookClient(nb, timeout=120, kernel_name="python3").execute()
    except Exception as e:                        # no kernel installed: not Anvil's problem
        if "kernel" in str(e).lower() and "No such" in str(e):
            pytest.skip(str(e))
        raise
    text = lambda c: "".join(o.get("text", "") for o in c.outputs)
    assert text(nb.cells[2]).strip() == "[0. 2. 4.]"
    assert "hello from Anvil 3.0000" in text(nb.cells[3])
