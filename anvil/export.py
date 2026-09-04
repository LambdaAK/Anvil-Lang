"""`anvil export file.anvil`: an Anvil function as a C library and header, for use outside Python.

    anvil export examples/digits_reader.anvil -o build/      →  build/digits_reader.h, build/libdigits_reader.dylib

The function's arguments must have concrete shapes (`fn read(image: [28, 28])`). Everything the
function reads from the rest of the program (a model, its trained weights from `load`, constants)
is computed once, when exporting, and built into the library as constant data: the library needs
no weights file and no Python, and a call computes only the function. From C:

    #include "digits_reader.h"
    float image[784], probs[10];
    digits_reader_read(image, probs);

Link with `cc app.c -Lbuild -ldigits_reader -Wl,-rpath,build`. Arrays are row-major. Calls from
several threads are safe (they take turns: the compiled code works in static buffers).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from . import ast as A
from .diagnostics import AnvilError, fmt_shape
from .ir import I32
from .parser import parse
from .source import SourceFile


def c_type(dtype) -> str:
    return "int32_t" if dtype == I32 else "float"


def c_ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if s and not s[0].isdigit() else "_" + s


def signature(path: str, text: str, fn: str | None, consts: dict | None):
    """The function's name and its arguments' (dtype, shape), from the annotations."""
    from .driver import load_prelude
    from .elaborate import Elaborator
    src = SourceFile(path, text)
    tree = parse(src)
    fns = [s for s in tree.body if isinstance(s, A.FnDecl)]
    if not fns:
        raise AnvilError("there is no function to export (`fn name(args) = …`)", None)
    decl = fns[-1] if fn is None else next((d for d in fns if d.name.id == fn), None)
    if decl is None:
        raise AnvilError(f"there is no function `{fn}`", None, help="its functions: " + ", ".join(d.name.id for d in fns))
    elab = Elaborator(source=src, consts=consts)
    elab.run(tree, load_prelude())
    sig = []
    for p in decl.params:
        if p.type is None:
            raise AnvilError(f"give `{p.name}` a type, e.g. `{p.name}: [28, 28]`: an exported function's arguments "
                           f"need their shapes", p.span)
        try:
            dtype, shape = elab.eval_type(p.type, elab.globals)
        except AnvilError:
            raise AnvilError(f"every dimension of `{p.name}` needs a size when exporting (numbers or constants, "
                           f"not shape variables)", p.span)
        sig.append(("tensor", shape, dtype))
    return decl.name.id, tuple(sig)


def export(path: str, out_dir: str | None = None, fn: str | None = None, name: str | None = None,
           consts: dict | None = None, seed: int = 0) -> list[str]:
    """Write <name>.h and lib<name>.dylib; returns their paths."""
    from .backend.aarch64 import generate
    from .backend.toolchain import find_cc
    from .pyapi import lower
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.endswith("\n"):
        text += "\n"
    fn_name, sig = signature(path, text, fn, consts)
    lowered = lower(text, path, fn_name, sig, seed=seed, consts=consts)
    if lowered.consts:
        raise AnvilError(f"`{fn_name}` returns a value known at compile time; an exported function returns tensors", None)
    lib = c_ident(name or os.path.splitext(os.path.basename(path))[0])
    func = f"{lib}_{c_ident(fn_name)}"
    out_dir = os.path.abspath(out_dir or os.path.dirname(path))
    os.makedirs(out_dir, exist_ok=True)
    asm, gen = generate(lowered.prog, seed=seed, library=True)
    used = {b.id for b in gen.used.values()}

    args, sizes, sets, doc_in, doc_out = [], [], [], [], []
    for b, an in zip(lowered.inputs, lowered.arg_names):
        arg = c_ident(an)
        args.append(f"const {c_type(b.dtype)} *{arg}")
        sizes.append(f"#define {func.upper()}_{arg.upper()}_SIZE {max(1, b.numel)}   /* {b.dtype}{fmt_shape(b.shape)} */")
        doc_in.append(f"{an}: {b.dtype}{fmt_shape(b.shape)}")
        if b.id in used:
            sets.append(f"    anvil_{b.name} = (void *){arg};")
    outs = [b for b in lowered.outputs if b is not None]
    for b, on in zip(outs, lowered.out_names):
        args.append(f"{c_type(b.dtype)} *{on}")
        sizes.append(f"#define {func.upper()}_{on.upper()}_SIZE {max(1, b.numel)}   /* {b.dtype}{fmt_shape(b.shape)} */")
        doc_out.append(f"{b.dtype}{fmt_shape(b.shape)}")
        sets.append(f"    anvil_{b.name} = (void *){on};")
    proto = f"void {func}({', '.join(args)})"
    rel = os.path.relpath(path)
    header = f"""// {lib}.h — the Anvil function `{fn_name}` from {os.path.basename(path)}, compiled by `anvil export`.
//
//     {fn_name}({', '.join(doc_in)}) -> {', '.join(doc_out)}
//
// Link with lib{lib}.dylib (cc app.c -L<dir> -l{lib} -Wl,-rpath,<dir>). Arrays are row-major and
// are read and written in place; the library holds everything else the function needs (its
// weights included). Calls from several threads take turns.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {{
#endif

{chr(10).join(sizes)}

{proto};

#ifdef __cplusplus
}}
#endif
"""
    slots = sorted({f"extern void *anvil_{b.name};" for b in list(lowered.inputs) + outs if b.id in used})
    wrapper = f"""// The C entry point of {lib}: points the compiled code at the caller's arrays, then runs it.
#include <pthread.h>
#include "{lib}.h"
{chr(10).join(slots)}
extern int anvil_entry(void);
static pthread_mutex_t anvil_lock = PTHREAD_MUTEX_INITIALIZER;

{proto} {{
    pthread_mutex_lock(&anvil_lock);
{chr(10).join(sets)}
    anvil_entry();
    pthread_mutex_unlock(&anvil_lock);
}}
"""
    h_path = os.path.join(out_dir, f"{lib}.h")
    lib_path = os.path.join(out_dir, f"lib{lib}.dylib")
    with open(h_path, "w") as f:
        f.write(header)
    with tempfile.TemporaryDirectory() as d:
        s_path, c_path = os.path.join(d, f"{lib}.s"), os.path.join(d, f"{lib}_api.c")
        with open(s_path, "w") as f:
            f.write(asm)
        with open(c_path, "w") as f:
            f.write(wrapper)
        cc = find_cc()
        obj = os.path.join(d, "fn.o")
        r = subprocess.run([cc, "-arch", "arm64", "-c", "-x", "assembler", s_path, "-o", obj], capture_output=True, text=True)
        if r.returncode == 0:
            r = subprocess.run([cc, "-arch", "arm64", "-O2", "-shared", f"-I{out_dir}", c_path, obj, "-o", lib_path,
                                "-lz", "-framework", "Accelerate", "-install_name", f"@rpath/lib{lib}.dylib"],
                               capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"anvil export: building the library failed:\n{r.stderr}")
    return [h_path, lib_path]
