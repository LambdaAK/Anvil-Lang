"""The `anvil` command-line tool."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time

from . import __version__, ir
from .diagnostics import AnvilError, Style, use_color

USAGE = """\
anvil — a tiny language for differentiable tensor programs, compiled to ARM64 assembly or CUDA

usage:
  anvil run   <file.anvil>             compile to native code and run it
  anvil build <file.anvil> [-o exe]    build a native executable
  anvil asm   <file.anvil> [-o out.s]  write the generated assembly (use -o - for stdout)
  anvil cuda  <file.anvil> [-o out.cu] write the program as CUDA C++ (one .cu file, no other sources)
  anvil metal <file.anvil> [-o out.mm] write the program for the Apple GPU (Metal kernels + Objective-C++)
  anvil ir    <file.anvil>             show the optimized kernel IR
  anvil export <file.anvil> [-o dir]   a function (--fn NAME, else the last) as a C library and header,
                                   with what it reads from the program (trained weights) built in
  anvil cost  <file.anvil>             what the program will cost before it runs: parameters, memory,
                                   FLOPs and memory traffic, line by line
  anvil check <file.anvil>             type- and shape-check only (--json: errors and every name's
                                   shape, for editors; --stdin: read the text from stdin)
  anvil repl                         try Anvil a line at a time

options:
  --interp      run on the NumPy reference interpreter instead of native code
  --metal       run or build for the Apple GPU (Metal)
  --cuda        run or build for an NVIDIA GPU (needs nvcc)
  --cuda-emulate  compile the CUDA code as C++ and run it on the CPU, one thread at a time
  --set N=V     give the program's `const N` the value V instead (repeatable)
  --seed N      random seed (default 0)
  -O0           disable kernel fusion and other optimizations
  --profile     time every kernel and print a breakdown at exit
  --check       stop at the first nan a computation makes, and say which line made it and how
                (--check=inf: at the first infinity too)
  -v            print compile statistics

environment:
  ANVIL_THREADS   worker threads for large kernels (default: performance cores, at most 8)
"""


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def compile_program(path: str, optimize: bool, seed: int, consts: dict | None = None):
    from .driver import compile_file
    return compile_file(path, optimize=optimize, seed=seed, consts=consts)


def parse_set(item: str):
    """`NAME=VALUE` from --set: VALUE is an integer, a number, true/false, or else a string."""
    name, eq, text = item.partition("=")
    name = name.strip()
    if not eq or not name.isidentifier():
        raise ValueError(f"--set expects NAME=VALUE, got `{item}`")
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return name, text[1:-1]                    # `--set OPT='"adam"'` and `--set OPT=adam` agree
    if text in ("true", "false"):
        return name, text == "true"
    for conv in (int, float):
        try:
            return name, conv(text)
        except ValueError:
            pass
    return name, text


def memory_report(prog: ir.Program, used_ids=None, arena=None) -> dict:
    """Bytes by kind of buffer; temporaries that share the arena count once, as the arena."""
    by_kind: dict[str, int] = {}
    shared = arena.offsets if arena is not None else {}
    for b in prog.buffers:
        if b.root is not b:
            continue
        if used_ids is not None and b.id not in used_ids:
            continue
        if b.id in shared:
            continue
        kind = {"var": "temp", "grad": "temp", "scalar": "temp"}.get(b.kind, b.kind)
        by_kind[kind] = by_kind.get(kind, 0) + b.nbytes
    if arena is not None and arena.size:
        by_kind["temp"] = by_kind.get("temp", 0) + arena.size
    return by_kind


def cache_dir() -> str:
    d = os.environ.get("ANVIL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "anvil")
    os.makedirs(d, exist_ok=True)
    return d


def build_native(compiled, seed: int, exe_path: str | None, asm_path: str | None = None, profile=False,
                 check: str | None = None):
    from .backend.aarch64 import generate
    from .backend.toolchain import assemble_and_link
    asm, gen = generate(compiled.program, seed=seed, profile=profile, check=check)
    if exe_path is None:
        key = hashlib.sha256(asm.encode()).hexdigest()[:16]
        exe_path = os.path.join(cache_dir(), f"prog-{key}")
        if os.path.exists(exe_path):
            return exe_path, asm, gen, True
    with tempfile.TemporaryDirectory() as d:
        s = asm_path or os.path.join(d, "program.s")
        with open(s, "w") as f:
            f.write(asm)
        assemble_and_link(s, exe_path)
    return exe_path, asm, gen, False


def build_cuda(compiled, seed: int, exe_path: str | None, emulate: bool):
    from .backend.cuda import build, generate
    src = generate(compiled.program, seed=seed)
    if exe_path is None:
        key = hashlib.sha256((src + str(emulate)).encode()).hexdigest()[:16]
        exe_path = os.path.join(cache_dir(), f"cuda-{key}")
        if os.path.exists(exe_path):
            return exe_path
    build(src, exe_path, emulate)
    return exe_path


def print_warnings(compiled, color):
    for w in compiled.warnings:
        print(w.render(color), file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] in ("--version", "version"):
        print(f"anvil {__version__}")
        return 0
    if argv[0] == "repl":
        from .repl import main as repl_main
        return repl_main()
    argv = ["--check=nan" if a == "--check" else a for a in argv]      # (a bare --check takes no value)
    p = argparse.ArgumentParser(prog="anvil", add_help=False)
    p.add_argument("command", choices=["run", "build", "asm", "cuda", "metal", "ir", "check", "cost", "export"])
    p.add_argument("file")
    p.add_argument("-o", dest="output")
    p.add_argument("--interp", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--metal", action="store_true")
    p.add_argument("--cuda-emulate", dest="cuda_emulate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", dest="consts", action="append", default=[])
    p.add_argument("-O0", dest="noopt", action="store_true")
    p.add_argument("-v", dest="verbose", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fn")
    p.add_argument("--name")
    p.add_argument("--check", nargs="?", const="nan", choices=["nan", "inf"])
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--no-opt", dest="noopt2", action="store_true")
    try:
        args = p.parse_args(argv)
    except SystemExit:
        print(USAGE, file=sys.stderr)
        return 2
    color = use_color(sys.stderr)
    st = Style(color)
    if args.command == "check" and args.json:
        # for editors: diagnostics and the type of every name, as JSON (see anvil/ide.py)
        import json
        from .ide import analyze
        text = sys.stdin.read() if args.stdin else None
        if text is None and not os.path.exists(args.file):
            print(json.dumps({"diagnostics": [], "hovers": [], "error": f"no such file: {args.file}"}))
            return 1
        print(json.dumps(analyze(args.file, text), ensure_ascii=False))
        return 0
    if not os.path.exists(args.file):
        print(f"{st.red('error')}{st.bold(f': no such file: {args.file}')}", file=sys.stderr)
        return 1
    optimize = not (args.noopt or args.noopt2 or args.check)     # --check: one kernel per operation, one line each
    if args.command == "export":
        from .export import export
        try:
            consts = dict(parse_set(item) for item in args.consts)
            files = export(args.file, args.output, fn=args.fn, name=args.name, consts=consts, seed=args.seed)
        except AnvilError as e:
            print(e.render(color), file=sys.stderr)
            return 1
        except (RuntimeError, ValueError) as e:
            print(f"{st.red('error')}: {e}", file=sys.stderr)
            return 1
        for f in files:
            print(f"{st.green('✓')} wrote {os.path.relpath(f)}", file=sys.stderr)
        return 0
    try:
        consts = dict(parse_set(item) for item in args.consts)
    except ValueError as e:
        print(f"{st.red('error')}{st.bold(': ' + str(e))}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    try:
        compiled = compile_program(args.file, optimize, args.seed, consts)
    except AnvilError as e:
        print(e.render(color), file=sys.stderr)
        return 1
    except RecursionError:
        print(f"{st.red('error')}: program is nested too deeply", file=sys.stderr)
        return 1
    t_front = time.perf_counter() - t0
    print_warnings(compiled, color)
    prog = compiled.program

    if args.command == "check":
        nk = len(ir.all_kernels(prog))
        print(f"{st.green('✓')} {args.file}: ok ({nk} kernels)", file=sys.stderr)
        return 0

    if args.command == "ir":
        print(ir.fmt_program(prog))
        return 0

    if args.command == "cost":
        from .cost import report
        print(report(compiled, args.file))
        return 0

    if args.command == "asm":
        from .backend.aarch64 import generate
        asm, _ = generate(prog, seed=args.seed)
        out = args.output or os.path.splitext(args.file)[0] + ".s"
        if out == "-":
            sys.stdout.write(asm)
        else:
            with open(out, "w") as f:
                f.write(asm)
            print(f"{st.green('✓')} wrote {out} ({asm.count(chr(10))} lines)", file=sys.stderr)
        return 0

    if args.command == "cuda":
        from .backend.cuda import generate as generate_cuda
        src = generate_cuda(prog, seed=args.seed)
        out = args.output or os.path.splitext(args.file)[0] + ".cu"
        if out == "-":
            sys.stdout.write(src)
        else:
            with open(out, "w") as f:
                f.write(src)
            print(f"{st.green('✓')} wrote {out} ({src.count(chr(10))} lines, {len(ir.all_kernels(prog))} kernels); "
                  f"build it with: nvcc -O3 -o prog {out} -lz", file=sys.stderr)
        return 0

    if args.command == "metal":
        from .backend.metal import generate as generate_metal
        src = generate_metal(prog, seed=args.seed)
        out = args.output or os.path.splitext(args.file)[0] + ".mm"
        if out == "-":
            sys.stdout.write(src)
        else:
            with open(out, "w") as f:
                f.write(src)
            print(f"{st.green('✓')} wrote {out} ({src.count(chr(10))} lines, {len(ir.all_kernels(prog))} kernels)", file=sys.stderr)
        return 0

    if args.command in ("run", "build") and args.metal:
        from .backend.metal import MetalError, build as build_metal, generate as generate_metal
        try:
            src = generate_metal(prog, seed=args.seed)
            if args.command == "build":
                exe = args.output or os.path.splitext(args.file)[0]
            else:
                exe = os.path.join(cache_dir(), "metal-" + hashlib.sha256(src.encode()).hexdigest()[:16])
            if args.command == "build" or not os.path.exists(exe):
                build_metal(src, exe)
        except MetalError as e:
            print(f"{st.red('error')}: {e}", file=sys.stderr)
            return 1
        if args.command == "build":
            print(f"{st.green('✓')} built {exe} (Metal)", file=sys.stderr)
            return 0
        try:
            return subprocess.run([exe]).returncode
        except KeyboardInterrupt:
            return 130

    if args.command in ("run", "build") and (args.cuda or args.cuda_emulate):
        from .backend.cuda import CudaToolchainError
        exe = (args.output or os.path.splitext(args.file)[0]) if args.command == "build" else None
        try:
            exe = build_cuda(compiled, args.seed, exe, emulate=args.cuda_emulate)
        except CudaToolchainError as e:
            print(f"{st.red('error')}: {e}", file=sys.stderr)
            return 1
        if args.command == "build":
            print(f"{st.green('✓')} built {exe} (CUDA{', emulated on the CPU' if args.cuda_emulate else ''})",
                  file=sys.stderr)
            return 0
        try:
            return subprocess.run([exe]).returncode
        except KeyboardInterrupt:
            return 130

    if args.command == "run" and args.interp:
        from .interp import AnvilRuntimeError, ProgramExit, run_program
        try:
            run_program(prog, seed=args.seed, check=args.check)
        except ProgramExit as e:
            return e.code
        except AnvilRuntimeError as e:
            sys.stdout.flush()
            print(f"{st.red('runtime error')}: {e}", file=sys.stderr)
            return 1
        return 0

    from .backend.toolchain import ToolchainError
    exe = None
    if args.command == "build":
        exe = args.output or os.path.splitext(args.file)[0]
    try:
        t1 = time.perf_counter()
        exe, asm, gen, cached = build_native(compiled, args.seed, exe, profile=args.profile, check=args.check)
        t_back = time.perf_counter() - t1
    except ToolchainError as e:
        print(f"{st.red('error')}: {e}", file=sys.stderr)
        return 1
    if args.command == "build" or args.verbose:
        mem = memory_report(prog, set(gen.used), gen.arena)
        nk = len(gen.kernels)
        parts = ", ".join(f"{k} {human_bytes(v)}" for k, v in sorted(mem.items(), key=lambda kv: -kv[1]))
        if gen.arena.unshared > gen.arena.size:
            parts += f" (sharing saved {human_bytes(gen.arena.unshared - gen.arena.size)} of temporaries)"
        print(f"{st.green('✓')} {'built ' + exe if args.command == 'build' else 'compiled'}: {nk} kernels, "
              f"{asm.count(chr(10))} lines of assembly, memory: {parts or 'none'} "
              f"({t_front + t_back:.2f}s)", file=sys.stderr)
    if args.command == "build":
        return 0
    try:
        r = subprocess.run([exe])
    except KeyboardInterrupt:
        return 130
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
