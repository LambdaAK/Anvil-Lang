"""Assemble and link generated assembly with the system toolchain."""
from __future__ import annotations

import os
import shutil
import subprocess


class ToolchainError(Exception):
    pass


def find_cc() -> str:
    for cc in (os.environ.get("ANVIL_CC"), "cc", "clang"):
        if cc and shutil.which(cc):
            return cc
    raise ToolchainError("no C toolchain found (need `cc` or `clang` to assemble and link); "
                         "on macOS run `xcode-select --install`")


def assemble_and_link(asm_path: str, exe_path: str):
    cc = find_cc()
    cmd = [cc, "-arch", "arm64", "-x", "assembler", asm_path, "-o", exe_path, "-lz", "-framework", "Accelerate"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ToolchainError(f"assembler/linker failed ({' '.join(cmd)}):\n{r.stderr.strip()}")
    return exe_path
