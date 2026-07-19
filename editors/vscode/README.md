# Anvil for VS Code and Cursor

Support for [Anvil](../../README.md) (`.anvil` files), driven by the Anvil compiler itself:

- **Errors as you type.** The program is checked on every edit (about 50 ms for the MNIST example).
  Shape errors are underlined where they happen. An error inside the standard library is underlined
  at the call in your file that led to it.
- **Shapes on hover.** Shapes are known at compile time, so hovering over any name shows what it
  holds: `f32[64, 784]`, `param f32[784, 128]`, `index j < 10`, a constant's value, a function's
  signature. A model shows its parameters and how many numbers they hold. A function's parameter
  shows every shape it is called with (`f32[64, 784] | f32[10000, 784]`).
- **Shapes after definitions.** `h = relu(x @ W)` is displayed as `h: f32[64, 128] = relu(x @ W)`
  (inlay hints; turn them off with `"anvil.inlayHints": false`).
- **Syntax highlighting.** It covers:
  - index definitions, `where` ranges, shapes in annotations
  - sampling (`~ normal(…)`), reductions, layers, optimizers, `static for`
  - f-strings, and Unicode operators (`∇ √ → ≤ ≥ ≠ ·`)
- **Snippets.** `def` (an index definition), `fn`, `model`, `train`, `param`, `sfor`.

The extension finds the compiler on its own: `bin/anvil` in the Anvil checkout above the file, else
`anvil` on the PATH, else `python3 -m anvil`. To use another one, set `"anvil.path"`. Set
`"anvil.checkOnType": false` to check only on open and save.

Install:

```
python3 editors/vscode/package_vsix.py
cursor --install-extension editors/vscode/anvil-language-0.3.1.vsix     # or: code --install-extension …
```

Test (no editor needed; it runs the extension against a stand-in for the editor API):

```
node editors/vscode/test/extension.test.js
```
