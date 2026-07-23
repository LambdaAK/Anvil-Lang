// Runs extension.js against a stand-in for the `vscode` module: open an Anvil file, then check the
// diagnostics, hovers and inlay hints it produces, and that an edit with a shape error is reported.
//     node editors/vscode/test/extension.test.js        (exits non-zero on failure)
"use strict";

const assert = require("assert");
const fs = require("fs");
const Module = require("module");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..", "..");

// ---------------------------------------------------------------------------- a fake vscode
class Position {
  constructor(line, character) { this.line = line; this.character = character; }
}
class Range {
  constructor(l1, c1, l2, c2) { this.start = new Position(l1, c1); this.end = new Position(l2, c2); }
  contains(p) {
    const after = (a, b) => a.line > b.line || (a.line === b.line && a.character >= b.character);
    return after(p, this.start) && after(this.end, p);
  }
}
class EventEmitter {
  constructor() { this.listeners = []; this.event = (f) => this.listeners.push(f); }
  fire(x) { this.listeners.forEach((f) => f(x)); }
  dispose() {}
}
const handlers = {};
const collection = new Map();
const providers = {};
const settings = {};
const vscode = {
  Position, Range, EventEmitter,
  Diagnostic: class { constructor(range, message, severity) { Object.assign(this, { range, message, severity }); } },
  DiagnosticSeverity: { Error: 0, Warning: 1 },
  DiagnosticRelatedInformation: class { constructor(location, message) { Object.assign(this, { location, message }); } },
  Location: class { constructor(uri, range) { Object.assign(this, { uri, range }); } },
  MarkdownString: class {
    constructor() { this.value = ""; }
    appendCodeblock(code, lang) { this.value += "```" + lang + "\n" + code + "\n```\n"; return this; }
  },
  Hover: class { constructor(contents, range) { Object.assign(this, { contents, range }); } },
  InlayHint: class { constructor(position, label, kind) { Object.assign(this, { position, label, kind }); } },
  InlayHintKind: { Type: 1 },
  window: { showWarningMessage: (m) => { throw new Error("unexpected warning: " + m); } },
  workspace: {
    textDocuments: [],
    getConfiguration: () => ({ get: (k, d) => (k in settings ? settings[k] : d) }),
    onDidOpenTextDocument: (f) => (handlers.open = f),
    onDidSaveTextDocument: (f) => (handlers.save = f),
    onDidChangeTextDocument: (f) => (handlers.change = f),
    onDidCloseTextDocument: (f) => (handlers.close = f),
  },
  languages: {
    createDiagnosticCollection: () => ({
      set: (uri, list) => collection.set(uri.toString(), list),
      delete: (uri) => collection.delete(uri.toString()),
      dispose() {},
    }),
    registerHoverProvider: (lang, p) => (providers.hover = p),
    registerInlayHintsProvider: (lang, p) => (providers.inlay = p),
  },
};
const resolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request === "vscode" ? "vscode" : resolve.call(this, request, ...rest);
};
require.cache.vscode = { id: "vscode", filename: "vscode", loaded: true, exports: vscode };

// ---------------------------------------------------------------------------- the test
function doc(file, text, version) {
  return {
    fileName: file, languageId: "anvil", version,
    uri: { scheme: "file", toString: () => "file://" + file },
    getText: () => text,
  };
}

async function until(cond, what) {
  for (let i = 0; i < 300; i++) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error("timed out waiting for " + what);
}

function hoverAt(d, line, character) {
  const h = providers.hover.provideHover(d, new Position(line, character));
  return h ? h.contents.value : null;
}

(async () => {
  const ext = require(path.join(ROOT, "editors", "vscode", "extension.js"));
  const file = path.join(ROOT, "examples", "mnist.anvil");
  assert.strictEqual(ext.anvilCommand(file).cmd, path.join(ROOT, "bin", "anvil"));   // the checkout's bin/anvil
  ext.activate({ subscriptions: [] });

  const text = fs.readFileSync(file, "utf8");
  const lines = text.split("\n");
  const d1 = doc(file, text, 1);
  handlers.open(d1);
  await until(() => collection.has(d1.uri.toString()), "diagnostics");
  assert.deepStrictEqual(collection.get(d1.uri.toString()), []);

  const xLine = lines.findIndex((l) => l.startsWith("X = images.reshape"));
  assert.match(hoverAt(d1, xLine, 0), /f32\[60000, 784\]/);
  const netLine = lines.findIndex((l) => l.startsWith("net = MLP()"));
  assert.match(hoverAt(d1, netLine, 1), /MLP \(model, 101,770 parameters\)/);
  const fwdLine = lines.findIndex((l) => l.includes("fn forward"));
  assert.match(hoverAt(d1, fwdLine, lines[fwdLine].indexOf("relu") + 1), /fn relu/);

  const all = new Range(0, 0, lines.length, 0);
  const hints = providers.inlay.provideInlayHints(d1, all).map((h) => `${h.position.line}:${h.label}`);
  assert.ok(hints.includes(`${xLine}:: f32[60000, 784]`), hints.join("\n"));
  const batchLine = lines.findIndex((l) => l.includes("for x, y in batches"));
  assert.ok(hints.includes(`${batchLine}:: f32[64, 784]`), hints.join("\n"));

  // an edit that breaks the shapes: reported after the debounce, at the right place
  const broken = text.replace("x |> l1 |> relu |> l2", "x |> l2 |> relu |> l1");
  const d2 = doc(file, broken, 2);
  handlers.change({ document: d2 });
  await until(() => (collection.get(d2.uri.toString()) || []).length > 0, "an error");
  const [err] = collection.get(d2.uri.toString());
  assert.strictEqual(err.severity, vscode.DiagnosticSeverity.Error);
  assert.match(err.message, /shape|cannot multiply|dimension/i);
  assert.strictEqual(err.range.start.line, fwdLine);
  // hints from the old version are not shown at positions of the new one
  assert.deepStrictEqual(providers.inlay.provideInlayHints(d1, all), []);

  settings.inlayHints = false;
  assert.deepStrictEqual(providers.inlay.provideInlayHints(d2, all), []);
  console.log("extension ok");
  process.exit(0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
