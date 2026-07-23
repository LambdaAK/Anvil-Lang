// Anvil for VS Code and Cursor: compile errors as you type, the shape of any tensor on hover, and
// shapes after definitions (`h = relu(x @ W)` shows `: f32[64, 128]`).
//
// Everything comes from `anvil check --json --stdin file.anvil` (see anvil/ide.py), which elaborates the
// program without generating code: about 50 ms for the MNIST example. No dependencies.
"use strict";

const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

const DEBOUNCE_MS = 350;

/** The command that runs anvil for a file: the `anvil.path` setting, a checkout's bin/anvil above the
 *  file, `anvil` on the PATH, or `python3 -m anvil`. */
function anvilCommand(file) {
  const configured = vscode.workspace.getConfiguration("anvil").get("path");
  if (configured) return { cmd: configured, args: [] };
  let dir = path.dirname(file);
  for (let i = 0; i < 12; i++) {
    const candidate = path.join(dir, "bin", "anvil");
    if (fs.existsSync(candidate) && fs.existsSync(path.join(dir, "anvil", "__init__.py"))) {
      return { cmd: candidate, args: [] };
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  const onPath = (process.env.PATH || "").split(path.delimiter).some((d) => d && fs.existsSync(path.join(d, "anvil")));
  return onPath ? { cmd: "anvil", args: [] } : { cmd: "python3", args: ["-m", "anvil"] };
}

function toRange(r) {
  return new vscode.Range(r[0], r[1], r[2], Math.max(r[3], r[2] === r[0] ? r[1] + 1 : r[3]));
}

function toDiagnostic(d, uri) {
  let message = d.message;
  if (d.label) message += `: ${d.label}`;
  for (const n of d.notes || []) message += `\nnote: ${n}`;
  if (d.help) message += `\nhelp: ${d.help}`;
  const severity = d.severity === "warning" ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Error;
  const diag = new vscode.Diagnostic(toRange(d.range), message, severity);
  diag.source = "anvil";
  if (d.related && d.related.length) {
    diag.relatedInformation = d.related.map(
      (r) => new vscode.DiagnosticRelatedInformation(new vscode.Location(uri, toRange(r.range)), r.message || "here"));
  }
  return diag;
}

function contains(r, pos) {
  const [l1, c1, l2, c2] = r;
  if (pos.line < l1 || pos.line > l2) return false;
  if (pos.line === l1 && pos.character < c1) return false;
  if (pos.line === l2 && pos.character > c2) return false;
  return true;
}

function activate(context) {
  const diagnostics = vscode.languages.createDiagnosticCollection("anvil");
  const results = new Map();     // uri -> {hovers, version}
  const running = new Map();     // uri -> child process
  const timers = new Map();      // uri -> timeout
  const hintsChanged = new vscode.EventEmitter();
  let warned = false;

  function analyze(doc) {
    if (doc.languageId !== "anvil" || doc.uri.scheme !== "file") return;
    const key = doc.uri.toString();
    const old = running.get(key);
    if (old) old.kill();
    const { cmd, args } = anvilCommand(doc.fileName);
    const version = doc.version;
    const child = cp.spawn(cmd, [...args, "check", "--json", "--stdin", doc.fileName], {
      cwd: path.dirname(doc.fileName),
      env: Object.assign({}, process.env, { NO_COLOR: "1" }),
    });
    running.set(key, child);
    let out = "";
    let err = "";
    child.stdout.on("data", (b) => (out += b));
    child.stderr.on("data", (b) => (err += b));
    child.on("error", (e) => {
      running.delete(key);
      if (!warned) {
        warned = true;
        vscode.window.showWarningMessage(
          `Anvil: could not run \`${cmd}\` (${e.message}). Set "anvil.path" to the anvil command (bin/anvil in a checkout).`);
      }
    });
    child.on("close", () => {
      if (running.get(key) === child) running.delete(key);
      if (child.killed) return;
      let data;
      try {
        data = JSON.parse(out);
      } catch (e) {
        if (err.trim()) console.error(`anvil check failed: ${err}`);
        return;
      }
      diagnostics.set(doc.uri, (data.diagnostics || []).map((d) => toDiagnostic(d, doc.uri)));
      results.set(key, { hovers: data.hovers || [], version });
      hintsChanged.fire();
    });
    child.stdin.on("error", () => {});
    child.stdin.end(doc.getText());
  }

  function schedule(doc) {
    if (doc.languageId !== "anvil") return;
    if (!vscode.workspace.getConfiguration("anvil").get("checkOnType", true)) return;
    const key = doc.uri.toString();
    clearTimeout(timers.get(key));
    timers.set(key, setTimeout(() => analyze(doc), DEBOUNCE_MS));
  }

  context.subscriptions.push(
    diagnostics,
    hintsChanged,
    vscode.workspace.onDidOpenTextDocument(analyze),
    vscode.workspace.onDidSaveTextDocument(analyze),
    vscode.workspace.onDidChangeTextDocument((e) => schedule(e.document)),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      const key = doc.uri.toString();
      diagnostics.delete(doc.uri);
      results.delete(key);
      clearTimeout(timers.get(key));
    }),
    vscode.languages.registerHoverProvider("anvil", {
      provideHover(doc, pos) {
        const r = results.get(doc.uri.toString());
        if (!r) return null;
        // the innermost recorded name under the cursor
        let best = null;
        for (const h of r.hovers) {
          if (!contains(h.range, pos)) continue;
          if (!best || (h.range[2] - h.range[0]) * 1e6 + (h.range[3] - h.range[1]) <
                       (best.range[2] - best.range[0]) * 1e6 + (best.range[3] - best.range[1])) best = h;
        }
        if (!best) return null;
        const md = new vscode.MarkdownString();
        md.appendCodeblock(best.text, best.text.startsWith("fn ") || best.text.startsWith("model ") ? "anvil" : "text");
        return new vscode.Hover(md, toRange(best.range));
      },
    }),
  );

  if (vscode.languages.registerInlayHintsProvider) {
    context.subscriptions.push(
      vscode.languages.registerInlayHintsProvider("anvil", {
        onDidChangeInlayHints: hintsChanged.event,
        provideInlayHints(doc, range) {
          if (!vscode.workspace.getConfiguration("anvil").get("inlayHints", true)) return [];
          const r = results.get(doc.uri.toString());
          if (!r || r.version !== doc.version) return [];      // stale positions would land in the wrong place
          const hints = [];
          for (const h of r.hovers) {
            if (!h.inlay) continue;
            const end = new vscode.Position(h.range[2], h.range[3]);
            if (!range.contains(end)) continue;
            const hint = new vscode.InlayHint(end, `: ${h.inlay}`, vscode.InlayHintKind.Type);
            hint.tooltip = "shape (from the Anvil compiler)";
            hints.push(hint);
          }
          return hints;
        },
      }),
    );
  }

  for (const doc of vscode.workspace.textDocuments) analyze(doc);
}

function deactivate() {}

module.exports = { activate, deactivate, anvilCommand };
