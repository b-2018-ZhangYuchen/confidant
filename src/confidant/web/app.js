// Confidant's web UI: upload a transcript, read it locally, and show a report.
//
// Everything a transcript or a report contains is put on the page with textContent,
// never as HTML. A chat can hold anything, including markup, and the model's answers
// quote it; treating either as HTML would let a message run as script on this page.
//
// The page talks only to the routes beside it (/api/...). Safety notices are shown as
// the server wrote them, first and in full: their wording is reviewed in safety.py, and
// re-creating it here would be a second copy to drift.

"use strict";

const $ = (id) => document.getElementById(id);

const state = { filename: null, busy: false, plain: false, lastText: null };

// -- building the page --------------------------------------------------------------

function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  for (const child of children) {
    if (child) node.append(child);
  }
  return node;
}

function section(title, ...children) {
  return el("section", { className: "part" }, el("h3", { text: title }), ...children);
}

function list(items) {
  const ul = el("ul");
  for (const item of items) ul.append(el("li", { text: item }));
  return ul;
}

function badge(text, kind) {
  return el("span", { className: `badge ${kind}`, text });
}

function quotes(evidence) {
  const box = el("div", { className: "evidence" });
  for (const item of evidence) {
    box.append(
      el("blockquote", { text: item.quote }),
      el("p", { className: "why", text: item.why_it_matters }),
    );
  }
  return box;
}

function personality(report) {
  const traits = el("div");
  for (const trait of report.traits) {
    traits.append(
      el(
        "article",
        { className: "card" },
        el(
          "h4",
          { text: trait.name },
          badge(`${trait.confidence} confidence`, `confidence-${trait.confidence}`),
        ),
        el("p", { text: trait.reading }),
        quotes(trait.evidence),
      ),
    );
  }
  const parts = [
    el("p", { className: "headline", text: report.headline }),
    section("How they communicate", el("p", { text: report.communication_style })),
    section("What stands out", traits),
    section("What they seem to want", el("p", { text: report.what_they_seem_to_want })),
  ];
  for (const [title, items] of [
    ["Green flags", report.green_flags],
    ["Worth watching", report.worth_watching],
    ["Worth asking", report.questions_to_ask],
  ]) {
    if (items.length) parts.push(section(title, list(items)));
  }
  parts.push(
    el(
      "p",
      { className: "confidence" },
      badge(`Overall confidence: ${report.confidence}`, `confidence-${report.confidence}`),
    ),
    el("p", { className: "caveat", text: `Caveat: ${report.caveat}` }),
  );
  return parts;
}

function flags(report, footnotes) {
  const parts = [el("p", { className: "headline", text: report.summary })];
  if (!report.flags.length) {
    parts.push(el("p", { text: "Nothing in this transcript rises to a red flag." }));
  }
  for (const flag of report.flags) {
    const card = el(
      "article",
      { className: `card severity-${flag.severity}` },
      el(
        "h4",
        {},
        badge(flag.severity, `severity-${flag.severity}`),
        document.createTextNode(` ${flag.category.replaceAll("_", " ")}`),
      ),
      el("p", { text: flag.behavior }),
      quotes(flag.evidence),
    );
    // As in the terminal: a charitable gloss is never shown on a danger flag, even if
    // the model supplied one.
    if (flag.innocent_reading && flag.severity !== "danger") {
      card.append(el("p", { className: "innocent", text: `Could also be: ${flag.innocent_reading}` }));
    }
    parts.push(card);
  }
  parts.push(
    el(
      "p",
      { className: "confidence" },
      badge(`Confidence: ${report.confidence}`, `confidence-${report.confidence}`),
    ),
  );
  for (const note of footnotes || []) parts.push(el("p", { className: "footnote", text: note }));
  return parts;
}

// -- showing results ----------------------------------------------------------------

function show(title, { escalation = null, parts = [], text = null, usage = null } = {}) {
  $("result").hidden = false;
  $("result-title").textContent = title;

  const notice = $("escalation");
  notice.replaceChildren();
  notice.hidden = !escalation;
  if (escalation) {
    // The same fixed sentences as the terminal's notice, laid out for a page rather than
    // re-wrapped from its 88-column text.
    notice.append(
      el("h3", { text: "Read this first" }),
      el("p", { className: "notice-headline", text: escalation.headline }),
      list(escalation.steps),
      ...escalation.routing_lines.map((line) => el("p", { text: line })),
    );
  }

  $("report").replaceChildren(...parts);
  state.lastText = text;
  $("plain").textContent = text || "";
  $("as-text").hidden = !text;
  setPlain(state.plain && Boolean(text));

  const line = $("usage");
  line.hidden = !(usage && usage.summary);
  line.textContent = usage && usage.summary ? usage.summary : "";
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setPlain(on) {
  $("plain").hidden = !on;
  $("report").hidden = on;
  $("as-text").setAttribute("aria-pressed", String(on));
  $("as-text").textContent = on ? "Show as a report" : "Show as plain text";
}

function failure(title, body, status) {
  const error = (body && body.error) || {};
  const message =
    error.message ||
    `The server answered ${status} without saying why. Is confidant serve still running?`;
  show(title, {
    parts: [el("p", { className: "error", text: message })],
    usage: body && body.usage,
  });
}

// -- talking to the server ----------------------------------------------------------

function transcript() {
  const text = $("text").value;
  if (!text.trim()) return null;
  const body = { text, filename: state.filename };
  const owner = $("owner").value.trim();
  const match = $("match").value.trim();
  if (owner) body.owner = owner;
  if (match) body.match = match;
  if ($("dates").value) body.dates = $("dates").value;
  body.redact = $("redact")
    .value.split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  return body;
}

async function call(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    // A non-JSON answer is reported by failure() with the status alone.
  }
  return { ok: response.ok, status: response.status, data };
}

async function run(button, waiting, task) {
  if (state.busy) return;
  const body = transcript();
  if (!body) {
    show("Nothing to read yet", {
      parts: [el("p", { text: "Choose a file, drop one on the page, or paste a transcript." })],
    });
    return;
  }
  state.busy = true;
  const buttons = document.querySelectorAll(".actions button");
  for (const b of buttons) b.disabled = true;
  const label = button.textContent;
  button.textContent = waiting;
  try {
    await task(body);
  } catch (exc) {
    failure("Could not reach Confidant", { error: { message: String(exc) } }, 0);
  } finally {
    for (const b of buttons) b.disabled = false;
    button.textContent = label;
    state.busy = false;
  }
}

function withRegion(path) {
  const region = $("region").value.trim();
  return region ? `${path}?region=${encodeURIComponent(region)}` : path;
}

async function readLocally(body) {
  const { ok, status, data } = await call("/api/stats", body);
  const summary = $("summary");
  if (!ok) {
    summary.hidden = true;
    failure("This transcript could not be read", data, status);
    return;
  }
  summary.hidden = false;
  summary.textContent = data.note ? `${data.text}\n${data.note}` : data.text;
  // Fill in the names the parser found, so the owner can see who it thinks is who and
  // correct it before paying for a read.
  if (!$("owner").value) $("owner").placeholder = data.owner;
  if (!$("match").value) $("match").placeholder = data.match;
}

async function preview(body) {
  const { ok, status, data } = await call("/api/redact", body);
  if (!ok) return failure("This transcript could not be read", data, status);
  show("What would be sent", {
    parts: [
      el("p", { text: `Replaced: ${data.replaced}. The placeholders are put back before you read a report.` }),
      el("pre", { className: "plain", text: data.transcript }),
    ],
  });
}

function reader(path, title, render) {
  return async (body) => {
    const { ok, status, data } = await call(path, body);
    if (!ok) return failure(title, data, status);
    show(title, {
      escalation: data.escalation,
      parts: render(data),
      text: data.text,
      usage: data.usage,
    });
  };
}

// -- wiring -------------------------------------------------------------------------

async function load(file) {
  state.filename = file.name;
  $("text").value = await file.text();
  $("summary").hidden = true;
  await run($("stats"), "Reading…", readLocally);
}

function wire() {
  $("file").addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) load(file);
  });

  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const [file] = event.dataTransfer.files;
    if (file) load(file);
  });

  $("text").addEventListener("input", () => {
    // Pasted text is not the file it replaced.
    state.filename = null;
    $("summary").hidden = true;
  });

  $("stats").addEventListener("click", (e) => run(e.target, "Reading…", readLocally));
  $("preview").addEventListener("click", (e) => run(e.target, "Redacting…", preview));
  $("analyze").addEventListener("click", (e) =>
    run(e.target, "Waiting for Claude…", reader("/api/analyze", "Personality read", (d) => personality(d.report))),
  );
  $("flags").addEventListener("click", (e) =>
    run(
      e.target,
      "Waiting for Claude…",
      reader(withRegion("/api/flags"), "Red-flag check", (d) => flags(d.report, d.footnotes)),
    ),
  );
  $("as-text").addEventListener("click", () => {
    state.plain = !state.plain;
    setPlain(state.plain);
  });
}

document.addEventListener("DOMContentLoaded", wire);
