// Confidant's web UI: upload a transcript, read it locally, and show a report; see who
// is saved, their profiles, and who is worth a message today.
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

const state = { filename: null, busy: false, plain: false, lastText: null, parsedMatch: null };

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

function flagCard(flag, tag = null) {
  const card = el(
    "article",
    { className: `card severity-${flag.severity}` },
    el(
      "h4",
      {},
      badge(flag.severity, `severity-${flag.severity}`),
      document.createTextNode(` ${flag.category.replaceAll("_", " ")}`),
      tag && el("span", { className: "meta", text: tag }),
    ),
    el("p", { text: flag.behavior }),
    quotes(flag.evidence),
  );
  // As in the terminal: a charitable gloss is never shown on a danger flag, even if
  // the model supplied one.
  if (flag.innocent_reading && flag.severity !== "danger") {
    card.append(el("p", { className: "innocent", text: `Could also be: ${flag.innocent_reading}` }));
  }
  return card;
}

function flags(report, footnotes) {
  const parts = [el("p", { className: "headline", text: report.summary })];
  if (!report.flags.length) {
    parts.push(el("p", { text: "Nothing in this transcript rises to a red flag." }));
  }
  for (const flag of report.flags) parts.push(flagCard(flag));
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

function plural(count, word, many = `${word}s`) {
  return `${count} ${count === 1 ? word : many}`;
}

// The server sends local times without a zone, as the store keeps them. Cutting the
// string, rather than parsing it, keeps them local and in the terminal's format.
function minute(iso) {
  return iso ? iso.slice(0, 16).replace("T", " ") : "—";
}

function day(iso) {
  return iso ? iso.slice(0, 10) : "—";
}

function readWhen(current, messages) {
  const at = day(current.read_at);
  return current.stale ? `${at}, at ${current.messages_read} of ${messages} messages` : at;
}

function conversationCard(entry, name) {
  const card = el(
    "article",
    { className: "card" },
    el("h4", { text: `#${entry.id} ${entry.source || "(no source)"}` }),
    el("p", {
      className: "meta",
      text: `${plural(entry.messages, "message")}, last ${minute(entry.last_message)}`,
    }),
  );
  if (!entry.match_messages) {
    card.append(el("p", { text: `Nothing from ${name} to read.` }));
    return card;
  }
  const read = entry.personality;
  if (read) {
    card.append(
      el("p", { text: read.report.headline }),
      list(read.report.traits.map((t) => `${t.name} (${t.confidence} confidence)`)),
      el("p", {
        className: "meta",
        text: `Read ${readWhen(read, entry.messages)}, ${read.report.confidence} confidence.`,
      }),
    );
  } else {
    card.append(el("p", { className: "meta", text: "Not read yet." }));
  }
  const check = entry.flags;
  if (check) {
    card.append(
      el("p", {
        className: "meta",
        text:
          `Checked ${readWhen(check, entry.messages)}, ${plural(check.report.flags.length, "flag")}, ` +
          `${check.report.confidence} confidence.`,
      }),
    );
  }
  return card;
}

function profile(data, onUpdate) {
  const name = data.person;
  const conversations = data.conversations;
  const messages = conversations.reduce((sum, c) => sum + c.messages, 0);
  const parts = [
    el("p", {
      className: "headline",
      text: `${plural(conversations.length, "conversation")}, ${plural(messages, "message")}`,
    }),
  ];
  if (!conversations.length) {
    parts.push(el("p", { text: `Nothing saved yet. Save a conversation above under ${name}.` }));
    return parts;
  }

  parts.push(section("Conversations", ...conversations.map((c) => conversationCard(c, name))));

  // Side by side, tagged with where each came from, as in the terminal: merging the
  // reads into one would be a claim no single quote supports.
  for (const [title, pick] of [
    ["Green flags", (r) => r.green_flags],
    ["Worth watching", (r) => r.worth_watching],
    ["Worth asking", (r) => r.questions_to_ask],
  ]) {
    const items = conversations.flatMap((c) =>
      c.personality ? pick(c.personality.report).map((item) => `${item}  [#${c.id}]`) : [],
    );
    if (items.length) parts.push(section(title, list(items)));
  }

  const checked = conversations.filter((c) => c.flags).length;
  const red = [];
  if (!checked) red.push(el("p", { text: "No conversation has been checked yet." }));
  else if (!data.flags.length) {
    red.push(
      el("p", { text: `Nothing in ${plural(checked, "checked conversation")} rises to a red flag.` }),
    );
  }
  for (const flag of data.flags) red.push(flagCard(flag, `#${flag.conversation}`));
  for (const note of data.footnotes) red.push(el("p", { className: "footnote", text: note }));
  parts.push(section("Red flags", ...red));

  if (data.pending) {
    const button = el("button", {
      className: "primary",
      text: `Bring ${plural(data.pending, "read")} up to date`,
    });
    button.type = "button";
    button.addEventListener("click", () => onUpdate(button));
    parts.push(
      el(
        "div",
        { className: "part" },
        el("p", { text: `${plural(data.pending, "read")} missing or out of date.` }),
        button,
        el("p", {
          className: "hint",
          text: "This sends the redacted conversations that need it to Claude, as confidant profile --update does.",
        }),
      ),
    );
  }
  return parts;
}

function notice(escalation, node = el("div", { className: "escalation" })) {
  // The same fixed sentences as the terminal's notice, laid out for a page rather than
  // re-wrapped from its 88-column text.
  node.replaceChildren(
    el("h3", { text: "Read this first" }),
    el("p", { className: "notice-headline", text: escalation.headline }),
    list(escalation.steps),
    ...escalation.routing_lines.map((line) => el("p", { text: line })),
  );
  return node;
}

function reasons(items) {
  const ul = list(items);
  ul.className = "reasons";
  return ul;
}

function contact(c, { open = false } = {}) {
  const box = el("div", { className: "contact" });
  const headline = el("strong", { text: c.headline });
  if (!c.reasons.length) {
    box.append(el("p", {}, headline));
  } else if (open) {
    box.append(el("p", {}, headline), reasons(c.reasons));
  } else {
    // Everyone not suggested gets one line, as in the terminal; the reasons are a
    // click away rather than a second request.
    box.append(el("details", {}, el("summary", {}, headline), reasons(c.reasons)));
  }
  return box;
}

function digest(data) {
  if (!data.contacts.length) {
    return [el("p", { text: "No one saved yet. Save a conversation above to start." })];
  }
  const parts = [];
  // A danger-tier notice comes before anything about timing, as it does in the terminal.
  for (const c of data.contacts.filter((c) => c.state === "safety")) {
    parts.push(
      el("div", { className: "contact" }, el("p", {}, el("strong", { text: c.headline }))),
      notice(c.escalation),
      reasons(c.reasons),
    );
  }
  const suggested = data.contacts.filter((c) => c.suggested);
  parts.push(
    el("h3", { text: suggested.length ? "Worth a message today" : "No one is due a message today." }),
    ...suggested.map((c) => contact(c, { open: true })),
  );
  const others = data.contacts.filter((c) => !c.suggested && c.state !== "safety");
  if (others.length) parts.push(el("h3", { text: "Not today" }), ...others.map((c) => contact(c)));
  if (data.caveat) parts.push(el("p", { className: "caveat", text: data.caveat }));
  return parts;
}

// -- showing results ----------------------------------------------------------------

function show(title, { escalation = null, parts = [], text = null, usage = null } = {}) {
  $("result").hidden = false;
  $("result-title").textContent = title;

  const box = $("escalation");
  box.replaceChildren();
  box.hidden = !escalation;
  if (escalation) notice(escalation, box);

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

// Every call to the server goes through here, each with its method and route written
// out where it is made, so the tests can check every one against the routes the API has.
async function request(method, path, { body = undefined, query = {} } = {}) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) if (value) params.set(key, value);
  const url = params.toString() ? `${path}?${params}` : path;
  const options = { method };
  if (body !== undefined) {
    options.headers = { "content-type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let data = null;
  try {
    data = await response.json();
  } catch {
    // A non-JSON answer is reported by failure() with the status alone.
  }
  return { ok: response.ok, status: response.status, data };
}

// One action at a time: a second click while Claude is answering would pay twice.
async function busy(button, waiting, task) {
  if (state.busy) return;
  state.busy = true;
  const buttons = document.querySelectorAll("main button");
  for (const b of buttons) b.disabled = true;
  const label = button.textContent;
  button.textContent = waiting;
  try {
    await task();
  } catch (exc) {
    failure("Could not reach Confidant", { error: { message: String(exc) } }, 0);
  } finally {
    for (const b of buttons) b.disabled = false;
    button.textContent = label;
    state.busy = false;
  }
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
  await busy(button, waiting, () => task(body));
}

function region() {
  return { region: $("region").value.trim() };
}

function person(name, suffix = "") {
  return `/api/people/${encodeURIComponent(name)}${suffix}`;
}

async function readLocally(body) {
  const { ok, status, data } = await request("POST", "/api/stats", { body });
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
  state.parsedMatch = data.match;
  $("save-name").placeholder = data.match;
}

async function preview(body) {
  const { ok, status, data } = await request("POST", "/api/redact", { body });
  if (!ok) return failure("This transcript could not be read", data, status);
  show("What would be sent", {
    parts: [
      el("p", { text: `Replaced: ${data.replaced}. The placeholders are put back before you read a report.` }),
      el("pre", { className: "plain", text: data.transcript }),
    ],
  });
}

function reader(send, title, render) {
  return async (body) => {
    const { ok, status, data } = await send(body);
    if (!ok) return failure(title, data, status);
    show(title, {
      escalation: data.escalation,
      parts: render(data),
      text: data.text,
      usage: data.usage,
    });
  };
}

function saveStatus(text, isError = false) {
  const line = $("save-status");
  line.hidden = false;
  line.className = isError ? "hint error" : "hint";
  line.textContent = text;
}

async function save(body) {
  const name = $("save-name").value.trim() || state.parsedMatch || body.match;
  if (!name) {
    saveStatus("Say who this conversation is with, or read it locally first so Confidant can tell.", true);
    return;
  }
  // create=true is `confidant add NAME FILE`: someone new is added only once the
  // transcript has been read and found to be with them.
  const { ok, data } = await request("POST", person(name, "/conversations"), {
    body,
    query: { create: "true" },
  });
  if (!ok) {
    const error = (data && data.error) || {};
    if (error.kind === "wrong-person") {
      saveStatus(
        `This is a conversation with ${error.match}, not ${name}. If that is how ${name} ` +
          `appears in it, put "${error.match}" in "Their name in it" under Options and save again.`,
        true,
      );
    } else {
      saveStatus(error.message || "The conversation could not be saved.", true);
    }
    return;
  }
  // The terminal's wording. Pasted text has no file name, so it is just "it".
  const stored = data.conversation;
  const source = stored.source;
  const lines = data.created ? [`Added ${data.person}.`] : [];
  if (data.outcome === "new") {
    lines.push(`Saved ${source || "it"} as #${stored.id} (${plural(stored.messages, "message")}).`);
  } else if (data.outcome === "extended") {
    const from = source ? ` from ${source}` : "";
    lines.push(
      `Added ${plural(data.added, "new message")}${from} to #${stored.id} ` +
        `(${stored.messages} in all).`,
    );
  } else {
    lines.push(`${source || "It"} is already saved, as #${stored.id}.`);
  }
  saveStatus(lines.join(" "));
  await refresh();
}

async function openProfile(name) {
  const { ok, status, data } = await request("GET", person(name, "/profile"), { query: region() });
  if (!ok) return failure(name, data, status);
  showProfile(data);
}

function showProfile(data, { updated = null, usage = null } = {}) {
  const parts = profile(data, (button) => busy(button, "Waiting for Claude…", () => update(data.person)));
  if (updated !== null) parts.unshift(el("p", { text: `Updated ${plural(updated, "read")}.` }));
  show(data.person, { escalation: data.escalation, parts, text: data.text, usage });
}

async function update(name) {
  const { ok, status, data } = await request("POST", person(name, "/profile/update"), {
    query: region(),
  });
  if (!ok) return failure(name, data, status);
  showProfile(data.profile, { updated: data.updated, usage: data.usage });
  await refresh();
}

async function forget(name, conversations) {
  // The same question `confidant remove` asks, since this cannot be undone.
  const sure = window.confirm(
    `Forget ${name} and ${plural(conversations, "saved conversation")}? ` +
      "Their messages and reads are overwritten in the file on this machine, and cannot be brought back.",
  );
  if (!sure) return;
  const { ok, status, data } = await request("DELETE", person(name));
  if (!ok) return failure(`Could not forget ${name}`, data, status);
  if ($("result-title").textContent === name) $("result").hidden = true;
  await refresh();
}

function peopleTable(people) {
  if (!people.length) {
    return el("p", { text: "No one saved yet. Read a conversation above and save it under their name." });
  }
  const head = el("tr");
  for (const [title, optional] of [
    ["Name", false],
    ["Conversations", true],
    ["Messages", true],
    ["Last message", false],
    ["", false],
  ]) {
    head.append(el("th", { className: optional ? "optional" : "", text: title }));
  }
  const rows = people.map((p) => {
    const view = el("button", { className: "quiet", text: "Profile" });
    view.type = "button";
    view.addEventListener("click", () => busy(view, "Opening…", () => openProfile(p.name)));
    const remove = el("button", { className: "quiet", text: "Forget" });
    remove.type = "button";
    remove.setAttribute("aria-label", `Forget ${p.name}`);
    remove.addEventListener("click", () => busy(remove, "Forgetting…", () => forget(p.name, p.conversations)));
    return el(
      "tr",
      {},
      el("td", { text: p.name }),
      el("td", { className: "num optional", text: String(p.conversations) }),
      el("td", { className: "num optional", text: String(p.messages) }),
      el("td", { className: "num", text: minute(p.last_message) }),
      el("td", { className: "row-actions" }, view, remove),
    );
  });
  return el("table", { className: "people" }, el("thead", {}, head), el("tbody", {}, ...rows));
}

async function loadPeople() {
  const { ok, status, data } = await request("GET", "/api/people");
  const box = $("people-list");
  if (!ok) {
    const message = (data && data.error && data.error.message) || `The server answered ${status}.`;
    box.replaceChildren(el("p", { className: "error", text: message }));
    return;
  }
  box.replaceChildren(peopleTable(data));
}

async function loadDigest() {
  // datetime-local gives "2026-04-14T09:00", which the API reads as `nudge --at` does.
  const query = { ...region(), at: $("digest-at").value };
  const { ok, status, data } = await request("GET", "/api/nudge", { query });
  const box = $("digest-body");
  if (!ok) {
    const message = (data && data.error && data.error.message) || `The server answered ${status}.`;
    box.replaceChildren(el("p", { className: "error", text: message }));
    return;
  }
  box.replaceChildren(...digest(data));
}

async function refresh() {
  await Promise.all([loadPeople(), loadDigest()]);
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
    run(
      e.target,
      "Waiting for Claude…",
      reader(
        (body) => request("POST", "/api/analyze", { body }),
        "Personality read",
        (d) => personality(d.report),
      ),
    ),
  );
  $("flags").addEventListener("click", (e) =>
    run(
      e.target,
      "Waiting for Claude…",
      reader(
        (body) => request("POST", "/api/flags", { body, query: region() }),
        "Red-flag check",
        (d) => flags(d.report, d.footnotes),
      ),
    ),
  );
  $("save").addEventListener("click", (e) => run(e.target, "Saving…", save));
  $("people-refresh").addEventListener("click", (e) => busy(e.target, "Refreshing…", refresh));
  $("digest-refresh").addEventListener("click", (e) => busy(e.target, "Refreshing…", loadDigest));
  $("digest-at").addEventListener("change", () => loadDigest());
  // Notices name the region's numbers, so a new region redraws the digest.
  $("region").addEventListener("change", () => loadDigest());
  $("as-text").addEventListener("click", () => {
    state.plain = !state.plain;
    setPlain(state.plain);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wire();
  refresh().catch((exc) => {
    $("people-list").replaceChildren(el("p", { className: "error", text: String(exc) }));
  });
});
