"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const state = { job: null, files: [], config: null, votes: {} };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tc = (t) => { const m = Math.floor(t / 60), s = (t - m * 60).toFixed(1).padStart(4, "0"); return `${String(m).padStart(2, "0")}:${s}`; };
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `Request failed (${res.status})`);
  return body;
}

async function init() {
  try {
    state.config = await api("/api/config");
    const h = await api("/api/health");
    $("#ver").textContent = h.version;
    const llm = state.config.llm;
    const pill = $("#llm-pill");
    pill.hidden = false;
    if (llm.available) {
      pill.textContent = `AI editor: ${llm.model}`; pill.classList.add("on");
    } else {
      pill.textContent = "AI editor off (heuristics only)";
      $("#use_llm").disabled = true;
      $("#llm-row").classList.add("disabled");
      $("#llm-row").title = "Set PROMOFINDER_LLM_PROVIDER and PROMOFINDER_LLM_API_KEY on the server to enable";
    }
  } catch (e) { /* page still works without config */ }
  const jid = new URLSearchParams(location.search).get("job");
  if (jid) loadJob(jid);
}

// ---------- input ----------
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); setFiles([...e.dataTransfer.files]); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file-input").click(); } });
$("#file-input").addEventListener("change", (e) => setFiles([...e.target.files]));
function setFiles(files) {
  state.files = files.filter((f) => /\.(zip|txt)$/i.test(f.name));
  $("#file-list").textContent = state.files.length ? state.files.map((f) => f.name).slice(0, 4).join(", ") + (state.files.length > 4 ? ` and ${state.files.length - 4} more` : "") : "";
}

function formData(useSample) {
  const fd = new FormData();
  if (useSample) fd.append("use_sample", "true");
  else state.files.forEach((f) => fd.append("files", f, f.name));
  ["min_len", "max_len", "top_k"].forEach((id) => fd.append(id, $("#" + id).value));
  fd.append("allow_stitched", $("#allow_stitched").checked);
  fd.append("use_llm", $("#use_llm").checked);
  return fd;
}

async function run(useSample) {
  const err = $("#form-error");
  err.hidden = true;
  if (!useSample && !state.files.length) { err.textContent = "Add a .zip or .txt episode files first, or use the sample show."; err.hidden = false; return; }
  busy(true, $("#use_llm").checked ? "Scoring cuts and asking the AI editor…" : "Scoring every possible cut…");
  try {
    const res = await api("/api/analyze", { method: "POST", body: formData(useSample) });
    history.replaceState(null, "", `?job=${res.job_id}`);
    render(res);
  } catch (e) {
    err.textContent = e.message; err.hidden = false;
  } finally { busy(false); }
}
$("#analyze-form").addEventListener("submit", (e) => { e.preventDefault(); run(false); });
$("#sample-btn").addEventListener("click", () => run(true));

async function loadJob(id) {
  busy(true, "Loading results…");
  try { render(await api(`/api/jobs/${id}`)); } catch (e) { $("#form-error").textContent = e.message; $("#form-error").hidden = false; }
  finally { busy(false); }
}

function busy(on, text) {
  $("#loading").hidden = !on;
  if (text) $("#loading-text").textContent = text;
  $("#run-btn").disabled = on; $("#sample-btn").disabled = on;
}

// ---------- results ----------
function render(res) {
  state.job = res;
  $("#empty").hidden = true;
  $("#results").hidden = false;
  $("#show-title").textContent = res.show.title;
  const st = res.stats, mins = Math.round(res.show.runtime_sec / 60);
  let stats = `${res.show.episodes} episodes, ${mins} min of story. Checked ${st.candidates.toLocaleString()} possible cuts in ${st.total_seconds.toFixed(2)} s.`;
  if (res.llm) stats += ` AI editor (${res.llm.model}) judged ${res.llm.judged} shortlisted cuts.`;
  if (res.llm && res.llm.errors && res.llm.errors.length) stats += " Some AI batches failed, so those cuts kept their heuristic score.";
  $("#run-stats").textContent = stats;
  ["csv", "edl", "json"].forEach((f) => { const a = $(`#exp-${f}`); a.href = `/api/jobs/${res.job_id}/export?format=${f}`; a.setAttribute("download", ""); });

  const meta = $("#show-meta");
  meta.hidden = false;
  meta.innerHTML = `<b>${esc(res.show.title)}</b><br>${res.show.episodes} episodes parsed` +
    (res.show.warnings.length ? `<ul>${res.show.warnings.slice(0, 5).map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "");

  const top = $("#tab-top");
  top.innerHTML = "";
  if (!res.promos.length) top.innerHTML = `<p class="muted">No cut in this script fits the length range. Try widening it.</p>`;
  res.promos.forEach((p, i) => top.appendChild(promoCard(p, i === 0)));

  const sel = $("#ep-select");
  sel.innerHTML = res.show.episode_list.map((e) => `<option value="${e.number}">Episode ${e.number} (${Math.round(e.duration)} s)</option>`).join("");
  sel.onchange = () => renderEpisode(sel.value);
  renderEpisode(sel.value);
  $("#eval-out").innerHTML = "";
  $("#key-input").hidden = res.source === "sample";
  selectTab("top");
}

function renderEpisode(num) {
  const list = $("#ep-list");
  list.innerHTML = "";
  const ep = state.job.show.episode_list.find((e) => String(e.number) === String(num));
  if (ep && ep.logline) list.insertAdjacentHTML("beforeend", `<p class="muted">${esc(ep.logline)}</p>`);
  const picks = state.job.per_episode[num] || [];
  if (!picks.length) list.insertAdjacentHTML("beforeend", `<p class="muted">No cut in this episode fits the length range.</p>`);
  picks.forEach((p, i) => list.appendChild(promoCard(p, i === 0)));
}

function promoCard(p, isTop) {
  const el = $("#promo-tpl").content.firstElementChild.cloneNode(true);
  if (isTop) el.classList.add("top");
  $(".rank", el).textContent = p.rank;
  const eps = [...new Set(p.segments.map((s) => s.episode))];
  const where = eps.length === 1 ? `Episode ${eps[0]}` : `Episodes ${eps.join(" and ")}`;
  $(".promo-title", el).innerHTML = `${where}, ${p.duration.toFixed(1)} s` +
    (p.type === "stitched" ? `<span class="tag stitched">${p.segments.length} moments joined</span>` : "") +
    (p.llm ? `<span class="tag ai">AI: ${esc(p.llm.verdict)}</span>` : "") +
    `<small>${p.segments.map((s) => `Ep ${s.episode} ${s.start_tc} to ${s.end_tc}`).join(", then ")}</small>`;
  $(".score-num", el).textContent = Math.round(p.score * 100);

  const rib = $(".ribbons", el);
  const total = p.duration;
  p.segments.forEach((s, i) => {
    const D = s.episode_duration || s.end;
    const L = (x) => `${(x / D) * 100}%`;
    const hookW = i === 0 ? Math.min(3, s.end - s.start) : 0;
    const cliffW = i === p.segments.length - 1 ? Math.min(4.5, s.end - s.start) : 0;
    rib.insertAdjacentHTML("beforeend", `<div class="ribbon"><span>Ep ${s.episode}</span>
      <div class="track" role="img" aria-label="Cut from ${s.start_tc} to ${s.end_tc} of a ${Math.round(D)} second episode">
        <div class="cut" style="left:${L(s.start)};width:${L(s.end - s.start)}"></div>
        ${hookW ? `<div class="hz" style="left:${L(s.start)};width:${L(hookW)}"></div>` : ""}
        ${cliffW ? `<div class="cz" style="left:${L(s.end - cliffW)};width:${L(cliffW)}"></div>` : ""}
      </div><span class="tc">${s.start_tc} to ${s.end_tc}</span></div>`);
  });

  const lineText = (b) => (b ? (b.kind === "DIALOGUE" ? `${b.speaker}: “${b.text}”` : b.text) : "");
  $(".line.open q", el).textContent = lineText(p.opening) || "(no clear hook found)";
  $(".line.end q", el).textContent = lineText(p.ending) || "(no clear ending beat found)";

  $(".meters", el).innerHTML = [["hook", "Hook", p.scores.hook], ["cliff", "Cliffhanger", p.scores.cliff], ["standalone", "Makes sense alone", p.scores.standalone]]
    .map(([k, label, v]) => `<div class="meter ${k}">${label} <b>${Math.round(v * 100)}</b><div class="bar"><i style="width:${v * 100}%"></i></div></div>`).join("");
  $(".why", el).innerHTML = p.reasons.map((r) => `<li>${esc(r)}</li>`).join("");
  $(".warn", el).innerHTML = p.warnings.map((r) => `<li>${esc(r)}</li>`).join("");

  const tr = $(".transcript", el);
  const tog = $(".toggle", el);
  tog.addEventListener("click", () => {
    const open = tr.hidden;
    if (open && !tr.dataset.ready) { tr.innerHTML = transcriptHTML(p); tr.dataset.ready = "1"; }
    tr.hidden = !open; tog.setAttribute("aria-expanded", String(open)); tog.textContent = open ? "Hide script" : "Show script";
  });
  $(".copy", el).addEventListener("click", async (e) => {
    const text = p.segments.map((s) => `Ep ${s.episode}  ${s.start_tc} - ${s.end_tc}`).join("\n");
    try { await navigator.clipboard.writeText(text); e.target.textContent = "Copied"; } catch { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy timecodes"), 1500);
  });
  const up = $(".vote.up", el), down = $(".vote.down", el);
  const mark = (v) => { up.setAttribute("aria-pressed", String(v === 1)); down.setAttribute("aria-pressed", String(v === -1)); };
  mark(state.votes[p.id]);
  [[up, 1], [down, -1]].forEach(([btn, v]) => btn.addEventListener("click", async () => {
    try {
      await api("/api/feedback", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ job_id: state.job.job_id, promo_id: p.id, vote: v }) });
      state.votes[p.id] = v; mark(v);
    } catch (e) { btn.textContent = "Not saved"; }
  }));
  return el;
}

function transcriptHTML(p) {
  let out = "", offset = 0;
  p.segments.forEach((s, i) => {
    if (i) out += `<p class="cutmark">Cut to episode ${s.episode}, ${s.start_tc}</p>`;
    const segLen = s.end - s.start;
    s.blocks.forEach((b) => {
      const rel = b.start - s.start;
      const cls = (i === 0 && rel < 3) ? "inhook" : (i === p.segments.length - 1 && rel >= segLen - 4.5) ? "incliff" : "";
      const body = b.kind === "DIALOGUE"
        ? `<span class="who">${esc(b.speaker)}</span>${b.emotions.length || b.modifiers.length ? `<span class="emo">${esc([...b.emotions, ...b.modifiers].join(", "))}</span>` : ""}<div>“${esc(b.text)}”</div>`
        : `<span class="x">${esc(b.text)}</span>`;
      out += `<div class="tline k-${b.kind} ${cls}"><span class="t">${tc(offset + rel)}</span><div>${body}</div></div>`;
    });
    offset += segLen;
  });
  return out;
}

// ---------- tabs ----------
function selectTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  ["top", "episodes", "eval"].forEach((t) => ($(`#tab-${t}`).hidden = t !== name));
  $(".legend").hidden = name === "eval";
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));

// ---------- evaluation ----------
$("#eval-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  const f = $("#key-input").files[0];
  if (f) fd.append("answer_key", f);
  const out = $("#eval-out");
  out.innerHTML = `<p class="muted">Comparing…</p>`;
  try {
    const m = await api(`/api/jobs/${state.job.job_id}/evaluate`, { method: "POST", body: fd });
    const e1 = m.per_episode, o = m.overall;
    out.innerHTML = `
      <div class="metrics">
        <div class="metric"><b>${pct(e1["hit@3_strict"])}</b><span>of episodes have an editor's promo in the top 3 picks</span></div>
        <div class="metric"><b>${pct(e1["hit@1_strict"])}</b><span>of episodes: the first pick is the editor's promo</span></div>
        <div class="metric"><b>${pct(e1.hook_start_within_3s)}</b><span>of first picks start within 3 s of the editor's hook</span></div>
        <div class="metric"><b>${pct(e1.cliff_end_within_3s)}</b><span>of first picks end within 3 s of the editor's cut</span></div>
        <div class="metric"><b>${pct(e1.top1_trap_rate)}</b><span>of first picks fall for a trap moment (lower is better)</span></div>
        <div class="metric"><b>${pct(o.precision_strict)}</b><span>of the show's top ${o.k} match an editor's promo</span></div>
      </div>
      ${m.baseline ? `<h2>Compared with a simple rule</h2>
      <p class="muted">The rule: take the last ~35 seconds of each episode, where writers put the cliffhanger.</p>
      <table class="compare"><thead><tr><th>First pick per episode</th><th>PromoFinder</th><th>Simple rule</th></tr></thead><tbody>
        <tr><td>Matches the editor's promo</td><td>${pct(e1["hit@1_strict"])}</td><td>${pct(m.baseline.per_episode["hit@1_strict"])}</td></tr>
        <tr><td>Overlaps it at least a little (IoU ≥ ${m.thresholds.loose_iou})</td><td>${pct(e1["hit@1_loose"])}</td><td>${pct(m.baseline.per_episode["hit@1_loose"])}</td></tr>
        <tr><td>Starts within 3 s of the editor's hook</td><td>${pct(e1.hook_start_within_3s)}</td><td>${pct(m.baseline.per_episode.hook_start_within_3s)}</td></tr>
        <tr><td>Falls for a trap moment</td><td>${pct(e1.top1_trap_rate)}</td><td>${pct(m.baseline.per_episode.top1_trap_rate)}</td></tr>
      </tbody></table>` : ""}
      <p class="muted">A match means the cuts overlap by at least half (IoU ≥ ${m.thresholds.strict_iou}). ${m.gold_count} hand-picked promos, ${m.trap_count} trap moments.</p>
      <table><thead><tr><th>Episode</th><th>First pick overlap</th><th>Best of top 3</th><th>Match in top 3</th></tr></thead><tbody>
      ${m.episode_rows.map((r) => `<tr><td>${r.episode}</td><td>${pct(r.top1_iou)}</td><td>${pct(r.best_of_3_iou)}</td><td class="${r["hit@3"] ? "ok" : "no"}">${r["hit@3"] ? "Yes" : "No"}</td></tr>`).join("")}
      </tbody></table>`;
  } catch (err) { out.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
});

init();
