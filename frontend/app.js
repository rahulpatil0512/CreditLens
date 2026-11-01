/* =====================================================================
   Credit Risk Intelligence — front-end logic (vanilla JS, no build step)
   Talks to the FastAPI backend served from the same origin.
   ===================================================================== */
"use strict";

const API = window.CREDIT_API_BASE || "";

const LABELS = {
  AMT_INCOME_TOTAL: "Annual income",
  AMT_CREDIT: "Loan amount",
  AMT_ANNUITY: "Installment (annuity)",
  AMT_GOODS_PRICE: "Goods price",
  AGE_YEARS: "Age",
  YEARS_EMPLOYED: "Years employed",
  EXT_SOURCE_1: "External score 1",
  EXT_SOURCE_2: "External score 2",
  EXT_SOURCE_3: "External score 3",
  CREDIT_INCOME_RATIO: "Loan-to-income",
  ANNUITY_INCOME_RATIO: "Installment-to-income",
  CREDIT_TERM: "Credit term",
  CNT_CHILDREN: "Children",
  CNT_FAM_MEMBERS: "Family members",
  NAME_CONTRACT_TYPE: "Contract type",
  NAME_EDUCATION_TYPE: "Education",
  NAME_FAMILY_STATUS: "Family status",
  NAME_INCOME_TYPE: "Income type",
  NAME_HOUSING_TYPE: "Housing",
};
const SCENARIO_LABELS = {
  income_drop_20: "Income −20%",
  income_drop_40: "Income −40%",
  unemployment_shock: "Job loss",
  rate_hike: "Rate rise",
};
const NUMERIC_FIELDS = [
  "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
  "AGE_YEARS", "YEARS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
];
const SELECT_FIELDS = ["NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE"];
const EXT_FIELDS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"];
const PRESET_STYLE = {
  "DEMO-LOW-001": { name: "Strong", color: "var(--approve)" },
  "DEMO-MID-002": { name: "Moderate", color: "var(--accent)" },
  "DEMO-REV-004": { name: "Borderline", color: "var(--review)" },
  "DEMO-HIGH-003": { name: "Weak", color: "var(--reject)" },
};
const PRESET_ORDER = ["DEMO-LOW-001", "DEMO-MID-002", "DEMO-REV-004", "DEMO-HIGH-003"];

const state = {
  thresholds: { approve_below: 0.12, reject_above: 0.18 },
  contractType: "Cash loans",
  demo: [],
  modelInfo: null,
  record: null,
  liveTimer: null,
  liveSeq: 0,
};

/* ---------------------------------------------------------------- utils */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const label = (f) => LABELS[f] || f;
const pct = (x, d = 1) => (x == null || isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
const money = (x) => (x == null || isNaN(x) ? "—" : Math.round(x).toLocaleString("en-US"));
const compact = (x) => {
  if (x == null || isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return (x / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (x / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return (x / 1e3).toFixed(1) + "K";
  return money(x);
};
const signed = (x, d = 1) => `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x * 100).toFixed(d)} pp`;
const icon = (path) => `<svg viewBox="0 0 24 24">${path}</svg>`;
const ICONS = {
  why: '<path d="M3 3v18h18"/><path d="M7 14l3-3 3 3 5-5"/>',
  cf: '<path d="M4 4v6h6"/><path d="M20 20v-6h-6"/><path d="M20 10A8 8 0 0 0 6.3 5.3L4 8m16 8l-2.3 2.7A8 8 0 0 1 4 14"/>',
  stress: '<path d="M13 2L3 14h9l-1 8 10-12h-9z"/>',
  text: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  doc: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
};

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  let res;
  try {
    res = await fetch(API + path, opts);
  } catch {
    throw new Error("Cannot reach the API. Is the server running?");
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (Array.isArray(j.detail)) {
        msg = j.detail.map((d) => `${label(d.loc?.[d.loc.length - 1])}: ${d.msg}`).join(" · ");
      } else if (j.detail) msg = j.detail;
    } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "error" ? 6000 : 3200);
}

function decisionFor(pd) {
  const { approve_below, reject_above } = state.thresholds;
  return pd < approve_below ? "Approve" : pd > reject_above ? "Reject" : "Review";
}

/* ---------------------------------------------------------------- theme */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("cri-theme"); } catch { /* storage blocked */ }
  const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");
  $("#themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("cri-theme", next); } catch { /* ignore */ }
  });
}

/* ---------------------------------------------------------------- router */
const loaders = {};
function route() {
  // "#assess/<audit_id>" deep-links a stored decision into the assessor.
  const [name, arg] = (location.hash || "#assess").slice(1).split("/");
  const page = $(`#page-${name}`) ? name : "assess";
  if (page === "assess" && arg) openRecordInAssessor(decodeURIComponent(arg));
  $$(".page").forEach((p) => { p.hidden = p.id !== `page-${page}`; });
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === page));
  if (loaders[page]) loaders[page]();
  window.scrollTo({ top: 0 });
}

/* ---------------------------------------------------------------- status */
async function loadHealth() {
  try {
    const h = await api("/health");
    const m = h.model || {};
    setStatus("model", m.available ? "ok" : "bad",
      m.available ? `Model v${m.model_version} · AUC ${m.test_auc} · ${m.data_source} data` : "Model not trained");
    const r = h.rag || {};
    const sem = (r.embedder || "").startsWith("st:");
    setStatus("rag", r.available ? "ok" : "bad",
      r.available ? `Retrieval · ${sem ? "semantic" : "lexical"}` : "RAG index missing");
    const l = h.llm || {};
    setStatus("llm", l.available ? "ok" : "warn",
      l.available ? `Narrative · LLM (${l.provider})` : "Narrative · template");
  } catch (e) {
    setStatus("model", "bad", "API offline");
    setStatus("rag", "bad", "—");
    setStatus("llm", "bad", "—");
  }
}
function setStatus(key, cls, text) {
  $(`#dot-${key}`).className = `dot ${cls}`;
  $(`#st-${key}`).textContent = text;
}

/* ================================================================ ASSESS */
function buildForm(options) {
  const cats = options.categorical || {};
  for (const f of SELECT_FIELDS) {
    $(`#${f}`).innerHTML = (cats[f] || []).map((v) => `<option>${esc(v)}</option>`).join("");
  }
  const contracts = cats.NAME_CONTRACT_TYPE || ["Cash loans", "Revolving loans"];
  $("#seg-contract").innerHTML = contracts.map((c) =>
    `<button type="button" role="radio" data-v="${esc(c)}">${esc(c)}</button>`).join("");
  $$("#seg-contract button").forEach((b) => b.addEventListener("click", () => {
    setContract(b.dataset.v);
    scheduleLive();
  }));

  $("#extScores").innerHTML = EXT_FIELDS.map((f, i) => `
    <div class="ext-row" id="row-${f}">
      <label for="${f}">Score ${i + 1}</label>
      <input type="range" id="${f}" min="0" max="1" step="0.01" value="0.5">
      <output id="out-${f}">0.50</output>
      <label class="na"><input type="checkbox" id="na-${f}"> n/a</label>
    </div>`).join("");
  for (const f of EXT_FIELDS) {
    $(`#${f}`).addEventListener("input", () => { $(`#out-${f}`).textContent = (+$(`#${f}`).value).toFixed(2); });
    $(`#na-${f}`).addEventListener("change", () => {
      $(`#row-${f}`).classList.toggle("off", $(`#na-${f}`).checked);
      $(`#${f}`).disabled = $(`#na-${f}`).checked;
      scheduleLive();
    });
  }

  const form = $("#applicantForm");
  form.addEventListener("input", () => { updateDerived(); scheduleLive(); });
  form.addEventListener("submit", (e) => { e.preventDefault(); runAssessment(); });
}

function setContract(v) {
  state.contractType = v;
  $$("#seg-contract button").forEach((b) => {
    b.classList.toggle("on", b.dataset.v === v);
    b.setAttribute("aria-checked", b.dataset.v === v);
  });
}

function fillForm(a) {
  for (const f of NUMERIC_FIELDS) if (a[f] != null) $(`#${f}`).value = a[f];
  for (const f of SELECT_FIELDS) if (a[f] != null) $(`#${f}`).value = a[f];
  if (a.NAME_CONTRACT_TYPE) setContract(a.NAME_CONTRACT_TYPE);
  for (const f of EXT_FIELDS) {
    const na = a[f] == null;
    $(`#na-${f}`).checked = na;
    $(`#row-${f}`).classList.toggle("off", na);
    $(`#${f}`).disabled = na;
    if (!na) { $(`#${f}`).value = a[f]; $(`#out-${f}`).textContent = (+a[f]).toFixed(2); }
  }
  updateDerived();
  scheduleLive(0);
}

function readForm() {
  const a = {};
  for (const f of NUMERIC_FIELDS) {
    const raw = $(`#${f}`).value;
    a[f] = raw === "" ? null : Number(raw);
  }
  for (const f of SELECT_FIELDS) a[f] = $(`#${f}`).value;
  a.NAME_CONTRACT_TYPE = state.contractType;
  for (const f of EXT_FIELDS) a[f] = $(`#na-${f}`).checked ? null : Number($(`#${f}`).value);
  return a;
}

function validate(a) {
  let ok = true;
  for (const f of ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]) {
    const bad = !(a[f] > 0);
    $(`#${f}`).classList.toggle("invalid", bad);
    if (bad) ok = false;
  }
  for (const [f, lo] of [["CNT_CHILDREN", 0], ["CNT_FAM_MEMBERS", 1]]) {
    const bad = a[f] == null || a[f] < lo || a[f] > 20;
    $(`#${f}`).classList.toggle("invalid", bad);
    if (bad) ok = false;
  }
  return ok;
}

function updateDerived() {
  const a = readForm();
  $("#v-age").textContent = `${a.AGE_YEARS} yrs`;
  $("#v-emp").textContent = `${a.YEARS_EMPLOYED} yrs`;
  const ok = a.AMT_INCOME_TOTAL > 0;
  $("#r-cir").textContent = ok && a.AMT_CREDIT ? `${(a.AMT_CREDIT / a.AMT_INCOME_TOTAL).toFixed(2)}×` : "—";
  $("#r-air").textContent = ok && a.AMT_ANNUITY ? pct(a.AMT_ANNUITY / a.AMT_INCOME_TOTAL) : "—";
  $("#r-term").textContent = a.AMT_ANNUITY > 0 && a.AMT_CREDIT ? `${(a.AMT_CREDIT / a.AMT_ANNUITY).toFixed(1)}` : "—";
}

function scheduleLive(delay = 300) {
  clearTimeout(state.liveTimer);
  $(".live").classList.add("loading");
  state.liveTimer = setTimeout(liveScore, delay);
}

async function liveScore() {
  const a = readForm();
  if (!validate(a)) {
    $(".live").classList.remove("loading");
    $("#livePd").innerHTML = "—";
    setBadge($("#liveBadge"), null, "Check inputs");
    return;
  }
  const seq = ++state.liveSeq;
  try {
    const r = await api("/predict", {
      applicant: a, include_explanation: false, include_counterfactual: false,
      include_rag: false, include_llm: false, persist: false,
    });
    if (seq !== state.liveSeq) return;
    $("#livePd").innerHTML = `${pct(r.pd)}<small>PD</small>`;
    setBadge($("#liveBadge"), r.decision);
  } catch (e) {
    if (seq !== state.liveSeq) return;
    $("#livePd").textContent = "—";
    setBadge($("#liveBadge"), null, "Error");
  } finally {
    if (seq === state.liveSeq) $(".live").classList.remove("loading");
  }
}

function setBadge(el, decision, fallback = "") {
  el.className = `badge ${decision || ""}`;
  el.textContent = decision || fallback;
}

function buildPresets() {
  const byId = Object.fromEntries(state.demo.map((d) => [d.applicant_id, d]));
  const list = PRESET_ORDER.filter((id) => byId[id]).map((id) => byId[id]);
  $("#presets").innerHTML = list.map((d) => {
    const st = PRESET_STYLE[d.applicant_id] || { name: d.applicant_id, color: "var(--accent)" };
    return `<button class="preset" type="button" data-id="${esc(d.applicant_id)}" title="${esc(d.label_hint || "")}">
      <i style="background:${st.color}"></i>${esc(st.name)}</button>`;
  }).join("");
  $$("#presets .preset").forEach((b) => b.addEventListener("click", () => {
    $$("#presets .preset").forEach((x) => x.classList.toggle("active", x === b));
    fillForm(byId[b.dataset.id]);
  }));
}

async function runAssessment() {
  const a = readForm();
  if (!validate(a)) { toast("Please fix the highlighted fields.", "error"); return; }
  const btn = $("#runBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Assessing…`;
  renderSkeleton();
  try {
    const rec = await api("/predict", { applicant: a });
    state.record = rec;
    history.replaceState(null, "", `#assess/${rec.audit_id}`);
    renderRecord(rec);
  } catch (e) {
    $("#results").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${icon('<path d="M5 12h14M13 6l6 6-6 6"/>')} Run full assessment`;
  }
}

function renderSkeleton() {
  $("#results").innerHTML = `
    <div class="card"><div class="skeleton" style="height:180px"></div></div>
    <div class="card"><div class="skeleton" style="height:22px;width:40%;margin-bottom:16px"></div>
      ${'<div class="skeleton" style="height:18px;margin-bottom:10px"></div>'.repeat(5)}</div>
    <div class="card"><div class="skeleton" style="height:120px"></div></div>`;
}

/* --- result rendering (also used for audit records) --- */
function renderRecord(rec) {
  const out = $("#results");
  out.innerHTML = "";
  out.append(heroCard(rec));
  if (rec.top_factors) out.append(shapCard(rec));
  out.append(cfCard(rec));
  const st = document.createElement("div");
  st.className = "card";
  st.id = "stressCard";
  out.append(st);
  loadStress(rec.audit_id ? { audit_id: rec.audit_id } : { applicant: rec.applicant });
  const row = document.createElement("div");
  row.className = "row-2";
  if (rec.narrative) row.append(narrativeCard(rec));
  if (rec.rag_evidence) row.append(evidenceCard(rec));
  if (row.children.length) out.append(row);
  requestAnimationFrame(() => requestAnimationFrame(animateBars));
}

function animateBars() {
  $$("[data-w]").forEach((el) => { el.style.width = el.dataset.w; });
  $$(".needle[data-rot]").forEach((el) => { el.style.transform = `rotate(${el.dataset.rot}deg)`; });
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function gaugeSVG(pdv, thr) {
  const MAX = 0.4, cx = 140, cy = 128, r = 100;
  const pt = (f) => {
    const t = Math.PI * Math.min(Math.max(f, 0), 1);
    return [cx - r * Math.cos(t), cy - r * Math.sin(t)];
  };
  const arc = (f0, f1, color) => {
    const [x0, y0] = pt(f0), [x1, y1] = pt(f1);
    return `<path d="M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}" stroke="${color}" stroke-width="20" fill="none"/>`;
  };
  const a = thr.approve_below / MAX, b = thr.reject_above / MAX;
  const tick = (f, txt) => {
    // End labels sit under the arc ends; inner labels sit just outside the arc.
    if (f === 0 || f === 1) {
      return `<text class="gauge-tick" x="${cx + (f ? r : -r)}" y="${cy + 22}" text-anchor="middle">${txt}</text>`;
    }
    const t = Math.PI * f, rr = r + 22;
    return `<text class="gauge-tick" x="${(cx - rr * Math.cos(t)).toFixed(1)}" y="${(cy - rr * Math.sin(t) + 3).toFixed(1)}" text-anchor="middle">${txt}</text>`;
  };
  const rot = Math.min(pdv / MAX, 1) * 180;
  return `<svg viewBox="0 0 280 200" role="img" aria-label="PD gauge ${pct(pdv)}">
    ${arc(0, a, "var(--approve)")}${arc(a, b, "var(--review)")}${arc(b, 1, "var(--reject)")}
    ${tick(0, "0%")}${tick(a, pct(thr.approve_below, 0))}${tick(b, pct(thr.reject_above, 0))}${tick(1, "40%+")}
    <g class="needle" style="transform-origin:${cx}px ${cy}px;transform:rotate(0deg)" data-rot="${rot.toFixed(1)}">
      <line x1="${cx}" y1="${cy}" x2="${cx - r + 14}" y2="${cy}" stroke="var(--text)" stroke-width="4" stroke-linecap="round"/>
    </g>
    <circle cx="${cx}" cy="${cy}" r="9" fill="var(--text)"/><circle cx="${cx}" cy="${cy}" r="4" fill="var(--surface)"/>
    <text class="gauge-pd" x="${cx}" y="${cy + 46}" text-anchor="middle">${pct(pdv)}</text>
    <text class="gauge-sub" x="${cx}" y="${cy + 64}" text-anchor="middle">PROBABILITY OF DEFAULT</text>
  </svg>`;
}

const DECISION_COPY = {
  Approve: "Risk is below the approval threshold derived from the break-even PD.",
  Review: "Risk sits in the buffer band around break-even — refer to a credit officer.",
  Reject: "Risk is above the rejection threshold; expected loss outweighs expected margin.",
};

function heroCard(rec) {
  const thr = rec.thresholds || state.thresholds;
  const rk = rec.risk || {};
  const when = rec.created_at ? new Date(rec.created_at).toLocaleString() : "";
  return el(`<div class="card hero ${rec.decision}">
    <div class="gauge">${gaugeSVG(rec.pd, thr)}</div>
    <div class="hero-info">
      <div class="decision-line">
        <span class="decision-big ${rec.decision}">${rec.decision}</span>
        ${rec.audit_id ? `<span class="audit-chip" title="Copy audit ID" onclick="copyId('${esc(rec.audit_id)}')">${icon(ICONS.copy)}${esc(rec.audit_id)}</span>` : ""}
      </div>
      <p class="hero-caption">${DECISION_COPY[rec.decision] || ""}${when ? ` <span class="muted">· ${esc(when)}</span>` : ""}</p>
      <div class="kpis">
        <div class="kpi"><span>Expected loss</span><strong>${money(rk.expected_loss)}</strong><small>PD × LGD × EAD</small></div>
        <div class="kpi"><span>Exposure (EAD)</span><strong>${compact(rk.ead)}</strong><small>assumption</small></div>
        <div class="kpi"><span>LGD</span><strong>${pct(rk.lgd, 0)}</strong><small>assumption</small></div>
        <div class="kpi"><span>Break-even PD</span><strong>${thr.breakeven_pd != null ? pct(thr.breakeven_pd) : "—"}</strong><small>± ${thr.review_buffer != null ? pct(thr.review_buffer, 0) : "—"} review band</small></div>
      </div>
    </div>
  </div>`);
}

window.copyId = (id) => {
  navigator.clipboard?.writeText(id).then(() => toast(`Copied ${id}`, "ok"), () => toast(id));
};

function fmtFeatureValue(f, v) {
  if (v == null) return "missing";
  if (typeof v === "string") return v;
  if (f.startsWith("EXT_SOURCE")) return v.toFixed(2);
  if (f === "ANNUITY_INCOME_RATIO") return pct(v);
  if (f === "CREDIT_INCOME_RATIO") return `${v.toFixed(2)}×`;
  if (f === "AGE_YEARS" || f === "YEARS_EMPLOYED") return `${v} yrs`;
  if (f.startsWith("AMT_")) return money(v);
  return String(v);
}

function shapCard(rec) {
  const factors = rec.top_factors || [];
  const max = Math.max(...factors.map((f) => Math.abs(f.contribution)), 1e-6);
  const rows = factors.map((f) => {
    const up = f.contribution > 0;
    const w = `${(Math.abs(f.contribution) / max) * 50}%`;
    return `<div class="shap-row">
      <div class="shap-label" title="${esc(f.feature)}">${esc(label(f.feature))}<small>${esc(fmtFeatureValue(f.feature, f.value))}</small></div>
      <div class="shap-track"><div class="shap-bar ${up ? "up" : "down"}" data-w="${w}"></div></div>
      <div class="shap-val ${up ? "up" : "down"}">${up ? "+" : "−"}${Math.abs(f.contribution).toFixed(2)}</div>
    </div>`;
  }).join("");
  const reasons = rec.adverse_action_reasons || [];
  return el(`<div class="card">
    <div class="card-head">
      <h3>${icon(ICONS.why)} Why this decision</h3>
      <div class="legend"><span><i style="background:var(--reject)"></i>Raises risk</span><span><i style="background:var(--approve)"></i>Lowers risk</span></div>
    </div>
    <div class="shap">${rows}</div>
    ${rec.decision !== "Approve" && reasons.length ? `<div class="note warn" style="margin-top:14px">${icon(ICONS.info)}
      <div><b>Principal reasons (adverse action):</b> ${reasons.map((r) => esc(label(r.feature))).join(", ")}.</div></div>` : ""}
    <p class="muted" style="margin-top:12px;font-size:12px">SHAP contributions in log-odds from the uncalibrated model; calibration rescales the final PD but doesn't change the ranking.</p>
  </div>`);
}

function cfCard(rec) {
  const cf = rec.counterfactuals;
  let body;
  if (rec.decision === "Approve" || !cf) {
    body = `<div class="note">${icon(ICONS.info)}<div>Already at the target decision — no change to the loan terms is needed.</div></div>`;
  } else if (!cf.feasible || !(cf.counterfactuals || []).length) {
    body = `<div class="note warn">${icon(ICONS.info)}<div>${esc(cf.reason || "No feasible change found.")}
      <div class="muted" style="margin-top:4px">Only actionable loan terms (loan amount, installment, goods price) are varied, by up to 50%.</div></div></div>`;
  } else {
    body = `<div class="cf-list">${cf.counterfactuals.map((c, i) => `
      <div class="cf ${i === 0 ? "best" : ""}">
        ${i === 0 ? '<span class="tag">Smallest change</span>' : ""}
        ${c.changes.map((ch) => `<div class="cf-change"><b>${esc(label(ch.feature))} ${ch.pct_change > 0 ? "+" : ""}${ch.pct_change}%</b>
          <span>${money(ch.from)} → <strong>${money(ch.to)}</strong></span></div>`).join("")}
        <div class="cf-foot"><span>PD <span class="cf-pd">${pct(c.resulting_pd)}</span></span>
          <button class="btn btn-sm" type="button" data-cf="${i}">Apply</button></div>
      </div>`).join("")}</div>`;
  }
  const card = el(`<div class="card">
    <div class="card-head"><h3>${icon(ICONS.cf)} What would change the outcome</h3>
      <span class="hint">Counterfactuals on actionable terms only</span></div>${body}</div>`);
  $$("[data-cf]", card).forEach((b) => b.addEventListener("click", () => {
    const c = cf.counterfactuals[+b.dataset.cf];
    const a = { ...rec.applicant };
    for (const ch of c.changes) a[ch.feature] = ch.to;
    location.hash = "#assess";
    fillForm(a);
    $$("#presets .preset").forEach((x) => x.classList.remove("active"));
    toast("Counterfactual applied — re-running assessment", "ok");
    runAssessment();
  }));
  return card;
}

async function loadStress(body) {
  const card = $("#stressCard");
  if (!card) return;
  card.innerHTML = `<div class="card-head"><h3>${icon(ICONS.stress)} Scenario stress test</h3></div>
    <div class="loading-row"><span class="spinner dark"></span> Running scenarios…</div>`;
  try {
    const r = await api("/stress-test", body);
    renderStress(card, r.applicant);
  } catch (e) {
    card.innerHTML = `<div class="card-head"><h3>${icon(ICONS.stress)} Scenario stress test</h3></div><p class="muted">${esc(e.message)}</p>`;
  }
}

function renderStress(card, st) {
  const thr = state.thresholds;
  const rows = [{ scenario: "baseline", description: "Current application", ...st.baseline, pd_delta: 0 }, ...st.scenarios];
  const scale = Math.max(...rows.map((r) => r.pd), thr.reject_above * 1.25) * 1.08;
  const pos = (v) => `${(v / scale) * 100}%`;
  const html = rows.map((r) => {
    const d = r.pd_delta;
    const cls = Math.abs(d) < 0.0005 ? "flat" : d > 0 ? "up" : "down";
    const isBase = r.scenario === "baseline";
    return `<div class="sc-row">
      <div class="sc-name"><b>${isBase ? "Baseline" : esc(SCENARIO_LABELS[r.scenario] || r.scenario)}</b><small>${esc(r.description)}</small></div>
      <div class="sc-track">
        <div class="sc-bar ${r.decision}" data-w="${pos(r.pd)}"></div>
        <div class="sc-mark" style="left:${pos(thr.approve_below)}"></div>
        <div class="sc-mark" style="left:${pos(thr.reject_above)}"></div>
        ${isBase ? "" : `<div class="sc-mark base" style="left:${pos(st.baseline.pd)}"></div>`}
      </div>
      <div class="sc-meta"><b>${pct(r.pd)}</b>${isBase ? "" : `<span class="delta ${cls}">${signed(d)}</span>`}
        <span class="badge ${r.decision}">${r.decision}</span></div>
    </div>`;
  }).join("");
  const flips = st.scenarios.filter((s) => s.decision_changed).length;
  const rateHike = st.scenarios.find((s) => s.scenario === "rate_hike");
  card.innerHTML = `<div class="card-head"><h3>${icon(ICONS.stress)} Scenario stress test</h3>
      <span class="hint">${flips ? `${flips} scenario${flips > 1 ? "s" : ""} change the decision` : "Decision holds in every scenario"}</span></div>
    <div class="sc">${html}</div>
    <p class="muted" style="margin-top:14px;font-size:12px">Thin markers show the approve / reject thresholds; the dark marker is the baseline PD.
    Feature-shock sensitivity analysis, not regulatory macro stress testing.${rateHike && rateHike.pd_delta < 0
      ? " The rate-rise shock raises the installment, which the model reads as a shorter credit term — a pattern it learned from the data (shorter terms default less), so PD can fall." : ""}</p>`;
  requestAnimationFrame(() => requestAnimationFrame(animateBars));
}

function narrativeCard(rec) {
  const n = rec.narrative || {};
  const lines = (n.text || "").split("\n").filter(Boolean);
  const main = lines.filter((l) => !l.startsWith("Assumption:") && !l.startsWith("Note:"));
  const assumptions = lines.filter((l) => l.startsWith("Assumption:")).map((l) => l.replace(/^Assumption:\s*/, ""));
  const prettify = (s) => s.replace(/\b(EXT_SOURCE_\d|AMT_[A-Z_]+|CREDIT_[A-Z_]+|ANNUITY_[A-Z_]+|NAME_[A-Z_]+|AGE_YEARS|YEARS_EMPLOYED|CNT_[A-Z_]+)\b/g, (m) => label(m));
  const body = n.mode === "llm"
    ? `<div class="narrative">${esc(n.text)}</div>`
    : `<ul class="narrative-lines">${main.map((l) => `<li>${esc(prettify(l))}</li>`).join("")}</ul>`;
  return el(`<div class="card">
    <div class="card-head"><h3>${icon(ICONS.text)} Explanation</h3>
      <span class="badge plain">${n.mode === "llm" ? `LLM · ${esc(n.model || "")}` : "Deterministic template"}</span></div>
    ${body}
    ${assumptions.length ? `<details class="assumptions"><summary>Assumptions (${assumptions.length})</summary>
      <ul>${assumptions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>` : ""}
    <p class="muted" style="margin-top:12px;font-size:12px">Decision-support prototype — not a compliance certification.</p>
  </div>`);
}

function evidenceCard(rec) {
  const ev = rec.rag_evidence || [];
  const mode = rec.rag_meta?.retrieval_mode;
  return el(`<div class="card">
    <div class="card-head"><h3>${icon(ICONS.doc)} Policy evidence</h3>
      ${mode ? `<span class="badge plain">${esc(mode)} retrieval</span>` : ""}</div>
    ${ev.length ? `<div class="evidence">${ev.map(evidenceItem).join("")}</div>`
      : `<p class="muted">${esc(rec.rag_meta?.error || "No evidence retrieved.")}</p>`}
  </div>`);
}

function evidenceItem(e, full = false) {
  const s = Math.max(0, Math.min(1, e.score));
  const text = full ? mdLite(e.text) : `<p>${esc(stripMd(e.snippet ?? e.text))}</p>`;
  return `<div class="ev">
    <div class="ev-head"><div class="ev-src">${icon(ICONS.doc)}<span>${esc(prettySource(e.source))}</span></div>
      <div class="score"><div class="score-bar"><i style="width:${(s * 100).toFixed(0)}%"></i></div>${e.score.toFixed(3)}</div></div>
    <div class="ev-text">${text}</div>
  </div>`;
}
const stripMd = (s) => String(s || "").replace(/(^|\s)#{1,6}\s+/g, "$1").replace(/\*\*|__|`|^>\s*|\s>\s/g, " ").replace(/\s+/g, " ").trim();
const prettySource = (s) => String(s || "").replace(/\.(md|txt|pdf)$/i, "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

/* Minimal, safe markdown: headings, bullets, bold, paragraphs. */
function mdLite(src) {
  const lines = esc(src).split("\n");
  let html = "", inList = false;
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
  for (const raw of lines) {
    const l = raw.trim();
    if (/^[-*]\s+/.test(l)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(l.replace(/^[-*]\s+/, ""))}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (!l) continue;
    if (/^#{1,6}\s/.test(l)) html += `<h4>${inline(l.replace(/^#+\s*/, ""))}</h4>`;
    else html += `<p>${inline(l)}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

/* ============================================================= PORTFOLIO */
async function runPortfolio() {
  const n = +$("#pfSize").value;
  const btn = $("#pfRun");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Running…`;
  $("#pfOut").innerHTML = `<div class="card"><div class="loading-row"><span class="spinner dark"></span>
    Scoring ${n.toLocaleString()} applicants under ${Object.keys(SCENARIO_LABELS).length + 1} scenarios…</div></div>`;
  try {
    const r = await api("/stress-test", { portfolio_sample: n });
    renderPortfolio(r.portfolio);
  } catch (e) {
    $("#pfOut").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run scenarios";
  }
}

function renderPortfolio(p) {
  const b = p.baseline;
  const maxEl = Math.max(...p.scenarios.map((s) => Math.abs(s.expected_loss_increase_pct)), 1);
  const rows = p.scenarios.map((s) => `<tr>
      <td><b>${esc(SCENARIO_LABELS[s.scenario] || s.scenario)}</b><div class="muted" style="font-size:12px">${esc(s.description)}</div></td>
      <td class="num">${pct(s.mean_pd, 2)}</td>
      <td class="num"><span class="delta ${s.mean_pd_delta > 0 ? "up" : s.mean_pd_delta < 0 ? "down" : "flat"}">${signed(s.mean_pd_delta, 2)}</span></td>
      <td class="num">${compact(s.total_expected_loss)}</td>
      <td class="num"><span class="delta ${s.expected_loss_increase_pct > 0 ? "up" : "down"}">${s.expected_loss_increase_pct > 0 ? "+" : ""}${s.expected_loss_increase_pct}%</span></td>
      <td class="num">${pct(s.approve_rate)} <span class="muted" style="font-size:12px">(${signed(s.approve_rate - b.approve_rate)})</span></td>
    </tr>`).join("");
  const bars = p.scenarios.map((s) => `<div class="hbar">
      <div class="hbar-label">${esc(SCENARIO_LABELS[s.scenario] || s.scenario)}</div>
      <div class="hbar-track"><div class="hbar-fill risk" data-w="${(Math.max(s.expected_loss_increase_pct, 0) / maxEl) * 100}%"></div></div>
      <div class="hbar-val">${s.expected_loss_increase_pct > 0 ? "+" : ""}${s.expected_loss_increase_pct}%</div></div>`).join("");
  $("#pfOut").innerHTML = `
    <div class="tiles">
      <div class="tile"><span>Applicants</span><strong>${b.n.toLocaleString()}</strong><small>held-out test split</small></div>
      <div class="tile"><span>Mean PD</span><strong>${pct(b.mean_pd, 2)}</strong><small>baseline</small></div>
      <div class="tile"><span>Expected loss</span><strong>${compact(b.total_expected_loss)}</strong><small>sum of PD × LGD × EAD</small></div>
      <div class="tile"><span>Approve rate</span><strong style="color:var(--approve)">${pct(b.approve_rate)}</strong><small>baseline</small></div>
      <div class="tile"><span>Reject rate</span><strong style="color:var(--reject)">${pct(b.reject_rate)}</strong><small>baseline</small></div>
    </div>
    <div class="row-2" style="grid-template-columns:1.6fr 1fr">
      <div class="card"><div class="card-head"><h3>${icon(ICONS.stress)} Scenario impact</h3></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Scenario</th><th class="num">Mean PD</th><th class="num">Δ PD</th><th class="num">Exp. loss</th><th class="num">Δ EL</th><th class="num">Approve rate</th></tr></thead>
          <tbody>${rows}</tbody></table></div></div>
      <div class="card"><div class="card-head"><h3>${icon(ICONS.why)} Expected-loss increase</h3></div>
        <div class="hbars">${bars}</div>
        <p class="muted" style="margin-top:14px;font-size:12px">LGD and EAD are labelled assumptions. Scenario sensitivity, not regulatory macro stress testing.</p></div>
    </div>`;
  requestAnimationFrame(() => requestAnimationFrame(animateBars));
}

/* ================================================================ POLICY */
const RAG_SUGGESTIONS = [
  "When must an application be referred to manual review?",
  "What are the affordability requirements?",
  "Which reasons must be given to a declined applicant?",
  "How are protected attributes handled?",
  "Model governance and monitoring expectations",
];
function buildPolicy() {
  $("#ragChips").innerHTML = RAG_SUGGESTIONS.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
  $$("#ragChips .chip").forEach((c) => c.addEventListener("click", () => { $("#ragQuery").value = c.textContent; runRag(); }));
  $("#ragForm").addEventListener("submit", (e) => { e.preventDefault(); runRag(); });
}
async function runRag() {
  const q = $("#ragQuery").value.trim();
  if (q.length < 3) { toast("Type at least 3 characters.", "error"); return; }
  $("#ragOut").innerHTML = `<div class="card"><div class="loading-row"><span class="spinner dark"></span> Searching…</div></div>`;
  try {
    const r = await api("/rag/query", { query: q, top_k: +$("#ragK").value });
    $("#ragOut").innerHTML = `<div class="card">
      <div class="card-head"><h3>${icon(ICONS.doc)} ${r.results.length} passages</h3>
        <span class="badge plain">${esc(r.retrieval_mode)} · ${esc(r.embedder)}</span></div>
      <div class="evidence">${r.results.map((x) => evidenceItem(x, true)).join("")}</div></div>`;
  } catch (e) {
    $("#ragOut").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
  }
}

/* ================================================================= AUDIT */
async function loadAudit() {
  $("#auditOut").innerHTML = `<div class="card"><div class="loading-row"><span class="spinner dark"></span> Loading…</div></div>`;
  try {
    const r = await api("/audit?limit=50");
    renderAudit(r);
  } catch (e) {
    $("#auditOut").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
  }
}

function donut(counts) {
  const order = [["Approve", "var(--approve)"], ["Review", "var(--review)"], ["Reject", "var(--reject)"]];
  const total = order.reduce((s, [k]) => s + (counts[k] || 0), 0) || 1;
  const C = 2 * Math.PI * 48;
  let off = 0;
  const segs = order.map(([k, c]) => {
    const len = ((counts[k] || 0) / total) * C;
    const s = `<circle cx="65" cy="65" r="48" fill="none" stroke="${c}" stroke-width="16"
      stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 65 65)"/>`;
    off += len;
    return s;
  }).join("");
  return `<div class="donut-wrap"><div class="donut"><svg viewBox="0 0 130 130">
      <circle cx="65" cy="65" r="48" fill="none" stroke="var(--surface-2)" stroke-width="16"/>${segs}
      <text x="65" y="63" text-anchor="middle" style="font-size:22px;font-weight:800;fill:var(--text)">${total === 1 && !counts.Approve && !counts.Review && !counts.Reject ? 0 : total}</text>
      <text x="65" y="80" text-anchor="middle" style="font-size:10px;fill:var(--muted);font-weight:600">DECISIONS</text></svg></div>
    <div class="donut-legend">${order.map(([k, c]) => `<div><i style="background:${c}"></i>${k}<span>${counts[k] || 0}</span></div>`).join("")}</div></div>`;
}

function renderAudit(r) {
  const by = r.stats.by_decision || {};
  const counts = Object.fromEntries(Object.entries(by).map(([k, v]) => [k, v.n]));
  const rows = r.records.map((x) => `<tr class="clickable" data-id="${esc(x.audit_id)}">
      <td>${esc(new Date(x.created_at).toLocaleString())}</td>
      <td class="mono">${esc(x.audit_id)}</td>
      <td><span class="badge ${x.decision}">${esc(x.decision)}</span></td>
      <td class="num">${pct(x.pd, 2)}</td>
      <td class="num">${money(x.amt_credit)}</td>
      <td class="num">${money(x.amt_income)}</td>
      <td class="num">${money(x.expected_loss)}</td>
      <td class="mono muted">v${esc(x.model_version)}</td></tr>`).join("");
  $("#auditOut").innerHTML = `
    <div class="row-2" style="grid-template-columns:minmax(280px,1fr) 2fr;margin-bottom:18px">
      <div class="card"><div class="card-head"><h3>${icon(ICONS.shield)} Decision mix</h3></div>${donut(counts)}</div>
      <div class="tiles" style="margin:0">
        <div class="tile"><span>Total decisions</span><strong>${r.stats.total}</strong><small>stored in SQLite</small></div>
        ${["Approve", "Review", "Reject"].map((k) => `<div class="tile"><span>${k} · mean PD</span>
          <strong>${by[k] ? pct(by[k].mean_pd, 2) : "—"}</strong><small>${by[k]?.n || 0} decisions</small></div>`).join("")}
      </div>
    </div>
    <div class="card"><div class="card-head"><h3>Recent decisions</h3><span class="hint">Click a row for the full record</span></div>
      ${r.records.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Time</th><th>Audit ID</th><th>Decision</th><th class="num">PD</th><th class="num">Loan</th><th class="num">Income</th><th class="num">Exp. loss</th><th>Model</th></tr></thead>
        <tbody>${rows}</tbody></table></div>` : `<p class="muted">No decisions yet — run an assessment first.</p>`}
    </div>`;
  $$("#auditOut tr.clickable").forEach((tr) => tr.addEventListener("click", () => openAudit(tr.dataset.id)));
}

async function openAudit(id) {
  try {
    const rec = await api(`/audit/${encodeURIComponent(id)}`);
    showDrawer(rec);
  } catch (e) { toast(e.message, "error"); }
}

function showDrawer(rec) {
  $("#drawerTitle").innerHTML = `Decision <span class="mono muted" style="font-size:13px">${esc(rec.audit_id)}</span>`;
  const a = rec.applicant || {};
  const keys = [...NUMERIC_FIELDS, ...EXT_FIELDS, "NAME_CONTRACT_TYPE", ...SELECT_FIELDS];
  $("#drawerBody").innerHTML = `
    <div class="card" style="display:flex;align-items:center;justify-content:space-between;gap:12px">
      <div><div class="decision-big ${rec.decision}" style="font-size:24px">${rec.decision}</div>
        <div class="muted">PD ${pct(rec.pd, 2)} · EL ${money(rec.risk?.expected_loss)} · ${esc(new Date(rec.created_at).toLocaleString())}</div></div>
      <button class="btn btn-primary" id="drawerLoad" type="button">Open in assessor</button>
    </div>
    <div class="card"><div class="card-head"><h3>Applicant</h3></div>
      <div class="kv">${keys.map((k) => `<div><span>${esc(label(k))}</span><b>${esc(fmtFeatureValue(k, a[k]))}</b></div>`).join("")}</div></div>
    <div class="card"><div class="card-head"><h3>Provenance</h3></div>
      <div class="kv">
        <div><span>Model version</span><b>${esc(rec.model_version)}</b></div>
        <div><span>Data source</span><b>${esc(rec.data_source)}</b></div>
        <div><span>Narrative</span><b>${esc(rec.narrative?.mode || "—")}</b></div>
        <div><span>Retrieval</span><b>${esc(rec.rag_meta?.retrieval_mode || "—")}</b></div>
      </div>
      <details style="margin-top:12px"><summary>Raw JSON record</summary><pre class="json">${esc(JSON.stringify(rec, null, 2))}</pre></details>
    </div>`;
  $("#drawerLoad").addEventListener("click", () => {
    closeDrawer();
    location.hash = `#assess/${rec.audit_id}`;
  });
  $("#drawerBackdrop").hidden = false;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
}
async function openRecordInAssessor(id) {
  if (state.record?.audit_id === id) return;
  renderSkeleton();
  try {
    const rec = await api(`/audit/${encodeURIComponent(id)}`);
    state.record = rec;
    fillForm(rec.applicant || {});
    $$("#presets .preset").forEach((x) => x.classList.remove("active"));
    renderRecord(rec);
  } catch (e) {
    $("#results").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
    toast(e.message, "error");
  }
}

function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#drawerBackdrop").hidden = true;
}

/* ================================================================= MODEL */
async function loadModel() {
  if (state.modelInfo) { renderModel(state.modelInfo); return; }
  $("#modelOut").innerHTML = `<div class="card"><div class="loading-row"><span class="spinner dark"></span> Loading…</div></div>`;
  try {
    state.modelInfo = await api("/model/info");
    renderModel(state.modelInfo);
  } catch (e) {
    $("#modelOut").innerHTML = `<div class="card empty small"><p>${esc(e.message)}</p></div>`;
  }
}

function renderModel(m) {
  const mt = m.metrics || {};
  const thr = m.thresholds || state.thresholds;
  const imp = (m.global_importance || []).slice(0, 12);
  const maxImp = Math.max(...imp.map((i) => i.importance), 1e-6);
  const fair = m.fairness_disparity || {};
  const groups = Object.entries(fair.groups || {});
  const MAXB = 0.4;
  const a = (thr.approve_below / MAXB) * 100, b = (thr.reject_above / MAXB) * 100;
  const ratio = fair.approval_rate_ratio_min_over_max;
  $("#modelOut").innerHTML = `
    <div class="tiles">
      <div class="tile"><span>Test ROC-AUC</span><strong>${mt.test_auc ?? "—"}</strong><small>LogReg baseline ${m.baseline_logreg_test_auc ?? "—"}</small></div>
      <div class="tile"><span>KS statistic</span><strong>${mt.test_ks ?? "—"}</strong><small>max separation</small></div>
      <div class="tile"><span>PR-AUC</span><strong>${mt.test_pr_auc ?? "—"}</strong><small>base rate ${pct(mt.base_rate, 2)}</small></div>
      <div class="tile"><span>Brier score</span><strong>${mt.brier_calibrated ?? "—"}</strong><small>raw ${mt.brier_raw ?? "—"}</small></div>
      <div class="tile"><span>Mean PD</span><strong>${pct(mt.mean_pd_calibrated, 2)}</strong><small>raw ${pct(mt.mean_pd_raw, 1)} → calibrated</small></div>
    </div>
    <div class="row-2" style="margin-bottom:18px">
      <div class="card"><div class="card-head"><h3>${icon(ICONS.why)} Global feature importance</h3><span class="hint">share of split gain</span></div>
        <div class="hbars">${imp.map((i) => `<div class="hbar"><div class="hbar-label" title="${esc(i.feature)}">${esc(label(i.feature))}</div>
          <div class="hbar-track"><div class="hbar-fill" data-w="${(i.importance / maxImp) * 100}%"></div></div>
          <div class="hbar-val">${pct(i.importance)}</div></div>`).join("")}</div></div>
      <div class="card calib"><div class="card-head"><h3>${icon(ICONS.stress)} Calibration</h3><span class="hint">${esc(m.calibration_method)} on a held-out split</span></div>
        ${m.calibration_plot_available ? `<img src="${API}/model/calibration-plot" alt="Calibration curve: raw vs calibrated PD" loading="lazy">` : `<p class="muted">Plot not available.</p>`}</div>
    </div>
    <div class="row-2" style="margin-bottom:18px">
      <div class="card"><div class="card-head"><h3>${icon(ICONS.shield)} Decision policy</h3><span class="hint">${esc(thr.mode)}</span></div>
        <div class="band-wrap">
          <span class="band-tick" style="left:${a}%">${pct(thr.approve_below)}</span>
          <span class="band-tick" style="left:${b}%">${pct(thr.reject_above)}</span>
          <div class="band"><div style="width:${a}%;background:var(--approve)">Approve</div>
            <div style="width:${b - a}%;background:var(--review)">Review</div>
            <div style="flex:1;background:var(--reject)">Reject</div></div>
          ${thr.breakeven_pd != null ? `<span class="band-tick bottom" style="left:${(thr.breakeven_pd / MAXB) * 100}%">break-even ${pct(thr.breakeven_pd)}</span>` : ""}
        </div>
        <div class="kv">
          <div><span>LGD (assumed)</span><b>${pct(m.risk_assumptions?.lgd, 0)}</b></div>
          <div><span>Annual margin (assumed)</span><b>${pct(m.risk_assumptions?.annual_margin, 0)}</b></div>
          <div><span>EAD proxy</span><b>${esc(label(m.risk_assumptions?.ead_source))}</b></div>
          <div><span>Review buffer</span><b>± ${thr.review_buffer != null ? pct(thr.review_buffer, 0) : "—"}</b></div>
        </div>
        <p class="muted" style="margin-top:12px;font-size:12px">Break-even PD* = margin / (margin + LGD). Single-period simplification.</p></div>
      <div class="card"><div class="card-head"><h3>${icon(ICONS.info)} Fairness check</h3><span class="hint">audit only — never a model input</span></div>
        ${groups.map(([attr, g]) => `<div class="table-wrap"><table>
          <thead><tr><th>${esc(attr)}</th><th class="num">n</th><th class="num">Approval rate</th><th class="num">Mean PD</th><th class="num">Observed default</th></tr></thead>
          <tbody>${Object.entries(g).map(([k, v]) => `<tr><td><b>${esc(k)}</b></td><td class="num">${v.n.toLocaleString()}</td>
            <td class="num">${pct(v.approval_rate)}</td><td class="num">${pct(v.mean_pd, 2)}</td><td class="num">${pct(v.observed_default_rate, 2)}</td></tr>`).join("")}</tbody></table></div>`).join("")}
        ${ratio != null ? `<div class="note ${ratio < 0.8 ? "warn" : ""}" style="margin-top:12px">${icon(ICONS.info)}<div>Approval-rate ratio (min / max) is <b>${ratio}</b>
          ${ratio >= 0.8 ? "— above the common four-fifths (0.8) screening heuristic." : "— below the four-fifths (0.8) screening heuristic; investigate."}
          <div class="muted" style="font-size:12px;margin-top:2px">At a reference cutoff of ${pct(fair.reference_approve_cutoff, 0)} PD.</div></div></div>` : ""}</div>
    </div>
    <div class="card"><div class="card-head"><h3>${icon(ICONS.doc)} Training details</h3></div>
      <div class="kv">
        <div><span>Primary model</span><b>${esc(m.primary_model)}</b></div>
        <div><span>Data source</span><b>${esc(m.data_source)}</b></div>
        <div><span>Model version</span><b>${esc(m.model_version)}</b></div>
        <div><span>Trained</span><b>${esc(new Date(m.trained_at_utc).toLocaleString())}</b></div>
        <div><span>Features</span><b>${(m.numeric_features || []).length} numeric · ${(m.categorical_features || []).length} categorical</b></div>
        <div><span>Random seed</span><b>${esc(m.random_seed)}</b></div>
      </div>
      <p class="muted" style="margin-top:12px;font-size:12px">${esc(m.notes || "")}</p></div>`;
  requestAnimationFrame(() => requestAnimationFrame(animateBars));
}

/* ================================================================== INIT */
async function init() {
  initTheme();
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  $("#pfRun").addEventListener("click", runPortfolio);
  $("#auditRefresh").addEventListener("click", loadAudit);
  $("#auditLookup").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("#auditId").value.trim();
    if (id) openAudit(id);
  });
  buildPolicy();
  loaders.audit = loadAudit;
  loaders.model = loadModel;
  window.addEventListener("hashchange", route);

  loadHealth();
  try {
    const [options, demo, info] = await Promise.all([
      api("/meta/options"), api("/meta/demo-applicants"), api("/model/info").catch(() => null),
    ]);
    if (info) { state.modelInfo = info; state.thresholds = info.thresholds; }
    buildForm(options);
    state.demo = demo;
    buildPresets();
    const first = demo.find((d) => d.applicant_id === "DEMO-MID-002") || demo[0];
    if (first) {
      fillForm(first);
      $(`#presets .preset[data-id="${first.applicant_id}"]`)?.classList.add("active");
    }
  } catch (e) {
    toast(e.message, "error");
    $("#results").innerHTML = `<div class="card empty"><h2>Backend not ready</h2><p>${esc(e.message)}</p>
      <p class="muted" style="margin-top:8px">Train the model first: <span class="mono">python scripts/train_model.py</span></p></div>`;
  }
  route();
}

document.addEventListener("DOMContentLoaded", init);
