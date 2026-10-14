"use strict";

const S = { meta: null, slate: null, players: [], track: null, byId: new Map(), slateById: new Map(), filter: "all" };
const view = document.getElementById("view");

const MARKET = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "3-pointers", stl: "Steals", blk: "Blocks", pra: "Pts + Reb + Ast",
  moneyline: "Moneyline", spread: "Spread", total: "Total points" };
const STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "pra"];

/* ---------------------------------------------------------------- utils */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f1 = (x) => (x == null ? "–" : Number(x).toFixed(1));
const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
const sign = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } } };
S.fmt = store.get("oddsFmt", "decimal");
const toDecimal = (am) => (am > 0 ? 1 + am / 100 : 1 + 100 / -am);
const price = (p) => (p == null || !Number.isFinite(+p) ? "–" : S.fmt === "decimal" ? toDecimal(+p).toFixed(2) : p > 0 ? "+" + p : String(p));
/* Accepts what people type from a bookmaker: decimal (1.85) or American (-110, +120). Returns American. */
function parseOdds(v) {
  const x = parseFloat(String(v).replace(",", "."));
  if (!Number.isFinite(x)) return null;
  if (Math.abs(x) >= 100) return x;
  if (x > 1 && x < 100) return x >= 2 ? Math.round((x - 1) * 100) : Math.round(-100 / (x - 1));
  return null;
}
const normCdf = (z) => { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; };
const time = (iso) => { try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); } catch { return ""; } };
const day = (iso) => { try { return new Date(iso).toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }); } catch { return iso; } };
const payout = (p) => (p > 0 ? p / 100 : 100 / -p);
const implied = (p) => (p > 0 ? 100 / (p + 100) : -p / (-p + 100));
const toAmerican = (prob) => { prob = Math.min(Math.max(prob, 0.01), 0.99); return prob >= 0.5 ? Math.round(-100 * prob / (1 - prob)) : Math.round(100 * (1 - prob) / prob); };

/* Negative binomial probabilities: same maths as the Python model */
function lgamma(x) {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function countProbs(mu, v, line) {
  mu = Math.max(mu, 0.05); v = Math.max(v, mu * 1.02);
  const r = mu * mu / (v - mu), p = r / (r + mu);
  const pmf = (k) => Math.exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1) + r * Math.log(p) + k * Math.log(1 - p));
  const cdf = (k) => { let s = 0; for (let i = 0; i <= k; i++) s += pmf(i); return Math.min(s, 1); };
  if (Number.isInteger(line)) { const push = pmf(line), under = line > 0 ? cdf(line - 1) : 0; return { over: 1 - under - push, under, push }; }
  const under = cdf(Math.floor(line)); return { over: 1 - under, under, push: 0 };
}
const variance = (stat, mu) => { const vp = S.meta?.var?.[stat]; return vp ? vp.a * Math.pow(Math.max(mu, 0.05), vp.b) : mu * 1.5; };

async function load(name) { const r = await fetch(`data/${name}.json`, { cache: "no-cache" }); if (!r.ok) throw new Error(name); return r.json(); }

/* ---------------------------------------------------------------- number line */
function numberLine(mu, sd, line, side) {
  const lo = Math.max(0, Math.min(mu - 2.4 * sd, line - sd)), hi = Math.max(mu + 2.4 * sd, line + sd);
  const x = (v) => ((v - lo) / (hi - lo)) * 100;
  const L = x(line), M = x(mu);
  const over = side === "over" || side === "home" ? `left:${L}%;right:0` : `left:0;width:${L}%`;
  return `<div class="nline" role="img" aria-label="Projection ${f1(mu)} against line ${line}">
    <div class="track"></div>
    <div class="band" style="left:${Math.max(0, x(mu - sd))}%;width:${Math.min(100, x(mu + sd)) - Math.max(0, x(mu - sd))}%"></div>
    <div class="over" style="${over}"></div>
    <div class="tick" style="left:${L}%"><span>Line ${line}</span></div>
    <div class="dot" style="left:${M}%"><span>${f1(mu)}</span></div>
  </div>`;
}

/* ---------------------------------------------------------------- data shaping */
function allOptions() {
  const out = [];
  for (const g of S.slate?.games || []) {
    if (g.started) continue;  // already tipped off: can't bet pre-game
    const matchup = `${g.away.abbr} @ ${g.home.abbr}`;
    for (const o of g.options || []) out.push({ ...o, g, matchup, kind: "game" });
    for (const p of g.players || []) for (const o of p.props || []) out.push({ ...o, g, p, matchup, kind: "prop" });
  }
  return out;
}

function betText(o, line = o.line) {
  const g = o.g;
  if (o.market === "moneyline") return `${(o.side === "home" ? g.home : g.away).abbr} to win`;
  if (o.market === "spread") return `${(o.side === "home" ? g.home : g.away).abbr} ${sign(line)}`;
  if (o.market === "total") return `${o.side === "over" ? "Over" : "Under"} ${line}`;
  return `${o.side === "over" ? "Over" : "Under"} ${line} ${MARKET[o.market].toLowerCase()}`;
}
function betwayLine(o) {
  if (o.bw_price == null) return "";
  return `<p class="bw small">Betway: ${esc(betText(o, o.bw_line ?? o.line))} at <b>${price(o.bw_price)}</b>,
    value <b class="${o.bw_ev > 0 ? "pos" : "neg"}">${sign(o.bw_ev, 2)}u</b></p>`;
}

function optionLine(o) {
  const gm = S.meta?.game_model || {};
  if (o.kind === "prop") return numberLine(o.p.proj[o.market], Math.sqrt(o.p.var[o.market]), o.line, o.side);
  if (o.market === "total") return numberLine(o.g.pred.total, gm.total?.sd || 18, o.line, o.side);
  if (o.market === "spread") {
    // show home margin vs the margin needed to cover
    const need = -(o.side === "home" ? o.line : -o.line);
    return numberLine(o.g.pred.margin, gm.margin?.sd || 13, need, "home").replace(`Line ${need}`, `${o.g.home.abbr} by ${f1(need)}`);
  }
  return "";
}

/* A logged pick (from track.json) in words */
function pickLabel(r) {
  const [away, home] = String(r.matchup || "").split(" @ ");
  const team = r.side === "home" ? home : away;
  if (r.market === "moneyline") return `${team} to win`;
  if (r.market === "spread") return `${team} ${sign(r.line)}`;
  const ou = r.side === "over" ? "Over" : "Under";
  return r.market === "total" ? `${ou} ${r.line} total points` : `${ou} ${r.line} ${MARKET[r.market].toLowerCase()}`;
}


/* ---------------------------------------------------------------- reasons behind a prediction */
function whyBlock(w, unitWord, title) {
  if (!w) return "";
  const rows = w.items.map((it) => `<li><b class="${it.amt > 0 ? "pos" : "neg"}">${sign(it.amt)}</b>
      <span><strong>${esc(it.label)}</strong><br><span class="muted small">${esc(it.detail)}</span></span></li>`).join("");
  return `<div class="why"><h3>${esc(title)}</h3>
    <ol class="why-list">
      <li class="start"><b>${f1(w.level)}</b><span><strong>Starting point</strong><br><span class="muted small">His usual level from season and recent averages</span></span></li>
      ${rows}
      ${Math.abs(w.other) >= 0.05 ? `<li><b class="${w.other > 0 ? "pos" : "neg"}">${sign(w.other)}</b><span><strong>Smaller factors</strong><br><span class="muted small">Everything else combined</span></span></li>` : ""}
      <li class="end"><b>${f1(w.pred)}</b><span><strong>Projection</strong> ${esc(unitWord)}</span></li>
    </ol></div>`;
}
function shortWhy(w) {
  if (!w || !w.items?.length) return "";
  return `<p class="small why-short">Why: ${w.items.slice(0, 3).map((it) => `<span class="${it.amt > 0 ? "pos" : "neg"}">${sign(it.amt)}</span> ${esc(it.label.toLowerCase())}`).join(", ")}</p>`;
}
function gameWhy(g) {
  if (!g.why) return "";
  const fav = g.pred.margin >= 0 ? g.home.abbr : g.away.abbr;
  const m = { ...g.why.margin }, t = g.why.total;
  return `<div class="why-games">
    ${whyBlock(m, `(${g.home.abbr} margin; negative means ${g.away.abbr} win)`, `Why ${fav} by ${f1(Math.abs(g.pred.margin))}`).replace("His usual level from season and recent averages", "Average home-court edge")}
    ${whyBlock(t, "total points", `Why ${f1(g.pred.total)} total points`).replace("His usual level from season and recent averages", "A typical NBA game")}
  </div>`;
}

/* ---------------------------------------------------------------- views */
function edgeCard(o) {
  const who = o.kind === "prop"
    ? `<h3>${esc(o.p.name)}</h3><p class="muted small">${esc(o.matchup)}, ${time(o.g.start)}${o.p.inj ? ` <span class="tag inj">${esc(o.p.inj)}</span>` : ""}</p>`
    : `<h3>${esc(o.g.away.name)} at ${esc(o.g.home.name)}</h3><p class="muted small">${time(o.g.start)}</p>`;
  return `<article class="panel edge ${o.flagged ? "hot" : ""}">
    <div class="edge-top"><div class="edge-who">${who}</div>
      <div class="edge-bet">${esc(betText(o))}<small>Best price ${price(o.price)} at ${esc(o.book || "a book")}</small></div></div>
    <div class="edge-stats">
      <span>Model <b>${pct(o.model_prob)}</b></span>
      <span>Book <b>${pct(o.fair_prob)}</b></span>
      <span>Edge <b class="${o.edge > 0 ? "pos" : "neg"}">${sign(o.edge * 100)}%</b></span>
      <span>Value <b class="${o.ev > 0 ? "pos" : "neg"}">${sign(o.ev, 2)}u</b></span>
    </div>
    ${optionLine(o)}
    ${o.kind === "prop" ? shortWhy(o.p.why?.[o.market]) : ""}
    ${betwayLine(o)}
  </article>`;
}

function renderEdges() {
  const opts = allOptions();
  if (!S.slate?.games?.length) {
    view.innerHTML = `<div class="empty"><h2>No games scheduled</h2><p>Projections appear here the day before games start.</p></div>`;
    return;
  }
  if (!opts.length) {
    view.innerHTML = `<div class="empty"><h2>No sportsbook lines yet</h2>
      <p>Lines load once a day when an Odds API key is set. Until then, open <a href="#games">Games</a> or <a href="#players">Players</a> and type in any line to check it against the projection.</p></div>`;
    return;
  }
  const f = S.filter;
  const pool = opts.filter((o) => f === "all" || (f === "props" ? o.kind === "prop" : o.kind === "game"));
  const flagged = pool.filter((o) => o.flagged).sort((a, b) => b.ev - a.ev);
  const check = pool.filter((o) => o.check).sort((a, b) => b.ev - a.ev);
  const near = pool.filter((o) => !o.flagged && !o.check && o.ev != null).sort((a, b) => b.edge - a.edge).slice(0, 6);
  const chips = [["all", "All"], ["props", "Player props"], ["game", "Game lines"]]
    .map(([k, l]) => `<button class="chip" data-filter="${k}" aria-pressed="${f === k}">${l}</button>`).join("");
  view.innerHTML = `
    <div class="section-head"><div><h2>${flagged.length ? `${flagged.length} edge${flagged.length > 1 ? "s" : ""} today` : "No strong edges today"}</h2>
      <p class="muted small">Model probability at least ${Math.round((S.meta?.edge_threshold || 0.05) * 100)} points above the book's, sorted by expected value per unit staked.</p></div>
      <div class="chips">${chips}</div></div>
    <div class="edges">${flagged.map(edgeCard).join("") || `<div class="empty"><p>Nothing clears the bar in this filter. Below are the closest calls.</p></div>`}</div>
    ${check.length ? `<div class="block"><div class="section-head"><div><h2>Check the news first</h2>
      <p class="muted small">The model disagrees with the book by a lot, the player is questionable, or has little history. Usually the book knows something (injury, role change). Not flagged.</p></div></div>
      <div class="edges">${check.slice(0, 10).map(edgeCard).join("")}</div></div>` : ""}
    ${near.length ? `<div class="block"><div class="section-head"><h2>Closest calls</h2></div><div class="edges">${near.map(edgeCard).join("")}</div></div>` : ""}`;
  view.querySelectorAll("[data-filter]").forEach((b) => b.onclick = () => { S.filter = b.dataset.filter; renderEdges(); });
}

function gameCard(g) {
  const hp = g.pred.home_win, L = g.lines || {};
  const fav = g.pred.margin >= 0 ? g.home.abbr : g.away.abbr;
  const bookSpread = L.spread ? `${g.home.abbr} ${sign(L.spread.point)}` : "No line";
  const modelSpread = `${fav} ${sign(-Math.abs(g.pred.margin))}`;
  const bookWin = L.ml ? pct(L.ml.sides.home.fair) : "–";
  const statRows = (side) => g.players.filter((p) => p.side === side).map((p) => {
    const flagged = (s) => (p.props || []).find((o) => o.market === s && o.flagged);
    const hit = (s) => (flagged(s) ? "prop-hit" : "");
    const cell = (s) => { const o = flagged(s); return f1(p.proj[s]) + (o ? `<small class="pick">${o.side === "over" ? "O" : "U"} ${o.line}</small>` : ""); };
    return `<tr class="clickable" data-player="${esc(p.id)}"><td>${esc(p.name)} ${p.inj ? `<span class="tag inj">${esc(p.inj)}</span>` : ""}${p.low_sample ? ` <span class="tag">new</span>` : ""}</td>
      <td>${f1(p.proj.min)}</td><td class="${hit("pts")}">${cell("pts")}</td><td class="${hit("reb")}">${cell("reb")}</td>
      <td class="${hit("ast")}">${cell("ast")}</td><td class="${hit("fg3m")}">${cell("fg3m")}</td><td class="${hit("pra")}">${cell("pra")}</td></tr>`;
  }).join("");
  const table = (side, team) => `<h3 style="margin-top:12px">${esc(team.name)}</h3>
    <div class="tbl-wrap"><table><thead><tr><th>Player</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th><th>3PM</th><th>PRA</th></tr></thead>
    <tbody>${statRows(side) || `<tr><td colspan="7" class="muted">No projections</td></tr>`}</tbody></table></div>`;
  const miss = (n) => (n >= 12 ? `<span class="tag inj">${Math.round(n)} min out</span>` : "");
  return `<article class="panel game">
    <button class="game-head" aria-expanded="false" data-game="${esc(g.id)}">
      <span class="team"><span class="abbr">${esc(g.away.abbr)}</span><span class="nm">${esc(g.away.name)} ${miss(g.missing.away)}</span></span>
      <span class="score">${Math.round(g.pred.away_score)}–${Math.round(g.pred.home_score)}<small>${g.started ? "Under way" : time(g.start)}</small></span>
      <span class="team home"><span class="abbr">${esc(g.home.abbr)}</span><span class="nm">${miss(g.missing.home)} ${esc(g.home.name)}</span></span>
    </button>
    <div class="winbar" title="Win probability"><i style="width:${(1 - hp) * 100}%;background:var(--book)"></i><i style="width:${hp * 100}%;background:var(--model)"></i></div>
    <div class="game-lines">
      <div><span class="muted">Win chance</span><b>${g.home.abbr} ${pct(hp)}</b><span class="muted small">Book ${bookWin}</span></div>
      <div><span class="muted">Spread</span><b>${esc(modelSpread)}</b><span class="muted small">Book ${esc(bookSpread)}</span></div>
      <div><span class="muted">Total</span><b>${f1(g.pred.total)}</b><span class="muted small">Book ${L.total ? L.total.point : "–"}</span></div>
    </div>
    <div class="game-body" hidden>${gameWhy(g)}${table("away", g.away)}${table("home", g.home)}
      <p class="muted small" style="margin-top:10px">Highlighted cells have a flagged prop. Tap a player to check any line.</p></div>
  </article>`;
}

function renderGames() {
  const games = S.slate?.games || [];
  if (!games.length) { view.innerHTML = `<div class="empty"><h2>No games scheduled</h2><p>Check back the day before the next game.</p></div>`; return; }
  view.innerHTML = `<div class="section-head"><div><h2>${games.length} game${games.length > 1 ? "s" : ""}</h2>
    <p class="muted small">Predicted score, away team first. Tap a game for player projections.</p></div></div>
    <div class="games">${games.map(gameCard).join("")}</div>`;
  view.querySelectorAll(".game-head").forEach((b) => b.onclick = () => {
    const body = b.parentElement.querySelector(".game-body"); const open = body.hidden;
    body.hidden = !open; b.setAttribute("aria-expanded", String(open));
  });
  view.querySelectorAll("[data-player]").forEach((r) => r.onclick = () => (location.hash = "p/" + r.dataset.player));
}

function renderPlayers(q = "") {
  const ql = q.trim().toLowerCase();
  const inSlate = (p) => (S.slateById.has(p.id) ? 1 : 0);
  let list = S.players.filter((p) => !ql || p.name.toLowerCase().includes(ql) || (p.team || "").toLowerCase() === ql);
  list = list.sort((a, b) => inSlate(b) - inSlate(a) || (b.avg.min || 0) - (a.avg.min || 0)).slice(0, ql ? 80 : 50);
  view.innerHTML = `<div class="section-head"><div><h2>Players</h2><p class="muted small">Search by name or team code (e.g. BOS). Playing today shows the projection; otherwise season averages.</p></div></div>
    <input class="search" type="search" placeholder="Search players" value="${esc(q)}" aria-label="Search players">
    <div class="prow" style="cursor:default;border:0;background:none;padding-top:14px;padding-bottom:0"><span class="muted small">Player</span><span class="muted small num">Pts</span><span class="muted small num">Reb</span><span class="muted small num">Ast</span></div>
    <div class="plist">${list.map((p) => {
      const sp = S.slateById.get(p.id), v = sp ? sp.proj : p.avg;
      return `<button class="prow" data-player="${esc(p.id)}"><span><b>${esc(p.name)}</b> <span class="muted small">${esc(p.team)} ${esc(p.pos || "")}</span>${sp ? ` <span class="tag">today</span>` : ""}</span>
        <span class="num">${f1(v.pts)}</span><span class="num">${f1(v.reb)}</span><span class="num">${f1(v.ast)}</span></button>`;
    }).join("") || `<div class="empty"><p>No player matches "${esc(q)}".</p></div>`}</div>`;
  const input = view.querySelector(".search");
  input.oninput = () => { const pos = input.selectionStart; renderPlayers(input.value); const i = view.querySelector(".search"); i.focus(); i.setSelectionRange(pos, pos); };
  view.querySelectorAll("[data-player]").forEach((r) => r.onclick = () => (location.hash = "p/" + r.dataset.player));
}

function renderPlayer(id) {
  const p = S.byId.get(id), sp = S.slateById.get(id);
  if (!p && !sp) { view.innerHTML = `<div class="empty"><h2>Player not found</h2><p><a href="#players">Back to players</a></p></div>`; return; }
  const name = p?.name || sp.name, g = sp?.game;
  const logs = p?.logs || [];
  const head = `<button class="back" onclick="history.back()">Back</button>
    <div class="pcard-head"><div><h2>${esc(name)}</h2><p class="muted">${esc(p?.team || "")} ${esc(p?.pos || sp?.pos || "")}${p ? `, ${p.gp} games this season` : ""}</p></div>
    ${sp?.inj ? `<span class="tag inj" title="${esc(sp.inj_note)}">${esc(sp.inj)}</span>` : ""}</div>`;
  const projBlock = sp ? `<div><h3>${esc(g.away.abbr)} @ ${esc(g.home.abbr)}, ${day(g.start)} ${time(g.start)}</h3>
      <div class="projgrid" style="margin-top:8px">${["min", ...STATS].map((s) => `<div class="proj"><b>${f1(sp.proj[s])}</b><span>${s === "min" ? "Minutes" : MARKET[s]}</span></div>`).join("")}</div>
      ${sp.low_sample ? `<p class="muted small" style="margin-top:6px">Fewer than 10 games of history, so treat this projection with extra caution.</p>` : ""}</div>
      <div id="why-box"></div>`
    : `<p class="muted">Not on the current slate. The line checker uses the recent average instead of a projection.</p>`;
  const opts = STATS.map((s) => `<option value="${s}">${MARKET[s]}</option>`).join("");
  view.innerHTML = `<div class="panel pcard">${head}${projBlock}
    <div class="checker"><h3>Check a line</h3>
      <div class="checker-row"><select id="ck-stat" aria-label="Stat">${opts}</select>
        <input id="ck-line" type="number" step="0.5" inputmode="decimal" placeholder="Line" aria-label="Line">
        <input id="ck-price" type="text" inputmode="decimal" placeholder="Odds, e.g. 1.85" aria-label="Odds" style="width:140px"></div>
      <div id="ck-out" class="result muted small">Enter the sportsbook line to see the chance of the over and under.</div>
      <div id="ck-line-viz"></div></div>
    <div><h3 id="bars-title">Last ${logs.length} games</h3><div class="bars" id="bars"></div></div>
    <div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Opp</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th><th>3PM</th><th>Stl</th><th>Blk</th></tr></thead>
      <tbody>${logs.map((l) => `<tr><td>${esc(l.d.slice(5))}</td><td>${esc(l.opp)}</td><td>${f1(l.min)}</td><td>${l.pts}</td><td>${l.reb}</td><td>${l.ast}</td><td>${l.fg3m}</td><td>${l.stl ?? "–"}</td><td>${l.blk ?? "–"}</td></tr>`).join("")}</tbody></table></div>
  </div>`;

  const statSel = document.getElementById("ck-stat"), lineIn = document.getElementById("ck-line"), priceIn = document.getElementById("ck-price");
  const recentMean = (s) => { const vals = logs.slice(0, 10).map((l) => (s === "pra" ? l.pts + l.reb + l.ast : l[s])); return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0; };
  const update = () => {
    const s = statSel.value, line = parseFloat(lineIn.value), pr = parseOdds(priceIn.value);
    const wb = document.getElementById("why-box");
    if (wb && sp?.why?.[s]) wb.innerHTML = whyBlock(sp.why[s], MARKET[s].toLowerCase(), `Why ${f1(sp.proj[s])} ${MARKET[s].toLowerCase()}`);
    const mu = sp ? sp.proj[s] : recentMean(s), v = sp ? sp.var[s] : variance(s, mu);
    const vals = logs.slice().reverse().map((l) => (s === "pra" ? l.pts + l.reb + l.ast : l[s]));
    const ref = Number.isFinite(line) ? line : mu, max = Math.max(...vals, ref, 1) * 1.1;
    document.getElementById("bars-title").textContent = `${MARKET[s]}, last ${vals.length} games`;
    document.getElementById("bars").innerHTML = vals.map((x) => `<i class="${Number.isFinite(line) && x > line ? "over" : ""}" style="height:${(x / max) * 100}%" title="${x}"></i>`).join("")
      + `<div class="ref" style="bottom:${(ref / max) * 100}%"></div>`;
    const out = document.getElementById("ck-out"), viz = document.getElementById("ck-line-viz");
    if (!Number.isFinite(line)) { out.innerHTML = `<span>${sp ? "Projection" : "Recent average"}: <b class="num">${f1(mu)}</b>. Enter the sportsbook line to see the chance of the over and under.</span>`; viz.innerHTML = ""; return; }
    const pr2 = countProbs(mu, v, line);
    let ev = "";
    if (pr != null) {
      const eo = pr2.over * payout(pr) - (1 - pr2.over - pr2.push), eu = pr2.under * payout(pr) - (1 - pr2.under - pr2.push);
      ev = `<span>Value at ${price(pr)}: over <b class="${eo > 0 ? "pos" : "neg"}">${sign(eo, 2)}u</b>, under <b class="${eu > 0 ? "pos" : "neg"}">${sign(eu, 2)}u</b></span>`;
    }
    out.classList.remove("muted");
    out.innerHTML = `<span>Over <b>${pct(pr2.over)}</b></span><span>Under <b>${pct(pr2.under)}</b></span>${pr2.push > 0.005 ? `<span>Push <b>${pct(pr2.push)}</b></span>` : ""}
      <span>Fair odds: over <b>${price(toAmerican(pr2.over))}</b></span>${ev}`;
    viz.innerHTML = numberLine(mu, Math.sqrt(v), line, pr2.over >= pr2.under ? "over" : "under");
  };
  [statSel, lineIn, priceIn].forEach((el) => (el.oninput = update));
  update();
}

function renderRecord() {
  const t = S.track || {}, m = S.meta || {}, pk = t.picks, pm = m.player_model || {}, gm = m.game_model || {};
  const all = pk?.all;
  const curve = pk?.curve || [];
  let svg = "";
  if (curve.length > 1) {
    const ys = curve.map((c) => c.units), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), W = 600, H = 180, pad = 8;
    const X = (i) => pad + (i / (curve.length - 1)) * (W - 2 * pad), Y = (v) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
    svg = `<svg class="curve panel" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Cumulative units won">
      <line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line)" stroke-width="1.5"/>
      <polyline fill="none" stroke="var(--model)" stroke-width="2.5" points="${curve.map((c, i) => `${X(i)},${Y(c.units)}`).join(" ")}"/></svg>`;
  }
  const mrow = (k, v) => `<tr><td>${MARKET[k] || k}</td><td>${v.wins}-${v.losses}${v.pushes ? "-" + v.pushes : ""}</td><td>${v.win_rate == null ? "–" : pct(v.win_rate)}</td><td class="${v.units >= 0 ? "pos" : "neg"}">${sign(v.units, 2)}</td><td>${v.roi == null ? "–" : sign(v.roi * 100) + "%"}</td></tr>`;
  const metrics = pm.metrics || {};
  const acc = Object.entries(metrics).map(([k, v]) => {
    const better = (1 - v.mae / v.baseline_mae) * 100;
    return `<tr><td>${MARKET[k] || "Minutes"}</td><td>${f1(v.mae)}</td><td>${f1(v.baseline_mae)}</td><td class="${better > 0 ? "pos" : "neg"}">${sign(better)}%</td></tr>`;
  }).join("");
  const live = t.projections;
  view.innerHTML = `
    <div class="section-head"><div><h2>Track record</h2><p class="muted small">Every flagged pick is logged before tip-off and graded after the game, at the odds shown at the time. One unit per pick.</p></div></div>
    ${all && all.n ? `<div class="kpis">
        <div class="panel kpi"><b>${all.wins}-${all.losses}${all.pushes ? "-" + all.pushes : ""}</b><span>Record</span></div>
        <div class="panel kpi"><b>${pct(all.win_rate)}</b><span>Win rate</span></div>
        <div class="panel kpi"><b class="${all.units >= 0 ? "pos" : "neg"}">${sign(all.units, 2)}</b><span>Units</span></div>
        <div class="panel kpi"><b class="${all.roi >= 0 ? "pos" : "neg"}">${sign(all.roi * 100)}%</b><span>Return per pick</span></div></div>
      <div class="block">${svg}</div>
      <div class="block"><h3>By market</h3><div class="tbl-wrap"><table><thead><tr><th>Market</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr></thead>
        <tbody>${Object.entries(pk.by_market).map(([k, v]) => mrow(k, v)).join("")}</tbody></table></div>
        <p class="muted small" style="margin-top:8px">At standard -110 odds you need to win about 52.4% to break even. Judge this after a few hundred picks, not a few days.</p></div>`
      : `<div class="empty"><h2>No graded picks yet</h2><p>Picks are graded the morning after each game. The record builds from the first day sportsbook lines are loaded.</p></div>`}
    ${pk?.recent?.length ? `<div class="block"><h3>Recent flagged picks</h3><div class="tbl-wrap"><table><thead><tr><th>Pick</th><th>Date</th><th>Odds</th><th>Edge</th><th>Result</th></tr></thead><tbody>
      ${pk.recent.map((r) => `<tr><td>${esc(r.player || r.matchup)}<br><span class="muted small">${esc(pickLabel(r))}</span></td><td class="nw">${esc(day(r.date + "T12:00:00"))}</td><td>${price(r.price)}</td><td>${sign(r.edge * 100)}%</td>
        <td class="${r.result === "win" ? "res-win" : r.result === "loss" ? "res-loss" : "muted"}">${r.result || "pending"}${r.actual != null && r.actual !== "" ? ` (${r.actual})` : ""}</td></tr>`).join("")}</tbody></table></div></div>` : ""}
    <div class="block"><h3>How accurate the projections are</h3>
      <p class="muted small">Average miss on games the model never saw (${esc(pm.valid_from || "")} to ${esc(pm.valid_to || "")}), compared with simply using each player's last-10-game average.</p>
      <div class="tbl-wrap"><table><thead><tr><th>Stat</th><th>Model miss</th><th>Last-10 avg miss</th><th>Better by</th></tr></thead><tbody>${acc}</tbody></table></div>
      ${live ? `<p class="small" style="margin-top:8px">This season, live: average miss ${f1(live.pts)} pts, ${f1(live.reb)} reb, ${f1(live.ast)} ast over ${live.n} player-games.</p>` : ""}
      ${gm.margin ? `<p class="small" style="margin-top:8px">Game model picked the winner in ${pct(gm.margin.winner_acc)} of test games (always picking the home team: ${pct(gm.margin.home_baseline_acc)}). Average miss on the final margin: ${f1(gm.margin.mae)} points, on the total: ${f1(gm.total.mae)}.</p>` : ""}
    </div>
    ${pm.calibration?.length ? `<div class="block"><h3>Are the percentages honest?</h3>
      <p class="muted small">When the model said an over had a given chance, how often it actually happened in test games.</p>
      <div class="tbl-wrap"><table><thead><tr><th>Model said</th><th>Happened</th><th>Samples</th></tr></thead><tbody>
      ${pm.calibration.map((c) => `<tr><td>${pct(c.pred)}</td><td>${pct(c.actual)}</td><td>${c.n.toLocaleString()}</td></tr>`).join("")}</tbody></table></div></div>` : ""}`;
}


/* ---------------------------------------------------------------- my bets (lines typed from the user's bookmaker) */
function slatePlayers() {
  const out = [];
  for (const g of S.slate?.games || []) for (const p of g.players || []) if (!g.started) out.push({ ...p, game: g });
  return out.sort((a, b) => a.name.localeCompare(b.name));
}

function evaluateBet(b) {
  // Returns {label, sub, sides:[{name, prob, odds, ev, fairBook}], mu, sd, line} or {error}
  const thr = S.meta?.edge_threshold || 0.05;
  let sides = [], label = "", sub = "", mu = null, sd = null, line = +b.line;
  if (b.kind === "prop") {
    const sp = S.slateById.get(b.player);
    const pl = S.byId.get(b.player);
    if (!sp && !pl) return { error: "Player not found in today's data." };
    const name = sp?.name || pl.name;
    const recent = () => { const v = (pl?.logs || []).slice(0, 10).map((l) => (b.stat === "pra" ? l.pts + l.reb + l.ast : l[b.stat])); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : 0; };
    mu = sp ? sp.proj[b.stat] : recent();
    const v = sp ? sp.var[b.stat] : variance(b.stat, mu);
    sd = Math.sqrt(v);
    const pr = countProbs(mu, v, line);
    label = name; sub = `${MARKET[b.stat]} ${line}${sp ? "" : " (not on today's slate, using recent average)"}`;
    sides = [{ name: `Over ${line}`, key: "over", prob: pr.over, push: pr.push, odds: b.o1 }, { name: `Under ${line}`, key: "under", prob: pr.under, push: pr.push, odds: b.o2 }];
  } else {
    const g = (S.slate?.games || []).find((x) => x.id === b.game);
    if (!g) return { error: "Game is no longer on the slate." };
    const gm = S.meta?.game_model || {}, sdm = gm.margin?.sd || 13, sdt = gm.total?.sd || 18;
    label = `${g.away.abbr} @ ${g.home.abbr}`;
    if (b.market === "moneyline") {
      const ph = normCdf(g.pred.margin / sdm);
      sub = "Moneyline"; mu = g.pred.margin; sd = sdm; line = 0;
      sides = [{ name: `${g.home.abbr} to win`, key: "home", prob: ph, odds: b.o1 }, { name: `${g.away.abbr} to win`, key: "away", prob: 1 - ph, odds: b.o2 }];
    } else if (b.market === "spread") {
      const ph = normCdf((g.pred.margin + line) / sdm);
      sub = `Spread, ${g.home.abbr} ${sign(line)}`; mu = g.pred.margin; sd = sdm;
      sides = [{ name: `${g.home.abbr} ${sign(line)}`, key: "home", prob: ph, odds: b.o1 }, { name: `${g.away.abbr} ${sign(-line)}`, key: "away", prob: 1 - ph, odds: b.o2 }];
      line = -line;
    } else {
      const po = normCdf((g.pred.total - line) / sdt);
      sub = `Total points ${line}`; mu = g.pred.total; sd = sdt;
      sides = [{ name: `Over ${line}`, key: "over", prob: po, odds: b.o1 }, { name: `Under ${line}`, key: "under", prob: 1 - po, odds: b.o2 }];
    }
  }
  const am = sides.map((x) => parseOdds(x.odds));
  const imp = am.map((a) => (a == null ? null : implied(a)));
  const fair = imp[0] != null && imp[1] != null ? [imp[0] / (imp[0] + imp[1]), imp[1] / (imp[0] + imp[1])] : imp;
  sides.forEach((x, i) => {
    x.am = am[i]; x.fair = fair[i];
    x.ev = am[i] == null ? null : x.prob * payout(am[i]) - (1 - x.prob - (x.push || 0));
    x.edge = fair[i] == null ? null : x.prob - fair[i];
    x.check = x.edge != null && x.edge > (S.meta?.max_edge || 0.2);
    x.good = x.ev != null && x.ev > 0 && (x.edge == null || x.edge >= thr) && !x.check;
  });
  return { label, sub, sides, mu, sd, line };
}

function myBetCard(b, i) {
  const r = evaluateBet(b);
  if (r.error) return `<article class="panel edge"><div class="edge-top"><p class="muted">${esc(r.error)}</p><button class="chip" data-del="${i}">Remove</button></div></article>`;
  const best = r.sides.filter((x) => x.good).sort((a, c) => c.ev - a.ev)[0];
  const sideRows = r.sides.map((x) => `<div class="side ${best === x ? "best" : ""}">
      <span class="side-name">${esc(x.name)}</span>
      <span>Model <b>${pct(x.prob)}</b></span>
      <span>Odds <b>${x.am == null ? "–" : price(x.am)}</b></span>
      <span>Book <b>${x.fair == null ? "–" : pct(x.fair)}</b></span>
      <span>Value <b class="${x.ev == null ? "" : x.ev > 0 ? "pos" : "neg"}">${x.ev == null ? "–" : sign(x.ev, 2) + "u"}</b></span>
    </div>`).join("");
  const gap = r.sides.find((x) => x.check);
  const keyFor = best ? best.key : (r.sides[0].prob >= r.sides[1].prob ? r.sides[0].key : r.sides[1].key);
  const viz = b.kind === "prop" || b.market === "total" ? numberLine(r.mu, r.sd, r.line, keyFor === "under" ? "under" : "over") : "";
  return `<article class="panel edge ${best ? "hot" : ""}">
    <div class="edge-top"><div class="edge-who"><h3>${esc(r.label)}</h3><p class="muted small">${esc(r.sub)}</p></div>
      <div class="edge-bet">${best ? esc(best.name) : gap ? "Check the news" : "No value"}<small>${best ? `value ${sign(best.ev, 2)}u per unit`
        : gap ? "model and Betway disagree by a lot, usually an injury or role change" : "at these odds"}</small></div></div>
    <div class="sides">${sideRows}</div>
    ${viz}
    <div><button class="chip" data-del="${i}">Remove</button></div>
  </article>`;
}

function renderMyBets() {
  const key = "mybets", all = store.get(key, {}), date = S.slate?.date || "none";
  const bets = all[date] || [];
  const save = (list) => { store.set(key, { [date]: list }); renderMyBets(); };
  const players = slatePlayers();
  const games = (S.slate?.games || []).filter((g) => !g.started);
  if (!games.length) { view.innerHTML = `<div class="empty"><h2>No games to bet on yet</h2><p>Once the next slate is loaded you can type in lines from your bookmaker here.</p></div>`; return; }
  const kind = S.mbKind || "prop";
  const statOpts = STATS.map((x) => `<option value="${x}">${MARKET[x]}</option>`).join("");
  const gameOpts = games.map((g) => `<option value="${esc(g.id)}">${esc(g.away.abbr)} @ ${esc(g.home.abbr)}, ${time(g.start)}</option>`).join("");
  const sorted = bets.map((b, i) => ({ b, i, r: evaluateBet(b) }))
    .sort((x, y) => Math.max(...(y.r.sides || []).map((s) => s.ev ?? -9)) - Math.max(...(x.r.sides || []).map((s) => s.ev ?? -9)));
  view.innerHTML = `
    <div class="section-head"><div><h2>My bets</h2>
      <p class="muted small">Type in lines from your Betway app (or any bookmaker) to see if they're worth it. Decimal odds like 1.85 or American like -110 both work. Saved on this device for today's slate.</p></div></div>
    <form class="panel checker mb-form" id="mb-form" autocomplete="off">
      <div class="chips" role="group" aria-label="Bet type">
        <button type="button" class="chip" data-kind="prop" aria-pressed="${kind === "prop"}">Player prop</button>
        <button type="button" class="chip" data-kind="game" aria-pressed="${kind === "game"}">Game line</button></div>
      ${kind === "prop" ? `
        <div class="checker-row">
          <input id="mb-player" list="mb-players" placeholder="Player" aria-label="Player" style="flex:1 1 220px;width:auto">
          <datalist id="mb-players">${players.map((p) => `<option value="${esc(p.name)} (${esc(p.side === "home" ? p.game.home.abbr : p.game.away.abbr)})"></option>`).join("")}</datalist>
          <select id="mb-stat" aria-label="Stat">${statOpts}</select>
          <input id="mb-line" type="text" inputmode="decimal" placeholder="Line, e.g. 24.5" aria-label="Line"></div>
        <div class="checker-row">
          <input id="mb-o1" type="text" inputmode="decimal" placeholder="Over odds" aria-label="Over odds">
          <input id="mb-o2" type="text" inputmode="decimal" placeholder="Under odds" aria-label="Under odds"></div>`
      : `
        <div class="checker-row">
          <select id="mb-game" aria-label="Game">${gameOpts}</select>
          <select id="mb-market" aria-label="Market"><option value="total">Total points</option><option value="spread">Spread (home team)</option><option value="moneyline">Moneyline</option></select>
          <input id="mb-line" type="text" inputmode="decimal" placeholder="Line" aria-label="Line"></div>
        <div class="checker-row">
          <input id="mb-o1" type="text" inputmode="decimal" placeholder="Over / home odds" aria-label="Over or home odds" style="width:150px">
          <input id="mb-o2" type="text" inputmode="decimal" placeholder="Under / away odds" aria-label="Under or away odds" style="width:150px"></div>`}
      <div class="checker-row"><button class="add" type="submit">Check bet</button><span id="mb-err" class="small neg" role="alert"></span></div>
    </form>
    <div class="edges block">${sorted.map(({ b, i }) => myBetCard(b, i)).join("") || `<div class="empty"><p>Bets you check appear here, best value first. Enter both over and under odds when you can: that lets the app strip out Betway's margin and show the true book probability.</p></div>`}</div>
    ${bets.length ? `<p style="margin-top:12px"><button class="chip" id="mb-clear">Clear all</button></p>` : ""}`;

  view.querySelectorAll("[data-kind]").forEach((btn) => (btn.onclick = () => { S.mbKind = btn.dataset.kind; renderMyBets(); }));
  view.querySelectorAll("[data-del]").forEach((btn) => (btn.onclick = () => { const l = bets.slice(); l.splice(+btn.dataset.del, 1); save(l); }));
  const clear = document.getElementById("mb-clear"); if (clear) clear.onclick = () => save([]);
  const mk = document.getElementById("mb-market");
  if (mk) { const lineIn = document.getElementById("mb-line"); const sync = () => { lineIn.disabled = mk.value === "moneyline"; lineIn.placeholder = mk.value === "spread" ? "Home spread, e.g. -4.5" : mk.value === "total" ? "Total, e.g. 226.5" : "No line"; }; mk.onchange = sync; sync(); }
  document.getElementById("mb-form").onsubmit = (e) => {
    e.preventDefault();
    const err = document.getElementById("mb-err"); err.textContent = "";
    const val = (id) => (document.getElementById(id)?.value || "").trim();
    const line = parseFloat(val("mb-line").replace(",", "."));
    const o1 = val("mb-o1"), o2 = val("mb-o2");
    if (parseOdds(o1) == null && parseOdds(o2) == null) { err.textContent = "Enter at least one price, like 1.85."; return; }
    let bet;
    if (kind === "prop") {
      const q = val("mb-player").replace(/\s*\([^)]*\)\s*$/, "").toLowerCase();
      const p = players.find((x) => x.name.toLowerCase() === q) || players.find((x) => x.name.toLowerCase().includes(q))
        || (q ? S.players.find((x) => x.name.toLowerCase().includes(q)) : null);
      if (!p) { err.textContent = "Pick a player from the list."; return; }
      if (!Number.isFinite(line)) { err.textContent = "Enter the line, like 24.5."; return; }
      bet = { kind, player: p.id, stat: val("mb-stat"), line, o1, o2 };
    } else {
      const market = val("mb-market");
      if (market !== "moneyline" && !Number.isFinite(line)) { err.textContent = "Enter the line."; return; }
      bet = { kind, game: val("mb-game"), market, line: market === "moneyline" ? 0 : line, o1, o2 };
    }
    save([...bets, bet]);
  };
}


/* ---------------------------------------------------------------- learning */
const UNIT = { pts: "pts", reb: "reb", ast: "ast", fg3m: "3pm", min: "min", margin: "margin", total: "total" };
const STAT_NAME = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "3-pointers", min: "Minutes", margin: "Game margin", total: "Game total" };
const when = (iso) => { try { return new Date(iso).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); } catch { return iso; } };
const issueUrl = (action, id) => `https://github.com/${S.meta?.repo || "Juniorcarl/nba-predictor"}/issues/new?title=${encodeURIComponent(`[${action}] ${id}`)}&body=${encodeURIComponent(
  action === "approve" ? `Approving ${id}. Tap "Submit new issue" to confirm.` : action === "undo" ? `Undoing ${id}. Tap "Submit new issue" to confirm.` : `Rejecting ${id}. Tap "Submit new issue" to confirm.`)}`;

function proposalCard(p) {
  const ev = p.evidence || {}, seg = ev.segment, targeted = ev.basis === "targeted";
  const per = targeted ? seg.per : ev.per || {};
  const rows = Object.entries(per).map(([k, v]) => {
    const ch = (v.after - v.before) / v.before;
    return `<tr><td>${STAT_NAME[k] || k}</td><td>${v.before.toFixed(2)}</td><td>${v.after.toFixed(2)}</td>
      <td class="${ch < 0 ? "pos" : ch > 0 ? "neg" : ""}">${sign(ch * 100, 1)}%</td></tr>`;
  }).join("");
  const headline = targeted ? -seg.gain : -ev.gain;
  return `<article class="panel edge hot">
    <div class="edge-top"><div class="edge-who"><p class="muted small">${esc(p.id)}, proposed ${when(p.created)}</p><h3>${esc(p.title)}</h3></div>
      <div class="edge-bet">${sign(headline * 100, targeted ? 1 : 2)}%<small>${targeted ? "miss in its games" : "average miss"}</small></div></div>
    <p>${esc(p.why)}</p>
    ${targeted ? `<p class="small">Tested on <b>${seg.n.toLocaleString()} ${esc(seg.label)}</b> the model hadn't seen. Across all games the change is ${sign(-ev.gain * 100, 2)}%, so it helps where it should without hurting elsewhere.</p>` : ""}
    <div class="tbl-wrap"><table><thead><tr><th>Average miss${targeted ? " in those games" : ""}</th><th>Now</th><th>With change</th><th>Change</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="muted small">It was better in both the earlier and the later half of the test period, so it isn't a one-off.</p>
    <div class="chips"><a class="add" href="${issueUrl("approve", p.id)}" target="_blank" rel="noopener">Approve</a>
      <a class="chip" href="${issueUrl("reject", p.id)}" target="_blank" rel="noopener">Reject</a></div>
  </article>`;
}

function groupPatterns(items) {
  const by = new Map();
  for (const p of items) {
    if (!by.has(p.segment + p.source)) by.set(p.segment + p.source, { ...p, parts: [] });
    by.get(p.segment + p.source).parts.push(p);
  }
  return [...by.values()].map((g) => {
    const words = g.parts.map((x) => `${x.bias > 0 ? "under" : "over"}-predicts ${({ pts: "points", reb: "rebounds", ast: "assists", min: "minutes" })[x.stat]} by ${Math.abs(x.bias).toFixed(1)}`);
    return { ...g, text: `${g.segment}: the model ${words.join(", ")} on average (${g.n.toLocaleString()} cases).` };
  });
}

function postmortemBlock(pm) {
  if (!pm) return `<div class="empty"><p>The first review appears the morning after the first games of the season.</p></div>`;
  const reasons = Object.entries(pm.reasons || {}).sort((a, b) => b[1] - a[1]);
  const total = reasons.reduce((a, [, n]) => a + n, 0) || 1;
  return `<div class="panel pcard">
    <h3>Games of ${esc(day(pm.date + "T12:00:00"))}</h3>
    <div class="kpis">
      <div class="kpi"><b>${f1(pm.mae.pts)}</b><span>Avg miss, points</span></div>
      <div class="kpi"><b>${f1(pm.mae.reb)}</b><span>Rebounds</span></div>
      <div class="kpi"><b>${f1(pm.mae.ast)}</b><span>Assists</span></div>
      <div class="kpi"><b>${pct(pm.share_minutes)}</b><span>of the points miss came from wrong minutes</span></div></div>
    ${reasons.length ? `<div><h3>Why the big misses happened (6+ points off)</h3>
      <div class="reasons">${reasons.map(([r, n]) => `<div class="reason"><span>${esc(r)}</span><i style="width:${(n / total) * 100}%"></i><b>${n}</b></div>`).join("")}</div></div>` : ""}
    <div><h3>Biggest misses</h3><div class="tbl-wrap"><table><thead><tr><th>Player</th><th>Pts exp / actual</th><th>Min exp / actual</th><th>Likely reason</th></tr></thead><tbody>
      ${(pm.biggest || []).map((b) => `<tr><td>${esc(b.player)}</td><td>${f1(b.p_pts)} / ${b.played ? f1(b.a_pts) : "DNP"}</td><td>${f1(b.p_min)} / ${b.played ? f1(b.a_min) : "–"}</td>
        <td><b>${esc(b.reason)}</b><br><span class="muted small">${esc(b.detail)}</span></td></tr>`).join("")}</tbody></table></div></div>
    <p class="muted small">${pm.n} players reviewed${pm.dnp ? `, ${pm.dnp} projected players didn't play` : ""}. Reasons are best guesses from the box score.</p>
  </div>`;
}

async function renderLearning() {
  if (!S.learning) {
    view.innerHTML = `<div class="empty"><p>Loading…</p></div>`;
    try { S.learning = await load("learning"); } catch { view.innerHTML = `<div class="empty"><h2>No learning data yet</h2><p>It appears after the next daily update.</p></div>`; return; }
  }
  const L = S.learning, cfg = L.config || { groups: [] }, pm = (L.postmortems || [])[0];
  const exp = (L.experiments || [])[0];
  const on = cfg.groups || [], all = Object.keys(L.groups || {});
  const pats = groupPatterns(L.patterns?.items || []);
  view.innerHTML = `
    <div class="section-head"><div><h2>Learning</h2>
      <p class="muted small">After every slate the model checks each prediction against what happened, looks for where it keeps missing, and tests possible fixes on games it hasn't seen. Nothing changes until you approve it.</p></div></div>

    <div class="block"><h2>${L.proposals?.length ? `${L.proposals.length} change${L.proposals.length > 1 ? "s" : ""} waiting for you` : "No changes waiting"}</h2>
      ${L.proposals?.length ? `<p class="muted small">Approve or Reject opens GitHub with the decision filled in. Tap "Submit new issue" and the model retrains with it within a few minutes.</p>
        <div class="edges">${L.proposals.map(proposalCard).join("")}</div>`
      : `<p class="muted small">${exp ? `The last test round (${when(exp.at)}) found nothing that clearly beats the current model. Tests run every Monday.` : "The first test round runs with the next update."}</p>`}</div>

    <div class="block"><h2>Last review</h2>${postmortemBlock(pm)}</div>

    <div class="block"><h2>Patterns being watched</h2>
      <p class="muted small">Groups of games where the model is consistently too high or too low.</p>
      ${pats.length ? `<div class="edges">${pats.map((p) => `<div class="panel edge"><p><b>${esc(p.text)}</b></p>
        ${p.fix ? `<p class="muted small">${esc(p.fix)}</p>` : ""}<p class="muted small">Seen in ${esc(p.source)}.</p></div>`).join("")}</div>`
      : `<div class="empty"><p>No consistent blind spots found right now.</p></div>`}</div>

    ${exp ? `<div class="block"><h2>Last test round</h2><p class="muted small">${when(exp.at)}. Each idea was trained and scored on games the model hadn't seen, against the current model.</p>
      <div class="tbl-wrap"><table><thead><tr><th>Idea</th><th>Change in avg miss</th><th>Result</th></tr></thead><tbody>
      ${exp.tested.slice().sort((a, b) => b.gain - a.gain).map((t) => `<tr><td>${esc(t.kind === "add_group" ? "Add " : t.kind === "remove_group" ? "Remove " : "Set ")}${esc(t.target.replace(/_/g, " "))}${t.value != null ? " = " + t.value : ""}</td>
        <td class="${t.gain > 0 ? "pos" : "neg"}">${sign(-t.gain * 100, 2)}%${t.segment && t.segment.gain != null ? `<br><span class="muted small">${sign(-t.segment.gain * 100, 1)}% in ${esc(t.segment.label)}</span>` : ""}</td><td>${t.passed ? "<b>Proposed</b>" : "Not enough"}</td></tr>`).join("")}</tbody></table></div></div>` : ""}

    <div class="block"><h2>Activity log</h2>
      <ol class="log">${(L.events || []).map((e) => `<li class="ev-${esc(e.kind)}"><time>${when(e.at)}</time><span>${esc(e.text)}</span>
        ${e.kind === "approved" && e.proposal && !(L.changes || []).find((c) => c.proposal === e.proposal && c.undone) ? ` <a href="${issueUrl("undo", e.proposal)}" target="_blank" rel="noopener">Undo</a>` : ""}</li>`).join("")
        || `<li><span class="muted">Nothing yet.</span></li>`}</ol></div>

    <div class="block"><h2>Current model</h2><p class="muted small">Version ${cfg.version || 1}. Settings: tree depth ${cfg.params?.max_depth}, at least ${cfg.params?.min_child_weight} games per pattern.</p>
      <div class="factors">${all.map((g) => `<div class="factor ${on.includes(g) ? "on" : ""}"><b>${on.includes(g) ? "On" : "Off"}</b>
        <span><strong>${esc(g.replace(/_/g, " "))}</strong>: ${esc(L.groups[g])}</span></div>`).join("")}</div>
      <p class="muted small" style="margin-top:8px">Always on: recent form, season averages, per-minute production, rest, home/away, team and opponent strength, pace, and injured teammates.</p></div>`;
}


/* ---------------------------------------------------------------- bold */
S.bank = store.get("bankroll", null);
S.riskBand = store.get("riskBand", "high");
const STAT_PLURAL = { pts: "points", reb: "rebounds", ast: "assists", fg3m: "threes", stl: "steals", blk: "blocks", pra: "pts + reb + ast" };
const dec = (p) => (1 / p).toFixed(2);

function valueLine(p, oddsStr) {
  const am = parseOdds(oddsStr);
  if (am == null) return `<span class="muted">Type the odds Betway is offering to see if it's worth it.</span>`;
  const d = toDecimal(am), ev = p * (d - 1) - (1 - p);
  if (ev <= 0) return `<span><b class="neg">Skip at ${d.toFixed(2)}</b>: below the fair price of ${dec(p)}, you'd lose ${Math.abs(ev).toFixed(2)} per unit on average.</span>`;
  const k = Math.max(0, (p * (d - 1) - (1 - p)) / (d - 1)) / 4;
  const amt = S.bank ? ` = <b>P${(S.bank * k).toFixed(2)}</b>` : "";
  return `<span><b class="pos">Value at ${d.toFixed(2)}</b>: +${ev.toFixed(2)} per unit on average. Suggested stake <b>${(k * 100).toFixed(1)}%</b> of bankroll${amt}.</span>`;
}

function oddsBox(id, p) {
  return `<div class="odds-box"><label>Betway odds <input data-p="${p}" data-out="${id}" type="text" inputmode="decimal" placeholder="e.g. 3.50"></label>
    <p id="${id}" class="small">${valueLine(p, "")}</p></div>`;
}

function chanceBlock(p, pn) {
  const w = (x) => Math.max(2, Math.min(100, x * 200));
  return `<div class="chance">
    <div><span class="muted small">Normal night</span><i class="bar n" style="width:${w(pn)}%"></i><b>${pct(pn)}</b></div>
    <div><span class="muted small">Tonight</span><i class="bar t" style="width:${w(p)}%"></i><b>${pct(p)}</b></div>
  </div>`;
}

function reasonList(items, cls) {
  return items.map((r) => `<li class="${cls}"><b>${esc(r.label)}</b>${r.amt != null ? ` <span class="${r.amt > 0 ? "pos" : "neg"}">${sign(r.amt)}</span>` : ""}<br><span class="muted small">${esc(r.detail)}</span></li>`).join("");
}

function boldCallCard(c, i) {
  const g = (S.slate?.games || []).find((x) => x.id === c.game_id);
  const when = g ? time(g.start) : "";
  return `<article class="panel bold-card">
    <div class="bold-head"><div><p class="muted small">${esc(c.team)} ${c.home ? "vs" : "@"} ${esc(c.opp)}${when ? ", " + when : ""}</p>
      <h3>${esc(c.player)} ${c.line}+ ${STAT_PLURAL[c.stat]}</h3>
      <p class="muted small">Projection ${f1(c.proj)}, usual ${f1(c.normal)}</p></div>
      <div class="bold-odds"><b>${pct(c.p)}</b><span>fair odds ${c.fair_odds.toFixed(2)}</span></div></div>
    ${chanceBlock(c.p, c.p_normal)}
    <p class="small"><b>${c.boost.toFixed(1)}× more likely than on a normal night.</b> ${c.band === "high" ? "High risk: expect to lose most of these." : "Medium risk."}</p>
    <ul class="reasons-list">${reasonList(c.reasons.for, "for")}${c.reasons.facts.map((f) => `<li class="fact">${esc(f)}</li>`).join("")}${reasonList(c.reasons.against, "against")}</ul>
    ${oddsBox(`bc${i}`, c.p)}
  </article>`;
}

function upsetCard(u, i) {
  return `<article class="panel bold-card">
    <div class="bold-head"><div><p class="muted small">${time(u.start)}</p><h3>${esc(u.team)} to beat ${esc(u.fav)}</h3>
      <p class="muted small">Model has ${esc(u.fav)} by ${f1(Math.abs(u.margin))}${u.book_p != null ? `. Bookmakers give ${esc(u.team)} ${pct(u.book_p)}` : ""}${u.betway_price != null ? `, Betway pays ${price(u.betway_price)}` : ""}</p></div>
      <div class="bold-odds"><b>${pct(u.p)}</b><span>fair odds ${u.fair_odds.toFixed(2)}</span></div></div>
    <ul class="reasons-list">${u.reasons.map((r) => `<li class="${r.helps ? "for" : "against"}"><b>${esc(r.label)}</b> <span class="${r.amt_dog > 0 ? "pos" : "neg"}">${sign(r.amt_dog)}</span><br><span class="muted small">${esc(r.detail)}</span></li>`).join("")}</ul>
    ${oddsBox(`up${i}`, u.p)}
  </article>`;
}

function multiCard(m, i) {
  return `<article class="panel bold-card">
    <div class="bold-head"><div><p class="muted small">${esc(m.matchup)}, ${time(m.start)}</p><h3>${m.legs.length}-leg multi</h3></div>
      <div class="bold-odds"><b>${pct(m.p)}</b><span>fair odds ${m.fair_odds.toFixed(2)}</span></div></div>
    <ol class="legs">${m.legs.map((l) => `<li><span>${esc(l.text)}</span><b>${pct(l.p)}</b></li>`).join("")}</ol>
    <p class="small">All legs together: <b>${pct(m.p)}</b>. Multiplying the legs gives ${pct(m.naive)}; ${m.lift > 1.005
      ? `they tend to hit together, so it's ${m.lift.toFixed(2)}× more likely than that.`
      : m.lift < 0.995 ? `they tend to work against each other, so it's less likely than that.` : `they're mostly independent.`}
      Only worth it if Betway pays more than <b>${m.fair_odds.toFixed(2)}</b>.</p>
    ${oddsBox(`mu${i}`, m.p)}
  </article>`;
}

async function renderBold() {
  if (!S.bold) {
    view.innerHTML = `<div class="empty"><p>Loading…</p></div>`;
    try { S.bold = await load("bold"); } catch { view.innerHTML = `<div class="empty"><h2>No bold calls yet</h2><p>They appear with the next daily update.</p></div>`; return; }
  }
  const B = S.bold, band = S.riskBand;
  const calls = (B.calls || []).filter((c) => band === "all" || c.band === band);
  const rec = B.record || {};
  const chips = [["high", "High risk (10–30%)"], ["medium", "Medium (30–48%)"], ["all", "All"]]
    .map(([k, l]) => `<button class="chip" data-band="${k}" aria-pressed="${band === k}">${l}</button>`).join("");
  view.innerHTML = `
    <div class="section-head"><div><h2>Bold</h2>
      <p class="muted small">Big nights and upsets that are clearly more likely than usual tonight, priced honestly. Every chance shown is checked against past games. Most of these lose; the ones that hit pay enough to make up for it, but only if the odds beat the fair price.</p></div></div>
    <div class="panel bold-settings">
      <div class="chips">${chips}</div>
      <label class="small">Bankroll (pula) <input id="bank" type="text" inputmode="decimal" value="${S.bank ?? ""}" placeholder="e.g. 500"></label>
      <p class="muted small">Stakes use a quarter of the Kelly formula: small bets, so a cold streak can't wipe you out. Saved on this phone only.</p>
    </div>

    <div class="block"><h2>Bold calls</h2>
      ${calls.length ? `<div class="edges">${calls.map(boldCallCard).join("")}</div>` : `<div class="empty"><p>${B.calls?.length ? "No calls at this risk level tonight. Try another level." : "No standout big-night spots on the current slate."}</p></div>`}</div>

    <div class="block"><h2>Upset picks</h2><p class="muted small">Underdogs the model gives a real chance to win.</p>
      ${B.upsets?.length ? `<div class="edges">${B.upsets.map(upsetCard).join("")}</div>` : `<div class="empty"><p>No upset worth a look on the current slate.</p></div>`}</div>

    <div class="block"><h2>Same-game multis</h2><p class="muted small">2–3 picks from one game. The chance accounts for how the picks are linked: a player who scores big usually also has a big PRA, and a team that wins usually got big games from its stars. Bookmakers take a bigger cut on multis, so check the price.</p>
      ${B.multis?.length ? `<div class="edges">${B.multis.map(multiCard).join("")}</div>` : `<div class="empty"><p>No multis that make sense on the current slate.</p></div>`}</div>

    <div class="block"><h2>Bold record</h2>
      ${rec.graded ? `<div class="tbl-wrap"><table><thead><tr><th>Type</th><th>Hit</th><th>Expected</th><th>Hit rate</th><th>Avg chance</th></tr></thead><tbody>
        ${["call", "upset", "multi"].filter((k) => rec[k]).map((k) => `<tr><td>${{ call: "Bold calls", upset: "Upsets", multi: "Multis" }[k]}</td><td>${rec[k].hits} of ${rec[k].n}</td><td>${f1(rec[k].expected)}</td><td>${pct(rec[k].hit_rate)}</td><td>${pct(rec[k].avg_p)}</td></tr>`).join("")}
        </tbody></table></div>
        <p class="muted small" style="margin-top:6px">If the page is honest, the number of hits should be close to the number expected. More hits than expected means the calls are better than their price.</p>
        ${rec.calibration?.length ? `<div class="tbl-wrap"><table><thead><tr><th>Page said</th><th>Happened</th><th>Calls</th></tr></thead><tbody>${rec.calibration.map((c) => `<tr><td>${pct(c.said)}</td><td>${pct(c.happened)}</td><td>${c.n}</td></tr>`).join("")}</tbody></table></div>` : ""}
        <h3 style="margin-top:14px">Recent</h3><div class="tbl-wrap"><table><tbody>${(rec.recent || []).map((r) => `<tr><td class="nw">${esc(day(r.date + "T12:00:00"))}</td><td>${esc(r.desc)}</td><td>${pct(r.p)}</td><td class="${r.result === "hit" ? "res-win" : r.result === "miss" ? "res-loss" : "muted"}">${r.result || "pending"}</td></tr>`).join("")}</tbody></table></div>`
      : `<div class="empty"><p>Bold calls are graded the morning after each game. The record starts with the first slate of the season.</p></div>`}
      ${B.tail_check?.pts ? `<p class="muted small" style="margin-top:10px">Tested on past games the model hadn't seen: when it said a big points night had a ${pct(B.tail_check.pts[1]?.cal)} chance, it happened ${pct(B.tail_check.pts[1]?.hit)} of the time (${(B.tail_check.pts[1]?.n || 0).toLocaleString()} cases).</p>` : ""}
    </div>`;

  view.querySelectorAll("[data-band]").forEach((b) => (b.onclick = () => { S.riskBand = b.dataset.band; store.set("riskBand", S.riskBand); renderBold(); }));
  const bank = document.getElementById("bank");
  bank.onchange = () => { const v = parseFloat(bank.value.replace(",", ".")); S.bank = Number.isFinite(v) && v > 0 ? v : null; store.set("bankroll", S.bank); renderBold(); };
  view.querySelectorAll("input[data-p]").forEach((inp) => (inp.oninput = () => {
    document.getElementById(inp.dataset.out).innerHTML = valueLine(parseFloat(inp.dataset.p), inp.value);
  }));
}

/* ---------------------------------------------------------------- routing */
function route() {
  const h = location.hash.replace("#", "") || "picks";
  const tab = h.startsWith("p/") ? "players" : h;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  if (h.startsWith("p/")) renderPlayer(decodeURIComponent(h.slice(2)));
  else if (h === "games") renderGames();
  else if (h === "players") renderPlayers();
  else if (h === "record") renderRecord();
  else if (h === "mybets") renderMyBets();
  else if (h === "learning") renderLearning();
  else if (h === "bold") renderBold();
  else renderEdges();
  window.scrollTo(0, 0);
}
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => (location.hash = b.dataset.tab)));
window.addEventListener("hashchange", route);

function renderFmtToggle() {
  const el = document.getElementById("fmt");
  el.innerHTML = `Odds shown as <button class="chip" data-fmt="decimal" aria-pressed="${S.fmt === "decimal"}">Decimal 1.91</button>
    <button class="chip" data-fmt="american" aria-pressed="${S.fmt === "american"}">American -110</button>`;
  el.querySelectorAll("[data-fmt]").forEach((b) => (b.onclick = () => { S.fmt = b.dataset.fmt; store.set("oddsFmt", S.fmt); renderFmtToggle(); route(); }));
}

(async function init() {
  try {
    [S.meta, S.slate] = await Promise.all([load("meta"), load("slate")]);
    route();
    [S.players, S.track] = await Promise.all([load("players"), load("track")]);
    S.players.forEach((p) => S.byId.set(p.id, p));
    for (const g of S.slate.games || []) for (const p of g.players || []) S.slateById.set(p.id, { ...p, game: g });
    route();
    const n = S.slate.games?.length || 0;
    document.getElementById("slate-label").textContent = S.slate.date
      ? `${n} game${n === 1 ? "" : "s"} on ${new Date(S.slate.date + "T12:00:00").toLocaleDateString([], { weekday: "long", day: "numeric", month: "short" })} (US date)`
      : "No upcoming games";
    renderFmtToggle();
    document.getElementById("updated").textContent = !S.meta.generated_at ? "Waiting for the first daily update." : `Updated ${new Date(S.meta.generated_at).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}. ${S.meta.has_odds ? "" : "No sportsbook lines in this update."}`;
  } catch (e) {
    console.error(e);
    view.innerHTML = document.getElementById("tpl-error").innerHTML;
  }
})();
