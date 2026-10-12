"use strict";

const S = { meta: null, slate: null, players: [], track: null, byId: new Map(), slateById: new Map(), filter: "all" };
const view = document.getElementById("view");

const MARKET = { pts: "Points", reb: "Rebounds", ast: "Assists", fg3m: "3-pointers", pra: "Pts + Reb + Ast",
  moneyline: "Moneyline", spread: "Spread", total: "Total points" };
const STATS = ["pts", "reb", "ast", "fg3m", "pra"];

/* ---------------------------------------------------------------- utils */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f1 = (x) => (x == null ? "–" : Number(x).toFixed(1));
const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
const sign = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const price = (p) => (p == null ? "–" : p > 0 ? "+" + p : String(p));
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
    const matchup = `${g.away.abbr} @ ${g.home.abbr}`;
    for (const o of g.options || []) out.push({ ...o, g, matchup, kind: "game" });
    for (const p of g.players || []) for (const o of p.props || []) out.push({ ...o, g, p, matchup, kind: "prop" });
  }
  return out;
}

function betText(o) {
  const g = o.g;
  if (o.market === "moneyline") return `${(o.side === "home" ? g.home : g.away).abbr} to win`;
  if (o.market === "spread") return `${(o.side === "home" ? g.home : g.away).abbr} ${sign(o.line)}`;
  if (o.market === "total") return `${o.side === "over" ? "Over" : "Under"} ${o.line}`;
  return `${o.side === "over" ? "Over" : "Under"} ${o.line} ${MARKET[o.market].toLowerCase()}`;
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

/* ---------------------------------------------------------------- views */
function edgeCard(o) {
  const who = o.kind === "prop"
    ? `<h3>${esc(o.p.name)}</h3><p class="muted small">${esc(o.matchup)}, ${time(o.g.start)}${o.p.inj ? ` <span class="tag inj">${esc(o.p.inj)}</span>` : ""}</p>`
    : `<h3>${esc(o.g.away.name)} at ${esc(o.g.home.name)}</h3><p class="muted small">${time(o.g.start)}</p>`;
  return `<article class="panel edge ${o.flagged ? "hot" : ""}">
    <div class="edge-top"><div class="edge-who">${who}</div>
      <div class="edge-bet">${esc(betText(o))}<small>${price(o.price)} at ${esc(o.book || "best book")}</small></div></div>
    <div class="edge-stats">
      <span>Model <b>${pct(o.model_prob)}</b></span>
      <span>Book <b>${pct(o.fair_prob)}</b></span>
      <span>Edge <b class="${o.edge > 0 ? "pos" : "neg"}">${sign(o.edge * 100)}%</b></span>
      <span>Value <b class="${o.ev > 0 ? "pos" : "neg"}">${sign(o.ev, 2)}u</b></span>
    </div>
    ${optionLine(o)}
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
      <span class="score">${Math.round(g.pred.away_score)}–${Math.round(g.pred.home_score)}<small>${time(g.start)}</small></span>
      <span class="team home"><span class="abbr">${esc(g.home.abbr)}</span><span class="nm">${miss(g.missing.home)} ${esc(g.home.name)}</span></span>
    </button>
    <div class="winbar" title="Win probability"><i style="width:${(1 - hp) * 100}%;background:var(--book)"></i><i style="width:${hp * 100}%;background:var(--model)"></i></div>
    <div class="game-lines">
      <div><span class="muted">Win chance</span><b>${g.home.abbr} ${pct(hp)}</b><span class="muted small">Book ${bookWin}</span></div>
      <div><span class="muted">Spread</span><b>${esc(modelSpread)}</b><span class="muted small">Book ${esc(bookSpread)}</span></div>
      <div><span class="muted">Total</span><b>${f1(g.pred.total)}</b><span class="muted small">Book ${L.total ? L.total.point : "–"}</span></div>
    </div>
    <div class="game-body" hidden>${table("away", g.away)}${table("home", g.home)}
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
      ${sp.low_sample ? `<p class="muted small" style="margin-top:6px">Fewer than 10 games of history, so treat this projection with extra caution.</p>` : ""}</div>`
    : `<p class="muted">Not on the current slate. The line checker uses the recent average instead of a projection.</p>`;
  const opts = STATS.map((s) => `<option value="${s}">${MARKET[s]}</option>`).join("");
  view.innerHTML = `<div class="panel pcard">${head}${projBlock}
    <div class="checker"><h3>Check a line</h3>
      <div class="checker-row"><select id="ck-stat" aria-label="Stat">${opts}</select>
        <input id="ck-line" type="number" step="0.5" inputmode="decimal" placeholder="Line" aria-label="Line">
        <input id="ck-price" type="number" step="1" inputmode="numeric" placeholder="Odds, e.g. -110" aria-label="American odds" style="width:140px"></div>
      <div id="ck-out" class="result muted small">Enter the sportsbook line to see the chance of the over and under.</div>
      <div id="ck-line-viz"></div></div>
    <div><h3 id="bars-title">Last ${logs.length} games</h3><div class="bars" id="bars"></div></div>
    <div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Opp</th><th>Min</th><th>Pts</th><th>Reb</th><th>Ast</th><th>3PM</th></tr></thead>
      <tbody>${logs.map((l) => `<tr><td>${esc(l.d.slice(5))}</td><td>${esc(l.opp)}</td><td>${f1(l.min)}</td><td>${l.pts}</td><td>${l.reb}</td><td>${l.ast}</td><td>${l.fg3m}</td></tr>`).join("")}</tbody></table></div>
  </div>`;

  const statSel = document.getElementById("ck-stat"), lineIn = document.getElementById("ck-line"), priceIn = document.getElementById("ck-price");
  const recentMean = (s) => { const vals = logs.slice(0, 10).map((l) => (s === "pra" ? l.pts + l.reb + l.ast : l[s])); return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0; };
  const update = () => {
    const s = statSel.value, line = parseFloat(lineIn.value), pr = parseFloat(priceIn.value);
    const mu = sp ? sp.proj[s] : recentMean(s), v = sp ? sp.var[s] : variance(s, mu);
    const vals = logs.slice().reverse().map((l) => (s === "pra" ? l.pts + l.reb + l.ast : l[s]));
    const ref = Number.isFinite(line) ? line : mu, max = Math.max(...vals, ref, 1) * 1.1;
    document.getElementById("bars-title").textContent = `${MARKET[s]}, last ${vals.length} games`;
    document.getElementById("bars").innerHTML = vals.map((x) => `<i class="${Number.isFinite(line) && x > line ? "over" : ""}" style="height:${(x / max) * 100}%" title="${x}"></i>`).join("")
      + `<div class="ref" style="bottom:${(ref / max) * 100}%"></div>`;
    const out = document.getElementById("ck-out"), viz = document.getElementById("ck-line-viz");
    if (!Number.isFinite(line)) { out.innerHTML = `${sp ? "Projection" : "Recent average"}: <b class="num">${f1(mu)}</b>. Enter the sportsbook line to see the chance of the over and under.`; viz.innerHTML = ""; return; }
    const pr2 = countProbs(mu, v, line);
    let ev = "";
    if (Number.isFinite(pr) && Math.abs(pr) >= 100) {
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

/* ---------------------------------------------------------------- routing */
function route() {
  const h = location.hash.replace("#", "") || "picks";
  const tab = h.startsWith("p/") ? "players" : h;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  if (h.startsWith("p/")) renderPlayer(decodeURIComponent(h.slice(2)));
  else if (h === "games") renderGames();
  else if (h === "players") renderPlayers();
  else if (h === "record") renderRecord();
  else renderEdges();
  window.scrollTo(0, 0);
}
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => (location.hash = b.dataset.tab)));
window.addEventListener("hashchange", route);

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
    document.getElementById("updated").textContent = `Updated ${new Date(S.meta.generated_at).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}. ${S.meta.has_odds ? "" : "No sportsbook lines in this update."}`;
  } catch (e) {
    console.error(e);
    view.innerHTML = document.getElementById("tpl-error").innerHTML;
  }
})();
