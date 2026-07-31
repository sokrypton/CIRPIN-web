// CIRPIN search — UI. All compute happens in worker.js.

// makeSec is TM-align's Cα-only secondary-structure assignment, reused here so
// the viewer draws real cartoons without a model or side chains.
const { makeSec, smoothSec } = await import('./src/tmalign.js?v=5e652e7b');
// The first-load hero. Dynamic, like the import above, because this file is loaded as a module with
// top-level await rather than with static imports.
const { ouroboros } = await import('./src/ouroboros.js?v=5e652e7b');
const { bestView, fillZoom } = await import('./src/orient.js?v=5e652e7b');
// The Cα cartoon renderer and the drag that turns it, which the atlas page imports too. It lived
// here until the atlas needed it; a second copy of either would have drifted from this one.
const { prep, fitOf, radiusAbout, hexToRgb, drawTraces, makeCamera, orbit, spectrumRgb,
  PAPER, PE_MAX, SIDE_INSET } = await import('./src/trace3d.js?v=5e652e7b');

// Declared up here, not beside the call that starts it: updateChrome() stops the animation once a
// structure loads, and updateChrome runs long before the end of this file. Declared at the bottom,
// every earlier call hit the temporal dead zone and threw — after setting the body class, so the
// page looked right and the hero silently never existed.
let hero = null;
let eggOpen = false;    // the snake game, if someone has found it

const $ = (id) => document.getElementById(id);
const worker = new Worker('./worker.js', { type: 'module' });

let MAX_HITS = 10;   // list depth; the scan is cheap, so this is a display choice
const PROGRES_CUTOFF = 0.6;  // CIRPIN.ipynb cell 9
const CIRPIN_CUTOFF = 0.9;

const INK = '#16202e';
const INK2 = '#5a6a7d';
const RULE = '#c3ccd8';
const CUT = '#d6006e';
const BOND = '#0e7c86';

// Domain colours. Four slots, validated with the dataviz palette checker under
// --pairs all (any two domains can end up adjacent in 3D, so adjacent-only is
// the wrong test): chroma floor, CVD separation and contrast all pass. Five did
// not — green/amber collapse under protanopia and indigo/blue are too close even
// with normal vision. 98% of SCOPe40 multi-domain chains have four or fewer, and
// the rest fall back to grey with the number still carrying identity.
// Eight, and validated rather than picked by eye: tools/palette_check.mjs reports every pair
// clearing 15 in OKLab for normal vision and 8 under deuteranope and protanope simulation, on the
// panel surface. The first seven are the atlas's class colours, so a domain here and a class there
// belong to one vocabulary; the eighth is the one addition that passed — a purple that looked fine
// collided with the blue at 4.3 under deuteranopia. Four was the old count, which sent a fifth
// domain straight to grey.
const DOMAIN_COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
  '#E69F00', '#56B4E9', '#333333', '#8B4513'];
const DOMAIN_EXTRA = '#8d99a8';
const domainColour = (i) => DOMAIN_COLOURS[i] ?? DOMAIN_EXTRA;

/**
 * How a domain id is shown, as opposed to what it is.
 *
 * A TED id is AF-<accession>-F1-model_v4_TED01. Everything except the accession and the domain
 * number is identical on all 3,466,144 rows: the AF- prefix, the fragment number, the model
 * version, and the word TED. That is 18 of 31 characters carrying no information, in a column
 * narrow enough that the rest wrapped onto two lines.
 *
 * So A0A2E0IL81-F1_TED01 becomes A0A2E0IL81_01. The fragment number goes too — TED was built on
 * F1 throughout, so it never varies here; if a database with real fragments ever arrives, this is
 * the line that has to know about it.
 *
 * The canonical id stays on the hit, because that is what addresses the AlphaFold model and what
 * hitUrl parses. Only the display is shortened.
 */
const displayId = (id) => {
  const ted = /^AF-(.+)-F\d+-model_v\d+_TED(\d+)$/.exec(id);
  return ted ? `${ted[1]}_${ted[2]}` : id;
};

/** Where a hit can be looked up, or null if nowhere known. */
function hitUrl(id) {
  const ted = /^AF-([0-9A-Z]+)-F\d+-model_v\d+_TED\d+$/.exec(id);
  if (ted) return `https://alphafold.ebi.ac.uk/entry/${ted[1]}`;
  // SCOPe ids are the sid the domain page is keyed on. The separator is "=", not "/": the path
  // form scop.berkeley.edu/sid/<id> is a plain 404, so every SCOPe link in the app was dead.
  // Checked in a browser rather than with curl — the site answers non-browser clients with a
  // bot-check page for both forms, so the two are indistinguishable from the command line.
  if (/^d[0-9a-z]{4}[0-9a-z_.][0-9a-z_]$/.test(id)) {
    return `https://scop.berkeley.edu/sid=${id}`;
  }
  return null;
}

// Databases the worker offers. Filled in from its ready message, because which ones exist and
// how big they are belongs with the data, not here.
//
// The size in the menu is what SELECTING it costs, not what the database weighs. AlphaFold TED
// occupies 277 MB of index, and that is the number this used to show — but a clustered index is
// read by byte range, so choosing it downloads centroids, cluster offsets, the PCA basis and the
// coarse chopping index, and nothing else. About 0.35 MB, then roughly 1 MB of codes per search.
// Advertising 277 MB was off by a factor of 800 and made the honest choice look expensive.
// Just the display names. The picker shows nothing else, so there is nothing else to carry.
// Versions in the label, because "SCOPe40" alone does not say what you searched. Each is taken from
// the data rather than assumed: SCOPe 2.08 from dir.des.scope.2.08-2023-01-06, CATH v4.4.0 from
// CathDomainList.v4.4.0 (16.12.2024), and AFDB model_v4 from the TED ids themselves, which are
// AF-<accession>-F1-model_v4_TED<nn>. A result is only reproducible against a stated version.
const DB_LABELS = {
  scope40: { label: 'SCOPe40 2.08' },
  cath40: { label: 'CATH40 v4.4' },
  afdb: { label: 'AlphaFoldDB-TED v4' },
};
let databases = [];
let activeDb = 'scope40';
let chains = [];
let activeChain = null;
let parseConfidence = 1;
let coassoc = null;   // {data, bins} co-association matrix from the consensus
let pendingDb = null;   // a switch asked for but not finished
const loadedDbs = new Set();
let DB_SIZE = 0;

let readyStatus = '';
let ready = false;
let reqId = 0;
let coords = null;        // [[x,y,z], ...] for the whole chain
let queryName = '';
let query = null;         // last 'result' message
let activeUnit = 0;
// What this session has pulled over the network, split by who did the pulling: the worker fetches
// the index, the weights and the coordinate ranges, and the page itself fetches structures from
// RCSB. Neither knows about the other, so the total is the sum.
let workerBytes = 0;
let pageBytes = 0;
// The one domain whose results are on screen, or null.
//
// Switching domain used to reveal that domain's hits immediately, because they were already computed
// and sitting in memory. That makes the table change under you from an act that reads as "look at
// this domain", and it means the same click sometimes shows results and sometimes does not, depending
// on whether that unit happens to have been scored already.
//
// One at a time, not a set of everything ever seen: with a set, going back to a domain searched
// earlier showed its hits straight away, so the rule held for some clicks and not others — which is
// worse than either rule on its own. Search is the one thing that puts results on screen; selecting
// a domain only changes which domain Search is about.
let revealedUnit = null;
let parsing = false;   // a parse is streaming its stages in
let hasParsed = false; // the user has asked for a parse on this chain
let killable = true;   // per-bar remove buttons, off while a parse streams
let selectedId = null;
let tab = 'domains';

// --- worker plumbing --------------------------------------------------------

worker.onmessage = (ev) => {
  const m = ev.data;
  // Every reply carries the worker's running byte count, so the total is current without any one
  // message type having to remember to report it.
  if (typeof m.netBytes === 'number' && m.netBytes > workerBytes) {
    workerBytes = m.netBytes;
    renderNet();
  }

  if (m.type === 'progress') { setProgress(m.label, m.detail, m.frac); return; }

  // A structure submitted before the models finished downloading is answered
  // twice — once parsed, once scored — and the scored half can arrive after the
  // user has moved on to something else. Anything carrying a stale id is a reply
  // to a question nobody is asking any more.
  if (m.requestId !== undefined && m.requestId !== reqId
      && (m.type === 'result' || m.type === 'parsed' || m.type === 'aligned')) return;

  /**
   * Anything the worker could not do.
   *
   * This branch did not exist, which meant every worker failure was silent: a database
   * that would not download, an AlphaFold model that no longer exists (UniProt has
   * retired about one TED entry in ten), a chain with no parsable Cα. The worker posted
   * the reason and the UI dropped it, leaving a stuck progress bar and a disabled
   * button as the only evidence. Failing loudly is the minimum; every control that a
   * request disables has to be released here too, or the page is dead until reloaded.
   */
  if (m.type === 'error') {
    setBusy(false);
    hideProgress();
    parsing = false;
    verifyState.running = false;
    $('parseBtn').disabled = false;
    parsing = false;
    syncParseBtn();
    syncFilterBtn();
    // The database menu shows what was asked for, not what loaded; put it back. Clearing the
    // pending switch first is what re-enables it: buildDbMenu disables the menu while one is in
    // flight, so a load that failed — a dropped connection fetching an index — would otherwise
    // leave the only control that could retry it disabled until the page was reloaded.
    pendingDb = null;
    buildDbMenu();
    renderSearchState();
    setMessage(m.message || 'Something went wrong.', true);
    return;
  }

  if (m.type === 'ready') {
    ready = true;
    // On the status dot, where it explains itself only if asked. Users do not need to
    // know what SIMD is; anyone wondering why a search takes 300 ms or 3 s does.
    $('status').title = m.accel
      ? 'Using the WebAssembly SIMD network (about 12x faster than the JavaScript path)'
      : 'Using the JavaScript network — this browser has no WebAssembly SIMD';
    databases = m.databases;
    activeDb = m.active;
    DB_SIZE = m.dbSize;
    for (const d of m.databases) if (d.loaded) loadedDbs.add(d.name);
    readyStatus = 'Ready';
    buildDbMenu();
    updateChrome();     // the database menu becomes usable now, structure or not
    hideProgress();
    return;
  }

  // Parsed and split, but not yet scored: the models were still downloading.
  // Everything that does not need them is live from here — the cartoon, the
  // domain bars, cropping, adding and removing domains.
  if (m.type === 'domainsParsed') {
    // Was also 'domainStep', one message per merge as the search-guided refinement worked, which the
    // bars animated through. That refinement is gone -- it decided domain counts with an uncalibrated
    // cosine -- and so is the CIRPIN/geometric toggle that briefly replaced it, so the parse now arrives
    // in one message and the domain bars are the only way to change it.
    domainSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    if (query) { query.domainSegs = domainSegs; query.nDomains = domainSegs.length; }
    activeUnit = Math.min(activeUnit, domainSegs.length - 1);
    if (m.type === 'domainsParsed') {
      parsing = false;
      hasParsed = true;
      parseMode = 'parsed';
      hideProgress();
      parsing = false;
      $('parseBtn').disabled = false;
      // Offers "Un-parse" now, since the chain has just been split.
      syncParseBtn();
      // markEdited also clears stale results and re-arms Search, which is right: the
      // hits on screen were computed for a different split.
      markEdited();
      pickColourMode(domainSegs.length > 1 ? 'domain' : 'spectrum');
      // The bars say how many there are, one per row.
      setMessage('');
    } else {
      // mid-parse: redraw only, no stale-result bookkeeping for a split that is
      // still being decided. Domain colours go on with the first split so the
      // over-parse and each merge are visible on the structure, not just the bars.
      renderDomains();
      pickColourMode(domainSegs.length > 1 ? 'domain' : chainColourMode);
      drawChain3d();
      if (m.note) setProgress(m.note, '', -1);
    }
    return;
  }

  if (m.type === 'parsed') {
    coords = m.coords;
    activeChain = m.chain;
    chains = m.chains;
    parseConfidence = m.confidence ?? 1;
    coassoc = m.coassoc || null;
    query = { nChain: m.nChain, nDomains: m.domainSegs.length, units: [], domainSegs: m.domainSegs };
    domainSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    autoSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    searchedSegs = null;
    searchedDb = null;
    activeUnit = 0;
    hasParsed = false;      // a new chain has not been asked about yet
    // Parse without being asked. The models were trained on domains, so a whole-chain search is the
    // out-of-distribution case rather than the neutral one -- searching a two-domain chain as one
    // unit asks the library a question it was never shown. It costs about 1.8s on a 200-residue
    // chain and runs while the structure is being looked at, before anyone presses Search, so the
    // wait is spent on something that was going to be waited for anyway. "Un-parse" is the way back
    // to one whole-chain domain for anyone who wants it.
    autoParseWanted = true;
    parseMode = 'none';
    chainState = null;
    selectedId = null;
    clearAlignment();
    $('hitsWrap').hidden = true;
    $('scoresFold').hidden = true;
    $('hitsHead').hidden = true;
    buildChainMenu();
    renderDomains();
    updateChrome();
    pickColourMode(m.domainSegs.length === 1 ? 'spectrum' : 'domain');
    drawChain3d();
    setBusy(false);
    hideProgress();
    // Once the structure is on screen, not before: the parse is the second thing that happens, so
    // the chain appears immediately and the split fills in behind it. Guarded by the flag so a
    // chain switch or a re-parse does not loop.
    if (autoParseWanted) {
      autoParseWanted = false;
      startParse();
    } else {
      syncParseBtn();
    }
    const nres = m.nChain;
    // Nothing. The id is in the field it was typed into, the residue count is on the bar, and the
    // structure itself has just appeared — three ways of already knowing what this line said.
    setMessage(chains.length > 1 ? `chain ${m.chain}` : '');
    // No "not scored yet, then Search". The Search button is filled and the results area is empty:
    // between them they already say it, and a sentence telling someone to press the one lit control
    // on the page is an instruction nobody needed.
    renderSearchState();
    return;
  }

  if (m.type === 'result') {
    coords = m.coords;
    query = m;
      domainSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    searchedSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    searchedDb = activeDb;
    if (!m.edited) autoSegs = m.domainSegs.map((sg) => sg.map((p) => [...p]));
    if (activeUnit >= m.nDomains) activeUnit = 0;
    // THE SEARCH ALREADY COMPUTED THE CONTRIBUTION NORMS. They come off the same forward pass that
    // produced the embedding, so caching them here means picking Contribution after a search costs
    // nothing at all -- no round trip, no second embed. Keyed exactly as requestContrib keys its own
    // asks, so it simply finds them.
    for (const u of m.units) {
      if (u.tooShort || !u.norms) continue;
      contribCache.set(`${queryName}|${u.index}|${segsKey(m.domainSegs)}`,
        packNorms(Array.from(u.residues), u.norms));
    }
    // The search that just finished was asked for while looking at this domain, so this is the one
    // whose results go on screen. The others are scored and waiting, a press away.
    revealedUnit = activeUnit;
    selectedId = null;
    clearAlignment();
    renderDomains();
    renderUnit();
    updateChrome();
    pickColourMode(m.nDomains === 1 ? 'spectrum' : 'domain');
    drawChain3d();
    setBusy(false);
    hideProgress();
    renderSearchState();
    resetVerify();
    return;
  }

  if (m.type === 'rehits') {
    revealedUnit = activeUnit;
    for (const u of m.units) {
      if (!u.hits || !query.units[u.index]) continue;
      query.units[u.index].hits = u.hits;
    }
    selectedId = null;
    clearAlignment();
    renderUnit();
    return;
  }

  if (m.type === 'verdict') {
    applyVerdict(m);
    return;
  }

  if (m.type === 'verifyDone') {
    verifyState.running = false;
    hideProgress();
    syncFilterBtn();
    // The filter drops hits below TM 0.5, and the one being looked at is often among them — it is
    // the low scorers that get looked at least and purged most. Landing on the new best hit is
    // what a search does; leaving three "pick a hit" placeholders and a message still quoting the
    // domain that was just removed is not. Only once, at the end: re-aligning after each of 86
    // individual purges would queue 86 TM-aligns for one answer.
    if (!selectedId) {
      const first = sortedHits.find((h) => h.hasCoords !== false);
      if (first) requestAlign(first);
    }
    return;
  }

  if (m.type === 'dbReady') {
    // Just how the index is read. The byte count used to be appended here too, which is now the
    // figure in the corner of the header — where it is visible rather than hidden in a tooltip.
    $('status').title = m.ivf
      ? 'Clustered index: only the nearest clusters are fetched per search'
      : 'Whole index held in memory';
    loadedDbs.add(m.db);
    pendingDb = null;
    activeDb = m.db;
    DB_SIZE = m.n;
    readyStatus = 'Ready';
    hideProgress();
    // Whatever the message line was saying belonged to the old database, and if it was an error
    // from a switch that failed, it would otherwise sit there in red over a page that now works.
    setMessage('');
    buildDbMenu();
    // Results belong to the database they were computed against, so switching
    // clears them rather than leaving numbers that mean something else.
    if (query) {
      $('hitsWrap').hidden = true;
      $('scoresFold').hidden = true;
      $('hitsHead').hidden = true;
      clearAlignment();
      // In the strip, not in the results column. Everything about what the page is doing now goes
      // to one place; the column below is for what came of it.
      setStatus(`Now searching ${m.label}`, 'ready');
      readyStatus = `Now searching ${m.label}`;
      renderSearchState();
    }
    return;
  }

  if (m.type === 'contrib') {
    const res = contribAsked;
    if (res && m.norms.length === res.length) {
      const packed = packNorms(res, m.norms);
      contribCache.set(m.key ?? contribKey(), packed);
      contribNorms = packed;
      if (chainColourMode === 'contrib') drawChain3d();
    }
    return;
  }

  if (m.type === 'aligned') { hideProgress(); renderAlignment(m); }
};

worker.postMessage({ type: 'init' });

// --- helpers ----------------------------------------------------------------

function setStatus(text, cls = '') { const s = $('status'); s.textContent = text; s.className = cls; }

/**
 * Progress feedback.
 *
 * Reported checkpoints are sparse — six per model per domain while embedding,
 * three across TM-align — so painting only on arrival leaves the bar sitting
 * still for a hundred milliseconds at a time, which reads as a hang. Instead a
 * requestAnimationFrame loop eases toward the last checkpoint and then keeps
 * creeping past it, asymptotically, toward where the next one is expected. The
 * creep rate comes from the size of the previous real step, so the estimate
 * tracks whatever the work is actually doing rather than a fixed guess.
 *
 * It never reaches the ceiling and never reaches 100% until the work reports
 * done, so it cannot claim to have finished early and it cannot go backwards.
 */
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)');
const prog = { shown: 0, target: 0, step: 0.08, raf: 0, last: 0 };

function paintProgress() {
  $('progBar').style.width = `${(prog.shown * 100).toFixed(1)}%`;
}

/**
 * Start a fresh job. The target only ever moves forward within a job, so it has
 * to be cleared between them — otherwise a completed job leaves the target at 1
 * and the next one's early fractions are all below the ceiling, pinning the bar
 * at 98.5% and never moving again.
 */
function resetProgress() {
  if (prog.raf) { cancelAnimationFrame(prog.raf); prog.raf = 0; }
  prog.shown = 0;
  prog.target = 0;
  prog.step = 0.08;
  paintProgress();
}

function animateProgress(now) {
  const dt = Math.min(0.1, (now - prog.last) / 1000);
  prog.last = now;
  // Creep at most nine tenths of one expected step past the last checkpoint,
  // and hold short of the end until completion is actually reported.
  const ceiling = Math.min(0.985, prog.target + prog.step * 0.9);
  if (prog.shown < ceiling) {
    prog.shown += (ceiling - prog.shown) * (1 - Math.exp(-2.6 * dt));
    paintProgress();
  }
  prog.raf = requestAnimationFrame(animateProgress);
}

function setProgress(label, detail, frac) {
  if (label === '' && detail === '') { hideProgress(); return; }
  $('progTrack').classList.add('on');
  $('status').innerHTML = `<b>${label}</b>${detail ? ` <span class="detail">${detail}</span>` : ''}`;
  $('status').className = '';

  if (typeof frac === 'number' && frac >= 0) {
    if (frac > prog.target) {
      prog.step = Math.max(0.02, Math.min(0.25, frac - prog.target));
      prog.target = frac;
    }
  }
  if (REDUCED.matches) {
    prog.shown = Math.max(prog.shown, prog.target);
    paintProgress();
    return;
  }
  if (!prog.raf) { prog.last = performance.now(); prog.raf = requestAnimationFrame(animateProgress); }
}

function hideProgress() {
  if (prog.raf) { cancelAnimationFrame(prog.raf); prog.raf = 0; }
  // Run to full before clearing, so the bar reads as finished rather than
  // vanishing from wherever it happened to be.
  prog.shown = 1;
  prog.target = 0;
  prog.step = 0.08;
  paintProgress();
  setTimeout(() => {
    if (prog.raf) return;                 // new work started in the meantime
    $('progTrack').classList.remove('on', 'indeterminate');
    prog.shown = 0;
    paintProgress();
  }, 220);
  if (readyStatus) setStatus(readyStatus, 'ready');
}

/** The running transfer total, in the corner of the header. */
function renderNet() {
  const mb = (workerBytes + pageBytes) / 1e6;
  $('netTotal').textContent = `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

function setMessage(text, isError = false) {
  const el = $('message');
  el.textContent = text;
  el.className = isError ? 'error' : '';
}
function setBusy(on) {
  $('fetchBtn').disabled = on;
  // Progress goes in the status strip, never in the message line.
  //
  // #message sits in the work column under the hero, so on the landing page a "Reading the
  // structure" there printed under the snake — while the strip at the top, which exists precisely
  // to say what the page is doing, sat on "Ready". One place for what is happening now, one place
  // for what came of it: the message line is for the result, and for errors, which have to stay put
  // and be read rather than scroll past in a status bar.
  if (on) setProgress('Reading the structure', '', 0.05);
}
const unit = () => query?.units?.[activeUnit] ?? null;

/**
 * Load a structure. Parsing only — scoring waits for Search.
 *
 * A file may hold several chains and the automatic split may not be the one you
 * want, so scoring immediately would spend a search on choices the user has not
 * made yet. Parsing needs no weights either, so this answers straight away even
 * while the models are still downloading.
 */
function submit(text, format, name) {
  queryName = name;
  activeUnit = 0;
  hasParsed = false;
  autoSegs = null;
  searchedSegs = null;
  searchedDb = null;
  chainState = null;
  revealedUnit = null;
  viewChosen = false;             // a new structure gets its own best view
  if (ready) resetProgress();
  setBusy(true);
  worker.postMessage({ type: 'parse', requestId: ++reqId, text, format });
}

/** Re-score with a given domain definition, hand-edited or reset to automatic. */
function rescore(segs, edited) {
  if (!coords) return;
  resetProgress();
  setBusy(true);
  setProgress('Re-scoring domains', '', 0.05);
  worker.postMessage({
    type: 'embed', requestId: ++reqId, coords, edited,
    domainSegs: segs.map((sg) => sg.map((p) => [...p])),
    maxHits: MAX_HITS, progresCutoff: PROGRES_CUTOFF, cirpinCutoff: CIRPIN_CUTOFF,
  });
}

// --- tabs -------------------------------------------------------------------

// Two tabs, because there are two questions: how the chain is carved up, and how
// one hit lines up against it. The three ways of showing an alignment are the
// same alignment drawn three ways, so they are a choice inside the second tab
// rather than three peers of it — the same shape the domain controls already
// use. The score plane is neither: it is context for the table, and lives with
// it further down the page.
const PANELS = { domains: 'panelDomains', align: 'panelAlign' };
const TABS = { domains: 'tabDomains', align: 'tabAlign' };

const ALN_VIEWS = { side: 'subSide', super: 'subSuper', map: 'subMap' };
const ALN_BUTTONS = { side: 'alnSide', super: 'alnSuper', map: 'alnMap' };
let alnView = 'side';

/**
 * Hide what cannot be used yet, rather than disabling it.
 *
 * A greyed control still invites a click and still has to be explained. Nothing
 * here is a mode the user can turn on early — the Alignment tab needs a hit to
 * have been aligned, "Contribution" needs the full-chain pass to have finished —
 * so the honest state is absent, not present-but-dead.
 */
function updateChrome() {
  const hasChain = !!coords;
  $('tabDomains').hidden = !hasChain;
  $('colourRow').hidden = !hasChain;
  // Not until there is something to search: choosing a database is a decision
  // about a query, and offering it against no query is offering nothing.
  $('dbRow').hidden = !hasChain;
  // Before anything is loaded there is nothing to view, so the viewer column is
  // not there either — an empty half-page explaining that it is empty is worse
  // than the input being the only thing on screen.
  document.body.classList.toggle('no-structure', !hasChain);
  // The hero is for the empty page. Once a structure is on screen it is not just hidden but
  // stopped, because an animation nobody can see is still a frame of work every 16ms.
  if (hasChain && hero) { hero.stop(); hero = null; }
  $('tabAlign').hidden = !alignAvailable;
  document.querySelector('.tabs').hidden = !hasChain;
  if (!alignAvailable && tab === 'align') showTab('domains');
}

function showTab(which) {
  tab = which;
  for (const k of Object.keys(PANELS)) {
    $(PANELS[k]).hidden = k !== which;
    $(TABS[k]).setAttribute('aria-selected', String(k === which));
  }
  // The bars can only be laid out once their pane has a width, so they are drawn on the way in
  // rather than while hidden. Cheap — a handful of rects per domain — and it is what guarantees
  // the editor is there whenever the tab showing it is.
  if (which === 'domains') drawBars();
  repaint();
}
Object.keys(TABS).forEach((k) => $(TABS[k]).addEventListener('click', () => showTab(k)));

function showAlnView(which) {
  alnView = which;
  for (const k of Object.keys(ALN_VIEWS)) {
    $(ALN_VIEWS[k]).hidden = k !== which;
    $(ALN_BUTTONS[k]).setAttribute('aria-pressed', String(k === which));
  }
  repaint();
}
Object.keys(ALN_BUTTONS).forEach((k) => {
  $(ALN_BUTTONS[k]).addEventListener('click', () => showAlnView(k));
});

function repaint() {
  if (tab === 'domains') { drawChain3d(); }
  else if (alnView === 'side') paintSide();
  else if (alnView === 'super') paintSuper();
  else paintMap();
  // Only costs a paint when it is on screen. This used to test `.open`, which is a
  // <details> property and undefined on the <section> it became, so the plane
  // stopped painting entirely.
  if (!$('scoresFold').hidden) paintScores();
}

/**
 * Tallest a plot may be.
 *
 * The viewer column is sticky, so anything taller than the window is permanently
 * half off-screen — a square canvas as wide as the column was exactly that. Views
 * are fitted to this instead and left narrower than the column when they have to
 * be, since a plot you can see all of beats a slightly larger one you cannot.
 */
// ONE height for every 3D view.
//
// The input panel used to get 0.44 of the window and the alignment views 0.62, on the reasoning that
// the input panel has a domain editor to fit underneath. The cost was that the canvas changed height
// when you switched tabs, so the structure jumped and resized for a reason that has nothing to do
// with the structure. Sharing the smaller of the two keeps the viewer still across modes -- and it
// is the value that leaves room for the editor, so nothing else has to move.
const VIEW_FRAC = 0.44;
const CHAIN_VIEW_FRAC = VIEW_FRAC;

const viewMaxH = (frac = VIEW_FRAC) => Math.max(240, Math.round(window.innerHeight * frac));


// --- domain editor ----------------------------------------------------------
// One bar per domain, each edited on its own. Domains are segment lists rather
// than a partition of the chain, which is how SCOPe actually defines them:
// linker residues belong to no domain, a domain can be discontinuous, and
// cropping one should not shove the removed residues into a neighbour. Two
// domains may overlap if that is genuinely what you want to search.
//
//   drag a segment            move it
//   drag either edge          crop or extend
//   double-click empty track  add a segment to that domain
//   + / −                     add or remove a whole domain

const SVGNS = 'http://www.w3.org/2000/svg';
const MIN_SEG = 4;            // coordsToGraph needs four Cα for the torsion

let autoParseWanted = false;  // a freshly loaded chain wants splitting without being asked
// 'none' before a parse, 'parsed' after one, 'unparsed' after Un-parse. What the button offers is
// derived from this rather than from the shape of the segments, which cannot tell "never parsed"
// from "parsed and the answer was one domain".
let parseMode = 'none';
let domainSegs = null;        // [[[s,e], ...], ...] — live, possibly unsearched
let autoSegs = null;
let searchedSegs = null;      // what the current results were actually computed from
let searchedDb = null;        // and which database they were computed against

const clampSeg = (a, b, n) => [Math.max(0, Math.min(a, b)), Math.min(n - 1, Math.max(a, b))];
const segsKey = (d) => JSON.stringify(d);

/**
 * True when the shown results no longer describe the current question.
 *
 * That is either the domains having been edited or the database having been
 * switched underneath them. Both make the numbers on screen answers to something
 * nobody asked, and both are fixed the same way — by scoring again.
 */
function isDirty() {
  if (!query || !domainSegs) return false;
  if (!searchedSegs) return true;                      // loaded but never scored
  if (searchedDb && searchedDb !== activeDb) return true;
  return segsKey(domainSegs) !== segsKey(searchedSegs);
}

/**
 * Editing no longer searches on its own.
 *
 * Re-embedding is a second or two per domain, so auto-scoring every drag made
 * three adjustments cost three searches nobody asked for, and threw the
 * intermediate ones away. Edits update the bars and the structure view, which
 * are cheap and local, and clear the results: hits computed from a different
 * domain definition are not the answer to the current one, and leaving them on
 * screen invites reading them as if they were.
 */
function markEdited() {
  revealedUnit = null;            // nothing on screen survives a change to the boundaries
  // Moving a boundary changes the graph, so it changes what every residue in it contributes. The
  // cache is keyed on the boundaries, so this is a fresh question rather than an invalidation.
  if (chainColourMode === 'contrib') requestContrib();
  renderDomains();
  drawChain3d();
  const dirty = isDirty();
  renderSearchState();
  $('hitsWrap').hidden = dirty;
  $('scoresFold').hidden = dirty;
  if (dirty) {
    if (searchedDb && searchedDb !== activeDb) {
      const lbl = (DB_LABELS[activeDb] || {}).label || activeDb;
    } else {
      // No sentence. The results are gone and Search is lit; that is the whole message.
    }
  }
}

/**
 * Has the chain actually been divided?
 *
 * A single domain spanning every residue is the undivided state, not a parse of one
 * domain — and it is what a freshly loaded chain always looks like. Anything else
 * (more than one domain, or one that has been cropped) is a real division and gets
 * the full editor.
 */
function hasDomains() {
  if (!query || !domainSegs) return false;
  // A parse that answered "one domain" is still an answer, and the editor is how you
  // disagree with it — so asking once opens it for good, whatever it found.
  if (hasParsed) return true;
  if (domainSegs.length !== 1) return true;
  return segsSize(domainSegs[0]) !== query.nChain;
}

function segsSize(segs) {
  const seen = new Set();
  for (const [a, b] of segs) for (let r = a; r <= b; r++) seen.add(r);
  return seen.size;
}

function renderDomains() {
  const box = $('domains');
  if (!query || !domainSegs) { box.hidden = true; return; }
  box.hidden = false;

  // The editor is on from the moment a structure loads.
  //
  // It used to appear only once the chain had been divided, on the reasoning that one bar spanning
  // everything reads as a result — as though a parser had looked and found one domain. But it also
  // meant the obvious thing to do with a chain you want to crop, drag its edge in, was not offered
  // until you had run a parser you may not want; and the auto parse button then hid itself once a
  // split existed, so there was no way back to it short of reloading the page. Both were the same
  // mistake: treating the automatic parse as the way in rather than as one of the tools.
  const divided = hasDomains();
  // Mid-parse the bars are the whole point — they are what shows the over-split collapsing — but
  // editing controls are not offered for a split still being decided.
  const settled = !parsing;
  $('domainBars').hidden = false;
  $('domainAxis').parentElement.hidden = false;
  $('domainsEyebrow').hidden = false;
  $('addDomain').hidden = !settled;
  $('parseBtn').hidden = false;
  $('domainNote').hidden = !settled;
  killable = settled;
  // Hidden, not disabled: before a parse there are no domains to colour by, and a
  // greyed-out control still invites a click. The seg group already drops the shared
  // border when its first button is hidden.
  // Colouring by domain still waits for a real division: with one domain spanning the chain it
  // would paint the whole thing one flat colour, which looks like a bug rather than an answer.
  $('colByDomain').hidden = !divided;

  const covered = new Set();
  for (const segs of domainSegs) for (const [a, b] of segs) for (let r = a; r <= b; r++) covered.add(r);
  const unassigned = query.nChain - covered.size;
  // Only what cannot be seen on the bars.
  //
  // The residue count and the number of domains were both a restatement: the count is in the header
  // line and the domains are the bars themselves, one per row, right underneath. What is left is the
  // two things the picture does not show — residues claimed by no domain, and a split the perturbed
  // runs disagreed about.
  const marginal = parseConfidence < 0.999
    ? `split is marginal · ${Math.round(parseConfidence * 100)}% of runs agreed` : '';
  const orphan = unassigned ? `${unassigned} residue${unassigned === 1 ? '' : 's'} in no domain` : '';
  $('domainNote').innerHTML = [orphan, marginal].filter(Boolean).join(' · ');
  $('addDomain').disabled = domainSegs.length >= 12;
  drawBars();
}

function drawBars() {
  const host = $('domainBars');
  // Never rebuild into a hidden pane.
  //
  // This function clears the rows and then paintBar measures the svg to lay them out — but an
  // element inside a hidden panel measures zero width, and paintBar cannot draw at zero, so the
  // bars ended up cleared and never redrawn. Reachable in one move: resize the window while the
  // Alignment tab is up. The resize handler calls this, the domain editor is hidden at the time,
  // and switching back to it showed an empty box where the boundary editor had been — which looks
  // exactly like the editor having been removed. showTab() repaints on the way back in, so
  // skipping here loses nothing.
  if (host.hidden || !host.offsetParent) return;
  host.textContent = '';
  if (!query || !domainSegs) return;
  const n = query.nChain;

  domainSegs.forEach((segs, di) => {
    const row = document.createElement('div');
    row.className = 'dbar';
    row.dataset.active = String(di === activeUnit);
    // The whole box selects, not only the number or the bar: the box is what is
    // outlined when selected, so the box is what a person aims at.
    row.addEventListener('click', () => selectDomain(di));

    // A dot in the domain's colour rather than its number: the rows are in order, so the number
    // said nothing, and the colour is the one thing about a domain that appears elsewhere — on the
    // structure, and in the score plane. It still selects, like the rest of the row.
    const num = document.createElement('span');
    num.className = 'n';
    num.title = `domain ${di + 1}`;
    const dot = document.createElement('i');
    dot.style.background = domainColour(di);
    num.appendChild(dot);
    num.addEventListener('click', () => selectDomain(di));
    row.appendChild(num);

    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('preserveAspectRatio', 'none');
    row.appendChild(svg);

    const size = segsSize(segs);
    const warn = document.createElement('span');
    warn.className = 'warn';
    if (size < MIN_SEG) {
      warn.textContent = '!';
      warn.title = `${size} residues — at least ${MIN_SEG} are needed to embed`;
    }
    row.appendChild(warn);


    const kill = document.createElement('button');
    kill.className = 'kill';
    kill.type = 'button';
    kill.textContent = '−';
    kill.title = 'remove this domain';
    kill.disabled = domainSegs.length < 2 || !killable;
    kill.addEventListener('click', (e) => {
      e.stopPropagation();
      domainSegs.splice(di, 1);
      if (activeUnit >= domainSegs.length) activeUnit = domainSegs.length - 1;
      commitDomains();
    });
    row.appendChild(kill);

    host.appendChild(row);
    paintBar(svg, segs, di, n);
  });

  drawAxis(n);
  drawConsensus(n);
}

/** Make one domain the one on screen. */
function selectDomain(di) {
  if (di === activeUnit) return;
  activeUnit = di;
  selectedId = null;
  clearAlignment();
  // The contribution is per domain, so selecting one asks for its numbers — from the cache if they
  // have been computed before, otherwise from the worker. Without this, switching domain left the
  // previous domain's shading on screen, or nothing at all on a domain never looked at.
  if (chainColourMode === 'contrib') requestContrib();
  // Whatever was on screen belonged to the domain being left, so it goes. Nothing is shown for the
  // domain being entered until Search is pressed, computed or not.
  revealedUnit = null;
  $('hitsHead').hidden = true;
  $('hitsWrap').hidden = true;
  $('scoresFold').hidden = true;
  renderSearchState();
  drawBars();
  drawChain3d();
}

function paintBar(svg, segs, di, n) {
  const w = svg.clientWidth || svg.getBoundingClientRect().width;
  if (!w) return;
  svg.dataset.di = String(di);   // so a live drag can repaint without re-deriving it
  const h = 22;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const xOf = (r) => (r / n) * w;
  const rOf = (x) => Math.round((x / w) * n);
  const colour = domainColour(di);
  const active = di === activeUnit;

  const track = document.createElementNS(SVGNS, 'rect');
  track.setAttribute('x', 0); track.setAttribute('y', 6);
  track.setAttribute('width', w); track.setAttribute('height', 10);
  track.setAttribute('fill', '#e3e8ee');
  svg.appendChild(track);

  svg.addEventListener('click', () => selectDomain(di));
  svg.addEventListener('dblclick', (e) => {
    // add a segment centred where the user clicked, in the largest free gap
    const r = svg.getBoundingClientRect();
    const at = rOf(e.clientX - r.left);
    if (segs.some(([a, b]) => at >= a && at <= b)) return;
    const half = Math.max(MIN_SEG, Math.round(n * 0.06));
    segs.push(clampSeg(at - half, at + half, n));
    segs.sort((p, q) => p[0] - q[0]);
    commitDomains();
  });

  segs.forEach((seg, si) => {
    const x = xOf(seg[0]);
    const bw = Math.max(2, xOf(seg[1] + 1) - x);

    const body = document.createElementNS(SVGNS, 'rect');
    body.setAttribute('class', 'seg');
    body.setAttribute('x', x); body.setAttribute('y', 4);
    body.setAttribute('width', bw); body.setAttribute('height', 14);
    body.setAttribute('fill', colour);
    body.setAttribute('fill-opacity', active ? 1 : 0.4);
    body.setAttribute('stroke', active ? INK : 'none');
    body.setAttribute('stroke-width', active ? 1 : 0);
    svg.appendChild(body);

    for (const [side, at] of [['lo', seg[0]], ['hi', seg[1] + 1]]) {
      const g = document.createElementNS(SVGNS, 'g');
      g.setAttribute('class', 'edge');
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('x1', xOf(at)); line.setAttribute('x2', xOf(at));
      line.setAttribute('y1', 1); line.setAttribute('y2', 21);
      line.setAttribute('stroke', active ? INK : '#8d99a8');
      line.setAttribute('stroke-width', 2);
      const grab = document.createElementNS(SVGNS, 'rect');
      grab.setAttribute('x', xOf(at) - 6); grab.setAttribute('y', 0);
      grab.setAttribute('width', 12); grab.setAttribute('height', h);
      grab.setAttribute('fill', 'transparent');
      grab.style.cursor = 'ew-resize';
      attachEdgeDrag(grab, svg, segs, si, side, n);
      g.appendChild(line); g.appendChild(grab);
      svg.appendChild(g);
    }
  });
}

/**
 * The consensus matrix: how often each pair of residues was parsed together.
 *
 * The confidence figure says how often the runs agreed on a domain count; this
 * says where they disagreed. A dark square on the diagonal is a domain every run
 * kept whole; how sharp its corner is tells you how firm the boundary is. The
 * committed boundaries are drawn over it, so a line sitting inside a smear is a
 * cut the parser picked out of several it could equally have picked.
 *
 * Drawn on the same grid as the bars, so a column sits above the residue it is
 * about and the boundaries line up with what is being dragged.
 */
function drawConsensus(n) {
  const row = $('coRow');
  const canvas = $('coassoc');
  if (!coassoc || !coassoc.bins) { row.hidden = true; return; }
  row.hidden = false;
  if (coassoc.runs) $('coRuns').textContent = String(coassoc.runs);

  // Square, and therefore narrower than the bars. A matrix stretched to the bar
  // width would put x and y on different scales, which turns the diagonal into a
  // slant and the domain blocks into rectangles — unreadable as a matrix, which is
  // the whole point of drawing one.
  const avail = canvas.parentElement.clientWidth;
  if (!avail) return;
  const side = Math.min(avail, Math.round(window.innerHeight * 0.26));
  canvas.style.width = `${side}px`;
  const p = prep(canvas, side);
  if (!p) return;
  const { ctx } = p;
  const wCss = side;

  const B = coassoc.bins;
  // Which domain each bin currently belongs to, so a block is drawn in the colour
  // that domain wears on the bars and in the structure. A cell whose two bins are
  // in different domains gets neutral ink instead: it is still saying "these were
  // parsed together this often", but it is not evidence about any one domain.
  const binOwner = binOwners(B, n);
  const rgbFor = [];
  for (let d = 0; d < (domainSegs || []).length; d++) rgbFor.push(hexToRgb(domainColour(d)));
  const NEUTRAL = [90, 106, 125];

  const img = ctx.createImageData(B, B);
  for (let a = 0; a < B; a++) {
    for (let b = 0; b < B; b++) {
      const i = a * B + b;
      const v = coassoc.data[i] / 255;
      const e = v ** 0.85;
      const same = binOwner[a] >= 0 && binOwner[a] === binOwner[b];
      const base = same ? (rgbFor[binOwner[a]] || NEUTRAL) : NEUTRAL;
      // a domain that is not the one being searched is muted, like the bars
      const mute = same && binOwner[a] !== activeUnit ? 0.55 : 1;
      const o = i * 4;
      for (let k = 0; k < 3; k++) {
        img.data[o + k] = Math.round(PAPER[k] + (base[k] - PAPER[k]) * e * mute);
      }
      img.data[o + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(B, B);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, wCss, side);

  // the split actually committed, over the top
  ctx.strokeStyle = CUT;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  const edges = new Set();
  for (const segs of domainSegs || []) {
    for (const [a, b] of segs) { edges.add(a); edges.add(b + 1); }
  }
  ctx.beginPath();
  for (const r of edges) {
    if (r <= 0 || r >= n) continue;
    const x = (r / n) * wCss;
    const y = (r / n) * side;
    ctx.moveTo(x, 0); ctx.lineTo(x, side);
    ctx.moveTo(0, y); ctx.lineTo(wCss, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // the box being dragged, and its mirror, since the matrix is symmetric
  if (coSel) {
    const px = (bin) => (bin / B) * wCss;
    const py = (bin) => (bin / B) * side;
    const [a0, a1] = [Math.min(coSel.a0, coSel.a1), Math.max(coSel.a0, coSel.a1) + 1];
    const [b0, b1] = [Math.min(coSel.b0, coSel.b1), Math.max(coSel.b0, coSel.b1) + 1];
    ctx.strokeStyle = CUT;
    ctx.lineWidth = 1.5;
    ctx.fillStyle = 'rgba(214,0,110,.16)';
    for (const [x0, x1, y0, y1] of [[a0, a1, b0, b1], [b0, b1, a0, a1]]) {
      ctx.fillRect(px(x0), py(y0), px(x1) - px(x0), py(y1) - py(y0));
      ctx.strokeRect(px(x0) + 0.5, py(y0) + 0.5, px(x1) - px(x0) - 1, py(y1) - py(y0) - 1);
    }
  }

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, wCss - 1, side - 1);
}

/** Which domain owns the residue at the middle of each bin. */
function binOwners(B, n) {
  const owner = new Int32Array(B).fill(-1);
  (domainSegs || []).forEach((segs, d) => {
    for (const [a, b] of segs) {
      const lo = Math.floor((a / n) * B);
      const hi = Math.min(B - 1, Math.floor((b / n) * B));
      for (let k = lo; k <= hi; k++) owner[k] = d;
    }
  });
  return owner;
}

/**
 * Drag a box on the matrix to make its residues a domain.
 *
 * The matrix is symmetric, so a box at (rows, columns) names two stretches of
 * chain and the same statement appears twice — the box and its mirror are both
 * drawn. On the diagonal that is one contiguous range. Off the diagonal it is two
 * separated stretches, and a bright off-diagonal block is precisely the signal
 * that they belong together: dragging it builds the discontinuous domain the
 * picture is telling you about, which is otherwise fiddly to enter by hand.
 *
 * Released selections go to the domain being searched, so the loop is: look at
 * the blocks, box the one you believe, score it.
 */
let coSel = null;

const coBin = (e) => {
  const c = $('coassoc');
  const r = c.getBoundingClientRect();
  const B = coassoc.bins;
  return {
    a: Math.max(0, Math.min(B - 1, Math.floor(((e.clientX - r.left) / r.width) * B))),
    b: Math.max(0, Math.min(B - 1, Math.floor(((e.clientY - r.top) / r.height) * B))),
  };
};

/** Bin range -> residue range, clamped to the chain. */
const binsToRes = (lo, hi, B, n) => [
  Math.max(0, Math.floor((Math.min(lo, hi) * n) / B)),
  Math.min(n - 1, Math.ceil(((Math.max(lo, hi) + 1) * n) / B) - 1),
];

/** The two stretches a box names, merged if they touch. */
function selSegments() {
  const B = coassoc.bins;
  const n = query.nChain;
  const one = binsToRes(coSel.a0, coSel.a1, B, n);
  const two = binsToRes(coSel.b0, coSel.b1, B, n);
  const segs = [one, two].sort((p, q) => p[0] - q[0]);
  if (segs[1][0] <= segs[0][1] + 1) return [[segs[0][0], Math.max(segs[0][1], segs[1][1])]];
  return segs;
}

$('coassoc').addEventListener('pointerdown', (e) => {
  if (!coassoc || !query) return;
  e.preventDefault();
  const { a, b } = coBin(e);
  coSel = { a0: a, a1: a, b0: b, b1: b };
  try { $('coassoc').setPointerCapture(e.pointerId); } catch { /* not capturable */ }
  drawConsensus(query.nChain);
});

$('coassoc').addEventListener('pointermove', (e) => {
  if (!coassoc || !query) return;
  const { a, b } = coBin(e);
  if (coSel) {
    coSel.a1 = a;
    coSel.b1 = b;
    drawConsensus(query.nChain);
    // preview the selection on the structure, in place of the active domain
    const live = domainSegs.map((segs, d) => (d === activeUnit ? selSegments() : segs));
    drawChain3d(live);
    const size = selSegments().reduce((t, [x, y]) => t + (y - x + 1), 0);
    $('domainRead').textContent = `${size} residues in `
      + `${selSegments().length === 1 ? 'one stretch' : 'two stretches'} `
      + `\u2192 domain ${activeUnit + 1}`;
    return;
  }
  const B = coassoc.bins;
  const pct = Math.round((coassoc.data[b * B + a] / 255) * 100);
  const ra = Math.round((a + 0.5) * query.nChain / B) + 1;
  const rb = Math.round((b + 0.5) * query.nChain / B) + 1;
  $('domainRead').textContent = `residues ${ra} and ${rb} parsed together in ${pct}% of runs`;
});

addEventListener('pointerup', () => {
  if (!coSel) return;
  const dragged = coSel.a0 !== coSel.a1 || coSel.b0 !== coSel.b1;
  const segs = dragged ? selSegments() : null;
  coSel = null;
  if (!segs) { drawConsensus(query.nChain); return; }
  domainSegs[activeUnit] = segs.map((p) => [...p]);
  commitDomains();
});

$('coassoc').addEventListener('pointerleave', () => {
  if (!coSel) $('domainRead').textContent = '';
});

function drawAxis(n) {
  const svg = $('domainAxis');
  const w = svg.clientWidth || svg.getBoundingClientRect().width;
  if (!w) return;
  svg.setAttribute('viewBox', `0 0 ${w} 16`);
  svg.setAttribute('preserveAspectRatio', 'none');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const step = n > 600 ? 200 : n > 300 ? 100 : n > 120 ? 50 : 25;
  for (let r = 0; r <= n; r += step) {
    const t = document.createElementNS(SVGNS, 'text');
    t.setAttribute('x', (r / n) * w);
    t.setAttribute('y', 11);
    t.setAttribute('text-anchor', r === 0 ? 'start' : (r + step > n ? 'end' : 'middle'));
    t.setAttribute('font-family', 'ui-monospace, Menlo, monospace');
    t.setAttribute('font-size', '9.5');
    t.setAttribute('fill', INK2);
    t.textContent = String(r);
    svg.appendChild(t);
  }
}

/**
 * Drag one boundary of one segment.
 *
 * Dragging used to be on the whole bar and was removed for firing by accident: a click meant to
 * select a domain would nudge its boundary, and the nudge was indistinguishable from an intended
 * edit. It is back, narrowed to the two edge handles and gated on movement — until the pointer has
 * travelled DRAG_SLOP pixels nothing happens at all, so a click that happens to land on a handle
 * still just selects. Below that threshold the gesture is a click; above it, it is a drag, and
 * once it is a drag the click is suppressed so the row does not also change selection.
 *
 * The whole bar dragging as a unit is deliberately NOT restored: that is the gesture that made
 * accidents, and the edges are what anyone actually wants to move.
 */
const DRAG_SLOP = 3;

function attachEdgeDrag(grab, svg, segs, si, side, n) {
  grab.addEventListener('pointerdown', (e) => {
    e.stopPropagation();      // do not let the bar's click handler see this
    const box = svg.getBoundingClientRect();
    const startX = e.clientX;
    const before = [segs[si][0], segs[si][1]];
    let live = false;
    let pending = 0;

    const move = (ev) => {
      if (!live && Math.abs(ev.clientX - startX) < DRAG_SLOP) return;
      live = true;
      const at = Math.round(((ev.clientX - box.left) / box.width) * n);
      // The moving edge follows the pointer; the other stays. MIN_SEG apart, so a segment can
      // never be dragged shorter than coordsToGraph can embed.
      const seg = side === 'lo'
        ? clampSeg(Math.min(at, before[1] - MIN_SEG + 1), before[1], n)
        : clampSeg(before[0], Math.max(at - 1, before[0] + MIN_SEG - 1), n);
      segs[si] = seg;
      paintBar(svg, segs, Number(svg.dataset.di), n);
      // The bar is cheap; the structure is not. Coalescing its repaint to one a frame keeps a
      // drag smooth on a 400-residue chain instead of queueing a full cartoon render per
      // pointermove, which fires far faster than the screen refreshes.
      if (!pending) {
        pending = requestAnimationFrame(() => { pending = 0; drawChain3d(); });
      }
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (pending) { cancelAnimationFrame(pending); pending = 0; }
      if (!live) return;                       // never moved: leave it to the click
      segs.sort((p, q) => p[0] - q[0]);
      commitDomains();
      drawBars();
      renderUnit();
      // one click event follows a pointerup; swallow it so selection does not also change
      svg.addEventListener('click', (ce) => ce.stopPropagation(), { capture: true, once: true });
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    try { grab.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
  });
}

/**
 * Whether scoring is owed, shown on the toolbar button and explained by the line
 * where the table used to be.
 *
 * The button lives in the toolbar with everything else that starts work, so it is
 * always in the same place; it just goes dim when there is nothing to do and takes
 * the accent colour when there is.
 */
function renderSearchState() {
  const dirty = isDirty();
  // Owed because the domains changed, because this domain's results have not been asked for yet, or
  // because this domain has no scored unit at all.
  //
  // That last clause is the fix for a dead Search button. The press handler used to set
  // revealedUnit = activeUnit and call renderUnit(), which returns silently when there is no unit for
  // the active domain (unit() is a plain index lookup and can miss). The reveal therefore did
  // nothing, but the claim stuck, so owed went false and the button went dim with nothing on screen
  // and no way to ask again. Basing this on whether results EXIST rather than on an index equality
  // makes that state unreachable however the mismatch arose.
  const owed = dirty || (!!query && (revealedUnit !== activeUnit || !query.units?.[activeUnit]));
  const btn = $('searchBtn');
  btn.disabled = !owed;
  btn.classList.toggle('due', owed);
}

/** Clean up the edited definition and mark the results stale. */
function commitDomains() {
  const cleaned = domainSegs
    .map((segs) => segs.filter(([a, b]) => b - a + 1 >= MIN_SEG))
    .filter((segs) => segs.length && segsSize(segs) >= MIN_SEG);
  if (!cleaned.length) {
    setMessage('A structure needs at least one domain of four residues or more.', true);
    domainSegs = autoSegs.map((s) => s.map((p) => [...p]));
    markEdited();
    return;
  }
  domainSegs = cleaned;
  if (activeUnit >= domainSegs.length) activeUnit = domainSegs.length - 1;
  markEdited();
}

// Search-guided refinement re-parses the chain on screen, so it goes back to the
// worker rather than being applied to the split already displayed. Any hand edits
// are discarded by the re-parse, which is what "refine" has to mean — the whole
// point is that the parser proposes a different set of domains.
/**
 * Domains are parsed only when asked for.
 *
 * A chain arrives as one unit — every split is a guess, and making it before anyone
 * has looked at the structure means the guess is invisible. This runs the real thing:
 * over-split first, then keep each boundary only where the library agrees the pieces
 * are separate domains. The bars redraw at each stage, so the over-split appears as
 * more pieces than the chain has domains and then collapses.
 */
/** Ask the worker to split the chain. Shared by the automatic parse and the button. */
function startParse() {
  if (!coords || parsing) return;
  parsing = true;
  $('parseBtn').disabled = true;
  $('parseBtn').textContent = 'Parsing…';
  resetProgress();
  setProgress('Over-splitting the chain', 'then asking the library', 0.04);
  worker.postMessage({ type: 'parseDomains', requestId: ++reqId });
}

/** One whole-chain domain: the escape hatch from an automatic parse. */
function unParse() {
  if (!coords) return;
  parseMode = 'unparsed';
  domainSegs = [[[0, coords.length - 1]]];
  activeUnit = 0;
  markEdited();
  renderDomains();
  drawChain3d();
  renderSearchState();
  syncParseBtn();
}

/** Whether the current definition is a single domain covering the whole chain. */
function isWholeChain() {
  return !!coords && !!domainSegs && domainSegs.length === 1 && domainSegs[0].length === 1
    && domainSegs[0][0][0] === 0 && domainSegs[0][0][1] === coords.length - 1;
}

/**
 * The button offers the other mode, tracked explicitly rather than inferred from the segments.
 *
 * Inferring it from "is this one whole-chain domain" gets 12BK wrong: the parse runs, the retrieval
 * merge collapses three domains into one, and the result is indistinguishable from never having
 * parsed — so the button offered "Auto parse" for a chain that had just been parsed, and pressing it
 * repeated the same seconds of work for the same answer. parseMode records which of the two states
 * you are in, so the button always offers the other one.
 */
function syncParseBtn() {
  const btn = $('parseBtn');
  if (parsing) return;
  const canUnparse = parseMode === 'parsed' && !isWholeChain();
  const parsedToOne = parseMode === 'parsed' && isWholeChain();
  btn.textContent = parseMode === 'parsed' ? 'Un-parse' : 'Auto parse';
  btn.disabled = parsedToOne;
  btn.title = parseMode === 'parsed'
    ? (parsedToOne
      ? 'The parse found one domain covering the whole chain, so parsed and un-parsed are the same '
        + 'definition here — there is nothing to undo.'
      : 'Search the whole chain as one domain instead. The models were trained on domains, so that '
        + 'is the out-of-distribution case and the scores get less trustworthy on a multi-domain '
        + 'chain.')
    : 'Split the chain into domains. The models were trained on domains, so this is what they '
      + 'expect — a whole chain is the unusual case.';
  return canUnparse;
}

$('parseBtn').addEventListener('click', () => {
  if (parseMode === 'parsed') unParse();
  else startParse();
});

$('filterBtn').addEventListener('click', () => startVerify());

/**
 * How many hits to list.
 *
 * The 40 was a display default, not a limit of the method: the scan visits every entry
 * regardless and keeps a top-K, so K costs nothing but rows. Changing it re-scans from
 * the embeddings already computed — milliseconds — rather than re-running the search.
 */
$('hitCount').addEventListener('change', (e) => {
  MAX_HITS = Number(e.target.value) || 10;
  if (!query || !query.units || !query.units.length) return;
  resetVerify();
  setProgress(`Listing the top ${MAX_HITS}`, '', 0.1);
  worker.postMessage({ type: 'rescan', requestId: ++reqId, maxHits: MAX_HITS,
    opts: { progresCutoff: PROGRES_CUTOFF, cirpinCutoff: CIRPIN_CUTOFF } });
});

// --- the CIRPIN split ---------------------------------------------------------
//
// One alternative, offered as a choice between two named parses rather than a walk through a ranked
// pool. The pool version put 22 stops from five generators on a slider, which is an instrument for
// auditing a search, not a control for choosing a decomposition.
//
// 'geometric' is the shipped parse: spectral bisection of the Ca contact graph. On the 1,155-chain
// benchmark it is the best method measured, held-out ARI 0.6930.
//
// The CIRPIN-weighted split used to be offered here as a second opinion. Removed: on held-out unbiased
// chains it loses to the geometric parse (ARI 0.6118-0.6495 against 0.6859), and the three pairwise
// signals tried as a replacement edge weight -- node cosine, profile correlation, and the network's own
// edge-message norms -- all improve the pairwise discrimination and still produce a worse partition,
// because reweighting rescales the normalised cut that maxCut was fitted against. See test/pairwise.mjs
// and test/edgeweights.mjs; the parse is geometry only until something beats it.

$('addDomain').addEventListener('click', () => {
  const n = query.nChain;
  // place the new domain in the largest stretch no domain covers, so it does
  // not silently land on top of an existing one
  const covered = new Uint8Array(n);
  for (const segs of domainSegs) for (const [a, b] of segs) for (let r = a; r <= b; r++) covered[r] = 1;
  let best = null;
  let run = null;
  for (let r = 0; r <= n; r++) {
    if (r < n && !covered[r]) { if (run === null) run = r; } else if (run !== null) {
      if (!best || r - run > best[1] - best[0]) best = [run, r - 1];
      run = null;
    }
  }
  const seg = best && best[1] - best[0] + 1 >= MIN_SEG
    ? best
    : [Math.round(n * 0.4), Math.round(n * 0.6)];   // nothing free: overlap the middle
  domainSegs.push([clampSeg(seg[0], seg[1], n)]);
  activeUnit = domainSegs.length - 1;
  commitDomains();
});


$('searchBtn').addEventListener('click', () => {
  // Recompute when the boundaries changed OR when this domain has nothing scored. Revealing a unit
  // that does not exist is a no-op that used to leave the button disabled for good.
  if (isDirty() || !query?.units?.[activeUnit]) {
    rescore(domainSegs, segsKey(domainSegs) !== segsKey(autoSegs));
    return;
  }
  // Nothing has changed, so this domain has simply not been shown yet: reveal what is already
  // scored rather than recomputing an identical answer.
  revealedUnit = activeUnit;
  renderUnit();
  renderSearchState();
});

// --- hits table -------------------------------------------------------------

let sortKey = 'cirpin';
let sortDir = -1;
// The hits in the order the table is showing them, kept so that anything reasoning about "the top
// hit" means the one at the top of the screen rather than the top of the worker's CIRPIN list.
let sortedHits = [];

/**
 * Check the shortlist with TM-align once it is on screen, and drop what fails.
 *
 * The embedding search ranks; it does not prove. A hit can sit high on cosine and
 * still be a different fold, and TM-score 0.5 is the conventional line between the
 * two. Running the alignment over the top hits afterwards makes the list mean
 * something stricter — with the important detail that both the sequential and the
 * circularly-permuted alignment are scored, so a permuted match is not thrown away
 * for scoring badly in sequence order.
 *
 * Deliberately after the results appear, never before: the search is a second, the
 * checking is several, and nobody should wait on the second thing to see the first.
 */
const TM_FOLD = 0.5;
const verifyState = { running: false, done: 0, total: 0, purged: 0, unit: -1 };

/** A new search or a different domain clears any previous verdict. */
function resetVerify() {
  verifyState.running = false;
  verifyState.done = 0;
  verifyState.purged = 0;
  verifyState.total = 0;
  verifyState.unit = activeUnit;
  syncFilterBtn();
}

function startVerify() {
  const u = unit();
  verifyState.running = false;
  verifyState.done = 0;
  verifyState.purged = 0;
  verifyState.unit = activeUnit;
  if (!u || !u.hits || !u.hits.length || !coords) { syncFilterBtn(); return; }
  // Only hits whose structure can actually be fetched are judgeable; the rest have no
  // verdict and are left alone rather than quietly dropped.
  const judgeable = u.hits.filter((h) => h.hasCoords !== false);
  verifyState.total = judgeable.length;
  if (!judgeable.length) { syncFilterBtn(); return; }
  verifyState.running = true;
  syncFilterBtn();
  worker.postMessage({ type: 'verify', requestId: ++reqId,
    coords: unitCoords(u),
    hits: judgeable.map((h) => ({ id: h.id, row: h.row })) });
}

/**
 * The query domain's coordinates: the residues that domain actually claims.
 *
 * It used to hand back the WHOLE chain whenever there was only one domain, on the reasoning that one
 * domain means an undivided chain. That is false as soon as the domain is trimmed — a single cropped
 * domain is a real division, which hasDomains() already says in as many words. The embedding was
 * computed from the segments and so was correct, but TM-align got the full structure, so a trimmed
 * chain searched as the crop and then aligned as the whole thing.
 *
 * Every unit carries its residue list, including the undivided case where it is every residue, so
 * there is no case to special-case.
 */
function unitCoords(u) {
  return u.residues ? Array.from(u.residues, (r) => coords[r]) : coords;
}

function applyVerdict(m) {
  const u = query?.units?.[verifyState.unit];
  if (!u || !u.hits) return;
  verifyState.done = m.done;
  const h = u.hits.find((x) => x.id === m.id);
  if (h && m.tm !== null) {
    h.tmVerified = m.tm;
    h.tmSeqVerified = m.tmSeq;
    h.tmCpVerified = m.tmCp;
    if (m.tm < TM_FOLD) {
      u.hits = u.hits.filter((x) => x.id !== m.id);
      verifyState.purged++;
      if (selectedId === m.id) { selectedId = null; clearAlignment(); }
    }
    renderRows();   // survivors need their score drawn, not only the purges
  }
  // The bar, not just a line of text. This is the one job in the app that takes seconds per item and
  // reports item by item, so it is exactly what a progress bar is for — and the same bar the search
  // and the parse already use, rather than a second convention for the same idea.
  if (verifyState.total) {
    setProgress('Checking with TM-align', `${verifyState.done} of ${verifyState.total}`,
      verifyState.done / verifyState.total);
  }
}

function syncFilterBtn() {
  const btn = $('filterBtn');
  if (!btn) return;
  const u = unit();
  const nothingToDo = !u || !u.hits || !u.hits.length;
  // Gone once the list has been checked: every surviving row already carries its
  // score, so the button would only offer to recompute what is on screen. A new
  // search or a different domain resets the state and brings it back.
  btn.hidden = nothingToDo || (!verifyState.running && verifyState.done > 0);
  btn.disabled = verifyState.running || nothingToDo;
  btn.textContent = verifyState.running ? 'Filtering…' : 'TM-align filter';
}

/*
 * Live feedback goes in the status strip, so there is one place for it.
 *
 * The filter used to write its own running line into the results column, which meant two places
 * reported what the page was doing depending on which job was doing it — and the strip at the top,
 * which exists for exactly this, sat on "Ready" while a job ran for several seconds. applyVerdict
 * drives the bar directly, and there is no verdict line afterwards because the table already shows
 * it: the failures are gone from it and every row left carries its TM.
 */

function renderUnit() {
  const u = unit();
  if (!u) {
    // Nothing scored for this domain. Clear rather than return, so the panels never show the
    // previous domain's hits under the current domain's heading.
    $('hitsHead').hidden = true;
    $('hitsWrap').hidden = true;
    $('scoresFold').hidden = true;
    return;
  }

  if (u.tooShort) {
    // Fewer than four Cα, so coordsToGraph cannot build the torsion feature.
    $('hitsHead').hidden = true;
    $('hitsWrap').hidden = true;
    $('scoresFold').hidden = true;
    setMessage(`Domain ${activeUnit + 1} has only ${u.nres} residues; four is the minimum. `
      + 'Widen it on the bar.', true);
    repaint();
    return;
  }

  $('hitsHead').hidden = false;
  $('hitsWrap').hidden = false;
  $('scoresFold').hidden = false;
  // No count of candidates here.
  //
  // It read "15 permutation candidates of 15,176" above a table of ten rows, which invites the
  // arithmetic — where are the other five? — and the answer is that the table shows the top ten of
  // whichever column is sorted while the count is over everything searched. Two numbers that cannot
  // be reconciled by looking, to say something the Δ column already shows per row.

  renderRows();
  if (verifyState.unit !== activeUnit) resetVerify();
  syncFilterBtn();

  // Land on the top hit rather than on an empty panel.
  //
  // The first thing anyone does with a result set is look at the best match, so doing it for them
  // removes a click and makes the alignment view self-explanatory instead of three "pick a hit"
  // placeholders. It is the first ALIGNABLE hit, not simply the first: SCOPe has 50 domains with
  // no stored coordinates, and auto-selecting one of those would open with an error.
  //
  // Only for a result set nothing is selected in yet — re-renders happen when the TM filter
  // purges rows or a column is re-sorted, and quietly moving the user's selection then would be
  // worse than the empty panel this fixes.
  //
  // "Top hit" means top of the ordering on screen, which is not the same as top by CIRPIN once a
  // column has been chosen: picking u.hits[0] with the table sorted by Δ selected a row sitting at
  // position 75, so the alignment filled in while the table showed no selection at all.
  if (!selectedId || !u.hits.some((h) => h.id === selectedId)) {
    const first = sortedHits.find((h) => h.hasCoords !== false);
    if (first) requestAlign(first);
  }
  // Residue and domain counts are on the editor right below this, so the line
  // Only what the table cannot say for itself, which on a single-chain single-domain structure is
  // nothing: the row count is the rows, and the id is in the field it was typed into. A chain letter
  // or a domain number is worth keeping, because neither is visible from the results alone.
  const ch = chains.length > 1 ? `chain ${activeChain}` : '';
  const dom = query.nDomains > 1 ? `domain ${activeUnit + 1}` : '';
  setMessage([ch, dom].filter(Boolean).join(' · '));
  repaint();
}

function renderRows() {
  const u = unit();
  if (!u) return;
  const rows = sortedHits = [...u.hits].sort((a, b) => {
    const x = a[sortKey];
    const y = b[sortKey];
    if (typeof x === 'string') return sortDir * x.localeCompare(y);
    // ties fall back to CIRPIN rank, so the order is never arbitrary
    return x === y ? a.rank - b.rank : sortDir * (x < y ? -1 : 1);
  });

  document.querySelectorAll('#hitsHeadRow th').forEach((th) => {
    const on = th.dataset.sort === sortKey;
    if (on) th.setAttribute('aria-sort', sortDir < 0 ? 'descending' : 'ascending');
    else th.removeAttribute('aria-sort');
  });

  // The column only exists once there is something measured to put in it.
  $('hitsTable').classList.toggle('tm-on', verifyState.running || verifyState.done > 0);

  // Only the first MAX_HITS of the ACTIVE ordering.
  //
  // The worker now returns the union of the top MAX_HITS by CIRPIN, by Progres and by Δ — up to
  // three times as many rows as are shown. Slicing after the sort is what makes the list the true
  // top ten of whichever column is selected, instead of one list reshuffled.
  const shown = rows.slice(0, MAX_HITS);
  // Rank is a position in the list being looked at. Keeping the worker's CIRPIN rank would number
  // a Progres-sorted table 1, 4, 7, 12..., which reads as a bug.
  shown.forEach((h, i) => { h.shownRank = i + 1; h.pinned = false; });

  // The selected hit stays in the table even when the ordering pushes it out of the top MAX_HITS.
  //
  // Otherwise switching column silently drops it: the alignment panel goes on describing a
  // structure, the message goes on quoting its TM-score, and the row it refers to is nowhere on
  // screen with nothing selected. Pinning it keeps the two halves of the page talking about the
  // same thing. It carries its true position in this ordering, not a made-up one.
  if (selectedId && !shown.some((h) => h.id === selectedId)) {
    const at = rows.findIndex((h) => h.id === selectedId);
    if (at >= 0) {
      rows[at].shownRank = at + 1;
      rows[at].pinned = true;
      shown.push(rows[at]);
    }
  }

  const tbody = $('hits');
  tbody.textContent = '';
  for (const h of shown) {
    const tr = document.createElement('tr');
    tr.dataset.id = h.id;
    if (h.id === selectedId) tr.setAttribute('aria-selected', 'true');
    if (h.pinned) {
      tr.classList.add('pinned');
      tr.title = `Outside the top ${MAX_HITS} by this column — kept because it is what the `
        + 'alignment below shows.';
    }
    const shown = displayId(h.id);
    // A row that cannot be aligned says so in a mark as well as in grey, because colour alone
    // is not a signal, and it is made inert below rather than left to refuse the click.
    const name = h.hasCoords
      ? shown
      : `<span class="no-coords" title="No structure in the coordinate store, so this hit `
        + `cannot be aligned. Its scores are computed from the index and are unaffected."`
        + `>${shown}<span class="nc-mark" aria-hidden="true">◌</span>`
        + '<span class="sr">, no structure stored</span></span>'
    tr.innerHTML = `
      <td class="left">${h.shownRank}</td>
      <td class="left">${name}</td>
      <td class="narrow">${h.nres}</td>
      <td ${scoreCell(h.cirpin, CIRPIN_CUTOFF)}>${h.cirpin.toFixed(3)}</td>
      <td ${scoreCell(h.progres, PROGRES_CUTOFF)}>${h.progres.toFixed(3)}</td>
      <td ${deltaCell(h.delta)}>${fmtDelta(h.delta)}</td>
      <td class="tm tm-col">${tmCell(h)}</td>`;
    if (h.hasCoords === false) {
      tr.classList.add('inert');
      tr.setAttribute('aria-disabled', 'true');
    } else {
      tr.addEventListener('click', () => requestAlign(h));
    }
    tbody.appendChild(tr);
  }
}

/**
 * The measured TM-score, once the filter has run.
 *
 * Just the number. Which of the two alignments produced it belongs in the tooltip, not
 * in the cell: the CP column already carries that claim, and a score column that
 * sometimes contains a word sorts and scans worse than one that never does.
 */
function tmCell(h) {
  if (typeof h.tmVerified !== 'number') return '';
  // The sequential alignment is only run for hits near the 0.5 line, so the tooltip
  // says what was actually measured rather than implying a number nobody computed.
  const title = typeof h.tmSeqVerified === 'number'
    ? `permuted ${h.tmCpVerified.toFixed(3)}, sequential ${h.tmSeqVerified.toFixed(3)}`
    : `permuted fit ${h.tmCpVerified.toFixed(3)}; sequential not run — it only decides `
      + 'hits close to the 0.5 cutoff';
  return `<span title="${title}">${h.tmVerified.toFixed(3)}</span>`;
}

/*
 * Sorting: one direction per column, and clicking the column already in use does nothing.
 *
 * A toggle gives every column two states and so the table eight, and the ascending half of
 * those is a list of the worst matches — which is not a thing anyone came here to see. So each
 * column has the one order that means something: scores high to low, position and name in their
 * natural order. Re-clicking is deliberately inert rather than a reversal.
 *
 * Columns without data-sort are not sortable at all. TM is the case: the filter measures a
 * subset, so an ordering by it would be ranking measured hits against blanks.
 */
document.querySelectorAll('#hitsHeadRow th[data-sort]').forEach((th) => {
  th.addEventListener('click', () => {
    const k = th.dataset.sort;
    if (k === sortKey) return;
    sortKey = k;
    sortDir = (k === 'id' || k === 'rank') ? 1 : -1;
    renderRows();
  });
});

/**
 * Cell shading for the score columns.
 *
 * The tint goes behind the number, not on it: at this size coloured mono digits
 * lose contrast, and a value should stay in ink while colour sits behind it.
 * Teal for the two similarity scores, magenta for Δ because that is the colour a
 * permutation candidate already wears elsewhere in the UI.
 *
 * Each ramp breaks at the cutoff its own column is judged against — CIRPIN 0.9,
 * Progres 0.6, Δ 0.05, the three numbers CIRPIN.ipynb cell 9 uses to call
 * something a permutation candidate. A straight line from 0.5 to 1 spread the ink
 * evenly across a range where only one point matters, so 0.89 and 0.91 looked
 * alike; now the ramp is faint below the cutoff and steps to roughly three times
 * the tint above it, where the value starts to mean something. Values above their
 * cutoff also take a heavier weight, so the distinction is not colour alone.
 *
 * Δ is only shaded when positive. Negative Δ means Progres scored higher than
 * CIRPIN, which is not evidence of anything anyone acts on, so shading it would
 * imply a second pole that carries no meaning.
 */
const DELTA_CUTOFF = 0.05;     // CIRPIN.ipynb cell 9: score_diff
const TINT_MAX = 0.40;
const SCORE_EPS = 2e-3;        // a self-match scores exactly 1

/**
 * Where a score sits on the axis, 0 to 1.
 *
 * The same log of the distance from perfect that the score plane uses, so a cell
 * and a point agree about how good a score is. Linear tinting put everything above
 * 0.9 at nearly the same darkness — which is most of any hit list — while spending
 * most of its range on scores no one would look at.
 */
const scorePos = (v) => Math.log10(1 / Math.max(SCORE_EPS, 1 - v))
  / Math.log10(1 / SCORE_EPS);

const tintStyle = (rgb, a) => (a < 0.02 ? ''
  : `background-image:linear-gradient(rgba(${rgb},${a.toFixed(3)}),rgba(${rgb},${a.toFixed(3)}))`);

/** Faint below the column's cutoff, a step to strong above it. */
function sharpRamp(v, cutoff) {
  const p = scorePos(v);
  const pc = scorePos(cutoff);
  if (p <= 0) return 0;
  if (v < cutoff) return 0.28 * (p / pc);
  return 0.62 + 0.38 * Math.min(1, (p - pc) / Math.max(1e-6, 1 - pc));
}

function scoreCell(v, cutoff) {
  const t = sharpRamp(v, cutoff);
  return `class="${v >= cutoff ? 'over' : ''}" `
    + `style="${tintStyle('14,124,134', t * TINT_MAX)}"`;
}

/** Two decimals, and no sign on a difference that rounds away to nothing. */
function fmtDelta(d) {
  const a = Math.abs(d).toFixed(2);
  return a === '0.00' ? '0.00' : `${d > 0 ? '+' : '\u2212'}${a}`;
}

/**
 * Δ keeps a linear ramp: it is a difference between two scores, not a score, so
 * "distance from perfect" means nothing for it. Only shaded when positive —
 * negative Δ means Progres scored higher, which is not evidence of anything
 * anyone acts on, and shading it would imply a second pole that carries no
 * meaning.
 */
function deltaCell(d) {
  if (d <= 0) return 'class=""';
  const t = Math.min(1, d / (DELTA_CUTOFF * 4)) * (d >= DELTA_CUTOFF ? 1 : 0.45);
  return `class="${d >= DELTA_CUTOFF ? 'over' : ''}" `
    + `style="${tintStyle('214,0,110', t * TINT_MAX)}"`;
}

// --- score plane ------------------------------------------------------------
// CIRPIN against Progres for every database domain. Two continuous measures, so
// a scatter, one scale per axis. The bulk is muted so the shown hits and the
// cutoff box stay legible over it.

let scoreGeom = null;

function paintScores() {
  const u = unit();
  const canvas = $('scores');
  const wCss = canvas.clientWidth;
  if (!wCss) return;
  const p = prep(canvas, Math.round(wCss * 0.84));
  if (!p) return;
  const { ctx, w, h } = p;
  if (!u) return;

  const L = 46; const R = 10; const T = 10; const B = 34;
  const pw = w - L - R;
  const ph = h - T - B;
  // The full range, 0 to 1. It used to start at 0.4 and drop anything below, which
  // threw away exactly the points the plot exists to show: a candidate is one that
  // CIRPIN likes and Progres does not, so the best ones have the lowest Progres —
  // 0.25 here — and they were being drawn outside the axes. The log spacing makes
  // the whole range affordable, since 0 to 0.4 costs only 8% of the axis.
  const lo = 0;

  // Both axes are log of the distance from a perfect score, not log of the score.
  //
  // Everything worth looking at is bunched against 1.0 — the hits here run 0.85 to
  // 1.00 on CIRPIN — and a log of the score itself would spread the empty low end
  // and squeeze that corner further. Log of (1 - score) does the opposite: each
  // tick is a tenfold approach to 1, so 0.9 to 0.99 gets as much room as 0 to 0.9.
  //
  // The same transform on both axes, so the diagonal stays a straight diagonal and
  // "a candidate sits above it" still reads. Clamped just short of 1 because a
  // self-match scores exactly 1.000 and log(0) has nowhere to go.
  const EPS = 2e-3;
  const t = (v) => Math.log10(1 / Math.max(EPS, 1 - v));
  const t0 = t(lo);
  const span = t(1) - t0;
  const sx = (v) => L + ((t(v) - t0) / span) * pw;
  const sy = (v) => T + ph - ((t(v) - t0) / span) * ph;

  ctx.fillStyle = 'rgba(214, 0, 110, 0.07)';
  ctx.fillRect(sx(lo), sy(1), sx(PROGRES_CUTOFF) - sx(lo), sy(CIRPIN_CUTOFF) - sy(1));
  ctx.strokeStyle = 'rgba(214, 0, 110, 0.35)';
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.strokeRect(sx(lo), sy(1), sx(PROGRES_CUTOFF) - sx(lo), sy(CIRPIN_CUTOFF) - sy(1));
  ctx.setLineDash([]);

  ctx.strokeStyle = RULE;
  ctx.beginPath();
  ctx.moveTo(sx(lo), sy(lo));
  ctx.lineTo(sx(1), sy(1));
  ctx.stroke();

  ctx.fillStyle = 'rgba(154, 167, 181, 0.5)';
  for (let i = 0; i < u.allC.length; i++) {
    ctx.fillRect(sx(u.allP[i]) - 0.75, sy(u.allC[i]) - 0.75, 1.5, 1.5);
  }

  for (const hit of u.hits) {
    const sel = hit.id === selectedId;
    ctx.beginPath();
    ctx.arc(sx(hit.progres), sy(hit.cirpin), sel ? 5 : 3, 0, 2 * Math.PI);
    ctx.fillStyle = hit.isCP ? CUT : BOND;
    ctx.fill();
    if (sel) { ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
  ctx.fillStyle = INK2;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  // Round scores rather than round positions: the reader thinks in "0.95", not in
  // "1.3 decades from perfect". Each successive tick halves the gap to 1, which is
  // what the axis is doing, so they come out evenly spread.
  const TICKS = [0, 0.4, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998];
  const label = (v) => (v >= 0.998 ? '1' : String(v));
  for (const v of TICKS) {
    ctx.fillText(label(v), sx(v), T + ph + 6);
    ctx.beginPath();
    ctx.moveTo(sx(v), T + ph);
    ctx.lineTo(sx(v), T + ph + 3);
    ctx.strokeStyle = RULE;
    ctx.stroke();
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of TICKS) {
    ctx.fillText(label(v), L - 6, sy(v));
    ctx.beginPath();
    ctx.moveTo(L - 3, sy(v));
    ctx.lineTo(L, sy(v));
    ctx.stroke();
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Progres score — order dependent', L + pw / 2, h - 1);
  ctx.save();
  ctx.translate(11, T + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('CIRPIN score — permutation invariant', 0, 0);
  ctx.restore();

  scoreGeom = { sx, sy };
}

function hitNear(e) {
  const u = unit();
  if (!scoreGeom || !u) return null;
  const r = $('scores').getBoundingClientRect();
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;
  let best = null;
  for (const hit of u.hits) {
    const d = (scoreGeom.sx(hit.progres) - mx) ** 2 + (scoreGeom.sy(hit.cirpin) - my) ** 2;
    if (d < 100 && (!best || d < best.d)) best = { d, hit };
  }
  return best?.hit ?? null;
}

$('scores').addEventListener('pointermove', (e) => {
  const h = hitNear(e);
  $('scoresRead').textContent = h
    ? `${displayId(h.id)} · ${h.nres} res · CIRPIN ${h.cirpin.toFixed(3)} · Progres ${h.progres.toFixed(3)}`
      + `${h.isCP ? ' · candidate' : ''}`
    : '';
  $('scores').style.cursor = h ? 'pointer' : 'default';
});
$('scores').addEventListener('pointerleave', () => { $('scoresRead').textContent = ''; });
$('scores').addEventListener('click', (e) => {
  const h = hitNear(e);
  if (h) requestAlign(h);
});

// --- alignment --------------------------------------------------------------

function clearAlignment() {
  align = null;
  alignAvailable = false;
  superState = null;
  sideState = null;
  mapState = null;
  $('mapEmpty').hidden = false;
  $('mapEmpty').textContent = 'Pick a hit to align it against your structure.';
  $('mapRead').textContent = '';
  $('alignStats').hidden = true;
  $('superEmpty').hidden = false;
  $('superEmpty').textContent = 'Pick a hit to superimpose it on your structure.';
  $('sideEmpty').hidden = false;
  $('sideEmpty').textContent = 'Pick a hit to compare it with your structure.';
  $('legendHit').textContent = 'hit';
  for (const id of ['map', 'super', 'side']) {
    const c = $(id);
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }
  updateChrome();
}

function requestAlign(h) {
  if (!coords) return;
  if (!h.hasCoords) {
    // Reachable only when the store had not loaded at search time, so the row could not be
    // marked. Not an error and not red: the score is real, the structure is the gap. 50 of
    // SCOPe40's 15,176 domains are like this — the ones whose id ends in a number instead of a
    // chain, which the coordinate harvest could not resolve.
    setMessage(`${displayId(h.id)} scores as a hit, but its structure is not in the `
      + 'coordinate store, so there is nothing to align.');
    return;
  }
  selectedId = h.id;
  alignAvailable = true;
  // A re-render rather than setting the attribute on the matching row, because the row may not be
  // in the table: the selection can sit outside the top MAX_HITS of the current column, and it is
  // renderRows that pins it there. Patching the DOM instead marked nothing and showed nothing.
  renderRows();

  superState = null;
  sideState = null;
  mapState = null;
  $('alignStats').hidden = true;
  $('mapEmpty').hidden = false;
  $('mapEmpty').textContent = `Aligning ${displayId(h.id)}…`;
  $('superEmpty').hidden = false;
  $('superEmpty').textContent = `Aligning ${displayId(h.id)}…`;
  $('sideEmpty').hidden = false;
  $('sideEmpty').textContent = `Aligning ${displayId(h.id)}…`;
  $('legendHit').textContent = displayId(h.id);
  // a small starting fraction rather than an indeterminate bar, so the creep
  // has somewhere to go while TM-align's first pass runs
  resetProgress();
  setProgress('Aligning', h.id, 0.04);
  updateChrome();
  showTab('align');
  // Keep whichever of the three views was last chosen: someone comparing hit
  // after hit is usually looking at one of them, not cycling.
  showAlnView(alnView);

  // Already aligned? Show it, and do no work.
  //
  // An alignment is expensive — a coordinate fetch, then TM-align twice, about a second on a
  // 300-residue pair — and completely determined by the query, the domain boundaries and the hit.
  // Clicking back to a hit already looked at used to pay the whole cost again, which is what made
  // comparing three or four candidates feel slow. The key carries everything the result depends
  // on, so an edit to the domains or a different database simply never matches.
  const key = alignKey(h.id);
  const hit = alignCache.get(key);
  if (hit) {
    hideProgress();
    renderAlignment(hit);
    return;
  }

  const u = unit();
  // unitCoords, not a second copy of the expression: this and the TM filter both had the same wrong
  // condition written out twice, so both were wrong in the same way.
  worker.postMessage({ type: 'align', requestId: ++reqId, coords: unitCoords(u),
    id: h.id, row: h.row });
}

/**
 * What an alignment result is keyed on: this query, these boundaries, this database, this hit.
 *
 * Anything that would change the answer is in the key rather than triggering an eviction, so there
 * is no invalidation to get wrong — a stale entry is simply unreachable.
 */
const alignKey = (id) => `${queryName}|${activeUnit}|${segsKey(domainSegs)}|${activeDb}|${id}`;

// Bounded, because each entry holds the hit's coordinates and two sets of fitted ones — a few tens
// of KB. Forty is far more hits than anyone compares in a sitting, and the oldest goes first.
const ALIGN_CACHE_MAX = 40;
const alignCache = new Map();

function cacheAlignment(m) {
  const key = alignKey(m.id);
  if (alignCache.has(key)) alignCache.delete(key);      // re-insert so it counts as newest
  alignCache.set(key, m);
  while (alignCache.size > ALIGN_CACHE_MAX) {
    alignCache.delete(alignCache.keys().next().value);
  }
}

function renderAlignment(m) {
  alignAvailable = true;
  // Offered only for the selected hit, never in the table: a link in a row people
  // click to select is a link people open by accident.
  const url = hitUrl(m.id);
  $('aId').innerHTML = url
    ? `<a href="${url}" target="_blank" rel="noopener">${displayId(m.id)} \u2197</a>`
    : displayId(m.id);
  // The embedding scores for this same hit, from the row that was clicked. They are what put it in
  // the table, so the card can be read on its own: "the models call this the same fold at 0.997 and
  // 0.550, and TM-align finds 0.378 sequential against 0.612 permuted" is one thought.
  const row = (unit()?.hits || []).find((h) => h.id === m.id);
  $('aCirpin').textContent = row ? row.cirpin.toFixed(3) : '—';
  $('aProgres').textContent = row ? row.progres.toFixed(3) : '—';
  $('aTm').textContent = m.tm.toFixed(3);
  $('aTmCp').textContent = m.tmCp.toFixed(3);
  $('aDiff').textContent = `${m.tmDiff >= 0 ? '+' : '−'}${Math.abs(m.tmDiff).toFixed(3)}`;
  $('aDiffRow').classList.toggle('hi', m.tmDiff > 0);
  $('aCut').textContent = m.cpPoint ? `${m.cpPoint} residues` : 'none';
  // aligned count and RMSD are per-alignment, so applyMode() fills them in
  $('alignStats').hidden = false;

  /*
   * No note about CP winning.
   *
   * When it wins it says so by the numbers: the readout carries TM and TM cp side by side and Δ is
   * the difference between them, which is the same claim as a sentence about gaining 0.198 — except
   * the numbers are there whether it wins or not, so nothing has to appear and disappear to be read.
   */

  align = m;
  cacheAlignment(m);
  updateChrome();
  // Open on whichever alignment actually won, so the first thing shown is the
  // better answer rather than an arbitrary default.
  mapMode = m.prefer;
  applyMode();
  // Not repeated here. The alignment readout in the left column carries the id, TM, TM cp, the cut
  // and the aligned length, so saying three of those again on the right was the same sentence twice.
  setMessage('');
}

let align = null;
let mapMode = 'cp';

/** Point all three views at whichever alignment scored higher. */
function applyMode() {
  if (!align) return;
  const side = align[mapMode];
  // No choice offered: the alignment that scored higher is the one worth looking
  // at, and both scores are in the table below for anyone who wants the other.
  // Nothing here. Every part of this — which alignment is shown, its TM, where the cut fell — is in
  // the readout under the viewer, and stating it twice beside the tabs made the tabs look annotated
  // rather than the numbers look authoritative.

  $('aAli').textContent = `${side.nAligned} · ${side.rmsd.toFixed(1)} Å`;

  mapState = {
    id: align.id, xlen: align.xlen, ylen: align.ylen,
    map: side.map, mapW: align.mapW, mapH: align.mapH, mapStride: align.mapStride,
    path: side.path, cpPoint: mapMode === 'cp' ? align.cpPoint : 0,
  };
  // The query as given, and the hit brought to it — not the other way round.
  //
  // Both views used the FITTED query against the raw hit, which is TM-align's direction: the query
  // was rotated onto each hit, so it faced a different way for every candidate and the superposed
  // pair drifted around the frame as you clicked down the list. Drawing the fixed query and the
  // hit expressed in its frame means the query never moves, every hit arrives oriented to it, and
  // side-by-side gains something it never had — corresponding parts pointing the same way in both
  // panels, because both panels are now in one frame.
  superState = { query: align.queryFixed, target: side.targetFitted };
  sideState = {
    query: align.queryFixed, target: side.targetFitted, path: side.path, id: align.id,
  };
  $('mapEmpty').hidden = true;
  $('superEmpty').hidden = true;
  $('sideEmpty').hidden = true;
  repaint();
}


// --- alignment map ----------------------------------------------------------
// Residue-by-residue score field from the winning superposition: query on x,
// target on y, both in original numbering. A sequential alignment is one
// diagonal; a circular permutation splits into two offset segments, because the
// cut is undone in the permuted numbering the aligner works in but visible in
// the numbering the user reads. Sequential magnitude, so one hue light to dark.

let mapState = null;
const MAP_PAD = { L: 44, B: 30, T: 8, R: 8 };

function rampColour(v) {
  const e = Math.min(1, Math.max(0, v)) ** 0.75;   // lift the low end
  return [
    Math.round(237 + (14 - 237) * e),
    Math.round(240 + (124 - 240) * e),
    Math.round(244 + (134 - 244) * e),
  ];
}

function paintMap() {
  const m = mapState;
  const canvas = $('map');
  const wCss = canvas.clientWidth;
  if (!wCss) return;
  if (!m) { prep(canvas, Math.min(Math.round(wCss * 0.84), viewMaxH())); return; }

  const { L, B, T, R } = MAP_PAD;
  // Aspect ratio is the point — a permuted alignment is two segments at the same
  // slope — so when the plot is too tall to fit, both axes shrink together rather
  // than the taller one being squashed.
  const ratio = Math.min(2, Math.max(0.5, m.ylen / m.xlen));
  let plotW = wCss - L - R;
  let plotH = Math.round(plotW * ratio);
  const budget = viewMaxH() - T - B;
  if (plotH > budget) { plotW = Math.round(budget / ratio); plotH = budget; }
  const p = prep(canvas, plotH + T + B);
  if (!p) return;
  const { ctx } = p;

  const img = ctx.createImageData(m.mapW, m.mapH);
  for (let j = 0; j < m.mapH; j++) {
    for (let i = 0; i < m.mapW; i++) {
      const [r, g, b] = rampColour(m.map[j * m.mapW + i] / 255);
      const o = ((m.mapH - 1 - j) * m.mapW + i) * 4;   // y increases upward
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(m.mapW, m.mapH);
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, L, T, plotW, plotH);

  const sx = (i) => L + (i / m.xlen) * plotW;
  const sy = (j) => T + plotH - (j / m.ylen) * plotH;

  // A line rather than a dot per pair. The aligned pairs are a path, and drawing
  // them as a path shows what the dots could not: how long each run is, where it
  // breaks, and that a permuted alignment is two straight segments at the same
  // slope rather than a scatter that happens to lie near the diagonal. Broken
  // wherever either chain skips more than a couple of residues, so a gap stays a
  // gap instead of being bridged by a line nothing supports.
  ctx.strokeStyle = CUT;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let k = 0; k < m.path.length; k += 2) {
    const qi = m.path[k];
    const ti = m.path[k + 1];
    const brk = k === 0
      || Math.abs(qi - m.path[k - 2]) > 3 || Math.abs(ti - m.path[k - 1]) > 3;
    if (brk) ctx.moveTo(sx(qi), sy(ti)); else ctx.lineTo(sx(qi), sy(ti));
  }
  ctx.stroke();

  if (m.cpPoint) {
    ctx.strokeStyle = 'rgba(214,0,110,.55)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(sx(m.cpPoint), T);
    ctx.lineTo(sx(m.cpPoint), T + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.strokeRect(L + 0.5, T + 0.5, plotW - 1, plotH - 1);
  ctx.fillStyle = INK2;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  const tick = (n) => (n > 600 ? 200 : n > 300 ? 100 : n > 120 ? 50 : 25);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= m.xlen; i += tick(m.xlen)) ctx.fillText(String(i), sx(i), T + plotH + 6);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let j = 0; j <= m.ylen; j += tick(m.ylen)) ctx.fillText(String(j), L - 6, sy(j));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('your structure — residue', L + plotW / 2, plotH + T + B - 1);
  ctx.save();
  ctx.translate(11, T + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${m.id} — residue`, 0, 0);
  ctx.restore();

  mapState.geom = { plotW, plotH };
}

$('map').addEventListener('pointermove', (e) => {
  const m = mapState;
  if (!m?.geom) return;
  const r = $('map').getBoundingClientRect();
  const fx = (e.clientX - r.left - MAP_PAD.L) / m.geom.plotW;
  const fy = 1 - (e.clientY - r.top - MAP_PAD.T) / m.geom.plotH;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) { $('mapRead').textContent = ''; return; }
  const i = Math.min(m.xlen - 1, Math.floor(fx * m.xlen));
  const j = Math.min(m.ylen - 1, Math.floor(fy * m.ylen));
  const ci = Math.min(m.mapW - 1, Math.floor(i / m.mapStride));
  const cj = Math.min(m.mapH - 1, Math.floor(j / m.mapStride));
  $('mapRead').textContent = `your ${i + 1} · ${m.id} ${j + 1} · score `
    + `${(m.map[cj * m.mapW + ci] / 255).toFixed(3)}`;
});
$('map').addEventListener('pointerleave', () => { $('mapRead').textContent = ''; });

// --- Cα trace rendering -----------------------------------------------------
// Shared by the domain view and the superposition: depth-sorted segments in
// orthographic projection, no lighting model, just depth-keyed alpha and width.
// Both are draggable, and both share one rotation so turning one turns the
// other — the same molecule seen two ways should not need re-orienting.

// One camera for all three structure views, so switching from the superposition to side-by-side
// does not throw away the orientation you just found. An object rather than two loose variables
// because orbit() in src/trace3d.js mutates it, and that is the same gesture the atlas uses.
const cam = makeCamera();
// Whether the current structure has had its view chosen. Once is right: after that the
// orientation belongs to whoever has been dragging it.
let viewChosen = false;
let superState = null;
let sideState = null;
let alignAvailable = false;
let chainState = null;


function paintSuper() {
  const st = superState;
  if (!st) { prep($('super'), $('super').clientWidth); return; }
  const cutRgb = hexToRgb(CUT);
  const bondRgb = hexToRgb(BOND);
  if (!st.secQ) {
    st.secQ = smoothSec(makeSec(st.query, st.query.length / 3));
    st.secT = smoothSec(makeSec(st.target, st.target.length / 3));
  }
  drawTraces($('super'), [
    { coords: st.target, sec: st.secT, colourAt: () => ({ rgb: bondRgb }) },
    { coords: st.query, sec: st.secQ, colourAt: () => ({ rgb: cutRgb }) },
  ], { rot: cam.rot, zoom: cam.zoom, maxH: viewMaxH() });
}

const POSITION_RAMPS = { spectrum: spectrumRgb };

/**
 * The contribution ramp: pale to deep teal, six steps' worth as a continuous blend.
 *
 * Teal because --bond is already the colour of a measured quantity in this page, and because a
 * magnitude wants one hue getting darker rather than a rainbow — a rainbow says "position", which is
 * what the mode beside it already means.
 */
function contribRgb(t) {
  const u = Math.min(1, Math.max(0, t));
  const lo = [214, 232, 235];
  const hi = [8, 74, 84];
  return [0, 1, 2].map((k) => Math.round(lo[k] + (hi[k] - lo[k]) * u));
}

/**
 * Colour each residue by where it sits between the N and C terminus.
 *
 * The hue is a function of position in the whole chain, not of position within a
 * domain, so the ramp does not restart at every boundary. Domain membership still
 * shows: a residue no domain claims goes grey and a domain that is not the one
 * being searched is dimmed, so a crop is visible here exactly as it is in the
 * by-domain view. Without that, cropping a tail changed nothing on screen.
 */
function positionColourAt(n, ramp, owner) {
  // The ramp is rebased onto the SELECTED domain, not sliced out of a whole-chain ramp.
  //
  // Running it over the chain and showing only the selected domain's stretch means a domain in the
  // middle is drawn in middle colours -- green to yellow -- and its own N and C ends are never blue
  // and red. The direction is the thing the colour is for, and it was only legible for a domain that
  // happened to start at residue 0. Rebasing gives every domain the full ramp across itself; the
  // whole chain still gets the whole ramp when nothing is split out.
  const owned = [];
  for (let i = 0; i < n; i++) if (owner[i] === activeUnit) owned.push(i);
  const lut = new Array(n);
  const span = (m) => (m < 2 ? 0 : 1 / (m - 1));
  if (owned.length >= 2) {
    const step = span(owned.length);
    owned.forEach((i, k) => { lut[i] = ramp(k * step); });
    // Anything outside the selection is grey below, so its ramp value never shows.
    const mid = ramp(0.5);
    for (let i = 0; i < n; i++) if (!lut[i]) lut[i] = mid;
  } else {
    const step = span(n);
    for (let i = 0; i < n; i++) lut[i] = ramp(i * step);
  }
  const none = hexToRgb('#b3bcc7');
  // Dimming says "this part is not the part being searched". Before Parse nothing is claimed by a
  // domain, so every residue took the out-of-domain dim and the whole structure came up at 45% —
  // which, multiplied by the depth shading and the loop dim, drew an unparsed chain at a fifth of
  // its colour. With no domains there is nothing to set apart, so nothing is set back.
  const undivided = !owner.some((d) => d >= 0);
  return (i) => {
    if (undivided) return { rgb: lut[i], dim: 1 };
    const d = owner[i];
    if (d < 0) return { rgb: none, dim: 0.45 };
    // The ramp runs across the SELECTED domain, and everything else is grey.
    //
    // It used to paint every domain with the ramp and dim the unselected ones to 45%, which is two
    // claims at once: a dimmed rainbow still reads as position along the chain, so a chain of three
    // domains showed three overlapping N-to-C gradients and the colour no longer said where you
    // were. Grey says "not this one" without competing.
    return d === activeUnit ? { rgb: lut[i], dim: 1 } : { rgb: none, dim: 0.45 };
  };
}

const SIDE_UNALIGNED = { rgb: [186, 195, 206], dim: 0.5 };

/**
 * Two colourings, both restricted to the aligned residues.
 *
 * "Residue number" gives each chain its own N-to-C ramp. Because the panels are
 * in the same orientation, the same place in space then carries a different
 * colour on each side when the chains are permuted — a blue N-terminal helix in
 * yours sitting where a red C-terminal one sits in the hit. That is the clearest
 * statement of a circular permutation the view can make, so it is the default.
 *
 * "Alignment order" instead paints matched pairs the same colour. The permutation
 * shows up the other way round: the colours arrive out of order along the hit's
 * own sequence. Useful for following which piece went where once you already know
 * the pair is permuted.
 *
 * Unaligned residues stay grey either way. Colouring them would put hues on the
 * parts the alignment has nothing to say about, which is where a reader would
 * most easily be misled.
 */
function sideColours(st) {
  const nq = st.query.length / 3;
  const nt = st.target.length / 3;
  const pairs = [];
  for (let k = 0; k < st.path.length; k += 2) pairs.push([st.path[k], st.path[k + 1]]);
  pairs.sort((a, b) => a[0] - b[0]);

  // Each chain's aligned residues, in its own numbering, renumbered from 0.
  //
  // Colouring by raw position in the chain wastes the ramp: if only residues 40 to
  // 90 of 200 are aligned, everything shown falls in the middle quarter of the
  // spectrum and the ends are never used. Ranking within the aligned set instead
  // spends the full blue-to-red on the part being compared. The two chains are
  // ranked separately, each in its own order, which is what makes a permutation
  // visible — the same place in space lands at a different rank on each side.
  const rankOf = (nums) => {
    const sorted = [...new Set(nums)].sort((a, b) => a - b);
    const m = new Map();
    sorted.forEach((v, i) => m.set(v, sorted.length < 2 ? 0 : i / (sorted.length - 1)));
    return m;
  };
  const qRank = rankOf(pairs.map((p) => p[0]));
  const tRank = rankOf(pairs.map((p) => p[1]));

  const byIndexQ = new Array(nq).fill(null);
  const byIndexT = new Array(nt).fill(null);
  const byAlignQ = new Array(nq).fill(null);
  const byAlignT = new Array(nt).fill(null);
  pairs.forEach(([qi, ti], k) => {
    const shared = spectrumRgb(pairs.length < 2 ? 0 : k / (pairs.length - 1));
    if (qi >= 0 && qi < nq) {
      byAlignQ[qi] = shared;
      byIndexQ[qi] = spectrumRgb(qRank.get(qi));
    }
    if (ti >= 0 && ti < nt) {
      byAlignT[ti] = shared;
      byIndexT[ti] = spectrumRgb(tRank.get(ti));
    }
  });
  return { byIndexQ, byIndexT, byAlignQ, byAlignT };
}

let sideColourBy = 'alignment';

/**
 * The two chains in separate panels, same orientation, aligned parts coloured.
 *
 * The superposition answers "do these occupy the same space"; this answers "which
 * part went where", which is the question a circular permutation actually raises.
 */
function paintSide() {
  const st = sideState;
  const canvas = $('side');
  const wCss = canvas.clientWidth;
  if (!wCss) return;
  if (!st) { prep(canvas, Math.min(Math.round(wCss / 2), viewMaxH())); return; }

  if (!st.secQ) {
    st.secQ = smoothSec(makeSec(st.query, st.query.length / 3));
    st.secT = smoothSec(makeSec(st.target, st.target.length / 3));
    st.cols = sideColours(st);

    // One frame, one fit, centred on the query.
    //
    // Both chains are now in the query's coordinate frame, so a single mapping for both panels
    // makes corresponding residues land on the same screen position in each — which is the whole
    // point of putting them side by side. The centre is the query's own, so the query does not
    // move when a different hit is chosen; the radius grows only if the hit reaches further, so
    // nothing is clipped and a hit twice the size still looks twice the size.
    const fq = fitOf([st.query], 0.98);
    st.fit = { ...fq, r: Math.max(fq.r, radiusAbout(st.target, fq, 0.98)) };
  }

  const byIndex = sideColourBy === 'index';
  const lutQ = byIndex ? st.cols.byIndexQ : st.cols.byAlignQ;
  const lutT = byIndex ? st.cols.byIndexT : st.cols.byAlignT;
  const colQ = (i) => (lutQ[i] ? { rgb: lutQ[i] } : SIDE_UNALIGNED);
  const colT = (i) => (lutT[i] ? { rgb: lutT[i] } : SIDE_UNALIGNED);

  const half = Math.floor(wCss / 2);
  const side = Math.min(half, viewMaxH());
  const pad = (half - side) / 2;
  const into = prep(canvas, side);
  if (!into) return;

  drawTraces(canvas, [{ coords: st.query, sec: st.secQ, colourAt: colQ }],
    { into, fit: st.fit, inset: SIDE_INSET, box: { x: pad, y: 0, size: side },
      rot: cam.rot, zoom: cam.zoom });
  drawTraces(canvas, [{ coords: st.target, sec: st.secT, colourAt: colT }],
    { into, fit: st.fit, inset: SIDE_INSET, box: { x: half + pad, y: 0, size: side },
      rot: cam.rot, zoom: cam.zoom });

  const { ctx } = into;
  ctx.strokeStyle = RULE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(half + 0.5, 6);
  ctx.lineTo(half + 0.5, side - 6);
  ctx.stroke();
  ctx.font = '11px ui-monospace, Menlo, monospace';
  ctx.fillStyle = INK2;
  ctx.textBaseline = 'bottom';
  ctx.textAlign = 'left';
  ctx.fillText('your structure', 4, side - 2);
  ctx.textAlign = 'right';
  ctx.fillText(displayId(st.id), wCss - 4, side - 2);

  // No legend, no hint. The ramp is on the COLOUR BY control right below, the two panels are
  // labelled, and what a permutation looks like is the thing the picture is for.
}

for (const [id, mode] of [['sideByIndex', 'index'], ['sideByAlign', 'alignment']]) {
  $(id).addEventListener('click', () => {
    sideColourBy = mode;
    $('sideByIndex').setAttribute('aria-pressed', String(mode === 'index'));
    $('sideByAlign').setAttribute('aria-pressed', String(mode === 'alignment'));
    paintSide();
  });
}

let chainColourMode = 'domain';

/**
 * The query chain coloured by domain, or by position along it.
 *
 * Takes an optional definition so a parse in progress can be drawn before it is
 * committed, which is how the over-split and each merge appear as they happen.
 */
function drawChain3d(override) {
  const canvas = $('chain3d');
  const segsNow = override || domainSegs;
  if (!coords || !segsNow) { prep(canvas, canvas.clientWidth); return; }
  $('chainEmpty').hidden = true;

  if (!chainState || chainState.n !== coords.length) {
    const flat = new Float64Array(coords.length * 3);
    coords.forEach((c, i) => {
      flat[i * 3] = c[0]; flat[i * 3 + 1] = c[1]; flat[i * 3 + 2] = c[2];
    });
    chainState = { n: coords.length, flat, sec: smoothSec(makeSec(flat, coords.length)) };
    // A fold has no up, so the view is chosen rather than left to whatever orientation the file
    // happened to be deposited in. The widest two axes go across the box, which shows the most of
    // the structure, and the zoom takes out the slack the bounding sphere leaves — a flat or
    // elongated fold, which is most of them, was coming up at about 46% of its box.
    if (!viewChosen) {
      cam.rot = bestView(flat, false);
      const { r } = fitOf([flat]);
      cam.zoom = fillZoom(flat, cam.rot, r);
      viewChosen = true;
    }
  }
  // Per-residue owner, last domain wins where two overlap. Residues in no
  // domain stay grey, which is the honest picture after a crop.
  const owner = new Int32Array(coords.length).fill(-1);
  segsNow.forEach((segs, d) => {
    for (const [a, b] of segs) for (let r = a; r <= b; r++) owner[r] = d;
  });
  const rgbOf = segsNow.map((_, d) => hexToRgb(domainColour(d)));
  const none = hexToRgb('#b3bcc7');

  let colourAt;
  if (chainColourMode === 'contrib') {
    // One hue, light to dark, over the domain's own range — the same way the atlas carries a
    // magnitude. Residues outside the selected domain are grey: the number is only defined for the
    // graph that was embedded, and that graph is this domain.
    const cn = contribNorms;
    colourAt = (i) => {
      const d = owner[i];
      if (d !== activeUnit || !cn) return { rgb: none, dim: d < 0 ? 0.45 : 0.35 };
      const v = cn.byResidue.get(i);
      if (v === undefined) return { rgb: none, dim: 0.35 };
      const t = cn.hi > cn.lo ? (v - cn.lo) / (cn.hi - cn.lo) : 0.5;
      return { rgb: contribRgb(t), dim: 1 };
    };
  } else if (POSITION_RAMPS[chainColourMode]) {
    colourAt = positionColourAt(coords.length, POSITION_RAMPS[chainColourMode], owner);
  } else {
    colourAt = (i) => {
      const d = owner[i];
      if (d < 0) return { rgb: none, dim: 0.5 };
      // Non-active domains are dimmed rather than hidden: the point is to see
      // the whole chain and where the active piece sits inside it.
      return { rgb: rgbOf[d] ?? hexToRgb(DOMAIN_EXTRA), dim: d === activeUnit ? 1 : 0.4 };
    };
  }
  drawTraces(canvas, [{ coords: chainState.flat, sec: chainState.sec, colourAt }],
    { maxH: viewMaxH(CHAIN_VIEW_FRAC), rot: cam.rot, zoom: cam.zoom });

  // No legend. The domain rows below are the list of domains, each carrying its own colour, so a
  // key here would be the same list twice — and the one thing a key could add, naming the colours,
  // is what the rows now do in place.
}

// Each mode owns its note and its hint, so nothing has to be reset by hand.
// Just the button and the mode. Each used to carry a note and a hint printed beside the controls;
// every one of those was removed as a caption on a caption, so the fields went with them.
const COLOUR_MODES = [
  { id: 'colByDomain', mode: 'domain' },
  { id: 'colRainbow', mode: 'spectrum' },
  { id: 'colContrib', mode: 'contrib' },
];

/**
 * Per-residue contribution to the embedding, for the selected domain.
 *
 * Keyed on the domain's own boundaries, because that is what it is a property of: the same residue
 * contributes differently once the crop around it changes, since the graph it sits in changes with
 * it. Asked for once per domain and kept, so switching back is instant.
 */
const contribCache = new Map();
let contribNorms = null;         // {lo, hi, byResidue} for the active domain, once it arrives
let contribAsked = null;         // the residue list the outstanding request was for

const contribKey = () => `${queryName}|${activeUnit}|${segsKey(domainSegs)}`;

/** Residue list plus per-residue magnitudes into the {lo, hi, byResidue} the colouring reads. */
function packNorms(res, norms) {
  const byResidue = new Map();
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < norms.length; i++) {
    byResidue.set(res[i], norms[i]);
    if (norms[i] < lo) lo = norms[i];
    if (norms[i] > hi) hi = norms[i];
  }
  return { lo, hi, byResidue };
}

/**
 * The residues the selected domain claims, from the bars rather than from a scored unit.
 *
 * Taking them from query.units would make this mode wait for a search, which it has no reason to:
 * the contribution is a property of the domain's own graph, and the domain exists as soon as its
 * bar does.
 */
function activeResidues() {
  const segs = domainSegs && domainSegs[activeUnit];
  if (!segs) return null;
  const out = [];
  for (const [a, b] of segs) for (let r = a; r <= b; r++) out.push(r);
  return out.length >= 4 ? out : null;
}

function requestContrib() {
  if (!coords) return;
  const res = activeResidues();
  if (!res) return;
  const key = contribKey();
  const got = contribCache.get(key);
  if (got) { contribNorms = got; drawChain3d(); return; }
  contribNorms = null;
  contribAsked = res;
  worker.postMessage({ type: 'contrib', requestId: ++reqId,
    coords: res.map((r) => coords[r]), key });
}

function pickColourMode(mode) {
  // Asking for domain colours before the chain is divided would paint it one flat
  // colour and look like a bug, so that mode falls back until there is a split.
  if (mode === 'domain' && !hasDomains()) mode = 'spectrum';
  if (mode === 'contrib' && !activeResidues()) mode = 'spectrum';
  const m = COLOUR_MODES.find((o) => o.mode === mode) || COLOUR_MODES[0];
  chainColourMode = m.mode;
  for (const o of COLOUR_MODES) $(o.id).setAttribute('aria-pressed', String(o === m));
  if (m.mode === 'contrib') requestContrib();
  drawChain3d();
}

for (const m of COLOUR_MODES) {
  $(m.id).addEventListener('click', () => pickColourMode(m.mode));
}

/**
 * Which residue is under a point on the chain canvas.
 *
 * Repeats drawTraces' projection rather than recording it: the maths is six lines and
 * shares its inputs (cam.rot, cam.zoom, fitOf), so duplicating it here cannot
 * drift out of step with what was drawn as long as those inputs are the same. Returns
 * the nearest residue within a generous radius, or -1 — a Cα trace is thin and asking
 * for a pixel-exact hit would make the feature unusable.
 */
function residueAt(clientX, clientY) {
  const canvas = $('chain3d');
  if (!chainState || !coords) return -1;
  const rect = canvas.getBoundingClientRect();
  const wCss = canvas.clientWidth;
  const side = Math.min(wCss, viewMaxH(CHAIN_VIEW_FRAC));
  const box = { x: (wCss - side) / 2, y: 0, size: side };
  const { cx, cy, cz, r } = fitOf([chainState.flat]);
  const R = ((box.size / 2 - 10) / PE_MAX) * cam.zoom;
  const M = cam.rot;
  const px = clientX - rect.left;
  const py = clientY - rect.top;

  let best = -1;
  let bestD = 14 * 14;          // px, squared
  let bestZ = -Infinity;
  const arr = chainState.flat;
  for (let i = 0; i < chainState.n; i++) {
    const x = (arr[i * 3] - cx) / r;
    const y = (arr[i * 3 + 1] - cy) / r;
    const z = (arr[i * 3 + 2] - cz) / r;
    const vx = M[0] * x + M[1] * y + M[2] * z;
    const vy = M[3] * x + M[4] * y + M[5] * z;
    const vz = M[6] * x + M[7] * y + M[8] * z;
    const pe = 1 / (1.9 - vz * 0.55);
    const sx = box.x + box.size / 2 + vx * R * pe;
    const sy = box.y + box.size / 2 - vy * R * pe;
    const d = (sx - px) ** 2 + (sy - py) ** 2;
    // nearest wins, and among near-ties the one in front — clicking a helix should
    // select the turn facing you, not the one behind it
    if (d <= bestD && (d < bestD * 0.6 || vz > bestZ)) { best = i; bestD = d; bestZ = vz; }
  }
  return best;
}

/**
 * Click a residue to select the domain that owns it.
 *
 * The bars already select by number, which requires knowing which number is which.
 * Pointing at the thing on screen is the direct version — and on a circular
 * permutation, where sequence order and spatial arrangement disagree, it is the only
 * way to select what you are actually looking at.
 */
function pickResidueSelection(clientX, clientY) {
  const r = residueAt(clientX, clientY);
  if (r < 0) return;

  if (domainSegs && domainSegs.length > 1) {
    for (let d = 0; d < domainSegs.length; d++) {
      for (const [a, b] of domainSegs[d]) {
        if (r >= a && r <= b) { selectDomain(d); return; }
      }
    }
  }
}


/**
 * Rotate and zoom, shared across the three structure views.
 *
 * One camera for all of them, so switching from the superposition to side-by-side does not throw
 * away the orientation you just found. The gesture itself is orbit() in src/trace3d.js, which the
 * atlas uses as well — a matrix accumulated per drag rather than a yaw and a pitch, so the drag
 * never runs out of travel at the poles, and pointer capture so releasing outside the canvas ends
 * the drag rather than leaving a button believed held.
 */
for (const [id, paint] of [['super', paintSuper], ['side', paintSide], ['chain3d', drawChain3d]]) {
  const el = $(id);
  orbit(el, cam, paint, {
    zoomMin: 0.5,
    zoomMax: 8,
    // On the chain view a press that did not travel picks the domain under the pointer; the other
    // two views have nothing to select.
    onClick: id === 'chain3d' ? (e) => pickResidueSelection(e.clientX, e.clientY) : null,
    // Back to the view the structure was opened at, which is the one worth returning to — zoom 1
    // and an unrotated axis frame is not a view anyone asked for.
    onReset() {
      if (chainState) {
        cam.rot = bestView(chainState.flat, false);
        cam.zoom = fillZoom(chainState.flat, cam.rot, fitOf([chainState.flat]).r);
      } else {
        cam.zoom = 1;
      }
    },
  });
}

/**
 * Offer the chains this file holds, when it holds more than one.
 *
 * The first chain long enough to embed is used by default, which is what the
 * reference does; a complex is common enough that the one worth searching is
 * often not the one written first, so the choice is exposed rather than guessed.
 */
function buildChainMenu() {
  const sel = $('chainSelect');
  $('chainRow').hidden = chains.length < 2;
  if (chains.length < 2) return;
  sel.textContent = '';
  for (const c of chains) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.id} \u00b7 ${c.n} res`;
    sel.appendChild(opt);
  }
  sel.value = activeChain;
}

$('chainSelect').addEventListener('change', (e) => {
  resetProgress();
  setBusy(true);
  worker.postMessage({ type: 'useChain', requestId: ++reqId, chain: e.target.value });
});

/**
 * Build the database menu from what the worker says it has.
 *
 * Each option carries its size, because a download is the consequence of picking
 * it and a menu that hides that is a menu that surprises you. Options already in
 * memory say so instead.
 */
function buildDbMenu() {
  const sel = $('dbSelect');
  sel.textContent = '';
  // Just the name.
  //
  // It used to carry the domain count and the download size, on the reasoning that picking an
  // option starts a download and a menu that hides that is a menu that surprises you. But the
  // count is on the page already and the size is 0.4 MB — a number small enough that showing it
  // asks for a decision nobody needs to make. A picker should say what the choices are.
  for (const d of databases) {
    const info = DB_LABELS[d.name] || { label: d.name };
    const opt = document.createElement('option');
    opt.value = d.name;
    opt.textContent = info.label;
    sel.appendChild(opt);
  }
  sel.value = activeDb;
  sel.disabled = pendingDb !== null || !ready;
}

/**
 * Switching database, which the first time means downloading it.
 *
 * The menu is put back to the database actually in use until the new one has
 * loaded, so it never claims to be searching something it has not got, and a
 * failed load leaves it on the old one rather than in a state the results
 * contradict.
 */
$('dbSelect').addEventListener('change', (e) => {
  const name = e.target.value;
  if (name === activeDb) return;
  pendingDb = name;
  $('dbSelect').disabled = true;
  if (!loadedDbs.has(name)) {
    resetProgress();
    setProgress(`Loading ${(DB_LABELS[name] || {}).label || name}`, 'starting', 0.01);
  }
  worker.postMessage({ type: 'useDb', db: name });
});

// Once. This line was three copies of itself, so every resize ran the whole repaint three times.
addEventListener('resize', () => { drawBars(); repaint(); });

updateChrome();
buildDbMenu();

// After updateChrome, not before: it is what adds body.no-structure, and until that class is on the
// hero is display:none, so the canvas measures zero and the first frame has nothing to draw into.
hero = ouroboros($('heroCanvas'));

/**
 * Poke the animal and it comes alive.
 *
 * A click does it, and so does a drag — a drag also says which way to go, so the snake leaves in
 * the direction it was pulled, which is a better first move than always heading right. The drag
 * fires as soon as it has travelled far enough to be a drag rather than waiting for the release,
 * because the whole effect is that the thing reacts to being touched.
 *
 * The phase the hero was on is handed over, so the ring the game opens with is the ring that was
 * just clicked rather than a fresh one. The hero is stopped while the game is up rather than left
 * animating behind an opaque overlay, and rebuilt on the way out by the same call that made it.
 * The module is fetched on the first poke, so the easter egg costs nothing to anyone who never
 * finds it.
 */
{
  const heroCanvas = $('heroCanvas');
  const DRAG_MIN = 7;                  // px, past which a press is a drag and not a click
  let from = null;

  const wake = async (dir) => {
    if (eggOpen) return;
    eggOpen = true;
    from = null;
    /*
     * A stale cache can break this, so it must not break silently.
     *
     * snakegame.js is fetched on the first poke, long after ouroboros.js was loaded with the page,
     * so on a deploy the two can come from different versions: Pages serves assets with
     * max-age=600, and a browser holding the previous ouroboros.js hands the new game a module
     * without the exports it imports. The import rejects, and with no catch that was an unhandled
     * rejection and an easter egg that quietly did nothing for ten minutes after every deploy.
     *
     * Nothing here can fix a stale module — only time or a hard reload can. What it can do is say
     * so, and leave the hero running rather than stopped over a game that never opened.
     */
    let snakeGame;
    try {
      ({ snakeGame } = await import('./src/snakegame.js?v=5e652e7b'));
    } catch (err) {
      eggOpen = false;
      console.error('snake: could not load the game — most likely a cached module from a previous '
        + 'version. A hard reload fixes it, and so does waiting for the cache to expire.', err);
      return;
    }
    const phase = hero?.headInfo?.()?.progress ?? 0;
    // Where and how big the hero is, so the board opens on top of it at the same size. With the
    // phase as well, the first frame of the game is the last frame of the hero.
    const box = heroCanvas.getBoundingClientRect();
    hero?.stop();
    snakeGame({
      fromPhase: phase,
      side: Math.round(box.width),
      anchor: { left: box.left, top: box.top, width: box.width },
      dir,
      onClose: () => {
        eggOpen = false;
        hero = ouroboros(heroCanvas);
      },
    });
  };

  heroCanvas.addEventListener('pointerdown', (e) => { from = { x: e.clientX, y: e.clientY }; });
  heroCanvas.addEventListener('pointermove', (e) => {
    if (!from) return;
    const dx = e.clientX - from.x;
    const dy = e.clientY - from.y;
    if (Math.hypot(dx, dy) < DRAG_MIN) return;
    // the dominant axis of the pull, as a grid direction
    wake(Math.abs(dx) > Math.abs(dy) ? [Math.sign(dx), 0] : [0, Math.sign(dy)]);
  });
  heroCanvas.addEventListener('pointerup', () => { if (from) wake(null); });
  heroCanvas.addEventListener('pointercancel', () => { from = null; });
}

// --- input ------------------------------------------------------------------

/**
 * The window is the drop target.
 *
 * A dashed box in the layout cost 84 pixels of the column it sat in and was a
 * smaller target than the page it sat on. Dragenter and dragleave fire for every
 * element crossed, so a depth counter decides when the pointer has actually left
 * rather than merely moved between children.
 */
$('fileBtn').addEventListener('click', () => $('file').click());

let dragDepth = 0;
const veil = (on) => $('dropVeil').classList.toggle('on', on);

addEventListener('dragenter', (e) => {
  if (![...e.dataTransfer.types].includes('Files')) return;
  e.preventDefault();
  dragDepth++;
  veil(true);
});
addEventListener('dragover', (e) => {
  if ([...e.dataTransfer.types].includes('Files')) e.preventDefault();
});
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; veil(false); } });
addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  veil(false);
  const f = e.dataTransfer.files[0];
  if (f) readFile(f);
});

$('file').addEventListener('change', (e) => { if (e.target.files[0]) readFile(e.target.files[0]); });

function readFile(f) {
  const r = new FileReader();
  r.onload = () => submit(r.result, 'guess', f.name);
  r.onerror = () => setMessage(`Could not read ${f.name}.`, true);
  r.readAsText(f);
}

async function fetchPDB(rawId) {
  const id = rawId.trim().toUpperCase();
  if (!/^[0-9A-Z]{4}$/.test(id)) {
    setMessage('Enter a 4-character PDB ID, like 2CC6.', true);
    return;
  }
  setBusy(true);
  resetProgress();
  setProgress(`Fetching ${id}`, 'from RCSB', 0.05);
  setMessage('');                 // whatever the last structure left there is not about this one
  try {
    const r = await fetch(`https://files.rcsb.org/download/${id}.cif`);
    if (!r.ok) throw new Error(`RCSB returned ${r.status} for ${id}`);
    const text = await r.text();
    pageBytes += text.length;      // mmCIF is ASCII, so characters are bytes
    renderNet();
    submit(text, 'mmcif', id);
  } catch (err) {
    // An error is a result, not progress: it belongs in the message line, where it stays until
    // something replaces it, and the status strip goes back to Ready.
    hideProgress();
    setMessage(`${err.message}. Check the ID, or drop the file instead.`, true);
    setBusy(false);
  }
}

$('fetchBtn').addEventListener('click', () => fetchPDB($('pdbId').value));
$('pdbId').addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchPDB($('pdbId').value); });
document.querySelectorAll('.examples button').forEach((b) => {
  b.addEventListener('click', () => { $('pdbId').value = b.dataset.id; fetchPDB(b.dataset.id); });
});
