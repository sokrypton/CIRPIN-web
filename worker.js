// Embedding, dual-model search and structural alignment. Everything heavy
// runs here so the UI thread stays responsive.

import { loadWeights, embedGraph } from './src/cirpin.js?v=2c8b633b';
import { loadAccelerator } from './src/wasm.js?v=2c8b633b';
import { coordsToGraph, parseStructureChains, parseCoordsTxt, parseCIF }
  from './src/structure.js?v=2c8b633b';
import { loadBasis, projectQuery, scanCodes, scoreRows, unpackId, TED_ID_BYTES }
  from './src/ted.js?v=2c8b633b';
import { cpAlign, permuteCoords, applyTransform, applyInverseTransform }
  from './src/tmalign.js?v=2c8b633b';
import { parseDomains, domainCoords } from './src/domains.js?v=2c8b633b';
import { loadCodebook, decodeRecord, shardedStore, codebookId }
  from './src/coords.js?v=2c8b633b';
// One range reader for the whole app, and the invariant that a file is either range-read or
// fetched whole but never both. See src/fetchrange.js for the 45 MB read that made it necessary.
import { fetchRange as rangeOf, fetchWhole, fetchJSONWhole } from './src/fetchrange.js?v=2c8b633b';

let cirpinW = null;
let progresW = null;
// The SIMD forward pass, or null where the browser has none. Measured at ~11.7x on a
// 332-residue chain, so this is the difference between a five-second domain parse and a
// half-second one — but it is an accelerator, never a requirement: every call falls back
// to the JS implementation, which is the one checked against PyTorch.
let accel = null;
const EMBED_DIM = 128;   // CIRPIN's hidden width; the node sink writes this many floats per residue
const CIRPIN = 0;
const PROGRES = 1;

/**
 * Embed one graph with whichever implementation is available.
 *
 * The fallback is per call rather than per session: a chain long enough to overflow the
 * module's arena throws, and the right answer to that is to compute it in JS, not to
 * fail the search.
 */
function embed(model, graph, report, normSink) {
  const w = model === CIRPIN ? cirpinW : progresW;
  // normSink, when given, is filled with each residue's |node_dec| -- the per-residue contribution the
  // Contribution colouring shows. Both paths can produce it, so asking for it never forces the slower one:
  // the accelerator writes whole node vectors and the norm is taken here; the JS path has a norm sink of
  // its own. Free either way, because the forward pass was happening regardless.
  if (accel) {
    try {
      if (!normSink) return accel.embed(model, graph, report);
      const nv = new Float32Array(graph.n * EMBED_DIM);
      const e = accel.embed(model, graph, report, nv);
      for (let i = 0; i < graph.n; i++) {
        let ss = 0;
        for (let k = 0; k < EMBED_DIM; k++) { const v = nv[i * EMBED_DIM + k]; ss += v * v; }
        normSink[i] = Math.sqrt(ss);
      }
      return e;
    } catch {
      // fall through to JS
    }
  }
  return embedGraph(w, graph, report, normSink ?? null);
}
let codebook = null;          // shared by every coordinate store; see src/coords.js
let codebookHash = null;
let codebookPending = null;

/**
 * The step codebook every stored coordinate record is written against.
 *
 * One codebook for all databases, fetched at most once. Started during init so it has arrived
 * before any hit is clicked, but self-starting so a reader cannot depend on that.
 */
function coordCodebook() {
  if (codebook) return codebook;
  if (!codebookPending) {
    codebookPending = fetchBin('./data/coords-codebook.bin')
      .then(async (buf) => {
        const c = loadCodebook(buf);
        codebookHash = await codebookId(buf);
        codebook = c.cb;
        return codebook;
      });
  }
  return codebookPending;
}


/**
 * Where TED domain coordinates come from, if anywhere.
 *
 * Set this to a base prefix holding the store published by tools/harvest_ted_coords.py and
 * tools/shard_coords.py, and clicking an AlphaFold hit reads ~170 bytes by range instead of
 * fetching and parsing a whole model from AlphaFold DB. It also covers the roughly one entry in
 * ten whose UniProt accession has been retired, which has no model to fetch at all.
 *
 * Empty means "not available": every hit then takes the AlphaFold DB path, which is what
 * has always happened.
 *
 * At 11-bit VQ the blob is ~600 MB in shards of under 90 MB, which a GitHub Pages site can
 * host: the 1 GB cap is per site, and a second repo published to the same sokrypton.github.io
 * origin gets its own allowance — so this can be a sibling path rather than a third-party host,
 * with no CORS involved either way (Pages serves ranges with access-control-allow-origin: *).
 */
// Databases published to their own repository, name -> origin. Rewritten at deploy time; see
// tools/deploy.py --external. Empty here because a local checkout serves everything from beside the app,
// which is what the tests and a `python3 -m http.server` in web/ both depend on.
//
// The convention is CIRPIN-<database>/ with files named by ROLE rather than by database -- db.json,
// db-ids.bin, coords-000.bin -- because the repository name already says which database it is, and
// CIRPIN-ted/ted-ids.bin stutters. One origin per database means each is published, re-published and
// size-capped on its own: GitHub Pages allows 1 GiB per site, and TED alone is 841 MB.
const EXTERNAL = {"afdb": "https://sokrypton.github.io/CIRPIN-ted", "ecod40": "https://sokrypton.github.io/CIRPIN-ecod"};

/**
 * The databases this build knows about.
 *
 * Every one is the same thing on disk — PCA basis, int8 codes, ids, residue
 * counts — so there is one loader and one scan for all of them. What differs is
 * recorded in each one's metadata: how its ids are packed, and where a hit's
 * structure comes from. Adding a database is an entry here and a file of codes,
 * not a code path.
 *
 * SCOPe40 is 1 MB and loads at startup. AlphaFold is 277 MB and loads when asked
 * for. That is the only reason the two are treated differently anywhere.
 */
const LOCAL_SOURCES = {
  // No eager/lazy flag: which path a database takes is decided by whether its manifest has an ivf
  // key, and two of these carried a flag saying otherwise that nothing read.
  scope40: { prefix: './data/db/scope40', coords: './data/coords' },
  // CATH's S40 non-redundant set. 34,649 domains, so it scans in full like SCOPe40 rather than
  // needing a clustered index. Its coordinates sit beside its codes because they were built from
  // cathdb.info's own PDB files rather than from training_coords.
  cath40: { prefix: './data/db/cath40', coords: './data/db/cath40-coords' },
  // ECOD's F40 set, 448,232 domains. Clustered like TED rather than scanned like the two above, and
  // its ids are text rather than packed, which needs no new code path -- CATH's shape with TED's index.
  ecod40: { prefix: './data/db/ecod40', coords: './data/db/ecod40-coords' },
  afdb: { prefix: './data/ted/ted', coords: './data/ted/ted-coords' },
};

// An external database needs no entry of its own: the whole thing is addressed by one prefix, so a home
// substitutes for both halves. Pages serves byte ranges with access-control-allow-origin: *, which the
// range reads already depend on, so a cross-origin database needs nothing further.
const DB_SOURCES = Object.fromEntries(Object.entries(LOCAL_SOURCES).map(
  ([name, local]) => [name, EXTERNAL[name]
    ? { prefix: `${EXTERNAL[name]}/db`, coords: `${EXTERNAL[name]}/coords` }
    : local]));
const dbs = new Map();
let inputChains = null;   // the last parsed structure, chain id -> coords
let activeChain = null;   // which of them is on screen, so a re-parse can repeat it
let activeDb = 'scope40';
const db = () => dbs.get(activeDb);

function say(label, detail, frac) {
  postMessage({ type: 'progress', label, detail, frac });
}

// How much a database may be without a clustered index. SCOPe40 needs about 1.2 MB; anything an
// order of magnitude past that is a database that has not been through build_ivf.py.
const EAGER_CAP = 8_000_000;

// Every byte this worker pulls over the network. The point of the clustered index is that
// this number stays small, so it is worth being able to see it rather than infer it.
let netBytes = 0;

/**
 * Every reply carries the bytes fetched so far.
 *
 * The figure is only useful if it is current, and threading it through each of a dozen postMessage
 * sites by hand is how one of them ends up stale. Wrapping the send is one place instead.
 */
const post = (msg, transfer) => {
  const withBytes = msg && typeof msg === 'object' ? { ...msg, netBytes } : msg;
  if (transfer) postMessage(withBytes, transfer);
  else postMessage(withBytes);
};

// The three readers, each counting into netBytes. The work is in src/fetchrange.js; these only
// bind the counter, so no call site has to remember to.
const count = (n) => { netBytes += n; };
const fetchBin = (url, onProgress) => fetchWhole(url, count, onProgress);
const fetchJSON = (url) => fetchJSONWhole(url, count);
const fetchRange = (url, start, end) => rangeOf(url, start, end, count);

// An idealised α-helix Cα trace: 3.8 Å rise per residue, 100° per turn. Only
// used to give the JIT something to chew on, so it need not be a real protein —
// it just has to produce a comparable contact graph.
function syntheticChain(n) {
  const coords = [];
  for (let i = 0; i < n; i++) {
    const a = (i * 100 * Math.PI) / 180;
    coords.push([2.3 * Math.cos(a), 2.3 * Math.sin(a), 1.5 * i]);
  }
  return coords;
}

// V8 needs a few calls to tier the kernel up to TurboFan. Measured in Chrome
// on a 302-residue chain, the call-by-call cost is 900, 876, 1767, 1769, then
// 707 ms steady — so an unwarmed first search pays up to 2.5x. Warming here
// costs nothing in wall-clock because it overlaps the database download.
// Yield to the event loop so queued postMessages actually go out. Without this
// the whole warm-up is one uninterruptible block and the progress bar freezes
// for its duration.
// Yield the task queue without setTimeout. Chrome throttles timers to about one
// a second in a backgrounded tab, workers included, so a setTimeout(0) here
// turned warm-up into one layer per second whenever the user switched tabs
// during the load — the bar would sit on "warming up" for a minute. A
// MessageChannel round trip is not throttled and yields just as well.
const breatheChan = new MessageChannel();
const breathe = () => new Promise((r) => {
  breatheChan.port1.onmessage = () => r();
  breatheChan.port2.postMessage(0);
});

async function warmUp(report) {
  // Two sizes and enough repeats to get past the tier-up hump for both models.
  // Profiling a 302-residue chain in Chrome gave 900, 876, 1767, 1769 ms for
  // the first four calls and 707 steady, so anything under ~6 calls per model
  // leaves the first real query paying for the compile.
  const noop = () => {};
  const sizes = [90, 220];
  const reps = 6;
  let done = 0;
  const total = sizes.length * reps;
  for (const n of sizes) {
    const g = coordsToGraph(syntheticChain(n));
    for (let i = 0; i < reps; i++) {
      embed(CIRPIN, g, noop);
      embed(PROGRES, g, noop);
      done++;
      if (report) report(done / total);
      await breathe();
    }
  }
}

// Parsing a file and splitting it into domains needs no weights, so it can run
// while the 24 MB of models and databases is still downloading. Anything that
// embeds waits on this.
let modelsReady = false;
let resolveReady;
const readyPromise = new Promise((r) => { resolveReady = r; });

async function init() {
  // Two models plus SCOPe40 as codes: about 10 MB, where the float32 databases
  // this replaced were 24. Progress is tracked in bytes across all of it at once
  // so the bar does not restart per file.
  const EXPECTED = 5.6e6;
  let bytes = 0;
  const bump = (n) => {
    bytes += n;
    say('Loading models', `${(bytes / 1e6).toFixed(1)} MB of ~${(EXPECTED / 1e6).toFixed(0)} MB`,
      Math.min(0.9, bytes / EXPECTED));
  };

  say('Loading models', 'starting', 0);
  const [cMeta, cBin, pMeta, pBin] = await Promise.all([
    fetchJSON('./data/cirpin.json'), fetchBin('./data/cirpin.bin', bump),
    fetchJSON('./data/progres.json'), fetchBin('./data/progres.bin', bump),
  ]);
  cirpinW = loadWeights(cBin, cMeta);
  progresW = loadWeights(pBin, pMeta);

  // 9 KB, and it either works or it does not — no progress reporting, and no failure
  // path beyond carrying on without it.
  accel = await loadAccelerator('./data/cirpin.wasm', [cirpinW, progresW]);

  // The coordinate index says which SCOPe40 domains can be aligned. It is not
  // Both are started here rather than on first use, so that by the time a search finishes the
  // store knows which domains can be aligned and hasCoords is accurate. Neither is awaited:
  // together they are ~45 KB and must not hold up the ready message.
  coordCodebook();
  coordStore(DB_SOURCES.scope40.coords);

  const pending = loadCoded('scope40', bump);

  // Warming overlaps the download, so this costs no wall-clock.
  await warmUp((f) => say('Warming up the network',
    'compiling the kernel while data downloads', 0.5 + 0.4 * f));

  await pending;

  // Every database's index, so the menu can name and count the ones that are not
  // loaded. These are a few hundred bytes each — the codes are what costs.
  const catalogue = await Promise.all(Object.entries(DB_SOURCES).map(
    async ([name, src]) => {
      if (dbs.has(name)) {
        const d = dbs.get(name);
        return { name, label: d.label, n: d.n, loaded: true };
      }
      try {
        const m = await fetchJSON(`${src.prefix}.json`);
        return { name, label: m.label, n: m.n, loaded: false };
      } catch {
        // A database that is not deployed simply does not appear in the menu.
        return null;
      }
    },
  ));

  say('Ready', '', 1);
  // Releases anything that arrived before the weights did — a structure can be
  // parsed and split while this is still downloading, and it waits here.
  modelsReady = true;
  resolveReady();
  post({
    type: 'ready',
    // Whether the SIMD path is live. Worth reporting rather than inferring: if the
    // module fails to load the app still works, just ~12x slower, and that is exactly
    // the kind of silent regression nobody notices until it is old.
    accel: !!accel,
    databases: catalogue.filter(Boolean),
    active: activeDb,
    dbSize: dbs.get('scope40').n,
    label: dbs.get('scope40').label,
    bytes: cBin.byteLength + pBin.byteLength,
  });
}

/**
 * Load one coded database.
 *
 * Held whole rather than fetched in ranges: the scan touches every row, so there
 * is nothing to be lazy about, and the ids, residue counts and choppings are small
 * beside the codes.
 */
async function loadCoded(name, outerBump) {
  if (dbs.has(name)) return dbs.get(name);
  const src = DB_SOURCES[name];
  if (!src) throw new Error(`no such database: ${name}`);
  const base = src.prefix.slice(0, src.prefix.lastIndexOf('/') + 1);
  const meta = await fetchJSON(`${src.prefix}.json`);
  const basisJson = await fetchJSON(`${src.prefix}-basis.json`);
  const { dims, n } = meta;

  // What the progress bar is measuring against. A clustered index downloads almost
  // nothing at startup — centroids, cluster offsets and the coarse chopping index — so the
  // estimate has to reflect that or the bar finishes instantly and then sits there.
  const EXPECTED = meta.ivf
    ? meta.ivf.clusters * dims + (meta.ivf.clusters + 1) * 4 + 4096
    : n * dims * 2 + n * (meta.idFormat === 'text' ? 10 : TED_ID_BYTES)
      + n * 2 + (meta.choppings ? n * 6 : 0);
  let bytes = 0;
  const bump = outerBump || ((k) => {
    bytes += k;
    say(`Loading ${meta.label}`,
      `${(bytes / 1e6).toFixed(0)} MB of ~${(EXPECTED / 1e6).toFixed(0)} MB`,
      Math.min(0.97, bytes / EXPECTED));
  });
  if (!outerBump) say(`Loading ${meta.label}`, `${n.toLocaleString()} domains`, 0.01);

  // A clustered index is fetched by the byte range, one cluster at a time, so the code shards are
  // not downloaded here at all. What is: 131 KB of centroids and 16 KB of cluster offsets. The
  // metadata — ids, choppings — is read a block at a time too (see chopBase and metaBlock), so a
  // whole-database fetch happens nowhere on this path.
  let ivf = null;
  if (meta.ivf) {
    const cent = new Uint8Array(await fetchBin(`${base}${meta.ivf.centroids}`, bump));
    const starts = new Uint32Array(await fetchBin(`${base}${meta.ivf.offsets}`, bump));
    if (starts.length !== meta.ivf.clusters + 1) {
      throw new Error(`${name}: cluster offsets are the wrong length`);
    }
    ivf = {
      clusters: meta.ivf.clusters,
      nprobe: meta.ivf.nprobe ?? 16,
      cent,
      starts,
      // The path scheme lives in the manifest, so changing the fanout needs no code change.
      // See tools/split_clusters.py.
      clusterUrl: (j) => base + meta.ivf.clusterPath
        .replace('{hh}', ((j / (meta.ivf.clusterFanout ?? 256)) | 0).toString(16).padStart(2, '0'))
        .replace('{j}', String(j)),
      cache: new Map(),
    };
  }

  const shards = {};
  // The eager path is how the cache got poisoned in the first place: it downloads whole code shards
  // and the whole id table, and those complete entries then answer later range requests with 200
  // and the entire body. It is still the right thing for a small database -- SCOPe40 is 15,176 rows
  // and about 1.2 MB -- so it stays, with a ceiling. Past that a database needs a clustered index,
  // and saying so is more useful than silently pulling hundreds of megabytes.
  if (!ivf) {
    const whole = n * dims * 2 + n * (meta.idFormat === 'text' ? 10 : TED_ID_BYTES)
      + n * 2 + (meta.choppings ? n * 6 : 0);
    if (whole > EAGER_CAP) {
      throw new Error(`${name}: no clustered index, so loading it means downloading `
        + `${(whole / 1e6).toFixed(0)} MB whole, over the ${(EAGER_CAP / 1e6).toFixed(0)} MB `
        + 'ceiling. Run web/tools/build_ivf.py on it first.');
    }
  }
  for (const model of ['cirpin', 'progres']) {
    if (ivf) { shards[model] = []; continue; }
    shards[model] = [];
    for (const [file] of meta.models[model].shards) {
      shards[model].push(new Uint8Array(await fetchBin(`${base}${file}`, bump)));
    }
    const rows = shards[model].reduce((acc, sh) => acc + sh.length / dims, 0);
    if (rows !== n) throw new Error(`${name}/${model}: ${rows} rows, index says ${n}`);
  }
  if (ivf) {
    // nothing to validate against here; the ranges are checked when they are fetched
  }

  // With a clustered index the per-row metadata is fetched by range for the handful of
  // rows actually shown — a query needs ids for ~40 hits, not 26 MB of them. Blocks of
  // 4096 rows are cached, which is cheap because hits cluster: they came from the same
  // few clusters, so they mostly live in the same few blocks.
  let ids = null;
  let nres = null;
  let chopN = null;
  let chopS = null;
  let chopAt = null;
  let lazy = null;

  if (ivf && meta.ivf.chopBase) {
    const bases = new Uint32Array(await fetchBin(`${base}${meta.ivf.chopBase}`, bump));
    lazy = {
      block: meta.ivf.chopBlock,
      idBytes: meta.ivf.idBytes ?? TED_ID_BYTES,
      bases,
      // No nres: the residue count is the summed chopping length, which rowMeta recomputes, and
      // build_ivf.py deliberately stops writing the file. The entry here pointed at a
      // <prefix>-nres.bin that does not exist — harmless only because nothing ever asked for it.
      url: {
        ids: `${src.prefix}-ids.bin`,
        chopn: `${src.prefix}-chopn.bin`,
        chops: `${src.prefix}-chops.bin`,
      },
      cache: { ids: new Map(), chopn: new Map(), chops: new Map() },
    };
  } else {
    if (meta.idFormat === 'text') {
      const text = new TextDecoder().decode(await fetchBin(`${src.prefix}-ids.txt`, bump));
      ids = text.split('\n');
      if (ids.length !== n) throw new Error(`${name}: ${ids.length} ids, expected ${n}`);
    } else {
      ids = new Uint8Array(await fetchBin(`${src.prefix}-ids.bin`, bump));
      if (ids.length !== n * TED_ID_BYTES) throw new Error(`${name}: id blob is the wrong length`);
    }
    nres = new Uint16Array(await fetchBin(`${src.prefix}-nres.bin`, bump));
    if (nres.length !== n) throw new Error(`${name}: nres blob is the wrong length`);

    if (meta.choppings) {

    chopN = new Uint8Array(await fetchBin(`${src.prefix}-chopn.bin`, bump));
    chopS = new Uint16Array(await fetchBin(`${src.prefix}-chops.bin`, bump));
    if (chopN.length !== n) throw new Error(`${name}: chopping counts are the wrong length`);
    // Prefix sum rather than a stored offset table: 3.5 MB of counts downloads
    // instead of 14 MB of offsets, and the table is rebuilt here in a few ms.
    chopAt = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) chopAt[i + 1] = chopAt[i] + chopN[i];
    if (chopAt[n] * 2 !== chopS.length) {
      throw new Error(`${name}: segment count does not match the per-domain counts`);
    }
    }
  }

  const loaded = {
    name, label: meta.label, n, dims,
    structures: meta.structures,
    // kept so the classification lookup can be fetched lazily; everything else it needs is derived
    prefix: src.prefix,
    // where this database's coordinate store lives; '' means it has none
    coords: src.coords ?? '',
    idFormat: meta.idFormat,
    shardRows: meta.models.cirpin.shardRows,
    shards,
    basis: {
      cirpin: loadBasis(basisJson.models.cirpin),
      progres: loadBasis(basisJson.models.progres),
    },
    ids, nres, chopN, chopS, chopAt, ivf, lazy,
  };
  dbs.set(name, loaded);
  return loaded;
}

/** The id of one row, however this database stores them. */
function rowId(d, row) {
  return d.idFormat === 'text' ? d.ids[row] : unpackId(d.ids, row * TED_ID_BYTES);
}

/**
 * Rank one database's codes, then annotate the survivors with the other model.
 *
 * Only CIRPIN decides the order, so Progres is scored for the shown rows and for
 * the cutoff count, not for every domain. The count needs a predicate over the
 * whole set, but it is cheap: a row only costs a Progres dot product if its CIRPIN
 * score already cleared the cutoff, which almost none do.
 */
/**
 * One block of a fixed-width per-row array, by range, cached.
 *
 * `stride` is bytes per row. Blocks are 4096 rows, so an id block is 32 KB, a residue-count
 * block 8 KB, a chopping-count block 4 KB — fetched once each and reused by every hit that
 * lands in them.
 */
async function metaBlock(d, kind, blockIdx, stride) {
  const { lazy } = d;
  const hit = lazy.cache[kind].get(blockIdx);
  if (hit) return hit;
  const from = blockIdx * lazy.block * stride;
  const to = Math.min((blockIdx + 1) * lazy.block, d.n) * stride - 1;
  const b = await fetchRange(lazy.url[kind], from, to);
  lazy.cache[kind].set(blockIdx, b);
  return b;
}

/**
 * Everything about one row that the UI needs, fetched lazily.
 *
 * The chopping list is variable length, so its offset is the block's base from the coarse
 * prefix sum (3.4 KB, loaded at startup) plus the counts of the rows before this one inside
 * the block — which is why the counts are fetched a block at a time rather than a row at a
 * time.
 */
async function rowMeta(d, row) {
  const { lazy } = d;
  const bi = Math.floor(row / lazy.block);
  const within = row - bi * lazy.block;
  const [idB, cnB] = await Promise.all([
    metaBlock(d, 'ids', bi, lazy.idBytes),
    metaBlock(d, 'chopn', bi, 1),
  ]);
  const id = unpackId(idB, within * lazy.idBytes);
  let at = lazy.bases[bi];
  for (let r = 0; r < within; r++) at += cnB[r];
  const segs = cnB[within];
  const chopBytes = segs
    ? await fetchRange(lazy.url.chops, at * 4, (at + segs) * 4 - 1)
    : new Uint8Array(0);
  const chop = new Uint16Array(chopBytes.buffer, chopBytes.byteOffset, segs * 2);
  // The residue count is the summed chopping length, not a stored field. Checked on 20,000
  // rows: identical in every one, so the 6.9 MB nres array was storing what the choppings
  // already say.
  let nres = 0;
  for (let k = 0; k < segs; k++) nres += chop[k * 2 + 1] - chop[k * 2] + 1;
  return { id, nres, chop };
}


/**
 * One cluster's codes, both models, as one whole file.
 *
 * This used to be a byte range into a 45 MB shard, with a stitching loop because a cluster near a
 * shard boundary spanned two files. Both are gone: each cluster is its own file now, holding its
 * CIRPIN codes followed by its Progres codes.
 *
 * The reason is not tidiness. A range request may be answered out of a complete cached entry --
 * whole body, status 200 -- and the reader, having sliced it, returns the right answer while
 * transferring the entire shard. A 27 KB read became 45 MB and the only symptom was a number in a
 * corner. A whole-file read cannot be answered with too much, and it can be cached, which the
 * cache: 'no-store' mitigation had to give up.
 *
 * The split point needs no header: the offsets array is already downloaded, so the row count is
 * known before the file arrives.
 */
async function clusterCodes(d, model, j) {
  const key = `${j}`;
  let both = d.ivf.cache.get(key);
  if (!both) {
    const { dims } = d;
    const rows = d.ivf.starts[j + 1] - d.ivf.starts[j];
    const url = d.ivf.clusterUrl(j);
    const buf = new Uint8Array(await fetchWhole(url, count));
    if (buf.length !== rows * dims * 2) {
      throw new Error(`cluster ${j}: ${url} is ${buf.length} bytes, wanted ${rows * dims * 2} `
        + `for ${rows} rows of both models`);
    }
    both = { cirpin: buf.subarray(0, rows * dims), progres: buf.subarray(rows * dims) };
    // Cached per CLUSTER rather than per model-and-cluster: one file serves both, so the second
    // model used to be a second request for bytes already in hand.
    d.ivf.cache.set(key, both);
  }
  return both[model];
}

/**
 * Coarse-to-fine search over a clustered index.
 *
 * Score the 4096 centroids, probe the nearest few, fetch only those rows. Measured on all
 * 3,466,144 TED domains against an exhaustive scan: at nprobe 16 the top hit is always the
 * true one and 99.3% of the true top ten are found, for 0.9 MB of transfer instead of
 * 222 MB.
 *
 * Two numbers change meaning here, and the UI says so: the count of domains clearing both
 * cutoffs, and the score-plane cloud, are now over the region searched (~15k domains)
 * rather than the whole database. There is no whole-database pass left to compute them
 * from — that pass was the thing being removed.
 */
async function scoreOneIvf(cEmb, pEmb, opts, report) {
  const d = db();
  const { dims } = d;
  const pc = projectQuery(d.basis.cirpin, cEmb);
  const pp = projectQuery(d.basis.progres, pEmb);
  const maxHits = opts.maxHits ?? 25;
  const nprobe = opts.nprobe ?? d.ivf.nprobe;
  const t = performance.now();

  // rank the centroids with the same weights the scan uses
  const K = d.ivf.clusters;
  const cs = new Float32Array(K);
  for (let j = 0; j < K; j++) {
    let sc = pc.bias;
    const o = j * dims;
    for (let k = 0; k < dims; k++) sc += pc.w[k] * d.ivf.cent[o + k];
    cs[j] = sc;
  }
  const probe = Array.from({ length: K }, (_, j) => j)
    .sort((a, b) => cs[b] - cs[a]).slice(0, nprobe);

  // fetch the probed clusters and lay them out as one block, with a map back to real rows
  let total = 0;
  for (const j of probe) total += d.ivf.starts[j + 1] - d.ivf.starts[j];
  const cBlock = new Uint8Array(total * dims);
  const pBlock = new Uint8Array(total * dims);
  const rowOf = new Int32Array(total);
  let at = 0;
  for (let i = 0; i < probe.length; i++) {
    const j = probe[i];
    const [cc2, pc2] = await Promise.all([
      clusterCodes(d, 'cirpin', j), clusterCodes(d, 'progres', j)]);
    cBlock.set(cc2, at * dims);
    pBlock.set(pc2, at * dims);
    const from = d.ivf.starts[j];
    for (let r = 0; r < cc2.length / dims; r++) rowOf[at + r] = from + r;
    at += cc2.length / dims;
    if (report) report((i + 1) / probe.length);
  }


  // both cutoffs, over the searched region
  const cc = opts.cirpinCutoff ?? 0.9;
  const pcut = opts.progresCutoff ?? 0.6;
  let cpTotal = 0;
  for (let i = 0; i < total; i++) {
    let sc = pc.bias;
    const o = i * dims;
    for (let k = 0; k < dims; k++) sc += pc.w[k] * cBlock[o + k];
    if (sc <= cc) continue;
    let ps = pp.bias;
    for (let k = 0; k < dims; k++) ps += pp.w[k] * pBlock[o + k];
    if (ps < pcut) cpTotal++;
  }

  // the plane, over the same region; no stride needed at this size
  const allC = new Float32Array(total);
  const allP = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const o = i * dims;
    let a = pc.bias;
    let b = pp.bias;
    for (let k = 0; k < dims; k++) {
      a += pc.w[k] * cBlock[o + k];
      b += pp.w[k] * pBlock[o + k];
    }
    allC[i] = a;
    allP[i] = b;
  }

  // Both scores are already in hand for every candidate, so the three top lists cost one pass.
  const byC = new TopN(maxHits);
  const byP = new TopN(maxHits);
  const byD = new TopN(maxHits);
  for (let i = 0; i < total; i++) {
    const r = rowOf[i];
    byC.offer(allC[i], r, allC[i], allP[i]);
    byP.offer(allP[i], r, allC[i], allP[i]);
    byD.offer(allC[i] - allP[i], r, allC[i], allP[i]);
  }
  const { rows, scores, pScores } = unionOfTops([byC, byP, byD]);

  return {
    hits: await buildHits(d, rows, scores, pScores, opts),
    cpTotal, allC, allP, searched: total, tSearch: performance.now() - t,
  };
}

async function scoreOneCoded(cEmb, pEmb, opts, report) {
  const d = db();
  const { dims, shards, shardRows, n } = d;
  const pc = projectQuery(d.basis.cirpin, cEmb);
  const pp = projectQuery(d.basis.progres, pEmb);
  const maxHits = opts.maxHits ?? 25;

  const t = performance.now();

  const cc = opts.cirpinCutoff ?? 0.9;
  const pcut = opts.progresCutoff ?? 0.6;
  let cpTotal = 0;
  let row = 0;
  const byC = new TopN(maxHits);
  const byP = new TopN(maxHits);
  const byD = new TopN(maxHits);
  for (const codes of shards.cirpin) {
    const here = codes.length / dims;
    for (let i = 0; i < here; i++) {
      const o = i * dims;
      let sc = pc.bias;
      for (let k = 0; k < dims; k++) sc += pc.w[k] * codes[o + k];
      // Progres is now needed for EVERY row, not only those clearing the CIRPIN cutoff: the
      // Progres and Δ top lists are drawn from all of them. One extra dot product over 32 dims on
      // a database this size is a few milliseconds.
      const pShard = shards.progres[Math.floor((row + i) / shardRows)];
      const po = ((row + i) % shardRows) * dims;
      let ps = pp.bias;
      for (let k = 0; k < dims; k++) ps += pp.w[k] * pShard[po + k];
      const g = row + i;
      byC.offer(sc, g, sc, ps);
      byP.offer(ps, g, sc, ps);
      byD.offer(sc - ps, g, sc, ps);
      if (sc > cc && ps < pcut) cpTotal++;
    }
    row += here;
  }
  const { rows, scores, pScores } = unionOfTops([byC, byP, byD]);

  // The score plane needs a distribution, not every point. Sampled once it gets
  // large, because shipping 28 MB of scores to the UI thread for every query would
  // cost more than the plot can resolve.
  const STRIDE = Math.max(1, Math.floor(n / 20_000));
  const sampled = Math.floor((n + STRIDE - 1) / STRIDE);
  const allC = new Float32Array(sampled);
  const allP = new Float32Array(sampled);
  for (let k = 0; k < sampled; k++) {
    const r = k * STRIDE;
    const cShard = shards.cirpin[Math.floor(r / shardRows)];
    const pShard = shards.progres[Math.floor(r / shardRows)];
    const o = (r % shardRows) * dims;
    let cs = pc.bias;
    let ps = pp.bias;
    for (let j = 0; j < dims; j++) {
      cs += pc.w[j] * cShard[o + j];
      ps += pp.w[j] * pShard[o + j];
    }
    allC[k] = cs;
    allP[k] = ps;
  }
  const tSearch = performance.now() - t;

  return { hits: await buildHits(d, rows, scores, pScores, opts), cpTotal, allC, allP, tSearch };
}

/**
 * The n largest by some key, kept as they stream past.
 *
 * Small n against a large stream, so an insertion into a sorted array of n beats sorting the lot.
 */
class TopN {
  constructor(n) { this.n = n; this.items = []; }

  offer(key, row, cirpin, progres) {
    const a = this.items;
    if (a.length === this.n && key <= a[a.length - 1].key) return;
    let i = a.length;
    while (i > 0 && a[i - 1].key < key) i--;
    a.splice(i, 0, { key, row, cirpin, progres });
    if (a.length > this.n) a.pop();
  }
}

/**
 * The top n by EACH metric, unioned.
 *
 * The scan used to return the top n by CIRPIN alone, and the Progres score was then looked up for
 * exactly those rows. Sorting the table by Progres therefore reordered the CIRPIN top ten rather
 * than showing the Progres top ten — the column said one thing and meant another. Δ was worse
 * still: the pairs with the largest CIRPIN-minus-Progres gap are, almost by definition, not the
 * ones with the highest CIRPIN, so the most interesting hits in the whole database could not appear
 * however the user sorted.
 *
 * So all three lists are collected and the union returned. The UI sorts by whichever column is
 * active and shows the first n, which is then genuinely the top n of that metric.
 */
function unionOfTops(tops) {
  const seen = new Map();
  for (const t of tops) {
    for (const it of t.items) if (!seen.has(it.row)) seen.set(it.row, it);
  }
  const rows = [...seen.keys()];
  return {
    rows,
    scores: Float32Array.from(rows, (r) => seen.get(r).cirpin),
    pScores: Float32Array.from(rows, (r) => seen.get(r).progres),
  };
}

/** One row per hit, in the shape the UI reads. Shared by the exhaustive and clustered paths. */
async function buildHits(d, rows, scores, pScores, opts) {
  const cc = opts.cirpinCutoff ?? 0.9;
  const pcut = opts.progresCutoff ?? 0.6;
  const hits = [];
  // With a clustered index this is where the ids arrive: a block fetch per 4096 rows, and
  // the hits came from a handful of clusters so they mostly share blocks.
  const meta = d.lazy
    ? await Promise.all(Array.from(rows, (r) => rowMeta(d, r)))
    : null;
  // Which of these rows have a stored structure. One question, one store, both databases: for
  // AlphaFold the answer does not gate anything (a missing record falls back to AlphaFold DB),
  // so only a database without that fallback is asked.
  const store = d.structures === 'afdb' ? null : await coordStore(d.coords);
  const where = store
    ? await Promise.all(Array.from(rows, (r) => store.where(r)))
    : null;
  for (let k = 0; k < rows.length; k++) {
    const id = meta ? meta[k].id : rowId(d, rows[k]);
    hits.push({
      rank: k + 1,
      id,
      nres: meta ? meta[k].nres : d.nres[rows[k]],
      cirpin: scores[k],
      progres: pScores[k],
      delta: scores[k] - pScores[k],
      isCP: pScores[k] < pcut && scores[k] > cc,
      // The row travels with the hit because everything a hit needs later — its
      // chopping, its coordinates — is stored by row, and looking it up by id
      // would mean a map over millions of strings.
      row: rows[k],
      // 50 of SCOPe40's 15,176 domains have no stored coordinates. If the store has not
      // loaded, assume alignable rather than greying out a row that is fine — the click
      // reports the truth either way.
      hasCoords: !where || where[k] !== null,
    });
  }
  return hits;
}

// --- search -----------------------------------------------------------------// --- search -----------------------------------------------------------------

function dot(q, vecs, off, dim) {
  let s = 0;
  for (let k = 0; k < dim; k++) s += q[k] * vecs[off + k];
  return s;
}

/**
 * Rank by CIRPIN similarity, annotate every hit with its Progres similarity.
 * The gap between them is the signal: CIRPIN is circular-permutation
 * invariant, Progres is not, so a pair that scores high on CIRPIN and low on
 * Progres is a circular-permutation candidate. Cutoffs mirror the notebook
 * (CIRPIN.ipynb cell 9): Progres < 0.6 and CIRPIN > 0.9.
 */
async function scoreOne(coords, opts, report) {
  const t1 = performance.now();
  const graph = coordsToGraph(coords);
  // CIRPIN's per-residue norms come out of the pass that was happening anyway, so the Contribution
  // colouring costs one loop over n * 128 floats instead of a second forward pass -- which was 284 ms
  // on a 1491-residue chain through the accelerator and 3354 ms without it.
  const norms = new Float32Array(graph.n);
  const cEmb = embed(CIRPIN, graph, report && ((d, t) => report('CIRPIN', d, t)), norms);
  const pEmb = embed(PROGRES, graph, report && ((d, t) => report('Progres', d, t)));
  const tEmbed = performance.now() - t1;

  const r = db().ivf
    ? await scoreOneIvf(cEmb, pEmb, opts,
      report && ((f) => report('CIRPIN', Math.round(6 * f), 6)))
    : await scoreOneCoded(cEmb, pEmb, opts,
      report && ((f) => report('CIRPIN', Math.round(6 * f), 6)));
  return {
    n: graph.n,
    nEdges: graph.nEdges,
    embedding: cEmb,
    pEmbedding: pEmb,
    norms,
    searched: r.searched,
    hits: r.hits,
    cpTotal: r.cpTotal,
    allC: r.allC,
    allP: r.allP,
    timing: { embed: tEmbed, search: r.tSearch },
  };
}

/**
 * Split the query into structural domains, then search each one.
 *
 * Both models were trained on parsed domains, so a whole multi-domain chain is
 * out of distribution. Measured on SCOPe40 multi-domain chains, searching the
 * whole chain recovers 2 of 59 true domains at rank 1; searching the parsed
 * domains recovers 30. Splitting is a correctness requirement, not a polish.
 */
/** Residue indices covered by a list of [start, end] segments, in chain order. */
function residuesOf(segs, n) {
  const out = [];
  for (const [a, b] of segs) {
    for (let r = Math.max(0, a); r <= Math.min(n - 1, b); r++) out.push(r);
  }
  out.sort((x, y) => x - y);
  // a residue listed twice within one domain would be embedded twice
  return out.filter((v, i) => i === 0 || v !== out[i - 1]);
}

/**
 * Score each domain independently.
 *
 * Domains are explicit segment lists rather than a per-residue partition. That
 * is how SCOPe actually defines them: linker residues belong to no domain at
 * all, and a domain can be discontinuous. It also lets the user crop a domain
 * without the removed residues being forced into a neighbour, and lets two
 * domains overlap where that is genuinely what they want to search.
 */
async function unitsFromSegments(coords, flat, domainSegs, opts) {
  const n = coords.length;
  lastUnits = [];
  // Sequential rather than parallel: a clustered search fetches byte ranges, and two
  // domains of the same chain usually want overlapping clusters — going in order lets the
  // second one hit the cache the first one filled.
  const out = [];
  for (let index = 0; index < domainSegs.length; index++) {
    const segs = domainSegs[index];
    const residues = residuesOf(segs, n);
    if (residues.length < 4) {
      // coordsToGraph needs four Ca for the torsion feature
      lastUnits[index] = null;   // nothing embedded, nothing to re-scan
      out.push({
        index, residues: Int32Array.from(residues), segments: segs,
        nres: residues.length, tooShort: true,
      });
      continue;
    }
    const dc = domainCoords(flat, residues);
    const asPairs = [];
    for (let i = 0; i < residues.length; i++) {
      asPairs.push([dc[i * 3], dc[i * 3 + 1], dc[i * 3 + 2]]);
    }
    const steps = domainSegs.length * 2 * 6;
    const scored = await scoreOne(asPairs, opts, (model, done, total) => {
      const doneSteps = index * 12 + (model === 'Progres' ? 6 : 0) + done;
      say(`Embedding domain ${index + 1} of ${domainSegs.length}`,
        `${model} · layer ${done} of ${total}`, doneSteps / steps);
    });
    // Copies, not references: the CIRPIN embedding's buffer is TRANSFERRED to the UI
    // thread with the result, which detaches it here. Keeping the original meant a
    // re-scan read a neutered buffer and every score came back NaN.
    lastUnits[index] = {
      cEmb: new Float32Array(scored.embedding),
      pEmb: new Float32Array(scored.pEmbedding),
    };
    out.push({
      index, residues: Int32Array.from(residues), segments: segs,
      nres: residues.length, ...scored,
    });
  }
  return out;
}

/** Contiguous runs of one label, as [start, end] pairs. */
function labelsToSegments(labels, nDomains) {
  const out = Array.from({ length: nDomains }, () => []);
  let s = 0;
  for (let i = 1; i <= labels.length; i++) {
    if (i === labels.length || labels[i] !== labels[s]) {
      out[labels[s]].push([s, i - 1]);
      s = i;
    }
  }
  return out;
}

/** The parser's own answer, or the whole chain when it is too short to split. */
let lastConfidence = 1;
let lastCoassoc = null;

/**
 * The whole chain, one unit.
 *
 * Nothing is split until the user asks. A chain arrives as itself, which is the only
 * honest default: every automatic split is a guess, and the guess used to be made
 * before anyone had looked at the structure. 'parseDomains' runs the real thing.
 */
function autoSegments(coords) {
  lastConfidence = 1;
  lastCoassoc = null;
  return [[[0, coords.length - 1]]];
}

// --- domain parsing ----------------------------------------------------------
//
// WHAT USED TO BE HERE, AND WHY IT IS GONE. A search-guided refinement: over-parse deliberately
// (maxCut 0.12 against the 0.08 default), then merge two fragments when their union retrieved a
// better top hit than the weaker piece did. Held out on half the 260-chain benchmark it scored 74.8%
// exact domain count against the geometric parser's 69.9%, NDO 0.9379 against 0.9284 -- a real five
// points, so this is a deliberate loss and not a cleanup.
//
// It went because the decision was circular. It settled a question about the embedding's own input
// using the embedding's cosine, and that cosine is not a calibrated confidence: measured on 1,000
// SCOPe queries with the query's superfamily held out, the published 0.9 "same fold" cutoff is 32%
// precise, and no higher cutoff rescues it (0.95 is 48%). Random self-avoiding walks clear 0.9
// unanimously. So the merge was asking an oracle that does not know, and when it was wrong it was
// wrong catastrophically rather than marginally: on 12BK it stitched three domains the geometric
// parser had found unanimously -- confidence 1.000, sizes 225/89/68 -- back into a single chain. Two
// guards were added for that case (an absolute 0.9 floor, and only merging domains in contact) and
// they fixed 12BK without fixing the reasoning.
//
// What replaces it is the threshold ladder: parseDomains' own maxCut is a granularity dial, and
// sweeping it costs about one parse because maxCut is only an accept filter (see domains.js
// buildLadder). The page gets every decomposition the contact graph supports and the user picks the
// count. Over the 260-chain benchmark that offers the true count on 88% of chains, against 65% for
// any fixed threshold -- so the five points are recoverable by the person who can see the structure,
// and nothing can silently collapse a correct parse again.
//
// The geometric parser's own error profile, unchanged and worth keeping in view: split PLACEMENT is
// right to within 20 residues on 24 of 25 chains, and the count is exact 58.5% of the time. The
// count is the whole problem, which is exactly why it is now the thing being offered rather than
// decided.

async function runQuery(coords, opts, presetSegs) {
  const t0 = performance.now();
  const flat = new Float64Array(coords.length * 3);
  coords.forEach((c, i) => {
    flat[i * 3] = c[0]; flat[i * 3 + 1] = c[1]; flat[i * 3 + 2] = c[2];
  });

  const domainSegs = presetSegs || autoSegments(coords, flat, opts);
  const tParse = performance.now() - t0;

  const units = await unitsFromSegments(coords, flat, domainSegs, opts);
  return {
    netBytes,
    nChain: coords.length,
    nDomains: units.length,
    units,
    domainSegs,
    tParse,
  };
}

/**
 * Fetch a TED domain's structure from AlphaFold DB and cut the domain out.
 *
 * AlphaFold has moved to v6 while these domains and their choppings were defined
 * on v4, so what comes back is a newer prediction of the same protein. That is
 * fine for alignment: the sequence is unchanged, so residue numbering is
 * unchanged, and the chopping selects the same residues. It is a different model
 * of the domain, not a different domain — worth knowing when a TM-score is a
 * hundredth off what the paper reports, and not worth 832 MB of foldcomp and a
 * decoder to avoid.
 *
 * Versions are tried newest first and the winner is remembered, because the whole
 * database is currently on one version and probing per hit would double the
 * requests.
 */
// Measured on 60 random TED accessions: 90% have a v6 model, none is v4-only, and
// 10% have none at all because UniProt has since retired the entry. So v6 first,
// v4 as a cheap safety net, and a miss costs two requests rather than three.
const AFDB_VERSIONS = [6, 4];



let afdbVersion = null;
const afdbCache = new Map();

async function afdbModel(accession) {
  if (afdbCache.has(accession)) return afdbCache.get(accession);
  const tryOrder = afdbVersion
    ? [afdbVersion, ...AFDB_VERSIONS.filter((v) => v !== afdbVersion)]
    : AFDB_VERSIONS;
  let lastStatus = 0;
  for (const v of tryOrder) {
    const url = `https://alphafold.ebi.ac.uk/files/AF-${accession}-F1-model_v${v}.cif`;
    const r = await fetch(url);
    if (r.ok) {
      afdbVersion = v;
      const text = await r.text();
      // Counted like everything else. It was not, so the byte readout was a lower bound that
      // omitted the largest thing a single hit downloads.
      netBytes += text.length;
      // Bounded so a long browsing session cannot grow without limit; a full
      // model is tens of kilobytes and one accession can serve several domains.
      if (afdbCache.size > 64) afdbCache.clear();
      afdbCache.set(accession, text);
      return text;
    }
    lastStatus = r.status;
  }
  throw new Error(`AlphaFold has no model for ${accession}. UniProt has retired about `
    + 'one in ten of the entries TED was built on, so some hits cannot be aligned. '
    + `(last status ${lastStatus})`);
}

/** Cα of one TED domain, as a flat Float64Array. */
async function afdbDomainCoords(row) {
  const d = db();
  // One range fetch for this row's id and chopping when the index is clustered; the block
  // is almost always already cached, because the hit list came through here first.
  const m = d.lazy ? await rowMeta(d, row) : null;
  const id = m ? m.id : rowId(d, row);
  const accession = id.slice(3, id.indexOf('-F1-'));
  say('Fetching', `${accession} from AlphaFold DB`, 0.1);
  const text = await afdbModel(accession);

  const seq = [];
  const coords = parseCIF(text, { seq });
  if (!coords.length) throw new Error(`could not read a model for ${accession}`);

  // Residues wanted, from this domain's chopping
  const want = new Set();
  const segs = m ? m.chop.length / 2 : d.chopN[row];
  const a0 = m ? 0 : d.chopAt[row];
  const src = m ? m.chop : d.chopS;
  for (let k = 0; k < segs; k++) {
    const lo = src[(a0 + k) * 2];
    const hi = src[(a0 + k) * 2 + 1];
    for (let r = lo; r <= hi; r++) want.add(r);
  }

  const keep = [];
  for (let i = 0; i < coords.length; i++) if (want.has(seq[i])) keep.push(coords[i]);
  if (keep.length < 4) {
    throw new Error(`${id}: only ${keep.length} of ${want.size} domain residues are in `
      + 'the current AlphaFold model');
  }
  // d.nres is null on the clustered path -- the residue count comes from the chopping instead, which
  // is what rowMeta already recomputed above. Reading d.nres[row] here threw for every AlphaFold hit
  // that got this far.
  const expect = m ? m.nres : d.nres[row];
  if (keep.length !== expect) {
    // The sequence should be identical across releases, so this means the
    // chopping and the model have genuinely diverged. Usable, but say so.
    say('Fetching', `${id}: ${keep.length} residues, expected ${expect}`, 0.2);
  }
  const out = new Float64Array(keep.length * 3);
  keep.forEach((c, i) => { out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2]; });
  return out;
}

// --- coordinates on demand ---------------------------------------------------

/** Per-residue CIRPIN vectors for a flat Ca array, through the accelerator when it is loaded. */
function nodeVectorsOf(flat, n) {
  if (!cirpinW || !n) return null;
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
  const g = coordsToGraph(pairs);
  const nv = new Float32Array(g.n * EMBED_DIM);
  if (accel) accel.embed(0, g, null, nv);
  else embedGraph(cirpinW, g, null, null, nv);
  return nv;
}

/**
 * One database's coordinate store, built once per base prefix.
 *
 * Both databases go through here. A store is a manifest, a block base table and the arithmetic
 * in src/coords.js; what differs between SCOPe and AlphaFold TED is the prefix and nothing else.
 * Returns null when no store is configured, or when its files are not reachable.
 */
/**
 * A database's classification lookup: which class, fold and family each row belongs to.
 *
 * Fetched on first use rather than at load, and cached per prefix. Nothing needs it until a hit is
 * selected, and it is between 0.4 MB (SCOPe, CATH) and 2.7 MB (ECOD) -- worth nothing at all to a search,
 * so making a search wait for it would be a pure loss.
 *
 * `paths` is flat, `levels.length` entries per path, and `names` is interned per level: ECOD40 has 448,232
 * domains over 34,748 distinct paths drawn from 32,446 names, so a name per row would be ~20 MB where an
 * index per row is 0.9 MB. See tools/build_class.py.
 *
 * Failure is null, not an exception. A missing lookup should cost the card a few rows, not the alignment
 * the card is describing.
 */
const classifications = new Map();
function classStore(prefix) {
  if (!prefix) return null;
  if (classifications.has(prefix)) return classifications.get(prefix);
  const p = (async () => {
    const meta = await fetchJSON(`${prefix}-class.json`);
    const buf = await fetchBin(`${prefix}-class.bin`);
    const view = meta.bits === 16 ? new Uint16Array(buf) : new Uint32Array(buf);
    if (view.length !== meta.rows) {
      throw new Error(`${prefix}-class.bin has ${view.length} rows, the manifest says ${meta.rows}`);
    }
    return { ...meta, view, sentinel: meta.bits === 16 ? 0xFFFF : 0xFFFFFFFF };
  })().catch(() => null);
  classifications.set(prefix, p);
  return p;
}

/** The hierarchy one row sits in: {scheme, code, levels: [{level, name}]}, or null. */
async function classOf(d, row) {
  if (!d || row == null || row < 0) return null;
  const c = await classStore(d.prefix);
  if (!c || row >= c.rows) return null;
  const at = c.view[row];
  if (at === c.sentinel) return null;
  const n = c.levels.length;
  return {
    scheme: c.scheme,
    code: c.codes[at],
    levels: c.levels.map((lv, i) => ({ level: lv, name: c.names[i][c.paths[at * n + i]] })),
  };
}

const stores = new Map();
function coordStore(prefix) {
  if (!prefix) return null;
  if (stores.has(prefix)) return stores.get(prefix);
  const p = (async () => {
    const manifest = await fetchJSON(`${prefix}.json`);
    await coordCodebook();
    if (manifest.codebook && codebookHash && manifest.codebook !== codebookHash) {
      // Refuse rather than decode: the records would come out as plausible structures that are
      // not the ones asked for, and nothing downstream could tell.
      throw new Error(`${prefix} was encoded against codebook ${manifest.codebook}, `
        + `this build has ${codebookHash}`);
    }
    const dir = prefix.slice(0, prefix.lastIndexOf('/') + 1);
    const base = new Uint32Array(await fetchBin(`${dir}${manifest.base}`));
    return shardedStore(manifest, base, fetchRange, (f) => `${dir}${f}`);
  })().catch(() => null);
  stores.set(prefix, p);
  return p;
}

/**
 * One hit's Cα coordinates from its database's store, or null if it has none.
 *
 * This is the whole reader. SCOPe40 and AlphaFold TED reach it identically; the only difference
 * is what a caller does with null, which is why that decision is left to them — SCOPe reports it
 * as unalignable, AlphaFold falls back to fetching the model from the EBI.
 */
async function storedCoords(d, row) {
  const store = await coordStore(d.coords);
  if (!store) return null;
  try {
    const w = await store.where(row);
    if (!w) return null;
    const b = await fetchRange(w.url, w.start, w.end);
    return decodeRecord(b.buffer, b.byteOffset, await coordCodebook());
  } catch {
    return null;
  }
}

/**
 * Align the query against one hit, both sequentially and with circular
 * permutation. Mirrors CIRPIN.ipynb cell 11, which runs TM-align twice and
 * reports the difference.
 *
 * Both alignments are returned in full — score field, path and fitted
 * coordinates — so the UI can switch between them rather than overlaying them.
 * Everything is expressed in ORIGINAL query numbering, including the permuted
 * one, which is the whole point: the cut is undone in the numbering the aligner
 * works in and only visible in the numbering the user reads.
 */
/**
 * TM-score one hit and nothing else.
 *
 * runAlign also builds two score fields and keeps the fitted coordinates, which is
 * most of its cost and all pointless when the only question is "is this the same fold
 * at all". Both alignments are still run: a circular permutation scores badly
 * sequentially, and rejecting it on the sequential score is exactly the mistake this
 * whole program exists to avoid.
 */
async function scoreAlignOnly(xa, xlen, id, row) {
  const d = db();
  const ya = (await storedCoords(d, row))
    ?? (d.structures === 'afdb' ? await afdbDomainCoords(row) : null);
  if (!ya) throw new Error(`no coordinates stored for ${id}`);
  const ylen = ya.length / 3;

  // One alignment, once: cpAlign with -fast.
  //
  // cpAlign duplicates the query head to tail, so shift 0 is the unpermuted case and
  // the separate sequential run is very nearly redundant — over 37 pairs it scored
  // higher only 3 times and by at most 0.0094. -fast costs up to 0.021 against the full
  // search. Both are accepted deliberately: this is a shortlist filter, and a hit
  // within about 0.02 of the 0.5 line may fall on the wrong side of it.
  //
  // Nothing is hidden by that: selecting a hit runs both alignments at full settings
  // and reports each score, so the exact answer for any hit that matters is one click
  // away. What this buys is roughly 1.8x, from -fast, over a list of 40.
  //
  // The sequential score is no longer dropped, because it was never actually saved: cpAlign's pass 2
  // computes it to decide whether the permutation is real, so reporting it costs nothing and the
  // shortlist can show both numbers instead of a null.
  const cp = cpAlign(xa, ya, xlen, ylen, { fast: true });
  return { tm: cp.linear.TM1, tmCp: cp.TM1, best: Math.max(cp.TM1, cp.linear.TM1), ylen };
}

/**
 * Check a list of hits with TM-align, reporting each verdict as it lands.
 *
 * The embedding search is a filter, not a proof: it can rank a structure highly that
 * TM-align puts below 0.5, which is the conventional line for "same fold". Running it
 * over the shortlist afterwards turns a ranked guess into a checked answer, and it is
 * affordable — about 0.2 s per hit at 300 residues — as long as it happens after the
 * results are already on screen rather than before.
 */
const TM_FOLD = 0.5;      // the conventional same-fold line
let verifyGen = 0;
let lastUnits = [];   // per-unit query embeddings, for re-scanning at a new depth

async function verifyHits(gen, queryCoords, hits) {
  const xlen = queryCoords.length;
  const xa = new Float64Array(xlen * 3);
  queryCoords.forEach((c, i) => { xa[i * 3] = c[0]; xa[i * 3 + 1] = c[1]; xa[i * 3 + 2] = c[2]; });
  let done = 0;
  for (const h of hits) {
    // Anything the user does bumps the generation: a stale check would purge rows
    // belonging to a search that is no longer on screen.
    if (gen !== verifyGen) return;
    let verdict = null;
    try {
      verdict = await scoreAlignOnly(xa, xlen, h.id, h.row);
    } catch {
      // No coordinates, or a model that would not download. Unjudged, so kept.
      verdict = null;
    }
    done++;
    if (gen !== verifyGen) return;
    post({ type: 'verdict', gen, id: h.id, done, total: hits.length,
      tm: verdict ? verdict.best : null, tmSeq: verdict ? verdict.tm : null,
      tmCp: verdict ? verdict.tmCp : null });
    // Let queued messages through, or a long check would freeze every other action.
    await breathe();
  }
  if (gen === verifyGen) post({ type: 'verifyDone', gen, total: hits.length });
}

async function runAlign(queryCoords, id, row) {
  const d = db();
  const ya = (await storedCoords(d, row))
    ?? (d.structures === 'afdb' ? await afdbDomainCoords(row) : null);
  if (!ya) throw new Error(`no coordinates stored for ${id}`);
  const ylen = ya.length / 3;
  const xa = new Float64Array(queryCoords.length * 3);
  queryCoords.forEach((c, i) => {
    xa[i * 3] = c[0]; xa[i * 3 + 1] = c[1]; xa[i * 3 + 2] = c[2];
  });
  const xlen = queryCoords.length;

  const t0 = performance.now();
  // -fast here too, matching the filter (see verifyHits). Not for speed: the filter measures every
  // hit with { fast: true }, so without this the same pair gets a different TM depending on which
  // path measured it -- a score in the table that changes when you click the row. Same reason the
  // C++ has one flag rather than two: the coarse initial search is either on or off for a
  // comparison, and mixing them makes two numbers that are not the same measurement.
  say('Aligning', `TM-align, sequential · ${xlen} vs ${ylen} residues`, 0.15);
  // ONE cpAlign, not a cpAlign and a tmAlign.
  //
  // cpAlign's pass 2 already runs the sequential alignment -- it has to, to decide whether the
  // permutation is real -- with exactly the arguments tmAlign would use and the same fast flag. Calling
  // tmAlign as well repeated 22-29% of the work for a bit-identical result; verified equal to six
  // decimal places on four pairs before this was changed.
  say('Aligning', `TM-align, sequential and permuted · ${xlen} vs ${ylen} residues`, 0.15);
  const cp = cpAlign(xa, ya, xlen, ylen, { fast: true });
  const plain = cp.linear;
  say('Aligning', 'building the score fields', 0.9);
  const ms = performance.now() - t0;

  // One score field per alignment, on a shared grid so the two are comparable.
  const MAX_CELLS = 480000;
  const stride = Math.max(1, Math.ceil(Math.sqrt((xlen * ylen) / MAX_CELLS)));
  const mw = Math.ceil(xlen / stride);
  const mh = Math.ceil(ylen / stride);

  function buildMode(res, shift) {
    // `shift` is the permutation offset: fitted index p holds original residue
    // (p + shift) % xlen. Zero for the sequential alignment.
    const xUse = shift ? permuteCoords(xa, xlen, shift) : xa;
    const fitted = applyTransform(xUse, xlen, res.t0, res.u0);
    const orig = (p) => (shift ? (p + shift) % xlen : p);

    const map = new Uint8Array(mw * mh);
    const d02 = res.d0A * res.d0A;
    for (let p = 0; p < xlen; p++) {
      const i = orig(p);
      if (i % stride !== 0) continue;
      const mx = i / stride;
      const qx = fitted[p * 3]; const qy = fitted[p * 3 + 1]; const qz = fitted[p * 3 + 2];
      for (let j = 0; j < ylen; j += stride) {
        const dx = qx - ya[j * 3];
        const dy = qy - ya[j * 3 + 1];
        const dz = qz - ya[j * 3 + 2];
        map[(j / stride) * mw + mx] = Math.round((1 / (1 + (dx * dx + dy * dy + dz * dz) / d02)) * 255);
      }
    }

    const path = new Int32Array(res.n_ali8 * 2);
    for (let k = 0; k < res.n_ali8; k++) {
      path[k * 2] = orig(res.m1[k]);
      path[k * 2 + 1] = res.m2[k];
    }

    // Fitted coordinates reordered into original numbering, so the
    // superposition draws the chain in the order the user reads.
    const drawn = new Float64Array(xlen * 3);
    for (let p = 0; p < xlen; p++) {
      const i = orig(p);
      drawn[i * 3] = fitted[p * 3];
      drawn[i * 3 + 1] = fitted[p * 3 + 1];
      drawn[i * 3 + 2] = fitted[p * 3 + 2];
    }

    // The superposition run the other way: the HIT brought into the query's frame.
    //
    // This is what the views draw. TM-align's own direction moves the query onto the hit, which
    // means the query is drawn differently for every hit — so a user clicking through candidates
    // re-reads a structure that has just rotated, and nothing can be compared by eye. Sending the
    // inverse leaves the query where it is and brings each hit to it.
    //
    // The permutation does not enter here: permuteCoords reorders residues without moving any of
    // them, so xUse and xa describe the same frame and the inverse lands in it either way.
    const targetInQueryFrame = applyInverseTransform(ya, ylen, res.t0, res.u0);

    return {
      tm: res.TM1,
      tmByQuery: res.TM2,
      nAligned: res.n_ali8,
      rmsd: res.rmsd0,
      map,
      path,
      fitted: drawn,
      targetFitted: targetInQueryFrame,
    };
  }

  const seq = buildMode(plain, 0);
  const withCp = buildMode(cp, cp.cpPoint);

  /**
   * CIRPIN's own view of the same comparison: cos(node_i, node_j) for every residue pair.
   *
   * On the SAME grid as the TM fields, so the two panels are pixel-for-pixel comparable -- that is the
   * whole point of showing them together. Independent of the alignment, so it is computed once rather
   * than per mode: the TM field asks "does residue i land on residue j under this superposition", this
   * asks "does the model think these two residues play the same role", and neither question mentions the
   * other's answer.
   *
   * Both structures are embedded on their own graph and the vectors share a weight set, so the cosine
   * between them is meaningful. Measured on six same-superfamily CATH pairs, this map separates the pairs
   * TM-align aligned from the rest at AUC 0.75-0.93 -- but its pixel correlation with the TM field is
   * only 0.15-0.41, so it finds the right region without reproducing the geometry. The disagreement is
   * the interesting part and the reason for two panels instead of one.
   *
   * Fixed 0..1 scale, clamping negatives to zero, rather than per-map normalisation: a map rescaled to
   * its own extremes would look equally bright for a good hit and a hopeless one.
   */
  function cosineField() {
    const qn = nodeVectorsOf(xa, xlen);
    const tn = nodeVectorsOf(ya, ylen);
    if (!qn || !tn) return null;
    const norm = (v, n) => {
      const o = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sq = 0;
        for (let k = 0; k < EMBED_DIM; k++) sq += v[i * EMBED_DIM + k] ** 2;
        o[i] = Math.sqrt(sq) || 1;
      }
      return o;
    };
    const nq = norm(qn, xlen);
    const nt = norm(tn, ylen);
    const out = new Uint8Array(mw * mh);
    for (let i = 0; i < xlen; i += stride) {
      const mx = i / stride;
      for (let j = 0; j < ylen; j += stride) {
        let d = 0;
        for (let k = 0; k < EMBED_DIM; k++) d += qn[i * EMBED_DIM + k] * tn[j * EMBED_DIM + k];
        const c = d / (nq[i] * nt[j]);
        out[(j / stride) * mw + mx] = c <= 0 ? 0 : Math.min(255, Math.round(c * 255));
      }
    }
    return out;
  }
  let cosMap = null;
  try {
    say('Aligning', 'CIRPIN residue similarity', 0.95);
    cosMap = cosineField();
  } catch (e) {
    cosMap = null;      // the panel is simply not offered
  }
  say('', '', 1);

  return {
    id,
    xlen,
    ylen,
    // The query exactly as it was handed in: what every view now draws for the left-hand side,
    // identical for every hit.
    queryFixed: xa,
    mapW: mw,
    mapH: mh,
    mapStride: stride,
    // null when the weights are not loaded, in which case the panel is not offered
    cosMap,
    cpPoint: cp.cpPoint,
    tm: plain.TM1,
    tmCp: cp.TM1,
    tmDiff: cp.TM1 - plain.TM1,
    prefer: cp.TM1 > plain.TM1 ? 'cp' : 'seq',
    seq,
    cp: withCp,
    target: ya,
    ms,
  };
}

/** Report a chain: its coordinates, the chains on offer, and an automatic split. */
async function postParsed(requestId, chain) {
  activeChain = chain;
  const coords = inputChains.get(chain);
  const flat = new Float64Array(coords.length * 3);
  coords.forEach((c, i) => {
    flat[i * 3] = c[0]; flat[i * 3 + 1] = c[1]; flat[i * 3 + 2] = c[2];
  });
  const domainSegs = autoSegments(coords);
  post({
    type: 'parsed',
    requestId,
    coords,
    chain,
    chains: [...inputChains].map(([id, c]) => ({ id, n: c.length })),
    domainSegs,
    confidence: lastConfidence,
    coassoc: lastCoassoc,
    nChain: coords.length,
  }, lastCoassoc ? [lastCoassoc.data.buffer] : []);
}

// --- messages ----------------------------------------------------------------

onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') { await init(); return; }

    if (msg.type === 'useDb') {
      const d = await loadCoded(msg.db);
      activeDb = msg.db;
      // No say('Ready') here: the UI finishes its own progress bar on dbReady and
      // puts the database name back in the header, and a progress message arriving
      // after that would overwrite it with the word "Ready".
      post({
        type: 'dbReady', db: msg.db, label: d.label, n: d.n,
        alignable: true, netBytes, ivf: !!d.ivf,
      });
      return;
    }

    // Parsing and splitting need no weights, so they answer immediately and
    // never score. Scoring is a separate request, because the split and the chain
    // are choices worth making before spending a search on them.
    if (msg.type === 'parse' || msg.type === 'useChain' || msg.type === 'embed'
        || msg.type === 'parseDomains' || msg.type === 'useDb') {
      verifyGen++;   // results are about to change; stop judging the old ones
    }

    if (msg.type === 'parse') {
      let chains;
      if (msg.format === 'coords') {
        chains = new Map([['A', { coords: parseCoordsTxt(msg.text) }]]);
      } else {
        chains = parseStructureChains(msg.text, msg.format || 'guess');
      }
      for (const [id, e] of [...chains]) if (e.coords.length < 4) chains.delete(id);
      if (!chains.size) throw new Error('no chain in that file has four or more Cα');
      inputChains = new Map([...chains].map(([id, e]) => [id, e.coords]));
      // The longest chain, not the first. The reference takes the first, but on a
      // complex that is often a peptide — 1A0N opens on a 14-residue chain A next
      // to a 58-residue chain B — and a search of it answers a question nobody
      // asked. The picker still offers every chain, so the reference's choice is
      // one click away when reproducing it matters.
      let best = null;
      for (const [id, c] of inputChains) if (!best || c.length > best[1]) best = [id, c.length];
      await postParsed(msg.requestId, best[0]);
      return;
    }

    if (msg.type === 'useChain') {
      if (!inputChains) throw new Error('no structure loaded');
      if (!inputChains.has(msg.chain)) throw new Error(`no chain ${msg.chain}`);
      await postParsed(msg.requestId, msg.chain);
      return;
    }

    if (msg.type === 'parseDomains') {
      if (!inputChains || !activeChain) throw new Error('no structure loaded');
      if (!modelsReady) {
        say('Waiting for the models', 'domain parsing needs the network', 0.02);
        await readyPromise;
      }
      const coords = inputChains.get(activeChain);
      const flat = new Float64Array(coords.length * 3);
      coords.forEach((c, i) => { flat[i * 3] = c[0]; flat[i * 3 + 1] = c[1]; flat[i * 3 + 2] = c[2]; });
      // THE SPECTRAL PARSER GIVES THE ANSWER. partition.js briefly did, on the strength of ARI 0.7471
      // against 0.6382 -- both measured on a 260-chain benchmark that contained no single-domain chains
      // at all. Rebuilt with 1,155 chains including 459 single-domain ones, the ordering reverses:
      //
      //                     ARI    ARI(disc)  count    k=1
      //   spectral        0.6930    0.5416   57.7%   0.766   <- best overall
      //   partition       0.6703    0.6181   60.9%   0.668
      //   pool oracle     0.9157    0.8114   84.5%   1.000
      //
      // The whole margin is k=1: single-domain chains are 40% of a representative set and this parser
      // is markedly better at leaving them alone. partition wins on discontinuous chains (+0.077) and
      // on exact count, but loses where it matters more.
      say('Domains', 'spectral parse of the contact graph', 0.3);
      const spectral = parseDomains(flat);
      lastConfidence = spectral.confidence ?? 1;
      lastCoassoc = spectral.coassoc ?? null;
      const domainSegs = labelsToSegments(spectral.labels, spectral.domains.length);

      say('', '', 1);
      post({ type: 'domainsParsed', requestId: msg.requestId, domainSegs });
      return;
    }

    if (msg.type === 'embed') {
      // An already-parsed chain being scored, under whatever split is on screen.
      if (!modelsReady) {
        say('Waiting for the models', 'they are still downloading', 0.02);
        await readyPromise;
      }
      const coords = msg.coords;
      const r = await runQuery(coords, msg, msg.domainSegs);
      post({ type: 'result', requestId: msg.requestId, coords, edited: !!msg.edited, ...r },
        r.units.flatMap((u) => (u.tooShort ? [u.residues.buffer]
          : [u.embedding.buffer, u.residues.buffer, u.allC.buffer, u.allP.buffer, u.norms.buffer])));
      return;
    }

    // Asking for a longer shortlist is a re-scan, not a re-search: the embeddings are
    // what cost a second each, and they have not changed. The scan itself is ~2 ms
    // over 15k entries, so any list length is effectively free once they are in hand.
    if (msg.type === 'rescan') {
      if (!lastUnits.length) throw new Error('nothing to re-scan');
      verifyGen++;   // the rows are about to change; drop any check in flight
      const out = [];
      for (let index = 0; index < lastUnits.length; index++) {
        const u = lastUnits[index];
        if (!u) { out.push({ index, hits: null }); continue; }
        const opts2 = { ...msg.opts, maxHits: msg.maxHits };
        const r = db().ivf
          ? await scoreOneIvf(u.cEmb, u.pEmb, opts2)
          : await scoreOneCoded(u.cEmb, u.pEmb, opts2);
        out.push({ index, hits: r.hits, cpTotal: r.cpTotal });
      }
      post({ type: 'rehits', requestId: msg.requestId, units: out });
      return;
    }

    if (msg.type === 'verify') {
      const gen = ++verifyGen;
      await verifyHits(gen, msg.coords, msg.hits);
      return;
    }

    /*
     * Per-residue contribution to the CIRPIN embedding.
     *
     * The readout is a plain sum over residues, so the vector a residue adds is exactly what it
     * contributes, and the norm of that vector is how far it moves the result. Everything after the
     * sum is nonlinear, so this is the last point at which "how much does this residue matter" has a
     * defined answer — which is why embedGraph offers the sink at all.
     *
     * Through the ACCELERATOR, which needed no new sink: data/cirpin.wasm writes the per-residue node
     * vector when given a pointer, and the contribution norm is exactly the L2 norm of that vector -- the
     * same quantity embedGraph's normSink computes, one step later. So the norms come out of the node sink
     * for free. 11.8x faster on a 1491-residue chain (3354 ms -> 284 ms), agreeing with the JS path to
     * 3.7e-6 relative. This used to read "the accelerator has no sink to fill", which stopped being true
     * when the node sink was added.
     *
     * Search is unaffected: the sink is a pointer the search path passes as null, and the rebuilt module
     * measures 0.24% off the previous one over 24,678 residues -- noise.
     */
    if (msg.type === 'contrib') {
      const graph = coordsToGraph(msg.coords);
      const norms = new Float32Array(graph.n);
      if (accel) {
        const nv = new Float32Array(graph.n * EMBED_DIM);
        accel.embed(0, graph, null, nv);
        for (let i = 0; i < graph.n; i++) {
          let ss = 0;
          for (let k = 0; k < EMBED_DIM; k++) { const v = nv[i * EMBED_DIM + k]; ss += v * v; }
          norms[i] = Math.sqrt(ss);
        }
      } else {
        embedGraph(cirpinW, graph, null, norms);
      }
      post({ type: 'contrib', requestId: msg.requestId, norms }, [norms.buffer]);
    }

    if (msg.type === 'align') {
      const r = await runAlign(msg.coords, msg.id, msg.row);
      // The classification rides with the alignment rather than in a message of its own: the card is
      // only on screen once an alignment has arrived, and the row number is already here.
      const cls = await classOf(db(), msg.row);
      post({ type: 'aligned', requestId: msg.requestId, cls, ...r }, [
        r.target.buffer, r.queryFixed.buffer,
        r.seq.map.buffer, r.seq.path.buffer, r.seq.fitted.buffer, r.seq.targetFitted.buffer,
        r.cp.map.buffer, r.cp.path.buffer, r.cp.fitted.buffer, r.cp.targetFitted.buffer,
        // conditional: an absent cosine map must not put a null in the transfer list
        ...(r.cosMap ? [r.cosMap.buffer] : []),
      ]);
    }
  } catch (err) {
    post({ type: 'error', requestId: msg.requestId, message: err.message });
  }
};
