// Structural domain parsing from Cα coordinates — no model, no weights.
//
// CIRPIN and Progres were both trained on parsed domains, so a whole
// multi-domain chain is out of distribution for them. Measured on SCOPe40
// two-domain chains, querying the whole chain recovers neither true domain at
// rank 1, while querying each domain separately recovers both. Splitting the
// query first is therefore not a refinement, it is a correctness requirement.
//
// Method: recursive spectral bisection of the Cα contact graph, with a
// PDP-style acceptance test on the interface.
//
// The classic parsers — DOMAK (1995), DomainParser (2000), PDP (2003) — all
// encode the same claim, that a domain has many internal contacts and few
// external ones. They differ in how they search for the cut. DOMAK and PDP
// enumerate one- and two-segment cut points in sequence, which caps how
// discontinuous a domain can be. The Fiedler vector of the graph Laplacian
// partitions on connectivity alone with no notion of sequence contiguity, so
// discontinuous domains fall out without enumerating anything; a weighted
// backbone edge is what keeps it from returning shredded, non-contiguous
// nonsense. Cost is one eigenvector of a sparse n x n Laplacian.


const DEFAULTS = {
  contactModel: 'ca',   // 'ca' plain Cα cut-off, or 'cb' the virtual-Cβ model
  cbOffset: 3.5,        // Å from Cα to the virtual Cβ ('cb' only)
  cbCut: 8.5,           // Å between virtual Cβ to count as a contact ('cb' only)
  contactDist: 8.0,     // Å between Cα to count as a contact ('ca' only)
  contactSoft: 1.0,     // Å of raised-cosine taper either side of it
  backboneWeight: 2.0,  // weight on (i, i+1); the price of one cut point
  minDomain: 40,        // residues; SCOPe's practical floor
  minSegment: 50,       // shorter runs get absorbed by whichever domain they touch most
  // Terminal trimming, OFF because it was measured and it does not work. Keep
  // reading before turning it on.
  //
  // The premise is sound: CATH and SCOPe discard residues that are not part of
  // the compact domain, and they do it at the ends — over 130 single-domain
  // SCOPe40 chains the median loses 2.8% and the mean 15.8%, and 99% of what
  // goes is leading or trailing rather than internal. Matching those boundaries
  // is worth real signal: querying with SCOPe's exact boundary retrieves the
  // true domain at rank 1 in 18 of 18 cases at mean similarity 0.9985, against
  // 17 of 18 at 0.9414 for the untrimmed split.
  //
  // But peeling terminal residues by long-range contact count is the wrong
  // instrument. It over-trims about three to one, and on the genuine-tail cases
  // it was designed for it makes everything worse: kept-residue IoU 0.845
  // against 0.917 untrimmed, boundaries within ten residues 75% against 93%,
  // and retrieval collapses to 10 of 18 at 0.7142. Tuning does not rescue it —
  // across trimDegree 2..6 and trimSep 2..5 the only setting that does not lose
  // ground is the one that trims nothing.
  //
  // The reason is headroom. With trimming off the split already lands within ten
  // residues of SCOPe's boundary 93% of the time, so there are only three to
  // seven residues to win, while this peeler removes about twelve. The precision
  // needed is finer than a contact-count threshold can deliver.
  //
  // Something better would have to use what this ignores: secondary-structure
  // continuity, or burial, or the fact that many "trimmed" regions are really a
  // neighbouring domain the 40% subset excludes rather than a tail at all —
  // terminal peeling cannot find those and should not try. Until then the manual
  // crop in the UI is the reliable route. See test/trim.mjs.
  // Terminal trimming, off. CATH and SCOPe both discard residues that are not part
  // of the compact domain — median 2.1% of a chain, 99% of it terminal — so it
  // looks like free accuracy, and two attempts have both cost more than they paid
  // on the 130-chain trim benchmark:
  //
  //   no trimming                      kept-residue IoU 0.911
  //   contact degree                                   0.845
  //   consensus stability, best tuning                 0.888
  //
  // Stability is the better signal of the two — a residue the perturbed runs kept
  // reassigning is a better candidate for removal than one with few contacts — but
  // it still over-trims: it removes about 700 residues SCOPe keeps in order to
  // catch 400 it drops. The reason both fail is that the residues a parse is
  // unsure about sit at domain boundaries as much as at chain ends, and on a
  // single-domain chain the sparse graph at the termini looks the same as a tail
  // that genuinely does not belong. See test/trim.mjs.
  trim: false,
  trimStable: 0.999,  // a terminal residue must be this settled to be kept
  trimDegree: 4,        // long-range contacts needed to count as packed in
  trimSep: 3,           // |i - j| above this counts as long-range
  trimKeep: 0.45,       // never peel a domain below this fraction of its size
  // Accept a split when the interface weight is below this fraction of the
  // smaller side's total incident weight — "how much of the smaller piece's
  // contact budget is spent on the other piece". PDP's size normalisation
  // (dividing by size^2/3 on the theory that interface is surface area) was
  // tried and measured no better than this plain ratio on SCOPe40, so the
  // simpler form is kept. Fitted in test/domains.mjs.
  // Normalised cut a split must come in under. On the ncut scale, re-fitted on the
  // 260-chain benchmark after acceptance moved off cut/min(vol): NDO 0.901 and the
  // right domain count on 57.7% of chains, against 0.901/55.4% at 0.07 and
  // 0.904/57.3% at 0.09 — within noise of each other on NDO, best on count here.
  maxCut: 0.08,
  scaleCut: 0,          // 1 scales the ceiling as n^(-1/3); 0 is a flat threshold
  // Compactness penalty, off. It is the term every method in the literature carries
  // and this one lacks, it does what it promises, and it is still not worth turning
  // on here. Over-splits fall monotonically with the weight — 56, 50, 39, 32 across
  // w = 0, 1, 2, 3 on the 260-chain benchmark — but under-splits rise in lockstep,
  // 52, 59, 73, 91, so the domain count never improves. Held out on half the
  // chains the NDO gain is real and transfers (0.8994 -> 0.9027 at w = 1) while the
  // count drops (56.2% -> 54.6%), and count is the metric with consequences: an
  // under-split hands the search a chimera, where a boundary ten residues off still
  // retrieves the right fold. Searching whole chains recovers 2 of 59 true domains
  // against 30 for parsed ones — that gap is about splitting, not precision.
  //
  // Third knob to trace the same frontier, after maxCut and scaleCut. Over- and
  // under-split failures fall on different chains, so any single scalar criterion
  // only slides the balance point along it.
  // How the domain count is decided. 'eigengap' reads it off the Laplacian
  // spectrum instead of thresholding a cut score — the one candidate that names a
  // count outright rather than sliding the over/under-split balance — and it is
  // markedly worse: count 40.0% against 58.5%, NDO 0.867 against 0.903, and 131
  // over-splits against 25 under. On two-domain chains it answers 3, 4, 5 or 7 more
  // often than 2.
  //
  // The rule assumes clusters that are nearly disconnected, so the first k
  // eigenvalues sit near zero with a visible jump after. Protein domains are not
  // like that: compact lumps sharing a real interface AND a covalent backbone that
  // runs straight through it, which leaves the spectrum smooth. With no gap to
  // find, 'the largest jump' is noise, and among seven eigenvalues noise usually
  // points above 2. Also 4x slower, needing seven power iterations rather than one.
  // Veto a split past the first when this many beta pairings cross it. Off, and the
  // reason is a lesson about the metrics rather than about the idea.
  //
  // The signal is the cleanest of anything tried here: 47% of spurious deeper
  // boundaries have two or more strand pairings across them against 21% of true
  // ones, and restricted to splits beyond the first — where over-splitting actually
  // happens — it delivers the only exact-count gain of the whole effort, 58.5% to
  // 64.2%, collapsing over-splits from 56 to 13, slightly faster too.
  //
  // Then measured by retrieval instead: whether each true SCOPe domain still comes
  // back at rank 1 from the library, its own PDB excluded. 47.8% to 42.6%, over 136
  // domains in 60 chains. It gains 5.7 points of count and loses 5.1 points of the
  // thing the parse is for, because it emits 114 domains where 130 were emitted
  // before — sixteen real domains stop being searched at all.
  //
  // NDO and exact count weight over- and under-splitting equally; searching does
  // not. An under-split withholds a domain entirely, an over-split still hands the
  // search a fragment of the right fold. Anything rejected here on count alone —
  // rgWeight in particular, which traded the same way — is worth re-checking on
  // retrieval before being believed.
  betaVeto: 0,
  countBy: 'threshold',
  eigenWant: 7,          // eigenvalues past the trivial one, for the gap
  // Compactness. Off, and now tested twice under opposite conditions.
  //
  // Pre-merge it gained NDO and lost exact count, so it was rejected on count. That
  // looked like a judgement in the wrong currency once the search-guided merge existed
  // — the merge repairs count, so a term that trades count for placement should have
  // become free. Screened again on that pipeline it peaked cleanly at rgWeight 1:
  // 84.0% count / NDO 0.9492 against 80.0% / 0.9437 with it off, an interior maximum
  // with a monotonic decline either side (rgWeight 5 collapses to 64% as the penalty
  // stops the over-parse splitting at all).
  //
  // On the held-out half it did not survive: 73.8% / 0.9401 against 74.8% / 0.9379 with
  // it off. Count down one chain, NDO up 0.0022, fragments 2.71 -> 2.50 and domains
  // 2.02 -> 1.94 — i.e. it pushes toward under-splitting, which is the direction that
  // withholds a domain from the search entirely.
  //
  // NDO has now risen with it in both halves while count fell, so the placement effect
  // is probably real and just too small to pay for. 50-chain screening cannot resolve
  // a two-chain difference; nothing under about +8 points there should be believed
  // without a held-out run.
  rgWeight: 0,
  // Runs to vote over, and the cost is linear in it: 45 ms a run on a 360-residue
  // chain, 70 ms on 524. Five measured on the 260-chain benchmark at NDO 0.903 —
  // the same as nine, and above 0.901 for a single pass — while closing the gap
  // between identical chains of 3MKU from 66 residues to 12. Nine closed it to 0
  // but cost 800 ms on a long chain, which is too much to pay on every load for a
  // dozen residues of boundary. Odd, so a majority vote cannot tie.
  //
  // Voting does not fix everything: 1AON's GroEL subunits agree to 0.07 Å and
  // still part company by 153 residues at any run count, which is why
  // `confidence` comes back alongside the answer instead of being smoothed over.
  consensus: 5,
  jitter: 0.4,       // Å of Gaussian noise per run
  cutJitter: 0.4,    // Å the contact cut-off moves per run
  coassocBins: 260,  // resolution of the consensus matrix handed to the UI
  maxDomains: 8,
  powerIters: 400,
  powerTol: 1e-9,
};

/** Weighted contact graph as CSR. Backbone edges are always present. */
/**
 * Contact graph with a tapered cut-off.
 *
 * A hard step at contactDist makes every edge weight a discontinuous function of
 * the coordinates, and the place that hurts is exactly the place that decides the
 * answer: a domain interface is by definition where contacts are sparsest, so a
 * handful of pairs crossing the threshold moves the normalised cut by tens of per
 * cent. 3MKU is the case that showed it — chains A and B are the same protein in
 * the same conformation, agreeing to 0.078 Å on every consecutive Cα, and one
 * split into two domains while the other did not, because their cut values landed
 * either side of 0.05.
 *
 * Tapering the weight from 1 to 0 with a raised cosine over contactSoft either
 * side makes the cut continuous in the coordinates. The same 0.078 Å of noise now
 * moves a weight by about 4% of one contact rather than by a whole one.
 */
/**
 * Virtual Cβ positions and the per-pair cut-off they are judged against.
 *
 * Taken from the lab's contact model (solab assets/js/contact.js), fitted to
 * ConFind over 151 native domains: a pseudo-side-chain 3.0 Å out for a helix
 * residue, 4.0 for a strand, 3.5 for a loop, along the direction the backbone
 * bends away from, with a contact at 8.0 Å between two helix residues and 8.5 Å
 * otherwise. Against ConFind that reaches F1 0.78 where a raw Cα cut-off manages
 * about 0.3, because it drops the pairs whose backbones pass close by while their
 * side chains point in opposite directions.
 */
/**
 * A virtual Cβ, one offset for every residue.
 *
 * From the lab's contact model (solab assets/js/contact.js): put a pseudo
 * side-chain out along the direction the backbone bends away from, and judge
 * contacts between those rather than between Cα. It drops the pairs whose
 * backbones pass close by while their side chains point in opposite directions,
 * which a raw Cα cut-off counts as touching.
 *
 * The original varies the offset and the cut-off by secondary structure, five
 * numbers fitted to ConFind. Here it is two, shared by every residue: this is
 * being asked a coarser question — is the interface between these two halves
 * cheap — and five parameters on 260 benchmark chains is more freedom than the
 * evidence supports. It also means no secondary-structure assignment is needed,
 * so this module stays independent of the aligner.
 */
function virtualCB(coords, n, off) {
  const cb = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    let vx = 0;
    let vy = 0;
    let vz = 0;
    if (i > 0) {
      vx += coords[i * 3] - coords[(i - 1) * 3];
      vy += coords[i * 3 + 1] - coords[(i - 1) * 3 + 1];
      vz += coords[i * 3 + 2] - coords[(i - 1) * 3 + 2];
    }
    if (i < n - 1) {
      vx += coords[i * 3] - coords[(i + 1) * 3];
      vy += coords[i * 3 + 1] - coords[(i + 1) * 3 + 1];
      vz += coords[i * 3 + 2] - coords[(i + 1) * 3 + 2];
    }
    let m = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (m < 1e-6) { vx = 0; vy = 0; vz = 1; m = 1; }
    cb[i * 3] = coords[i * 3] + (vx / m) * off;
    cb[i * 3 + 1] = coords[i * 3 + 1] + (vy / m) * off;
    cb[i * 3 + 2] = coords[i * 3 + 2] + (vz / m) * off;
  }
  return cb;
}

/**
 * LEARNED EDGE WEIGHTS, optional. opt.nodeVecs is CIRPIN's per-residue node_dec array (n x 128, from
 * embedGraph's nodeSink); when present, each contact weight is scaled by the cosine similarity of its
 * two residues' node vectors, raised to opt.simPower.
 *
 * Why this is not more of the same: every criterion tried on this graph so far read only geometry, and
 * geometry weights all pairs inside the cut-off almost identically. The node vectors do not -- measured
 * over 24 multi-domain chains, CONTACTING pairs within a domain have mean cosine 0.372 against 0.218
 * for contacting pairs across a boundary. That 0.154 gap is information the contact graph cannot hold,
 * produced by a network trained on real domains, and it arrives at exactly the place a normalised cut
 * needs it: the edge weight.
 *
 * Distinct from the retrieval merge that was removed. That asked a library whether two pieces looked
 * like one domain, using an uncalibrated similarity as a yes/no. This uses no library and no threshold
 * -- only the internal representation's own geometry, as a weight.
 */
function buildGraph(coords, n, opt) {
  const soft = opt.contactSoft ?? 0;
  // nodeVecs may be one array or several, one per jittered copy of the structure. With several, the
  // per-pair cosine is AVERAGED before being turned into a weight -- CIRPIN was trained with 1.0 A
  // coordinate noise, so jittered inputs are in distribution, and the learned split's failure mode is
  // spread in within-domain similarity, which is exactly what averaging shrinks if it is noise.
  const nvList = opt.nodeVecs ? (Array.isArray(opt.nodeVecs) ? opt.nodeVecs : [opt.nodeVecs]) : null;
  const nv = nvList;
  const simPow = opt.simPower ?? 6;
  // opt.simMatrix short-circuits the cosine: an n x n array of ready-made positive multipliers, so an
  // experiment can substitute any pairwise transform (centred, observed/expected, profile correlation)
  // without this function knowing which. The parser is identical in every case; only the weight differs.
  const simMat = opt.simMatrix ?? null;
  const simOf = simMat ? ((i, j) => simMat[i * n + j]) : (i, j) => {
    let acc = 0;
    for (const v of nvList) {
      let d = 0; let a = 0; let b = 0;
      for (let k = 0; k < 128; k++) {
        const x = v[i * 128 + k]; const y = v[j * 128 + k];
        d += x * y; a += x * x; b += y * y;
      }
      acc += d / (Math.sqrt(a * b) || 1);
    }
    const c = acc / nvList.length;
    // cosine into a positive multiplier; simPower sharpens the within/across contrast
    return ((1 + c) / 2) ** simPow;
  };
  const cb = opt.contactModel === 'cb' ? virtualCB(coords, n, opt.cbOffset) : null;
  const pos = cb || coords;
  const heads = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    const xi = pos[i * 3]; const yi = pos[i * 3 + 1]; const zi = pos[i * 3 + 2];
    for (let j = i + 1; j < n; j++) {
      let w = 0;
      if (j === i + 1) {
        w = opt.backboneWeight;
      } else if (!cb || j >= i + 3) {
        // the reference model ignores pairs closer than three in sequence, since a
        // virtual Cβ two apart is always within reach and says nothing about packing
        const base = cb ? opt.cbCut : opt.contactDist;
        const lo = base - soft;
        const hi = base + soft;
        const dx = xi - pos[j * 3];
        const dy = yi - pos[j * 3 + 1];
        const dz = zi - pos[j * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= lo * lo) w = 1;
        else if (d2 < hi * hi) {
          w = 0.5 * (1 + Math.cos(Math.PI * (Math.sqrt(d2) - lo) / (hi - lo)));
        }
      }
      if (w > 0 && (nv || simMat) && j !== i + 1) w *= simOf(i, j);   // backbone edges keep their fixed weight
      if (w > 0) { heads[i].push(j, w); heads[j].push(i, w); }
    }
  }
  const off = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) off[i + 1] = off[i] + heads[i].length / 2;
  const idx = new Int32Array(off[n]);
  const wt = new Float64Array(off[n]);
  for (let i = 0, p = 0; i < n; i++) {
    for (let k = 0; k < heads[i].length; k += 2) { idx[p] = heads[i][k]; wt[p] = heads[i][k + 1]; p++; }
  }
  return { n, off, idx, wt };
}

/**
 * One contact graph averaged over jittered copies of the structure.
 *
 * WHY AVERAGE THE GRAPH RATHER THAN THE ANSWERS. parseDomains' consensus runs the whole parse five
 * times on jittered coordinates and votes on the resulting labelings -- so each run applies its own
 * threshold to its own noisy graph, and the noise reaches the accept decision before anything is
 * averaged. Near the threshold that is exactly where it does damage: a cut at 0.0799 and one at
 * 0.0801 are the same cut, and voting on the outcome turns a coin flip into a domain count.
 *
 * Averaging here instead means the edge weights are expected contact strength under positional
 * uncertainty, the cut scores computed from them are smooth in the coordinates, and the threshold is
 * applied ONCE to stable numbers. It is also cheaper for a sweep: K graph builds up front and then a
 * single memoised set of splits, against K complete parses per rung.
 *
 * The weights are already continuous -- contactSoft tapers them with a raised cosine -- so this is a
 * refinement of the same idea rather than a new one. Edges are the union over jitters, with an edge
 * absent from a copy contributing zero, which is what makes a marginal contact fade rather than
 * flicker.
 */
function averagedGraph(flat, n, opt, k) {
  const acc = new Map();                 // i*n + j (i < j) -> summed weight
  for (let t = 0; t < k; t++) {
    const pts = t === 0 ? flat : jitterCoords(flat, opt.jitter, 0x51ed + t * 2654435761);
    const spread = t === 0 ? 0 : ((t % 3) - 1) * opt.cutJitter;
    const g = buildGraph(pts, n, { ...opt, contactDist: opt.contactDist + spread });
    for (let i = 0; i < n; i++) {
      for (let p = g.off[i]; p < g.off[i + 1]; p++) {
        const j = g.idx[p];
        if (j <= i) continue;
        const key = i * n + j;
        acc.set(key, (acc.get(key) || 0) + g.wt[p]);
      }
    }
  }
  const heads = Array.from({ length: n }, () => []);
  for (const [key, sum] of acc) {
    const i = Math.floor(key / n);
    const j = key % n;
    const w = sum / k;
    heads[i].push(j, w);
    heads[j].push(i, w);
  }
  const off = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) off[i + 1] = off[i] + heads[i].length / 2;
  const idx = new Int32Array(off[n]);
  const wt = new Float64Array(off[n]);
  for (let i = 0, p = 0; i < n; i++) {
    for (let q = 0; q < heads[i].length; q += 2) { idx[p] = heads[i][q]; wt[p] = heads[i][q + 1]; p++; }
  }
  return { n, off, idx, wt };
}

/**
 * Normalised cut of a GIVEN bipartition on a given graph.
 *
 * bestSplit finds a cut and scores it on the same graph. This scores a cut someone else chose, which
 * is what lets one split be evaluated against several jittered graphs -- the difference between
 * "average the answers" and "average the evidence for one answer".
 */
function ncutOn(g, A, B) {
  const side = new Map();
  for (const v of A) side.set(v, 0);
  for (const v of B) side.set(v, 1);
  let cut = 0;
  let volA = 0;
  let volB = 0;
  for (const [v, sv] of side) {
    for (let p = g.off[v]; p < g.off[v + 1]; p++) {
      const u = g.idx[p];
      const su = side.get(u);
      if (su === undefined) continue;      // outside this piece
      if (sv === 0) volA += g.wt[p]; else volB += g.wt[p];
      if (su !== sv) cut += g.wt[p];
    }
  }
  cut /= 2;
  if (volA <= 0 || volB <= 0) return Infinity;
  return cut / volA + cut / volB;
}

/**
 * Fiedler vector of the normalised Laplacian, restricted to `nodes`.
 *
 * Power-iterates on ((D^-1/2 W D^-1/2) + I) / 2, whose spectrum is shifted
 * into [0, 1] so the dominant eigenvalue is unambiguous. The top eigenvector
 * is known analytically (D^1/2 1), so it is deflated out at every step and
 * what remains converges to the Fiedler direction.
 */
function fiedler(g, nodes, opt) {
  const m = nodes.length;
  const local = new Map();
  nodes.forEach((v, i) => local.set(v, i));

  const deg = new Float64Array(m);
  for (let a = 0; a < m; a++) {
    const v = nodes[a];
    for (let p = g.off[v]; p < g.off[v + 1]; p++) {
      if (local.has(g.idx[p])) deg[a] += g.wt[p];
    }
  }
  const dsq = new Float64Array(m);
  const dinv = new Float64Array(m);
  let norm0 = 0;
  for (let a = 0; a < m; a++) {
    dsq[a] = Math.sqrt(deg[a]);
    dinv[a] = deg[a] > 0 ? 1 / dsq[a] : 0;
    norm0 += deg[a];
  }
  norm0 = Math.sqrt(norm0);
  if (norm0 === 0) return null;
  const v0 = new Float64Array(m);
  for (let a = 0; a < m; a++) v0[a] = dsq[a] / norm0;

  // deterministic, non-degenerate start
  let x = new Float64Array(m);
  for (let a = 0; a < m; a++) x[a] = Math.sin((a + 1) * 0.7391) + 0.1;
  const project = (vec) => {
    let d = 0;
    for (let a = 0; a < m; a++) d += vec[a] * v0[a];
    for (let a = 0; a < m; a++) vec[a] -= d * v0[a];
    let s = 0;
    for (let a = 0; a < m; a++) s += vec[a] * vec[a];
    s = Math.sqrt(s);
    if (s < 1e-300) return false;
    for (let a = 0; a < m; a++) vec[a] /= s;
    return true;
  };
  if (!project(x)) return null;

  let y = new Float64Array(m);
  for (let it = 0; it < opt.powerIters; it++) {
    // y = ((D^-1/2 W D^-1/2) x + x) / 2
    y.fill(0);
    for (let a = 0; a < m; a++) {
      const v = nodes[a];
      let s = 0;
      for (let p = g.off[v]; p < g.off[v + 1]; p++) {
        const b = local.get(g.idx[p]);
        if (b !== undefined) s += g.wt[p] * dinv[b] * x[b];
      }
      y[a] = (dinv[a] * s + x[a]) / 2;
    }
    if (!project(y)) return null;
    let diff = 0;
    for (let a = 0; a < m; a++) diff += Math.abs(y[a] - x[a]);
    const t = x; x = y; y = t;
    if (diff / m < opt.powerTol) break;
  }
  return x;
}

/**
 * Sweep one ordering of `nodes`, returning the prefix split with the lowest
 * normalised cut. Moving one node at a time and updating the interface
 * incrementally makes this O(edges) per ordering.
 */
/**
 * How much less compact a set of residues is than a real domain of its size.
 *
 * Globular proteins follow Rg ~= 2.2 N^0.38 Å closely, so the ratio of measured to
 * expected radius of gyration is a size-free measure of straggliness: about 1 for a
 * compact unit, well above for a piece that is merely cheap to cut off. 1 is the
 * floor — being tighter than expected is not a fault.
 *
 * This is the term every method in the literature carries and this one did not.
 * Normalised cut alone asks only whether an interface is thin; two halves joined by
 * a narrow waist score well even when one of them is a shell wrapped round the
 * other, which is exactly the shape a spurious split produces.
 */
function rgPenalty(sumX, sumY, sumZ, sumSq, count) {
  if (count < 2) return 1;
  const mean = (sumX * sumX + sumY * sumY + sumZ * sumZ) / (count * count);
  const rg = Math.sqrt(Math.max(0, sumSq / count - mean));
  const expected = 2.2 * (count ** 0.38);
  return Math.max(1, rg / expected);
}

function sweep(g, coords, nodes, order, degOf, volTotal, localOf, opt) {
  const m = nodes.length;
  const side = new Int8Array(m);
  let volA = 0;
  let cut = 0;
  let best = null;

  // Running moments so each candidate's radius of gyration costs O(1), not O(n).
  let ax = 0; let ay = 0; let az = 0; let aq = 0;
  let tx = 0; let ty = 0; let tz = 0; let tq = 0;
  for (const v of nodes) {
    const x = coords[v * 3]; const y = coords[v * 3 + 1]; const z = coords[v * 3 + 2];
    tx += x; ty += y; tz += z; tq += x * x + y * y + z * z;
  }

  for (let k = 0; k < m - 1; k++) {
    const a = order[k];
    const v = nodes[a];
    side[a] = 1;
    volA += degOf[a];
    const x = coords[v * 3]; const y = coords[v * 3 + 1]; const z = coords[v * 3 + 2];
    ax += x; ay += y; az += z; aq += x * x + y * y + z * z;
    for (let p = g.off[v]; p < g.off[v + 1]; p++) {
      const b = localOf.get(g.idx[p]);
      if (b === undefined) continue;
      // the edge leaves the interface if its other end is already in A
      cut += side[b] === 1 ? -g.wt[p] : g.wt[p];
    }
    const sizeA = k + 1;
    const sizeB = m - sizeA;
    if (sizeA < opt.minDomain || sizeB < opt.minDomain) continue;
    const volB = volTotal - volA;
    if (volA <= 0 || volB <= 0) continue;
    const ncut = cut / volA + cut / volB;
    let score = ncut;
    if (opt.rgWeight) {
      const pa = rgPenalty(ax, ay, az, aq, sizeA);
      const pb = rgPenalty(tx - ax, ty - ay, tz - az, tq - aq, sizeB);
      score = ncut * ((pa * pb) ** opt.rgWeight);
    }
    if (!best || score < best.score) {
      best = { score, ncut, cut, k, quality: cut / Math.min(volA, volB) };
    }
  }
  return best;
}

/**
 * Best bisection of `nodes`, taking the better of two candidate orderings.
 *
 * Measured separately on SCOPe40, the two have complementary failures. A
 * contiguous scan in chain order — DOMAK's 1995 move — places two-domain
 * boundaries within 20 residues 81% of the time but structurally cannot
 * produce a discontinuous domain. The Fiedler ordering finds discontinuous
 * splits and, when it does return two contiguous domains, its boundary is
 * within a median of 1 residue; it is the *choice* of when to split that it
 * gets wrong. Scoring both with the same normalised cut and keeping the
 * better one gets the contiguous accuracy without giving up discontinuity.
 */
/**
 * The smallest few non-trivial eigenvalues of the normalised Laplacian.
 *
 * Same shifted power iteration the Fiedler vector uses — the spectrum of
 * (D^-1/2 W D^-1/2 + I)/2 is ordered opposite to the Laplacian's, so its largest
 * eigenvalues are the Laplacian's smallest — extended by deflating every vector
 * found so far, not just the analytic one. Eigenvalues come back via the Rayleigh
 * quotient: a shifted value mu corresponds to lambda = 2(1 - mu).
 *
 * Wanted for the eigengap rule: with the Laplacian's eigenvalues in ascending
 * order, the number of clusters is conventionally read off the largest jump. The
 * appeal is that it names a count outright instead of thresholding a score, so it
 * is the one candidate that does not merely slide the over/under-split balance.
 */
function eigenSpectrum(g, nodes, opt, want) {
  const m = nodes.length;
  const local = new Map();
  nodes.forEach((v, i) => local.set(v, i));
  const deg = new Float64Array(m);
  for (let a = 0; a < m; a++) {
    const v = nodes[a];
    for (let p = g.off[v]; p < g.off[v + 1]; p++) if (local.has(g.idx[p])) deg[a] += g.wt[p];
  }
  const dsq = new Float64Array(m);
  const dinv = new Float64Array(m);
  let norm0 = 0;
  for (let a = 0; a < m; a++) {
    dsq[a] = Math.sqrt(deg[a]);
    dinv[a] = deg[a] > 0 ? 1 / dsq[a] : 0;
    norm0 += deg[a];
  }
  norm0 = Math.sqrt(norm0);
  if (norm0 === 0) return [];
  const basis = [new Float64Array(m)];
  for (let a = 0; a < m; a++) basis[0][a] = dsq[a] / norm0;

  const apply = (x, y) => {
    y.fill(0);
    for (let a = 0; a < m; a++) {
      const v = nodes[a];
      let acc = 0;
      for (let p = g.off[v]; p < g.off[v + 1]; p++) {
        const b = local.get(g.idx[p]);
        if (b !== undefined) acc += g.wt[p] * dinv[b] * x[b];
      }
      y[a] = (dinv[a] * acc + x[a]) / 2;
    }
  };
  const orthonormalise = (vec) => {
    for (const u of basis) {
      let d = 0;
      for (let a = 0; a < m; a++) d += vec[a] * u[a];
      for (let a = 0; a < m; a++) vec[a] -= d * u[a];
    }
    let sq = 0;
    for (let a = 0; a < m; a++) sq += vec[a] * vec[a];
    sq = Math.sqrt(sq);
    if (sq < 1e-300) return false;
    for (let a = 0; a < m; a++) vec[a] /= sq;
    return true;
  };

  const values = [];
  for (let want_i = 0; want_i < want; want_i++) {
    let x = new Float64Array(m);
    for (let a = 0; a < m; a++) x[a] = Math.sin((a + 1) * (0.7391 + 0.131 * want_i)) + 0.1;
    if (!orthonormalise(x)) break;
    let y = new Float64Array(m);
    for (let it = 0; it < opt.powerIters; it++) {
      apply(x, y);
      if (!orthonormalise(y)) return values;
      let diff = 0;
      for (let a = 0; a < m; a++) diff += Math.abs(y[a] - x[a]);
      const t = x; x = y; y = t;
      if (diff / m < opt.powerTol) break;
    }
    apply(x, y);
    let mu = 0;
    for (let a = 0; a < m; a++) mu += x[a] * y[a];
    values.push(2 * (1 - mu));            // back to the Laplacian's scale
    basis.push(Float64Array.from(x));
  }
  return values;
}

/** How many domains the spectrum suggests: the largest jump in the eigenvalues. */
function eigengapCount(g, nodes, opt) {
  const vals = eigenSpectrum(g, nodes, opt, opt.eigenWant);
  if (vals.length < 2) return 1;
  const lam = [0, ...vals];             // lambda_1 is exactly zero
  let bestK = 1;
  let bestGap = -Infinity;
  const top = Math.min(lam.length - 1, opt.maxDomains);
  for (let k = 1; k <= top; k++) {
    const gap = lam[k] - lam[k - 1];
    if (gap > bestGap) { bestGap = gap; bestK = k; }
  }
  return bestK;
}

/**
 * Residue pairs that look like a beta-sheet pairing, from Ca alone.
 *
 * Two Ca within 5.5 Å but at least five apart in sequence: at that separation a
 * helix cannot do it (i to i+3 is about 5.0 Å, i to i+4 about 6.2), so what remains
 * is overwhelmingly strand pairing. Deliberately not using a secondary-structure
 * assignment, which would make this module depend on the aligner for a distinction
 * the distance already draws.
 *
 * The point is a veto rather than a score. A sheet spanning a proposed boundary
 * means the two halves are hydrogen-bonded into one piece of structure, which is
 * evidence of a kind that a normalised cut cannot express: the interface may be
 * thin in contact terms and still be a continuous sheet.
 */
const BETA_MAX = 5.5;
const BETA_SEQ = 5;
function betaPairs(coords, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + BETA_SEQ; j < n; j++) {
      const dx = coords[i * 3] - coords[j * 3];
      const dy = coords[i * 3 + 1] - coords[j * 3 + 1];
      const dz = coords[i * 3 + 2] - coords[j * 3 + 2];
      if (dx * dx + dy * dy + dz * dz <= BETA_MAX * BETA_MAX) out.push(i, j);
    }
  }
  return out;
}

/** How many of those pairings a proposed split would break. */
function betaCrossing(pairs, A, B) {
  const inA = new Set(A);
  const inB = new Set(B);
  let crossing = 0;
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1];
    if ((inA.has(i) && inB.has(j)) || (inB.has(i) && inA.has(j))) crossing++;
  }
  return crossing;
}

function bestSplit(g, coords, nodes, opt) {
  const m = nodes.length;
  if (m < 2 * opt.minDomain) return null;

  const localOf = new Map();
  nodes.forEach((v, i) => localOf.set(v, i));
  const degOf = new Float64Array(m);
  let volTotal = 0;
  for (let a = 0; a < m; a++) {
    const v = nodes[a];
    for (let p = g.off[v]; p < g.off[v + 1]; p++) {
      if (localOf.has(g.idx[p])) degOf[a] += g.wt[p];
    }
    volTotal += degOf[a];
  }

  const candidates = [];

  // chain order: contiguous cuts only
  const chainOrder = Array.from({ length: m }, (_, a) => a);
  const c1 = sweep(g, coords, nodes, chainOrder, degOf, volTotal, localOf, opt);
  if (c1) candidates.push({ order: chainOrder, best: c1 });

  // Fiedler order: connectivity only, contiguity not required
  const f = fiedler(g, nodes, opt);
  if (f) {
    const fo = Array.from({ length: m }, (_, a) => a).sort((a, b) => f[a] - f[b]);
    const c2 = sweep(g, coords, nodes, fo, degOf, volTotal, localOf, opt);
    if (c2) candidates.push({ order: fo, best: c2 });
  }
  if (!candidates.length) return null;

  candidates.sort((a, b) => a.best.ncut - b.best.ncut);
  const { order, best } = candidates[0];
  const A = []; const B = [];
  for (let k = 0; k < m; k++) (k <= best.k ? A : B).push(nodes[order[k]]);
  A.sort((a, b) => a - b);
  B.sort((a, b) => a - b);
  return { A, B, quality: best.quality, ncut: best.ncut, score: best.score };
}

/**
 * Absorb short runs into a neighbouring domain.
 *
 * The Fiedler sweep places the boundary almost exactly — median error against
 * SCOPe is 1 residue — but it assigns interface residues one at a time by
 * eigenvector value, which shatters domains into many short runs. Real
 * discontinuous domains have long segments, so a minimum run length recovers
 * the intended partition without forbidding discontinuity. Each short run goes
 * to whichever other domain it shares the most contact weight with, which is
 * the same "belongs with what it touches" rule used for the split itself.
 */
/**
 * The largest normalised cut worth accepting, for a piece of this size.
 *
 * A flat threshold cannot be right across a fivefold range of chain length. For
 * two compact halves the interface grows as the two-thirds power of size while the
 * volume grows linearly, so a genuine boundary scores about n^(-1/3) — a real
 * interface in a 125-residue chain is intrinsically more expensive than the same
 * interface in a 600-residue one. Judging both against 0.08 rejects the small
 * ones: 1AQTA and 1A04A were turned down at 0.0837 and 0.0835, within 5% of the
 * line, while longer chains passed at the same score.
 *
 * maxCut keeps its meaning at REF residues, so the number that was fitted stays
 * interpretable; scaleCut at 0 recovers the old flat behaviour.
 */
const CUT_REF = 250;
function cutCeiling(size, opt) {
  if (!opt.scaleCut) return opt.maxCut;
  return opt.maxCut * ((CUT_REF / Math.max(1, size)) ** (opt.scaleCut / 3));
}

function absorbShortSegments(labels, n, nParts, g, opt) {
  let guard = 0;
  for (;;) {
    // shortest run below the floor
    // How much of each domain is present, so absorption cannot delete one whole.
    // minSegment (50) is deliberately larger than minDomain (40) to clean up
    // fragments inside a decomposition, but for a domain made of ONE segment the
    // two describe the same object, and the larger floor silently won: a 49-residue
    // domain was created, then absorbed away. 1A62A failed exactly there — a split
    // at ncut 0.077, comfortably inside the threshold, undone after the fact.
    const partLen = new Int32Array(nParts);
    for (let i = 0; i < n; i++) if (labels[i] >= 0) partLen[labels[i]]++;

    let bestStart = -1; let bestLen = Infinity; let bestEnd = -1;
    let s = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || labels[i] !== labels[s]) {
        const len = i - s;
        // a run that is its domain's only substance is judged against minDomain
        const floor = len === partLen[labels[s]] ? opt.minDomain : opt.minSegment;
        if (len < floor && len < bestLen) { bestLen = len; bestStart = s; bestEnd = i - 1; }
        s = i;
      }
    }
    if (bestStart < 0) return;

    const own = labels[bestStart];
    const weight = new Float64Array(nParts);
    for (let r = bestStart; r <= bestEnd; r++) {
      for (let p = g.off[r]; p < g.off[r + 1]; p++) {
        const j = g.idx[p];
        if (j >= bestStart && j <= bestEnd) continue; // inside the run
        weight[labels[j]] += g.wt[p];
      }
    }
    let target = -1; let bestW = 0;
    for (let d = 0; d < nParts; d++) {
      if (d === own) continue;
      if (weight[d] > bestW) { bestW = weight[d]; target = d; }
    }
    // bestW must be strictly positive: a run that touches no other domain has
    // nowhere to go, and relabelling it anyway would ping-pong forever once
    // only one domain is left.
    if (target < 0) return;
    for (let r = bestStart; r <= bestEnd; r++) labels[r] = target;
    if (++guard > n) return;
  }
}

/**
 * Peel loosely packed residues off the ends of each domain.
 *
 * Uses long-range contacts only: a residue in a flexible tail still touches its
 * sequence neighbours, so counting those would make every tail look packed. The
 * two ends are peeled alternately and the loop stops as soon as both are held
 * in by trimDegree contacts, so it removes a tail without eating into the core.
 */
function trimTerminals(labels, n, nParts, g, opt) {
  const inDomain = (r, d) => labels[r] === d;

  for (let d = 0; d < nParts; d++) {
    const members = [];
    for (let r = 0; r < n; r++) if (labels[r] === d) members.push(r);
    if (!members.length) continue;
    const floor = Math.max(opt.minDomain, Math.ceil(members.length * opt.trimKeep));

    // long-range contact count within this domain, kept current as we peel
    const deg = new Map();
    for (const r of members) {
      let k = 0;
      for (let p = g.off[r]; p < g.off[r + 1]; p++) {
        const j = g.idx[p];
        if (Math.abs(j - r) > opt.trimSep && inDomain(j, d)) k++;
      }
      deg.set(r, k);
    }

    let live = members.slice();
    while (live.length > floor) {
      const lo = live[0];
      const hi = live[live.length - 1];
      // take whichever end is looser, so a long tail on one side does not have
      // to wait for the other end to also fail
      const pick = (deg.get(lo) ?? 0) <= (deg.get(hi) ?? 0) ? lo : hi;
      if ((deg.get(pick) ?? 0) >= opt.trimDegree) break;
      labels[pick] = -1;
      deg.delete(pick);
      for (let p = g.off[pick]; p < g.off[pick + 1]; p++) {
        const j = g.idx[p];
        if (Math.abs(j - pick) > opt.trimSep && deg.has(j)) deg.set(j, deg.get(j) - 1);
      }
      live = pick === lo ? live.slice(1) : live.slice(0, -1);
    }
  }
}

function toSegments(sorted) {
  const segs = [];
  let s = sorted[0];
  let p = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] !== p + 1) { segs.push([s, p]); s = sorted[i]; }
    p = sorted[i];
  }
  segs.push([s, p]);
  return segs;
}

/**
 * Parse a Cα trace into domains.
 *
 * @param {Float64Array|number[][]} coords - flat stride 3, or [[x,y,z], ...]
 * @param {object} options - see DEFAULTS
 * @returns {{n: number, labels: Int32Array, domains: [{index, residues, segments, nres}]}}
 */
/**
 * Everything about a chain that does not depend on maxCut.
 *
 * maxCut is only an accept/stop filter -- it never reaches buildGraph or bestSplit, both of which are
 * pure functions of the coordinates and a residue subset. Sweeping the threshold therefore rebuilt
 * the O(n^2) contact graph once per rung and recomputed the same splits over and over: measured on a
 * 1,011-residue chain, a 13-rung sweep built 13 graphs and made 231 bestSplit calls for 11 DISTINCT
 * parts. Hoisting the graph and memoising the splits turns the whole sweep into roughly the cost of
 * one parse, which is what makes a live threshold slider possible rather than a recompute button.
 *
 * The memo key is the full residue membership, not a summary of it. A key like first+length collides
 * for discontinuous parts, and a collision here would silently hand back another part's split -- the
 * kind of bug that changes a parse without any symptom.
 */
function prepare(flat, n, opt) {
  const k = Math.max(1, opt.graphConsensus | 0);
  const sc = Math.max(1, opt.scoreConsensus | 0);
  // scoreConsensus keeps the jittered graphs so a split found on the unjittered one can be RESCORED
  // against each. That averages at the accept decision itself, which is a different place from
  // averaging the graph (before any split is found) or the labelings (after the threshold is applied).
  const jittered = [];
  if (sc > 1) {
    for (let t = 1; t < sc; t++) {
      const pts = jitterCoords(flat, opt.jitter, 0x7a19 + t * 40503);
      const spread = ((t % 3) - 1) * opt.cutJitter;
      jittered.push(buildGraph(pts, n, { ...opt, contactDist: opt.contactDist + spread }));
    }
  }
  return {
    flat,
    n,
    jittered,
    g: k > 1 ? averagedGraph(flat, n, opt, k) : buildGraph(flat, n, opt),
    bpairs: opt.betaVeto ? betaPairs(flat, n) : null,
    memo: new Map(),
  };
}

/** bestSplit, memoised on the exact part. */
function splitOf(prep, nodes, opt) {
  const key = nodes.join(',');
  if (prep.memo.has(key)) return prep.memo.get(key);
  const s = bestSplit(prep.g, prep.flat, nodes, opt);
  // Rescore the chosen cut on each jittered graph and average. A cut that only looks cheap because of
  // where the noise fell gets a worse average; a real interface keeps its score.
  if (s && prep.jittered.length) {
    let sum = s.ncut;
    let m = 1;
    for (const gj of prep.jittered) {
      const v = ncutOn(gj, s.A, s.B);
      if (Number.isFinite(v)) { sum += v; m++; }
    }
    s.ncut = sum / m;
    if (s.score !== undefined) s.score = s.ncut;
  }
  prep.memo.set(key, s);
  return s;
}

/**
 * The partition at one threshold: recursive bisection, splitting the cheapest acceptable interface
 * while both halves stay above the size floor.
 *
 * Split out of parseOnce so a threshold sweep can replay it against a shared cache. It is the same
 * code, not a copy -- parseOnce calls this too, because two implementations of the accept loop would
 * drift and the sweep would stop describing what the app actually produces.
 */
function partition(prep, opt) {
  const { g, flat, n, bpairs } = prep;
  const all = Array.from({ length: n }, (_, i) => i);
  let parts = [all];
  if (opt.countBy === 'eigengap') {
    // No threshold at all: the spectrum names a count, and the chain is bisected that many times,
    // always splitting whichever part has the cheapest cut.
    const target = eigengapCount(g, all, opt);
    while (parts.length < target) {
      let pick = null;
      for (let i = 0; i < parts.length; i++) {
        if (parts[i].length < 2 * opt.minDomain) continue;
        const sp = splitOf(prep, parts[i], opt);
        if (!sp) continue;
        const sc = sp.score ?? sp.ncut;
        if (!pick || sc < pick.sc) pick = { i, sp, sc };
      }
      if (!pick) break;
      parts = [...parts.slice(0, pick.i), pick.sp.A, pick.sp.B, ...parts.slice(pick.i + 1)];
    }
    return parts;
  }
  while (parts.length < opt.maxDomains) {
    let pick = null;
    for (let i = 0; i < parts.length; i++) {
      if (parts[i].length < 2 * opt.minDomain) continue;
      const s = splitOf(prep, parts[i], opt);
      // Only from the second split onward. Applied to the first as well it breaks
      // more correct two-domain chains than it saves — 21% of true boundaries have
      // two or more pairings across them — but 47% of the spurious deeper boundaries
      // do, and those are where over-splitting happens. Restricting it to splits
      // beyond the first is what makes it a gain rather than another trade.
      if (bpairs && parts.length > 1 && s
          && betaCrossing(bpairs, s.A, s.B) >= opt.betaVeto) continue;
      // Judge the split on the same quantity the sweep minimised. It used to
      // accept on cut/min(vol) while the sweep chose by normalised cut, so the
      // split that won the search was not the one the gate was measuring, and
      // near the threshold that mismatch decided the answer — see contactSoft.
      // The null check has to come FIRST. It used to sit on the next line, after `s.quality` had already
      // been read, so a part with no acceptable split threw instead of being skipped -- the guard on the
      // betaVeto line above shows the author knew `s` can be null. Reachable in shipped code whenever
      // bestSplit declines; found by feeding the graph unusual edge weights.
      if (!s) continue;
      const score = opt.acceptOn === 'quality' ? s.quality : (s.score ?? s.ncut);
      if (score > cutCeiling(parts[i].length, opt)) continue;
      if (!pick || score < pick.score) pick = { i, s, score };
    }
    if (!pick) break;
    parts = [...parts.slice(0, pick.i), pick.s.A, pick.s.B, ...parts.slice(pick.i + 1)];
  }
  return parts;
}

function parseOnce(coords, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  let flat;
  let n;
  if (ArrayBuffer.isView(coords)) { flat = coords; n = coords.length / 3; } else {
    n = coords.length;
    flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = coords[i][0]; flat[i * 3 + 1] = coords[i][1]; flat[i * 3 + 2] = coords[i][2];
    }
  }

  // WHY THE COUNT DOES NOT IMPROVE. Two analyses, over 258 benchmark chains, using
  // cutProfile to get the whole cut-score landscape rather than just its minimum.
  //
  // The two failure modes are different animals, and comparing their top-level
  // scores — which is what made this look like one badly-set threshold — hid it:
  //
  //                n    best cut   depth   basin width
  //   correct    152      0.052    3.29        0.027
  //   over        56      0.036    3.18        0.012
  //   under       50      0.101    1.92        0.047
  //
  // Under-splits are not a threshold mistake. Their best cut is expensive AND their
  // landscape is flat — depth 1.92 against 3.2 elsewhere, basins four times wider.
  // SCOPe's boundary is real but the Ca contact graph does not contain it, so no
  // criterion recovers them and raising the ceiling only ever buys over-splits.
  //
  // Over-splits do not fail at the top level at all: their best cut is the SHARPEST
  // of any group, 0.036, and correctly accepted. They fail on the second split.
  //
  // That suggested judging a split by how sharp a minimum it is in its own
  // landscape — scale-free, level-free, and biting only where the error is. The
  // signal is real and too weak: depth 3.23 for correct level-1 splits against 2.25
  // for spurious level-2 ones, but the distributions overlap so heavily that the
  // 25th percentile of the good ones equals the median of the bad ones. Requiring
  // depth >= 2.4 rejects 28 of 48 spurious splits and destroys 40 of 128 correct
  // ones. Net loss, at every threshold.
  //
  // Nine criteria have now been tried and every one either slides the over/under
  // balance or is dominated. The information needed to tell "two domains with an
  // intimate interface" from "one domain with a waist" does not appear to be in a
  // Ca contact graph, which is consistent with the field having moved to methods
  // that learn it from labelled domains (Merizo, Chainsaw). The honest use of this
  // parser's uncertainty is to report it — see `confidence` and the consensus
  // matrix — rather than to keep trying to resolve it.
  //
  // Where the remaining error lives, measured per chain on 25 two-domain cases:
  //
  //   the best available cut is within 20 residues of truth on 24 of 25 chains
  //
  // so placement is essentially solved and every failure is this accept decision.
  // Over the full 260, 56 chains over-split and 52 under-split — balanced, which is
  // why no threshold wins: moving maxCut trades one for the other, and every sweep
  // of it lands on a flat surface. Fixing this needs a different signal, not a
  // better-calibrated one — the eigengap choosing the count directly, or a
  // compactness term able to reject a cut that is cheap but leaves straggly halves,
  // which every method in the literature has and this one does not.
  const all = Array.from({ length: n }, (_, i) => i);
  if (n < 2 * opt.minDomain) {
    // too short to split, but still worth trimming: a single domain with a
    // flexible tail is exactly the case the models were not trained on
    const labels = new Int32Array(n);
    if (opt.trim) trimTerminals(labels, n, 1, buildGraph(flat, n, opt), opt);
    const kept = all.filter((r) => labels[r] >= 0);
    const res = kept.length >= opt.minDomain ? kept : all;
    return {
      n,
      labels,
      domains: [{ index: 0, residues: res, segments: toSegments(res), nres: res.length }],
    };
  }

  const prep = prepare(flat, n, opt);
  const { g } = prep;
  const parts = partition(prep, opt);

  return finish(prep, parts, opt);
}

/**
 * Turn a partition into the returned shape: clean it, then label it.
 *
 * Shared by parseOnce and the threshold ladder rather than copied, because a ladder that skipped
 * absorption or trimming would show the user a decomposition the app would never produce -- and the
 * whole point of the ladder is that the rung you pick is the parse you get.
 */
function finish(prep, partsIn, opt) {
  const { g, n } = prep;
  // Order domains by where they start, so indices read along the chain.
  const parts = partsIn.slice().sort((a, b) => a[0] - b[0]);
  const labels = new Int32Array(n);
  parts.forEach((residues, index) => { for (const r of residues) labels[r] = index; });

  absorbShortSegments(labels, n, parts.length, g, opt);
  if (opt.trim) trimTerminals(labels, n, parts.length, g, opt);

  // Rebuild from the cleaned labels. Absorption can empty a domain entirely, and
  // trimming marks residues -1, which belong to no domain at all.
  const byLabel = new Map();
  for (let r = 0; r < n; r++) {
    if (labels[r] < 0) continue;
    if (!byLabel.has(labels[r])) byLabel.set(labels[r], []);
    byLabel.get(labels[r]).push(r);
  }
  const kept = [...byLabel.entries()].sort((a, b) => a[1][0] - b[1][0]);
  const domains = kept.map(([, residues], index) => {
    for (const r of residues) labels[r] = index;
    return { index, residues, segments: toSegments(residues), nres: residues.length };
  });
  return { n, labels, domains };
}

// --- the threshold ladder ----------------------------------------------------

/**
 * Every decomposition the maxCut dial can reach, for the price of about one parse.
 *
 * maxCut is only an accept/stop filter, so the graph and the splits are shared across every rung: on
 * a 1,011-residue chain a 13-rung sweep used to build 13 graphs and call bestSplit 231 times for 11
 * distinct parts. Here it builds one graph and memoises the splits, so the sweep costs roughly what a
 * single parse costs and the UI can hand the whole ladder to the page and let a slider move through
 * it with no further computation.
 *
 * consensus is deliberately ignored -- one pass per rung. The sweep is itself a stability probe, and
 * five jittered graphs per rung would need five separate caches for a second one. Commit the chosen
 * rung through parseDomains to get the full consensus answer.
 */
export function buildLadder(coords, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  let flat;
  let n;
  if (ArrayBuffer.isView(coords)) { flat = coords; n = coords.length / 3; } else {
    n = coords.length;
    flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = coords[i][0]; flat[i * 3 + 1] = coords[i][1]; flat[i * 3 + 2] = coords[i][2];
    }
  }
  const lo = options.ladderLo ?? 0.02;
  // The full range, not the range that looked sensible. Capping at 0.14 hid every finer split, and
  // 22% of chains have two DIFFERENT decompositions with the same domain count -- a coarse three-way
  // split and a finer one that also happens to give three -- so a cap or a dedupe by count silently
  // drops a real alternative. The over-split levels are kept and marked instead: they come with
  // domains in three interleaved fragments, which is visible, whereas absence is not.
  const hi = options.ladderHi ?? 0.30;
  const step = options.ladderStep ?? 0.01;
  if (n < 2 * opt.minDomain) {
    // Too short to split at any threshold: one rung, one plateau. The plateaus array is always
    // present -- an early return that omitted it made every consumer guard for undefined.
    const one = parseOnce(flat, options);
    const segments = one.domains.map((d) => d.segments);
    const sig = segments.map((sg) => sg.map((x) => x.join('-')).join('+')).join('|');
    return {
      lo, hi, step, n,
      rungs: [{ maxCut: lo, count: one.domains.length, segments }],
      // segments included: every consumer reads them off the plateau, and omitting them here made
      // the short-chain case throw. Second time this early return has been missing a field the main
      // path has -- it exists precisely so callers need no special case, so it has to be complete.
      plateaus: [{ from: lo, to: hi, len: 1, count: one.domains.length, sig, segments }],
    };
  }
  const prep = prepare(flat, n, opt);
  const rungs = [];
  for (let i = 0; ; i++) {
    const mc = Math.round((lo + i * step) * 1000) / 1000;
    if (mc > hi + 1e-9) break;
    const parts = partition(prep, { ...opt, maxCut: mc });
    const r = finish(prep, parts, opt);
    rungs.push({ maxCut: mc, count: r.domains.length, segments: r.domains.map((d) => d.segments) });
  }
  // Plateaus keyed on the SPLIT, not the count: two rungs can both say three domains while cutting in
  // different places, and calling that a plateau would overstate how settled the answer is.
  const sigOf = (r) => r.segments.map((s) => s.map((x) => x.join('-')).join('+')).join('|');
  const plateaus = [];
  rungs.forEach((r) => {
    const prev = plateaus[plateaus.length - 1];
    const sig = sigOf(r);
    if (prev && prev.sig === sig) { prev.to = r.maxCut; prev.len++; } else {
      // The split travels with the plateau, so a caller never has to pair a plateau back to a rung.
      // A UI wants the distinct DECOMPOSITIONS -- exposing a raw maxCut slider gave twelve stops that
      // did nothing and one that mattered, which is a dial calibrated in the wrong unit.
      plateaus.push({ from: r.maxCut, to: r.maxCut, len: 1, count: r.count, sig, segments: r.segments });
    }
  });
  return { lo, hi, step, n, rungs, plateaus };
}

/** Extract one domain's coordinates, in chain order. */
export function domainCoords(coords, residues) {
  const flat = ArrayBuffer.isView(coords);
  const out = new Float64Array(residues.length * 3);
  residues.forEach((r, i) => {
    if (flat) {
      out[i * 3] = coords[r * 3]; out[i * 3 + 1] = coords[r * 3 + 1]; out[i * 3 + 2] = coords[r * 3 + 2];
    } else {
      out[i * 3] = coords[r][0]; out[i * 3 + 1] = coords[r][1]; out[i * 3 + 2] = coords[r][2];
    }
  });
  return out;
}

export { DEFAULTS as DOMAIN_DEFAULTS };


// --- consensus ---------------------------------------------------------------

/** Deterministic PRNG, so the same structure always parses the same way. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x9e3779b9) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, two coordinates at a time. */
function jitterCoords(flat, sigma, seed) {
  const out = Float64Array.from(flat);
  const r = rng(seed);
  for (let i = 0; i < out.length; i += 2) {
    const u = Math.max(r(), 1e-12);
    const v = r();
    const m = sigma * Math.sqrt(-2 * Math.log(u));
    out[i] += m * Math.cos(2 * Math.PI * v);
    if (i + 1 < out.length) out[i + 1] += m * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/**
 * Parse a chain several times under small perturbations and keep what survives.
 *
 * Some proteins have two bisections whose normalised cuts sit within coordinate
 * noise of each other, and a single pass reports whichever one won by a hair as
 * though it were certain. 3MKU is the case that exposed it: chains A and B are the
 * same protein in the same conformation, agreeing to 0.078 Å on every consecutive
 * Cα, and they picked different splits.
 *
 * Each run jitters the coordinates and shifts the contact cut-off, so a decision
 * that hinges on a handful of borderline contacts comes out differently across
 * runs while a real domain boundary does not. The modal domain count wins, then
 * each residue takes the majority label among the runs that agreed on that count.
 *
 * The spread is returned as well as the answer: `confidence` is the share of runs
 * that found the modal count, which is the honest way to say "this one is
 * marginal" instead of drawing a boundary and hoping.
 */
export function parseDomains(coords, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  const runs = Math.max(1, opt.consensus | 0);
  if (runs === 1) return parseOnce(coords, options);

  let flat;
  let n;
  if (ArrayBuffer.isView(coords)) { flat = coords; n = coords.length / 3; } else {
    n = coords.length;
    flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = coords[i][0]; flat[i * 3 + 1] = coords[i][1]; flat[i * 3 + 2] = coords[i][2];
    }
  }

  const results = [];
  for (let k = 0; k < runs; k++) {
    // The first run is the unperturbed one, so a stable chain gets exactly the
    // answer a single pass would have given.
    const jittered = k === 0 ? flat : jitterCoords(flat, opt.jitter, 0x9e37 + k * 7919);
    const spread = k === 0 ? 0 : ((k % 3) - 1) * opt.cutJitter;
    results.push(parseOnce(jittered, { ...options, contactDist: opt.contactDist + spread }));
  }

  const tally = new Map();
  for (const r of results) tally.set(r.domains.length, (tally.get(r.domains.length) || 0) + 1);
  let modal = results[0].domains.length;
  for (const [count, hits] of tally) {
    const bestHits = tally.get(modal);
    if (hits > bestHits || (hits === bestHits && count < modal)) modal = count;
  }
  const agreeing = results.filter((r) => r.domains.length === modal);

  // Majority label per residue. Domains come back ordered by where they start, so
  // index i means the same thing across runs.
  const labels = new Int32Array(n);
  const votes = new Int32Array(modal + 1);
  for (let i = 0; i < n; i++) {
    votes.fill(0);
    for (const r of agreeing) {
      const l = r.labels[i];
      votes[l < 0 ? modal : l]++;
    }
    let bestL = 0;
    for (let l = 1; l <= modal; l++) if (votes[l] > votes[bestL]) bestL = l;
    labels[i] = bestL === modal ? -1 : bestL;
  }

  // How often each residue ended up where the vote put it. A residue in the core of
  // a domain lands there in every run; one on a flexible tail or in a linker gets
  // shuffled between neighbours, or dropped, as the perturbation moves. That is a
  // trimming signal the contact-degree heuristic did not have — it could only ask
  // how many neighbours a residue has, not whether the parse was sure about it.
  const stability = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let hits = 0;
    for (const r of agreeing) if (r.labels[i] === labels[i]) hits++;
    stability[i] = hits / agreeing.length;
  }
  if (opt.trim) trimByStability(labels, stability, n, modal, opt);

  // The vote can leave a label with no residues at all — every run agreed on the
  // count, but no residue's majority landed on that particular index. Empty domains
  // are dropped and the rest renumbered, so `labels` has to follow the renumbering
  // or every label above the gap points one slot past its domain. Reachable only off
  // the default maxCut (5 of 260 chains at 0.12), but `domains[labels[i]]` is how
  // every consumer reads this, so the two must not disagree.
  const domains = [];
  const remap = new Int32Array(modal).fill(-1);
  for (let index = 0; index < modal; index++) {
    const residues = [];
    for (let i = 0; i < n; i++) if (labels[i] === index) residues.push(i);
    if (residues.length) {
      remap[index] = domains.length;
      domains.push({ index: domains.length, residues, segments: toSegments(residues), nres: residues.length });
    }
  }
  for (let i = 0; i < n; i++) if (labels[i] >= 0) labels[i] = remap[labels[i]];
  return {
    n,
    labels,
    domains,
    confidence: agreeing.length / results.length,
    stability,
    counts: [...tally].sort((a, b) => b[1] - a[1]),
    ...coassociation(results, n, opt.coassocBins),
  };
}

/**
 * How often each pair of residues landed in the same domain, over all runs.
 *
 * The confidence figure says how often the runs agreed; this says *where* they
 * disagreed. Blocks along the diagonal are the domains, and their edges show how
 * far the boundary wandered — a crisp corner means every run cut in the same
 * place, a smeared one means the cut point is genuinely uncertain and the number
 * on the bar is one of several defensible answers.
 *
 * Every run counts here, including those whose domain count lost the vote: a run
 * that split into three still says which residues it kept together, and throwing
 * it away would hide exactly the disagreement this is for.
 *
 * Binned to keep the matrix a fixed size — a 2000-residue chain would otherwise
 * be 4 MB to hand to the UI thread, and the screen cannot resolve it anyway. Each
 * bin votes with the label of its middle residue.
 */
function coassociation(results, n, bins) {
  const B = Math.min(n, bins);
  if (B < 2) return {};
  const mid = new Int32Array(B);
  for (let b = 0; b < B; b++) mid[b] = Math.min(n - 1, Math.floor((b + 0.5) * n / B));
  const out = new Uint8Array(B * B);
  const step = 255 / results.length;
  for (const r of results) {
    for (let a = 0; a < B; a++) {
      const la = r.labels[mid[a]];
      for (let b = a; b < B; b++) {
        if (r.labels[mid[b]] !== la) continue;
        out[a * B + b] += step;
        if (b !== a) out[b * B + a] += step;
      }
    }
  }
  return { coassoc: out, coassocBins: B };
}


/**
 * Drop terminal residues the runs were not sure about.
 *
 * CATH and SCOPe both discard residues that are not part of the compact domain,
 * and 99% of what they discard is terminal. Walking in from each end while the
 * runs disagree is a more direct test than counting contacts: a residue the parse
 * kept reassigning is one that does not clearly belong, whatever its coordination
 * number happens to be.
 *
 * Stops at the first confident residue rather than trimming the worst ones
 * anywhere, since a domain with a hole punched in the middle is not a domain.
 */
function trimByStability(labels, stability, n, nParts, opt) {
  for (let part = 0; part < nParts; part++) {
    const idx = [];
    for (let i = 0; i < n; i++) if (labels[i] === part) idx.push(i);
    if (idx.length < opt.minDomain) continue;
    const floor = Math.max(opt.minDomain, Math.ceil(idx.length * opt.trimKeep));
    let lo = 0;
    let hi = idx.length - 1;
    while (hi - lo + 1 > floor && stability[idx[lo]] < opt.trimStable) lo++;
    while (hi - lo + 1 > floor && stability[idx[hi]] < opt.trimStable) hi--;
    for (let k = 0; k < lo; k++) labels[idx[k]] = -1;
    for (let k = hi + 1; k < idx.length; k++) labels[idx[k]] = -1;
  }
}


/**
 * Retrieval as a parsing objective: tested, and dominated. Kept as a note because
 * it is an appealing idea and the app already ships everything it needs.
 *
 * The proposal was to score a candidate part by how strongly a curated domain
 * library "echoes" it back — its best cosine against the 15,176 SCOPe40
 * embeddings — and split only when both halves become more retrievable. Two
 * networks trained solely on single domains plus a resident library give a learned
 * notion of domain-ness for free, with nothing new to train.
 *
 * It works, and is not a size artefact. On 25 two-domain chains, with the query's
 * own PDB excluded from the library, a cut at SCOPe's boundary beat 10 random cuts
 * 20/25 times against a 9% chance rate, and beat its own size-matched mirror cut
 * (identical half sizes, different residues) 10/12. Retrieval density really is a
 * boundary signal independent of fragment length.
 *
 * It is also worse than the criterion below, on the same cuts: 23/25, mean rank
 * 1.24 against 1.60, mirror 12/12 against 10/12 — at roughly 500 times less
 * compute. Sixteen scoring rules were tried; the best was plain top-1 combined
 * across halves by mean (21/25, 1.36, 12/12). Deeper statistics were steadily
 * worse (top-3 1.72, top-10 2.12, top-50 3.24), so the signal lives entirely in
 * the single nearest relative and averaging it away destroys it.
 *
 * Two caveats if anyone revisits: pooling a subset of per-residue vectors does NOT
 * reproduce that subset's embedding (cosine 0.48, see test/pooling.mjs), so every
 * candidate needs a real forward pass; and relatives were left in the library,
 * which matches the real use case but makes this easier than a superfamily-level
 * holdout.
 */

/**
 * The normalised cut of one specific bipartition, [0, k) against [k, n).
 *
 * Exported so an externally proposed cut can be scored on exactly the graph and
 * criterion the parser itself uses — otherwise a comparison against some other
 * objective is really a comparison against a reimplementation of this one. Lower
 * is better, matching how bestSplit ranks candidates.
 */
export function cutScoreAt(coords, k, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  let flat;
  let n;
  if (ArrayBuffer.isView(coords)) { flat = coords; n = coords.length / 3; } else {
    n = coords.length;
    flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = coords[i][0]; flat[i * 3 + 1] = coords[i][1]; flat[i * 3 + 2] = coords[i][2];
    }
  }
  const g = buildGraph(flat, n, opt);
  let cut = 0;
  let volA = 0;
  let volB = 0;
  for (let i = 0; i < n; i++) {
    const inA = i < k;
    for (let e = g.off[i]; e < g.off[i + 1]; e++) {
      const w = g.wt[e];
      if (inA) volA += w; else volB += w;
      if (inA !== (g.idx[e] < k)) cut += w;
    }
  }
  cut /= 2;                       // each crossing edge is seen from both ends
  if (volA <= 0 || volB <= 0) return Infinity;
  return cut / volA + cut / volB;
}


/**
 * The normalised cut at every contiguous split point, in one pass.
 *
 * cutScoreAt rebuilds the graph per call, which is O(n^2) each and hopeless for a
 * whole profile. This builds it once and walks the chain, updating the interface
 * and the two volumes incrementally, so the entire profile costs what a single
 * sweep costs. Entries below minDomain from either end are Infinity.
 *
 * Exported for failure analysis: the value at the best cut turned out not to
 * separate chains that should be split from chains that should not, so the next
 * question is whether the SHAPE of this profile does — a real boundary ought to be
 * a sharp minimum, a spurious one a shallow dip in a flat landscape.
 */
export function cutProfile(coords, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  let flat;
  let n;
  if (ArrayBuffer.isView(coords)) { flat = coords; n = coords.length / 3; } else {
    n = coords.length;
    flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3] = coords[i][0]; flat[i * 3 + 1] = coords[i][1]; flat[i * 3 + 2] = coords[i][2];
    }
  }
  const g = buildGraph(flat, n, opt);
  const deg = new Float64Array(n);
  let volTotal = 0;
  for (let i = 0; i < n; i++) {
    for (let p = g.off[i]; p < g.off[i + 1]; p++) deg[i] += g.wt[p];
    volTotal += deg[i];
  }
  const out = new Float64Array(n + 1).fill(Infinity);
  let volA = 0;
  let cut = 0;
  for (let k = 1; k <= n - 1; k++) {
    const v = k - 1;                       // residue joining side A
    volA += deg[v];
    for (let p = g.off[v]; p < g.off[v + 1]; p++) {
      cut += g.idx[p] < v ? -g.wt[p] : g.wt[p];
    }
    if (k < opt.minDomain || n - k < opt.minDomain) continue;
    const volB = volTotal - volA;
    if (volA <= 0 || volB <= 0) continue;
    out[k] = cut / volA + cut / volB;
  }
  return out;
}
