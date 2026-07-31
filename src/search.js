// Brute-force cosine search over a pre-embedded CIRPIN database.
//
// Both the query and the database rows are L2-normalised, so similarity is a
// plain dot product. 15k x 128 is ~2M multiply-adds — sub-millisecond, no
// index structure needed. Scoring matches progres.py embedding_similarity:
// (1 + cos) / 2, running 0 (far) to 1 (close).

export function loadDatabase(bin, meta) {
  const n = meta.n;
  const dim = meta.dim;
  let vecs;
  if (meta.dtype === 'float16') {
    const u16 = new Uint16Array(bin, 0, n * dim);
    vecs = new Float32Array(n * dim);
    for (let i = 0; i < n * dim; i++) {
      const h = u16[i];
      const s = (h & 0x8000) >> 15;
      const e = (h & 0x7c00) >> 10;
      const f = h & 0x03ff;
      let v;
      if (e === 0) v = f * 2 ** -24;
      else if (e === 0x1f) v = f ? NaN : Infinity;
      else v = (1 + f / 1024) * 2 ** (e - 15);
      vecs[i] = s ? -v : v;
    }
  } else {
    vecs = new Float32Array(bin, 0, n * dim);
  }
  return { n, dim, vecs, ids: meta.ids, nres: meta.nres, notes: meta.notes || [] };
}

/**
 * @param {object} db - from loadDatabase
 * @param {Float32Array} query - 128-d L2-normalised embedding
 * @param {{maxHits?: number, minSimilarity?: number}} opts
 */
export function search(db, query, opts = {}) {
  const maxHits = opts.maxHits ?? 100;
  const minSimilarity = opts.minSimilarity ?? 0.8;
  const { n, dim, vecs } = db;

  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const off = i * dim;
    let s = 0;
    for (let k = 0; k < dim; k++) s += query[k] * vecs[off + k];
    scores[i] = (1 + s) / 2;
  }

  const idx = [];
  for (let i = 0; i < n; i++) if (scores[i] >= minSimilarity) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a]);

  return idx.slice(0, maxHits).map((i) => ({
    rank: 0,
    id: db.ids[i],
    similarity: scores[i],
    nres: db.nres[i],
    notes: db.notes[i] ?? '-',
  })).map((h, k) => ({ ...h, rank: k + 1 }));
}
