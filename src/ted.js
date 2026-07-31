// Search the AFDB-ClustR TED database: 3,466,144 domains, 32 bytes each.
//
// Storing these the way scope40 is stored would be 3.55 GB of float32 across the
// two models. Instead each embedding is 32 PCA coefficients quantised to a byte,
// which measured on this set costs 0.0032 of absolute score error — see
// web/tools/export_ted.py for how the codes are made and test/ted.mjs for what
// that error does to the answers.
//
// The scan never decodes a code. Similarity is (1 + q·x) / 2 over L2-normalised
// vectors, and with x ≈ zV + mu:
//
//     q·x = (q Vᵀ)·z + q·mu
//
// so the query is projected once into 32 dimensions and each domain costs 32
// multiply-adds instead of 128. Dequantising z inline as lo[d] + c·step[d] would
// be two more operations per dimension, so the query is pre-scaled instead: fold
// step into the projected query and lo into the constant, and a code byte enters
// the dot product as-is.

export const TED_DIMS = 32;
export const TED_ID_BYTES = 8;

/**
 * Prepare one model's basis for scanning.
 * @param {object} m one entry of ted-basis.json's `models`
 */
export function loadBasis(m) {
  const dims = m.lo.length;
  return {
    dims,
    mu: Float32Array.from(m.mu),
    basis: Float32Array.from(m.basis),   // dims x 128, row-major
    lo: Float32Array.from(m.lo),
    step: Float32Array.from(m.step),
  };
}

/**
 * Project a query into the scan's coordinates.
 *
 * Returns the per-dimension multipliers a code byte is weighted by, and the
 * constant every score carries. Then score(code) = bias + sum(w[d] * code[d]),
 * which is exactly (1 + q·x)/2 with x the dequantised reconstruction.
 */
export function projectQuery(B, q) {
  const { dims, mu, basis, lo, step } = B;
  const w = new Float32Array(dims);
  let bias = 0;
  for (let d = 0; d < dims; d++) {
    let z = 0;
    const o = d * 128;
    for (let k = 0; k < 128; k++) z += q[k] * basis[o + k];
    // q·(lo + c*step) summed over dimensions splits into a constant and a weight
    w[d] = z * step[d] * 0.5;
    bias += z * lo[d];
  }
  for (let k = 0; k < 128; k++) bias += q[k] * mu[k];
  return { w, bias: (1 + bias) * 0.5 };
}

/**
 * Scan every code and keep the best `maxHits`.
 *
 * A simple insertion into a sorted array of length maxHits beats a heap here: 40
 * is small, and after the first few thousand domains almost every candidate fails
 * the one comparison against the running floor and costs nothing more.
 *
 * @param {Uint8Array[]} shards code blocks in row order
 * @param {{w: Float32Array, bias: number}} p from projectQuery
 * @param {number} maxHits
 * @param {function(number):void} [onProgress] called with a 0..1 fraction
 */
export function scanCodes(shards, p, maxHits, dims, onProgress) {
  const { w, bias } = p;
  const bestScore = new Float32Array(maxHits).fill(-Infinity);
  const bestRow = new Int32Array(maxHits).fill(-1);
  let floor = -Infinity;
  let filled = 0;
  let row = 0;
  let done = 0;
  let total = 0;
  for (const s of shards) total += s.length / dims;

  for (const codes of shards) {
    const rowsHere = codes.length / dims;
    // Chunked so progress can be reported without testing a counter per row.
    const CHUNK = 200_000;
    for (let a = 0; a < rowsHere; a += CHUNK) {
      const end = Math.min(a + CHUNK, rowsHere);
      for (let i = a; i < end; i++) {
        const o = i * dims;
        let s = bias;
        for (let d = 0; d < dims; d++) s += w[d] * codes[o + d];
        if (s <= floor) continue;
        // shift down from the tail until this score is in place
        let j = filled < maxHits ? filled++ : maxHits - 1;
        while (j > 0 && bestScore[j - 1] < s) {
          bestScore[j] = bestScore[j - 1];
          bestRow[j] = bestRow[j - 1];
          j--;
        }
        bestScore[j] = s;
        bestRow[j] = row + i;
        if (filled === maxHits) floor = bestScore[maxHits - 1];
      }
      done += end - a;
      if (onProgress) onProgress(done / total);
    }
    row += rowsHere;
  }
  return { rows: bestRow.subarray(0, filled), scores: bestScore.subarray(0, filled) };
}

/**
 * Score specific rows only, for annotating one model's hits with the other's.
 *
 * The table shows CIRPIN and Progres side by side, but only CIRPIN decides the
 * ranking, so the Progres column needs 40 rows scored, not 3.47 million.
 */
export function scoreRows(shards, p, rows, dims, shardRows) {
  const { w, bias } = p;
  const out = new Float32Array(rows.length);
  for (let k = 0; k < rows.length; k++) {
    const r = rows[k];
    const codes = shards[Math.floor(r / shardRows)];
    const o = (r % shardRows) * dims;
    let s = bias;
    for (let d = 0; d < dims; d++) s += w[d] * codes[o + d];
    out[k] = s;
  }
  return out;
}

const B36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/**
 * Rebuild a TED id from its 8 packed bytes.
 *
 *   bit 0      accession is 10 characters rather than 6
 *   bits 1..8  TED index
 *   bits 9..   accession, base 36, most significant character first
 *
 * BigInt because the accession alone spans 52 bits and a Number would lose the
 * low ones. It runs once per displayed row, so the cost does not matter.
 */
export function unpackId(bytes, at) {
  let v = 0n;
  for (let k = TED_ID_BYTES - 1; k >= 0; k--) v = (v << 8n) | BigInt(bytes[at + k]);
  const long = Number(v & 1n);
  const ted = Number((v >> 1n) & 0xffn);
  v >>= 9n;
  let acc = '';
  for (let k = 0; k < (long ? 10 : 6); k++) {
    acc = B36[Number(v % 36n)] + acc;
    v /= 36n;
  }
  return `AF-${acc}-F1-model_v4_TED${String(ted).padStart(2, '0')}`;
}
