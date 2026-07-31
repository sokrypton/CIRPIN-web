// The Cα coordinate decoder — one implementation, every database.
//
// A stored domain is a self-describing record: centroid, residue count, three absolute anchor
// residues, then one 11-bit codebook index per residue. The index is into a codebook of step
// vectors expressed in the local frame of the previous three residues, which is the same
// information as (bond length, bond angle, dihedral) in the basis where quantisation error and
// coordinate error are the same thing.
//
// Because the record carries its own residue count, a reader needs nothing but the bytes and
// the codebook. That is what lets SCOPe40 and the hosted AlphaFold TED store share this decoder
// instead of one each: they differ in where their files are hosted and how large they are, not in
// what the bytes mean. Adding a database means publishing a store, not writing a reader.
//
// web/tools/coordcodec.py is the encoder and the reference; test/coords.mjs checks this against
// records it produced, so the two cannot drift silently.
//
// Half the size of the Cartesian int8 format it replaces (1.50 bytes per residue against 3.02)
// at the same fidelity: over 30 real query/hit pairs, mean |dTM| 0.00074 and worst 0.0062,
// where the old format scored 0.00037 and 0.0041 and neither moved a pair past 0.01.

export const FORMAT = 4;
export const BITS = 11;
export const ESCAPE = (1 << BITS) - 1;   // 2047, the one index that is not a code
export const SCALE = 100;                // 0.01 A absolute anchors
export const ANCHORS = 3;                // residues stored absolutely, to seed the first frame
const CB_SCALE = 1000;                   // codebook is int16 at 0.001 A
const HEADER = 14;                       // float32 x3 centroid + uint16 count

/**
 * Read a codebook file written by coordcodec.save_codebook.
 *
 * @param {ArrayBuffer} buf
 * @returns {{bits: number, n: number, cb: Float64Array}} cb is n*3, row-major
 */
export function loadCodebook(buf) {
  const dv = new DataView(buf);
  const bits = dv.getUint16(0, true);
  const n = dv.getUint16(2, true);
  if (bits !== BITS) {
    throw new Error(`codebook holds ${bits}-bit codes, this build expects ${BITS}`);
  }
  const cb = new Float64Array(n * 3);
  for (let i = 0; i < n * 3; i++) cb[i] = dv.getInt16(4 + i * 2, true) / CB_SCALE;
  return { bits, n, cb, buf };
}

/**
 * The digest a store records for the codebook its records were encoded against.
 *
 * Decoding against the wrong codebook does not produce garbage — it produces a valid-looking
 * structure that is not the right one, which is the worst failure available here. Training is
 * deterministic so there is normally only ever one codebook, but the file is regenerable and
 * therefore deletable, and a cache serving a stale one would otherwise be silent.
 *
 * Matches coordcodec.codebook_id: the first 16 hex characters of the file's SHA-256.
 */
export async function codebookId(buf) {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

/** Residue count of the record at `at`, without decoding it. */
export function recordCount(buf, at = 0) {
  return new DataView(buf).getUint16(at + 12, true);
}

/**
 * Decode one record to absolute coordinates.
 *
 * @param {ArrayBuffer} buf
 * @param {number} at - byte offset of the record start
 * @param {Float64Array} cb - codebook from loadCodebook().cb
 * @returns {Float64Array} 3n coordinates, x,y,z interleaved
 */
export function decodeRecord(buf, at, cb) {
  const dv = new DataView(buf);
  const ox = dv.getFloat32(at, true);
  const oy = dv.getFloat32(at + 4, true);
  const oz = dv.getFloat32(at + 8, true);
  const n = dv.getUint16(at + 12, true);
  const P = new Float64Array(n * 3);

  const nanch = Math.min(ANCHORS, n);
  let p = at + HEADER;
  for (let i = 0; i < nanch; i++) {
    P[i * 3] = dv.getInt16(p, true) / SCALE;
    P[i * 3 + 1] = dv.getInt16(p + 2, true) / SCALE;
    P[i * 3 + 2] = dv.getInt16(p + 4, true) / SCALE;
    p += 6;
  }

  // MSB-first bit reader, matching coordcodec.BitWriter
  let acc = 0;
  let nbits = 0;
  const get = (width) => {
    while (nbits < width) {
      acc = (acc * 256) + dv.getUint8(p++);
      nbits += 8;
    }
    nbits -= width;
    const div = 2 ** nbits;
    const v = Math.floor(acc / div);
    acc -= v * div;
    return v;
  };

  for (let i = ANCHORS; i < n; i++) {
    const code = get(BITS);
    if (code === ESCAPE) {
      for (let k = 0; k < 3; k++) {
        const raw = get(16);
        P[i * 3 + k] = (raw > 32767 ? raw - 65536 : raw) / SCALE;
      }
      continue;
    }
    // the frame from the three previous RECONSTRUCTED residues — identical arithmetic to the
    // encoder, including the collinear fallback, or the two walk apart
    const ax = P[(i - 1) * 3], ay = P[(i - 1) * 3 + 1], az = P[(i - 1) * 3 + 2];
    const bx = P[(i - 2) * 3], by = P[(i - 2) * 3 + 1], bz = P[(i - 2) * 3 + 2];
    const cx = P[(i - 3) * 3], cy = P[(i - 3) * 3 + 1], cz = P[(i - 3) * 3 + 2];
    let ux = ax - bx, uy = ay - by, uz = az - bz;
    const ul = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1;
    ux /= ul; uy /= ul; uz /= ul;
    let wx = bx - cx, wy = by - cy, wz = bz - cz;
    const dot = wx * ux + wy * uy + wz * uz;
    wx -= dot * ux; wy -= dot * uy; wz -= dot * uz;
    let wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (wl < 1e-8) {
      const sx = Math.abs(ux) < 0.9 ? 1 : 0;
      const sy = Math.abs(ux) < 0.9 ? 0 : 1;
      const t = sx * ux + sy * uy;
      wx = sx - t * ux; wy = sy - t * uy; wz = -t * uz;
      wl = Math.sqrt(wx * wx + wy * wy + wz * wz) || 1;
    }
    wx /= wl; wy /= wl; wz /= wl;
    const vx = uy * wz - uz * wy, vy = uz * wx - ux * wz, vz = ux * wy - uy * wx;
    const a = cb[code * 3], b = cb[code * 3 + 1], g = cb[code * 3 + 2];
    P[i * 3] = ax + a * ux + b * wx + g * vx;
    P[i * 3 + 1] = ay + a * uy + b * wy + g * vy;
    P[i * 3 + 2] = az + a * uz + b * wz + g * vz;
  }

  for (let i = 0; i < n; i++) {
    P[i * 3] += ox;
    P[i * 3 + 1] += oy;
    P[i * 3 + 2] += oz;
  }
  return P;
}

// --- locating records in a sharded store ------------------------------------
//
// Decoding is above; this is the other half of a store, and the two are separate because a
// store is only "a way to turn a row into a URL and a byte range". The SCOPe buckets locate
// through a JSON index; AlphaFold TED locates through this. A third database supplies its own
// locator and reuses decodeRecord unchanged.
//
// Two constraints shape it, both about what the user downloads:
//
//   * GitHub refuses any pushed file over 100 MB, so 600 MB of records arrive as shards. Records
//     never straddle one (tools/shard_coords.py cuts on cluster boundaries, which are record
//     boundaries), so one domain is still exactly one range request.
//   * A flat table of offsets would be 28 MB, and of lengths 13.9 MB, and the browser would
//     fetch all of it to read 170 bytes.
//
// The second is worth being precise about, because the length itself is nearly redundant: a
// record carries its own residue count and decodeRecord stops there, ignoring trailing bytes, so
// a reader could over-fetch a fixed window and never store a length at all. What it cannot
// derive is the OFFSET — that is the prefix sum of every preceding length, and residue count
// will not substitute because each escape adds 48 bits that the choppings say nothing about.
//
// So the table stays, and instead almost none of it is downloaded. Lengths are uint16 (records
// top out at 1,489 bytes, 44x under the ceiling) in blocks of 1024 rows, with a uint32 prefix
// sum giving each block's byte base. A query reads 13.5 KB of base table once and 2 KB per block
// it touches — call it 45 KB for a page of hits, against 13.9 MB. This is the same arrangement
// the row metadata already uses (chopBase in worker.js) for the same reason.

/** The shard holding a row. A handful of entries, so a linear walk from the end. */
export function shardFor(shards, row) {
  for (let i = shards.length - 1; i >= 0; i--) {
    if (row >= shards[i].row0) return shards[i];
  }
  return null;
}

/**
 * A coordinate store: rows in, byte ranges out.
 *
 * Every database uses this one. SCOPe40 and AlphaFold TED differ only in where their files are
 * hosted and how big they are, and size is handled here rather than by a second implementation:
 * an index small enough to fetch whole (SCOPe's is 30 KB) is fetched in one request, and a large
 * one is read a block at a time. Same format, same code, one decision.
 *
 * @param {object} manifest - from coordstore.publish(): {records, index, indexBlock, indexBytes,
 *   base, shards}
 * @param {Uint32Array} blockBase - global byte offset at each block start
 * @param {(url: string, from: number, to: number) => Promise<Uint8Array>} fetchRange
 * @param {(name: string) => string} url - resolves a manifest filename to a fetchable URL
 * @param {{eagerBytes?: number}} [opts] - eagerBytes overrides the whole-index threshold; the
 *   tests use it to drive both paths over the same data, since a database is otherwise only ever
 *   large enough for one of them.
 */
export function shardedStore(manifest, blockBase, fetchRange, url, opts = {}) {
  const BLOCK = manifest.indexBlock;
  const blocks = new Map();
  // 256 KB: below this, fetching the index whole costs less than the round trips its blocks
  // would take, and every row can then be answered without touching the network again.
  const eager = (manifest.indexBytes ?? manifest.records * 2) <= (opts.eagerBytes ?? 262144);

  async function lengths(bi) {
    const key = eager ? 0 : bi;
    let got = blocks.get(key);
    if (got) return got;
    const from = eager ? 0 : bi * BLOCK * 2;
    const rows = eager
      ? manifest.records
      : Math.min((bi + 1) * BLOCK, manifest.records) - bi * BLOCK;
    const b = await fetchRange(url(manifest.index), from, from + rows * 2 - 1);
    // copy rather than view: a fetched buffer is not guaranteed 2-byte aligned
    got = new Uint16Array(rows);
    for (let i = 0; i < rows; i++) got[i] = b[i * 2] | (b[i * 2 + 1] << 8);
    blocks.set(key, got);
    return got;
  }

  return {
    /**
     * Where a row's record lives, or null if it has none.
     *
     * The offset is the block's base plus the lengths of the rows before this one within the
     * block — which is why lengths arrive a block at a time rather than a row at a time.
     */
    async where(row) {
      if (row < 0 || row >= manifest.records) return null;
      const bi = Math.floor(row / BLOCK);
      const L = await lengths(bi);
      const within = eager ? row : row - bi * BLOCK;
      const len = L[within];
      if (!len) return null;
      // eager holds every row, so the prefix sum starts at zero; blocked starts at the
      // block's recorded base
      let at = eager ? 0 : blockBase[bi];
      for (let r = 0; r < within; r++) at += L[r];
      const sh = shardFor(manifest.shards, row);
      if (!sh) return null;
      const start = at - sh.base;      // offsets are global; shards are not
      return { url: url(sh.file), start, end: start + len - 1, length: len };
    },
  };
}
