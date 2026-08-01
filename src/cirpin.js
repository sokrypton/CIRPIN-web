// CIRPIN inference in plain JavaScript.
//
// Port of the EGNN in train_CIRPIN.py:257-386 / progres.py:93-229.
// Every block is Linear -> SiLU -> Linear (dropout is identity at eval), so
// the whole op set is: dense matvec, SiLU, gather, segment-sum, L2 normalize.
//
// Key optimisation — the per-edge Linear(257 -> 256) factorises exactly.
// Its input is cat(feats_i, feats_j, rel_dist), so splitting the weight into
// [W_i | W_j | w_d] gives
//
//     pre_ij = feats[i] @ W_iᵀ + feats[j] @ W_jᵀ + d²_ij * w_d + b
//
// The first two terms are per-NODE and computed once per layer (2·n·128·256)
// instead of once per EDGE (e·257·256) — for a 300-residue chain that is
// ~20M MACs rather than ~395M. This is algebraically exact, not an
// approximation. Only the second edge Linear (256 -> 64), which sits after
// the nonlinearity, is irreducibly per-edge.

import { coordsToGraph, N_FEATURES } from './structure.js?v=2c8b633b';

const HIDDEN = 128;       // hidden_dim
const EDGE_HIDDEN = 256;  // hidden_edge_dim
const M_DIM = 64;         // hidden_egnn_dim
const N_LAYERS = 6;
const EMB = 128;          // embedding_size

// --- weight loading ----------------------------------------------------------

function decodeF16(buf, count) {
  const u16 = new Uint16Array(buf, 0, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    let v;
    if (e === 0) v = f * 2 ** -24;
    else if (e === 0x1f) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * 2 ** (e - 15);
    out[i] = s ? -v : v;
  }
  return out;
}

/**
 * @param {ArrayBuffer} bin - contents of cirpin.bin
 * @param {object} meta - parsed cirpin.json
 */
export function loadWeights(bin, meta) {
  const all = meta.dtype === 'float16'
    ? decodeF16(bin, meta.count)
    : new Float32Array(bin, 0, meta.count);

  const get = (name) => {
    const t = meta.tensors[name];
    if (!t) throw new Error(`missing tensor ${name}`);
    const size = t.shape.reduce((a, b) => a * b, 1);
    return all.subarray(t.offset, t.offset + size);
  };

  // Weights and intermediates are float32. Measured against a float64 build:
  // identical speed to within 2% and identical accuracy (worst |Δembedding|
  // vs PyTorch 2.6e-7 either way — the reference is float32, so float64
  // compute buys nothing against it), at half the resident memory.
  // Dot products still accumulate in a JS number, i.e. float64; only the
  // stored values are float32.
  const f32 = (name) => Float32Array.from(get(name));

  // Transpose (out, in) -> (in, out) so the per-edge accumulation becomes a
  // saxpy over contiguous rows with the 64 accumulators held in L1.
  const transpose = (src, outDim, inDim) => {
    const t = new Float32Array(inDim * outDim);
    for (let o = 0; o < outDim; o++) {
      for (let k = 0; k < inDim; k++) t[k * outDim + o] = src[o * inDim + k];
    }
    return t;
  };

  const layers = [];
  for (let l = 0; l < N_LAYERS; l++) {
    const e0w = get(`layers.${l}.edge_mlp.0.weight`); // (256, 257)
    // Split the concat weight: columns [0..128) -> W_i, [128..256) -> W_j, [256] -> w_d
    const Wi = new Float32Array(EDGE_HIDDEN * HIDDEN);
    const Wj = new Float32Array(EDGE_HIDDEN * HIDDEN);
    const wd = new Float32Array(EDGE_HIDDEN);
    for (let o = 0; o < EDGE_HIDDEN; o++) {
      const row = o * (2 * HIDDEN + 1);
      Wi.set(e0w.subarray(row, row + HIDDEN), o * HIDDEN);
      Wj.set(e0w.subarray(row + HIDDEN, row + 2 * HIDDEN), o * HIDDEN);
      wd[o] = e0w[row + 2 * HIDDEN];
    }
    layers.push({
      Wi,
      Wj,
      wd,
      eb0: f32(`layers.${l}.edge_mlp.0.bias`),
      // (64, 256) -> (256, 64)
      ew3T: transpose(get(`layers.${l}.edge_mlp.3.weight`), M_DIM, EDGE_HIDDEN),
      eb3: f32(`layers.${l}.edge_mlp.3.bias`),
      nw0: f32(`layers.${l}.node_mlp.0.weight`), // (256, 192)
      nb0: f32(`layers.${l}.node_mlp.0.bias`),
      nw3: f32(`layers.${l}.node_mlp.3.weight`), // (128, 256)
      nb3: f32(`layers.${l}.node_mlp.3.bias`),
    });
  }

  return {
    encW: f32('node_enc.weight'), // (128, 68)
    encB: f32('node_enc.bias'),
    layers,
    ndW0: f32('node_dec.0.weight'),
    ndB0: f32('node_dec.0.bias'),
    ndW3: f32('node_dec.3.weight'),
    ndB3: f32('node_dec.3.bias'),
    gdW0: f32('graph_dec.0.weight'),
    gdB0: f32('graph_dec.0.bias'),
    gdW4: f32('graph_dec.4.weight'),
    gdB4: f32('graph_dec.4.bias'),
  };
}

// --- kernels -----------------------------------------------------------------

const silu = (x) => x / (1 + Math.exp(-x));

// out[o] = b[o] + sum_k x[xo + k] * W[o*inDim + k]   (W is (outDim, inDim) row-major)
//
// Four rows at a time, four accumulators. Not for the memory traffic — x is tiny and
// stays in cache either way — but for the dependency chain: with one accumulator each
// multiply-add waits on the previous one, so the loop is latency-bound at about four
// cycles per MAC. Measured on a 256x128 matvec, 1.20 GMAC/s one-at-a-time against
// 2.26 GMAC/s four-at-a-time, a 1.88x speedup for the same arithmetic.
//
// Each accumulator starts at its bias and adds the same products in the same order as
// the one-row version, so this is bit-identical, not merely close — checked at
// worst |delta| exactly 0 across every shape the network uses.
function matvec(out, outOff, x, xOff, W, b, outDim, inDim) {
  let o = 0;
  for (; o + 4 <= outDim; o += 4) {
    const r0 = o * inDim;
    const r1 = r0 + inDim;
    const r2 = r1 + inDim;
    const r3 = r2 + inDim;
    let s0 = b ? b[o] : 0;
    let s1 = b ? b[o + 1] : 0;
    let s2 = b ? b[o + 2] : 0;
    let s3 = b ? b[o + 3] : 0;
    for (let k = 0; k < inDim; k++) {
      const xv = x[xOff + k];
      s0 += xv * W[r0 + k];
      s1 += xv * W[r1 + k];
      s2 += xv * W[r2 + k];
      s3 += xv * W[r3 + k];
    }
    out[outOff + o] = s0;
    out[outOff + o + 1] = s1;
    out[outOff + o + 2] = s2;
    out[outOff + o + 3] = s3;
  }
  // tail: every real shape here is a multiple of four, but the kernel should not
  // silently depend on that
  for (; o < outDim; o++) {
    let s = b ? b[o] : 0;
    const w = o * inDim;
    for (let k = 0; k < inDim; k++) s += x[xOff + k] * W[w + k];
    out[outOff + o] = s;
  }
}

// --- forward -----------------------------------------------------------------

/**
 * Run CIRPIN over a graph from coordsToGraph().
 * @param {function} [onLayer] - called (done, total) after each EGNN layer, so a
 *   caller can report progress through what is a multi-second computation.
 * @param {Float32Array} [nodeSink] - length n*128; receives each residue's
 *   node_dec vector, the thing that gets summed. With these you can pool any
 *   subset and decode it via poolAndDecode, which is not the same as embedding
 *   that subset alone — see poolAndDecode for how far apart they are.
 * @param {Float32Array} [normSink] - length n; receives the L2 norm of each
 *   residue's node_dec vector. The readout is a plain sum over residues, so that
 *   vector is exactly what the residue adds to the pooled embedding, and its
 *   norm is how much the residue moves the result. Everything after the sum
 *   (graph_dec) is nonlinear, so this is the last point at which a per-residue
 *   contribution is well defined.
 * @returns {Float32Array} 128-d L2-normalised embedding
 */
/**
 * edgeSink, when supplied, receives the L2 NORM OF EACH EDGE MESSAGE -- Float32Array sized
 * N_LAYERS * nEdges, indexed layer-major by the caller's neighbour-list position.
 *
 * Why this is a different proposition from the node vectors. Comparing two node vectors is a similarity we
 * invent after the fact, and it has to guess which comparison the network would endorse; today's
 * measurements say cosine is a weak guess and profile correlation a slightly better one. The edge message
 * needs no guess. It IS the quantity the network propagates along that contact, computed from both
 * residues' current states and their distance, and its magnitude is how much this contact contributes to
 * its neighbour's update -- a learned edge weight, which is exactly the input a normalised cut wants.
 *
 * Note the message is DIRECTED: m_ij comes from Wi h_i + Wj h_j, so m_ij != m_ji. Callers wanting an
 * undirected weight must symmetrise.
 */
export function embedGraph(w, graph, onLayer, normSink, nodeSink, edgeSink) {
  const { n, x, nbrOffsets, nbrIndices, nbrDist2 } = graph;

  const feats = new Float32Array(n * HIDDEN);
  for (let i = 0; i < n; i++) {
    matvec(feats, i * HIDDEN, x, i * N_FEATURES, w.encW, w.encB, HIDDEN, N_FEATURES);
  }

  const A = new Float32Array(n * EDGE_HIDDEN);
  const B = new Float32Array(n * EDGE_HIDDEN);
  const h = new Float32Array(EDGE_HIDDEN);
  const mAcc = new Float32Array(M_DIM);
  const nodeIn = new Float32Array(HIDDEN + M_DIM);
  const tmp = new Float32Array(EDGE_HIDDEN);
  const next = new Float32Array(n * HIDDEN);

  const acc = new Float32Array(M_DIM);
  const nEdges = nbrIndices.length;

  for (let l = 0; l < N_LAYERS; l++) {
    const L = w.layers[l];
    // Hoist out of the edge loop: property loads on `L` inside the innermost
    // loop cost as much as the arithmetic.
    const { wd, eb0, eb3, ew3T } = L;

    // Per-node halves of the factorised edge linear.
    for (let i = 0; i < n; i++) {
      matvec(A, i * EDGE_HIDDEN, feats, i * HIDDEN, L.Wi, null, EDGE_HIDDEN, HIDDEN);
      matvec(B, i * EDGE_HIDDEN, feats, i * HIDDEN, L.Wj, null, EDGE_HIDDEN, HIDDEN);
    }

    for (let i = 0; i < n; i++) {
      mAcc.fill(0);
      const ai = i * EDGE_HIDDEN;
      const e0 = nbrOffsets[i];
      const e1 = nbrOffsets[i + 1];

      for (let e = e0; e < e1; e++) {
        const bj = nbrIndices[e] * EDGE_HIDDEN;
        const d2 = nbrDist2[e];

        // edge_mlp[0] + SiLU
        for (let o = 0; o < EDGE_HIDDEN; o++) {
          h[o] = silu(A[ai + o] + B[bj + o] + d2 * wd[o] + eb0[o]);
        }
        // edge_mlp[3]: saxpy over the transposed weight, 64 accumulators in L1
        acc.set(eb3);
        for (let k = 0; k < EDGE_HIDDEN; k += 2) {
          const h0 = h[k]; const h1 = h[k + 1];
          const b0 = k * M_DIM; const b1 = b0 + M_DIM;
          for (let o = 0; o < M_DIM; o += 4) {
            acc[o] += h0 * ew3T[b0 + o] + h1 * ew3T[b1 + o];
            acc[o + 1] += h0 * ew3T[b0 + o + 1] + h1 * ew3T[b1 + o + 1];
            acc[o + 2] += h0 * ew3T[b0 + o + 2] + h1 * ew3T[b1 + o + 2];
            acc[o + 3] += h0 * ew3T[b0 + o + 3] + h1 * ew3T[b1 + o + 3];
          }
        }
        // trailing SiLU, accumulated straight into m_i (sum over neighbours)
        if (edgeSink) {
          let ss = 0;
          for (let o = 0; o < M_DIM; o++) { const v = silu(acc[o]); mAcc[o] += v; ss += v * v; }
          edgeSink[l * nEdges + e] = Math.sqrt(ss);
        } else {
          for (let o = 0; o < M_DIM; o++) mAcc[o] += silu(acc[o]);
        }
      }

      // node_mlp over cat(feats, m_i), plus residual
      nodeIn.set(feats.subarray(i * HIDDEN, i * HIDDEN + HIDDEN), 0);
      nodeIn.set(mAcc, HIDDEN);
      matvec(tmp, 0, nodeIn, 0, L.nw0, L.nb0, EDGE_HIDDEN, HIDDEN + M_DIM);
      for (let o = 0; o < EDGE_HIDDEN; o++) tmp[o] = silu(tmp[o]);
      matvec(next, i * HIDDEN, tmp, 0, L.nw3, L.nb3, HIDDEN, EDGE_HIDDEN);
      for (let o = 0; o < HIDDEN; o++) next[i * HIDDEN + o] += feats[i * HIDDEN + o];
    }

    feats.set(next);
    if (onLayer) onLayer(l + 1, N_LAYERS);
  }

  // node_dec, then sum-pool over nodes. `pooled` stays float64: it accumulates
  // one term per residue, so unlike the other buffers its rounding error grows
  // with chain length, and at 128 elements it costs nothing to keep wide.
  const pooled = new Float64Array(HIDDEN);
  const nd = new Float32Array(HIDDEN);
  const nd2 = new Float32Array(HIDDEN);
  for (let i = 0; i < n; i++) {
    matvec(nd, 0, feats, i * HIDDEN, w.ndW0, w.ndB0, HIDDEN, HIDDEN);
    for (let o = 0; o < HIDDEN; o++) nd[o] = silu(nd[o]);
    matvec(nd2, 0, nd, 0, w.ndW3, w.ndB3, HIDDEN, HIDDEN);
    for (let o = 0; o < HIDDEN; o++) pooled[o] += nd2[o];
    if (normSink) {
      let ss2 = 0;
      for (let o = 0; o < HIDDEN; o++) ss2 += nd2[o] * nd2[o];
      normSink[i] = Math.sqrt(ss2);
    }
    if (nodeSink) nodeSink.set(nd2, i * HIDDEN);
  }

  const g1 = new Float32Array(HIDDEN);
  matvec(g1, 0, pooled, 0, w.gdW0, w.gdB0, HIDDEN, HIDDEN);
  for (let o = 0; o < HIDDEN; o++) g1[o] = silu(g1[o]);

  const raw = new Float32Array(EMB);
  matvec(raw, 0, g1, 0, w.gdW4, w.gdB4, EMB, HIDDEN);

  let ss = 0;
  for (let o = 0; o < EMB; o++) ss += raw[o] * raw[o];
  const inv = 1 / Math.max(Math.sqrt(ss), 1e-12);
  const out = new Float32Array(EMB);
  for (let o = 0; o < EMB; o++) out[o] = raw[o] * inv;
  return out;
}

/**
 * Pool a subset of residues from one full-chain pass and decode it.
 *
 * The readout is a plain sum, so summing a subset of node_dec vectors and
 * running graph_dec over it is well defined and costs nothing next to a second
 * forward pass. It is NOT the same as embedding that subset on its own: the node
 * vectors come from message passing over the whole chain's contact graph, so a
 * residue near a domain interface has already mixed in its neighbour's
 * structure. How much that matters is measured in test/pooling.mjs.
 *
 * @param {object} w - loaded weights
 * @param {Float32Array} nodeVecs - n*128 from embedGraph's nodeSink
 * @param {Int32Array|number[]} residues - which residues to pool
 * @returns {Float32Array} 128-d L2-normalised embedding
 */
export function poolAndDecode(w, nodeVecs, residues) {
  const pooled = new Float64Array(HIDDEN);
  for (const r of residues) {
    const off = r * HIDDEN;
    for (let o = 0; o < HIDDEN; o++) pooled[o] += nodeVecs[off + o];
  }
  const g1 = new Float32Array(HIDDEN);
  matvec(g1, 0, pooled, 0, w.gdW0, w.gdB0, HIDDEN, HIDDEN);
  for (let o = 0; o < HIDDEN; o++) g1[o] = silu(g1[o]);
  const raw = new Float32Array(EMB);
  matvec(raw, 0, g1, 0, w.gdW4, w.gdB4, EMB, HIDDEN);
  let ss = 0;
  for (let o = 0; o < EMB; o++) ss += raw[o] * raw[o];
  const inv = 1 / Math.max(Math.sqrt(ss), 1e-12);
  const out = new Float32Array(EMB);
  for (let o = 0; o < EMB; o++) out[o] = raw[o] * inv;
  return out;
}

/** Coordinates -> 128-d embedding. */
export function embedCoords(w, coords) {
  return embedGraph(w, coordsToGraph(coords));
}

/** Cosine similarity mapped to 0 (far) .. 1 (close), matching progres.py. */
export function embeddingSimilarity(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return (1 + s) / 2;
}
