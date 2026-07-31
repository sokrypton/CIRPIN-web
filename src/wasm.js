// The SIMD forward pass, when the browser has one.
//
// src/cirpin.js stays the reference implementation and the fallback: it is the thing
// checked against PyTorch, it needs no toolchain to read, and it runs where wasm SIMD
// does not. This module is the fast path, measured at ~11.7x on a 332-residue chain
// (675 ms -> 57 ms), which is the difference between a search you wait for and one you
// do not.
//
// Two implementations of one network can drift, so test/wasm.mjs asserts they agree on
// real chains and is the gate on changing either.

// No hand-assembled feature probe.
//
// There was one, and its bytes were wrong: it reported "no SIMD" on a runtime that had
// just executed the module successfully. The real artifact is the only honest detector —
// if the engine cannot validate simd128, instantiating a module full of v128 ops fails,
// and that is exactly the question being asked. One less thing to get subtly wrong.

const HIDDEN = 128;
const EMB = 128;
const EH = 256;
const MD = 64;

/**
 * Load the module and upload the weights.
 *
 * CIRPIN and Progres are the same architecture down to every dimension — both are
 * 1,063,680 parameters through the same six layers — so one module and one arena serve
 * both, with a base pointer per model. Two instances would be two copies of the same
 * code and a second 48 MB reservation for nothing.
 *
 * @param {ArrayBuffer|Response|string} src - the .wasm, or a URL to it
 * @param {object[]} models - weights from loadWeights(), in the order callers will index
 * @returns {Promise<object|null>} an accelerator, or null if this runtime cannot run it
 *   — no SIMD, wasm disabled, or the module missing. Callers fall back to cirpin.js.
 */
export async function loadAccelerator(src, models) {
  let instance;
  try {
    if (typeof src !== 'string') {
      instance = (await WebAssembly.instantiate(src)).instance;
    } else {
      try {
        // The fast path, but it insists on Content-Type: application/wasm and throws
        // otherwise — which many static servers get wrong, including Python's
        // http.server on some versions.
        instance = (await WebAssembly.instantiateStreaming(fetch(src))).instance;
      } catch {
        const buf = await (await fetch(src)).arrayBuffer();
        instance = (await WebAssembly.instantiate(buf)).instance;
      }
    }
  } catch {
    // Missing module, no SIMD, wasm disabled by policy — all the same answer here. The
    // JS path computes the same thing, only slower, so this is not worth a message.
    return null;
  }

  const { memory, arena, arena_size: arenaSize, forward } = instance.exports;
  if (!forward || !arena) return null;

  const base = arena() >>> 2;          // float index, not byte offset
  const limit = base + (arenaSize() >>> 2);
  const f32 = () => new Float32Array(memory.buffer);
  const u32 = () => new Uint32Array(memory.buffer);

  // Exactly the order forward.rs reads them. Changing one without the other is the
  // failure this pairing invites, so both sides name the sequence in the same words.
  const packed = (w) => {
    const parts = [w.encW, w.encB];
    for (const L of w.layers) {
      parts.push(L.Wi, L.Wj, L.wd, L.eb0, L.ew3T, L.eb3, L.nw0, L.nb0, L.nw3, L.nb3);
    }
    parts.push(w.ndW0, w.ndB0, w.ndW3, w.ndB3, w.gdW0, w.gdB0, w.gdW4, w.gdB4);
    return parts;
  };

  let p = base;
  const M = f32();
  const wBase = [];
  for (const w of models) {
    wBase.push(p);
    for (const part of packed(w)) { M.set(part, p); p += part.length; }
  }
  const weightsEnd = p;

  return {
    /**
     * One embedding. Same contract as embedGraph(): a graph in, a 128-d unit vector out.
     *
     * onLayer is accepted and called once at the end rather than per layer — the whole
     * pass is now faster than the interval that progress reporting existed to fill, and
     * the alternative is six boundary crossings to animate a bar for 57 ms.
     *
     * nodeSink mirrors embedGraph's: an n*128 Float32Array that receives each residue's node_dec
     * vector. Without it here, anything wanting per-residue vectors had to use the JS path and pay
     * about 10x — 706 ms on a 382-residue chain against 57 ms accelerated.
     */
    embed(model, graph, onLayer, nodeSink) {
      const { n, x, nbrOffsets, nbrIndices, nbrDist2 } = graph;
      let q = weightsEnd;
      const xPtr = q; q += x.length;
      const offPtr = q; q += nbrOffsets.length;
      const idxPtr = q; q += nbrIndices.length;
      const d2Ptr = q; q += nbrDist2.length;
      const outPtr = q; q += EMB;
      const nodePtr = nodeSink ? q : 0;
      if (nodeSink) q += n * HIDDEN;
      const scratch = q;
      // what forward.rs carves out of scratch, in floats
      const need = 2 * n * HIDDEN + 2 * n * EH + 2 * EH + 4 * HIDDEN + 2 * MD
        + (HIDDEN + MD);
      if (scratch + need > limit) {
        // A chain long enough to overflow a 48 MB arena. Rather than grow memory
        // mid-flight, say so and let the caller use the JS path.
        throw new Error(`chain of ${n} residues needs more arena than the module has`);
      }

      const F = f32();
      const U = u32();
      F.set(x, xPtr);
      U.set(nbrOffsets, offPtr);
      U.set(nbrIndices, idxPtr);
      F.set(nbrDist2, d2Ptr);

      forward(wBase[model] * 4, n, xPtr * 4, offPtr * 4, idxPtr * 4, d2Ptr * 4,
        outPtr * 4, scratch * 4, nodePtr * 4);

      if (nodeSink) nodeSink.set(f32().subarray(nodePtr, nodePtr + n * HIDDEN));
      if (onLayer) onLayer(6, 6);
      // A copy, not a view: the next call reuses this memory, and callers keep results.
      return f32().slice(outPtr, outPtr + EMB);
    },
  };
}
