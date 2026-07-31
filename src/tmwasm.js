// The TM-align wasm module, wrapped to look exactly like src/tmalign.js.
//
// Same call signatures, same returned shape, so worker.js can hold one or the other
// without knowing which. src/tmalign.js stays the reference: it is what the C++ parity
// test checks (1.1e-16), it needs no toolchain to read, and it runs where wasm does not.
//
// The speedup is modest and worth stating plainly: ~1.22x on tmAlign, ~1.34x on cpAlign,
// and slower than JS on some individual pairs. TM-align is branch-dense search code that
// V8 already compiles well, unlike the EGNN's dense matrix multiplication where the same
// exercise gave ~12x. test/tmalign_wasm.mjs is the gate on both.

const OUT_F64 = 22;   // TM1, TM2, rmsd0, d0A, d0B, t0[3], u0[9], cpPoint, cpAlnBest, lin[3]
const SEC_CODE = { H: 0, E: 1, T: 2, C: 3 };

/** Bytes ws_init consumes for one chain pair — mirrors its carve order. */
function wsBytes(xlen, ylen) {
  const minlen = Math.min(xlen, ylen);
  const cells = (xlen + 1) * (ylen + 1);
  return 8 * (48 + 2 * (ylen + 1))          // f64: four t/u pairs, two DP rows
    + 4 * (minlen * 3 * 4 + xlen * 3 + minlen + 3 + cells)  // f32: r1 r2 xtm ytm, xt, dis, xrot, score
    + 4 * (minlen * 2 + (ylen + 1) * 3 + minlen + 1)        // i32: kAli iAli, three invmaps, ifr
    + cells                                                  // u8: path
    + 256;                                                   // alignment slack
}

export async function loadTmAlign(src) {
  let instance;
  const imports = {
    // core has no libm without std. These are the host's, which is also what the JS path
    // calls, so the two implementations cannot disagree about a cosine.
    m: { atan2: Math.atan2, cos: Math.cos, sin: Math.sin, cbrt: Math.cbrt, pow: Math.pow },
  };
  try {
    if (typeof src !== 'string') {
      instance = (await WebAssembly.instantiate(src, imports)).instance;
    } else {
      try {
        instance = (await WebAssembly.instantiateStreaming(fetch(src), imports)).instance;
      } catch {
        const buf = await (await fetch(src)).arrayBuffer();
        instance = (await WebAssembly.instantiate(buf, imports)).instance;
      }
    }
  } catch {
    return null;
  }

  const { memory, arena, arena_size: arenaSize, tmalign_main: mainW, cp_align: cpW,
    make_sec_wasm: secW } = instance.exports;
  if (!mainW || !cpW) return null;
  const limit = arena() + arenaSize();
  const f32 = () => new Float32Array(memory.buffer);
  const f64 = () => new Float64Array(memory.buffer);
  const i32 = () => new Int32Array(memory.buffer);
  const u8 = () => new Uint8Array(memory.buffer);

  /** Lay out everything one alignment needs; throws if it will not fit. */
  function layout(xlen, ylen, dup) {
    const align8 = (b) => (b + 7) & ~7;
    let b = arena();
    const put4 = (n) => { b = align8(b); const at = b; b += n * 4; return at; };
    const put8 = (n) => { b = align8(b); const at = b; b += n * 8; return at; };
    const put1 = (n) => { const at = b; b += n; return at; };

    const L = {};
    L.x = put4(xlen * 3);
    L.y = put4(ylen * 3);
    L.secx = put1(xlen + 8);
    L.secy = put1(ylen + 8);
    if (dup) {
      L.xcp = put4(xlen * 2 * 3);
      L.xf = put4(xlen * 3);
      L.scp = put1(xlen * 2 + 8);
      L.x2y = put1(xlen * 2 + 8);
      L.sf = put1(xlen + 8);
    }
    L.outF = put8(OUT_F64 + 4);
    L.outI = put4(2 + (ylen + 1) + xlen + ylen + 16);
    if (dup) {
      L.outFb = put8(OUT_F64 + 4);
      // pass 1 aligns the doubled chain, so its m1 is 2*xlen long
      L.outIb = put4(2 + (ylen + 1) + 2 * xlen + ylen + 16);
    }
    b = align8(b);
    L.scratch = b; b += wsBytes(dup ? xlen * 2 : xlen, ylen);
    if (dup) { b = align8(b); L.scratch2 = b; b += wsBytes(xlen, ylen); }
    if (b > limit) {
      throw new Error(`alignment of ${xlen} vs ${ylen} needs more arena than the module has`);
    }
    return L;
  }

  const writeCoords = (at, a, n) => {
    const F = f32();
    const base = at >>> 2;
    for (let i = 0; i < n * 3; i++) F[base + i] = a[i];
  };
  const writeSec = (at, s, n) => {
    const U = u8();
    for (let i = 0; i < n; i++) U[at + i] = SEC_CODE[s[i]] ?? 3;
  };

  /** The common tail: pull one alignment's results out of the output arrays. */
  function harvest(L, xlen, ylen) {
    const F = f64();
    const I = i32();
    const of = L.outF >>> 3;
    const oi = L.outI >>> 2;
    const nAli8 = I[oi + 1];
    return {
      TM1: F[of], TM2: F[of + 1], rmsd0: F[of + 2], d0A: F[of + 3], d0B: F[of + 4],
      t0: F.slice(of + 5, of + 8),
      u0: F.slice(of + 8, of + 17),
      n_ali: I[oi], n_ali8: nAli8,
      invmap0: I.slice(oi + 2, oi + 2 + ylen + 1),
      m1: I.slice(oi + 2 + ylen + 1, oi + 2 + ylen + 1 + nAli8),
      m2: I.slice(oi + 2 + ylen + 1 + xlen, oi + 2 + ylen + 1 + xlen + nAli8),
    };
  }

  return {
    /** Secondary structure as the string the JS path produces. */
    makeSec(x, len) {
      const L = layout(len, 1, false);
      writeCoords(L.x, x, len);
      secW(L.x, len, L.secx);
      const U = u8();
      let out = '';
      for (let i = 0; i < len; i++) out += 'HETC'[U[L.secx + i]];
      return out;
    },

    tmAlign(xa, ya, xlen, ylen, opts = {}) {
      const L = layout(xlen, ylen, false);
      writeCoords(L.x, xa, xlen);
      writeCoords(L.y, ya, ylen);
      secW(L.x, xlen, L.secx);
      secW(L.y, ylen, L.secy);
      const rc = mainW(L.x, L.y, L.secx, L.secy, xlen, ylen, opts.fast ? 1 : 0,
        L.scratch, L.outF, L.outI);
      if (rc !== 0) throw new Error('There is no alignment between the two proteins');
      return harvest(L, xlen, ylen);
    },

    cpAlign(xa, ya, xlen, ylen, opts = {}) {
      const L = layout(xlen, ylen, true);
      writeCoords(L.x, xa, xlen);
      writeCoords(L.y, ya, ylen);
      secW(L.x, xlen, L.secx);
      secW(L.y, ylen, L.secy);
      const rc = cpW(L.x, L.y, L.secx, L.secy, xlen, ylen, opts.fast ? 1 : 0,
        L.xcp, L.scp, L.x2y, L.xf, L.sf,
        L.scratch, L.outF, L.outI, L.scratch2, L.outFb, L.outIb);
      if (rc !== 0) throw new Error('There is no alignment between the two proteins');
      const r = harvest(L, xlen, ylen);
      const F = f64();
      const of = L.outF >>> 3;
      return {
        ...r,
        cpPoint: F[of + 17],
        cpAlnBest: F[of + 18],
        linear: { TM1: F[of + 19], TM2: F[of + 20], n_ali8: F[of + 21] },
      };
    },
  };
}
