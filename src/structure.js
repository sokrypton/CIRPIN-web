// Structure parsing and CIRPIN graph construction.
//
// Ports progres.py: read_coords (:374-400) and coords_to_graph (:429-473).
// Cα-only, first chain, first model — matching the reference exactly.

// Modified amino acids are deposited as HETATM but are part of the polymer.
// Dropping them punches a hole in the chain: measured on 2IA9 and 1R2K, an
// ATOM-only parse leaves a 6.7-7.1 A gap between "consecutive" CA atoms where
// the real spacing is 3.9 A, which then feeds a bogus tau torsion and draws a
// phantom bond in the superposition view.
//
// The trap is that HETATM records often carry an atom named CA that is not a
// backbone carbon at all — a calcium ion is literally residue CA atom CA
// (seen in 1B1C), and ligands like SAH have one too (1XCJ). So membership in
// the polymer has to be established, never assumed:
//   mmCIF - label_seq_id is set only for polymer positions; ligands have '.'.
//   PDB   - MODRES records declare the substitution explicitly.
// A small dictionary of common modifications covers files that omit MODRES.
//
// Classification follows py2Dmol's isRealAminoAcid (web/utils.js:1082).
//
// NOTE: this deliberately diverges from progres.py, which is ATOM-only in the
// PDB path and uses Biopython's hetero flag (MSE is 'H_MSE') in the mmCIF
// path, so the released SCOPe40 embeddings were built without these residues.
// The measured cost of the divergence is a cosine of 0.9986 (2IA9) and 0.9995
// (1R2K) against an ATOM-only parse, with identical top-5 hits in both cases.

export const STANDARD_AMINO_ACIDS = new Set([
  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
]);

// Modified -> parent, for protein only (py2Dmol utils.js:1224).
// Modified amino acids that count as polymer residues, from the PDB Chemical Component Dictionary.
//
// This was an eight-entry hand table -- MSE, PTR, SEP, TPO, FME, HYP, PCA, ALY -- which covered
// selenomethionine and little else. About 4% of ECOD40's domains carry HETATM Ca atoms the old table
// dropped -- 16,892 residues recovered across the build -- and the residues involved run well past that
// list: MLY, CGU, CME, CSO, CSD, OCS, CSX, KCX. (An earlier draft of this comment said 22%, which came
// from a PDB-biased sample of the first 4,000 archive members, not the whole set.) Each one
// dropped is a gap in the chain, a bogus torsion angle across it, and a contact that should exist
// and does not -- so the embedding stops being the one the model was trained to produce.
//
// Imported AND re-exported, not `export ... from`: that form creates no local binding, so the two
// uses below would have been silent undefined-property failures at runtime while parsing fine.
// See web/tools/gen_modres.py for the provenance and for why the obvious distance test was not used.
import { MODIFIED_AMINO_ACIDS } from './modres.js?v=2c8b633b';

export { MODIFIED_AMINO_ACIDS };

export const CONTACT_DIST = 10.0;
export const POS_EMBED_DIM = 64;
export const POS_EMBED_FREQ_INV = 2000;
export const N_FEATURES = 68;

// --- parsing ----------------------------------------------------------------

// Fixed-column PDB. Follows progres.py:376-386 — CA only, first chain only,
// stop at ENDMDL — but additionally keeps HETATM residues declared as modified
// amino acids, either by a MODRES record or by the common-modification table.
export function parsePDB(text, opts = {}) {
  return firstChainOf(parsePDBChains(text), opts);
}

/**
 * Every chain in a PDB file, as chain id -> {coords, seq}.
 *
 * The reference takes the first chain and stops. This keeps going, because a
 * deposited entry is often a complex and the chain worth searching is not always
 * the one that happens to be written first — but which chain is used stays an
 * explicit choice rather than a heuristic, so parsePDB still returns the first.
 */
export function parsePDBChains(text) {
  const lines = text.split('\n');

  // MODRES: cols 13-15 modified name, 25-27 the standard residue it stands for.
  const modres = new Map();
  for (const line of lines) {
    if (line.startsWith('MODRES')) {
      const from = line.slice(12, 15).trim();
      const to = line.slice(24, 27).trim();
      if (from && STANDARD_AMINO_ACIDS.has(to)) modres.set(from, to);
    } else if (line.startsWith('ATOM  ') || line.startsWith('HETATM')) {
      break; // MODRES always precedes the coordinate section
    }
  }

  const out = new Map();
  for (const line of lines) {
    if (line.startsWith('ENDMDL')) break;    // first model only
    const isAtom = line.startsWith('ATOM  ');
    const isHet = line.startsWith('HETATM');
    if (!(isAtom || isHet) || line.slice(12, 16).trim() !== 'CA') continue;
    const resName = line.slice(17, 20).trim();
    if (isHet && !modres.has(resName) && !MODIFIED_AMINO_ACIDS.has(resName)) {
      continue; // a ligand that happens to carry an atom named CA
    }
    if (isAtom && !STANDARD_AMINO_ACIDS.has(resName)) continue;
    const ch = line[21];
    if (!out.has(ch)) out.set(ch, { coords: [], seq: [], last: null });
    const e = out.get(ch);
    const resSeq = line.slice(22, 27);        // number plus insertion code
    if (resSeq === e.last) continue;          // one CA per residue
    e.last = resSeq;
    e.coords.push([
      parseFloat(line.slice(30, 38)),
      parseFloat(line.slice(38, 46)),
      parseFloat(line.slice(46, 54)),
    ]);
    e.seq.push(parseInt(line.slice(22, 26), 10));
  }
  for (const e of out.values()) delete e.last;
  return out;
}

/** The first chain long enough to embed, with opts.seq filled in if asked. */
function firstChainOf(byChain, opts) {
  for (const e of byChain.values()) {
    if (e.coords.length < 4) continue;
    if (opts.seq) opts.seq.push(...e.seq);
    return e.coords;
  }
  const first = byChain.values().next().value;
  if (first && opts.seq) opts.seq.push(...first.seq);
  return first ? first.coords : [];
}

// mmCIF _atom_site loop. The reference uses Biopython's MMCIFParser, which
// yields first model / first chain / non-hetero residues / CA atoms; this
// reproduces that by filtering on group_PDB, model num, chain id and altloc.
/**
 * @param {string} text
 * @param {{seq?: number[]}} [opts] if opts.seq is given, each kept residue's
 *   number is appended to it, parallel to the returned coordinates. AlphaFold
 *   numbers residues by position in the UniProt sequence, which is what TED
 *   choppings are expressed in, so this is how a domain gets cut out of a model.
 */
export function parseCIF(text, opts = {}) {
  return firstChainOf(parseCIFChains(text), opts);
}

/** Every chain in an mmCIF file, as chain id -> {coords, seq}. */
export function parseCIFChains(text) {
  const lines = text.split('\n');
  let i = 0;
  let cols = null;
  let start = -1;

  while (i < lines.length) {
    if (/^\s*loop_/.test(lines[i])) {
      const names = [];
      let j = i + 1;
      while (j < lines.length && /^\s*_/.test(lines[j])) {
        names.push(lines[j].trim());
        j++;
      }
      if (names.length && names[0].startsWith('_atom_site.')) {
        cols = {};
        names.forEach((n, k) => { cols[n.trim().slice('_atom_site.'.length)] = k; });
        start = j;
        break;
      }
      i = j;
    } else {
      i++;
    }
  }
  if (!cols) return new Map();

  const need = ['Cartn_x', 'Cartn_y', 'Cartn_z'];
  for (const n of need) if (!(n in cols)) return new Map();
  const cAtom = cols.label_atom_id ?? cols.auth_atom_id;
  const cChain = cols.auth_asym_id ?? cols.label_asym_id;
  const cModel = cols.pdbx_PDB_model_num;
  const cGroup = cols.group_PDB;
  const cAlt = cols.label_alt_id ?? cols.auth_alt_id;
  const cSeq = cols.auth_seq_id ?? cols.label_seq_id;
  // An insertion code is part of a residue's identity, not decoration. Without it, 52 and 52A
  // are one residue and the second is dropped: thrombin's chymotrypsin numbering costs 1AHT
  // half of its light chain, 14 of 28, and antibodies lose 2-3% of every chain. The PDB path
  // has always keyed on columns 23-27, which is number and insertion code together; this is
  // the same identity expressed in mmCIF.
  const cIns = cols.pdbx_PDB_ins_code;
  const cLabelSeq = cols.label_seq_id;
  const cComp = cols.label_comp_id ?? cols.auth_comp_id;

  // _pdbx_struct_mod_residue declares this file's substitutions explicitly.
  const modResCif = parseModResCif(text);

  const out = new Map();
  let modelNum = null;

  for (let k = start; k < lines.length; k++) {
    const line = lines[k];
    if (!line.trim()) continue;
    if (/^\s*(#|loop_|data_|_)/.test(line)) break;

    const f = splitCIFRow(line);
    if (f.length < Object.keys(cols).length) continue;

    if (cAtom !== undefined && strip(f[cAtom]) !== 'CA') continue;
    if (cGroup !== undefined && f[cGroup] !== 'ATOM') {
      // Keep only HETATM that sits at a polymer position and is a known
      // modified amino acid. label_seq_id is '.' for every ligand, which is
      // what rejects the calcium ion and SAH cases.
      const inPolymer = cLabelSeq !== undefined
        && f[cLabelSeq] !== '.' && f[cLabelSeq] !== '?';
      const rn = cComp !== undefined ? strip(f[cComp]) : '';
      if (!inPolymer || !(MODIFIED_AMINO_ACIDS.has(rn) || modResCif.has(rn))) continue;
    }
    if (cAlt !== undefined && f[cAlt] !== '.' && f[cAlt] !== '?' && f[cAlt] !== 'A') continue;

    if (cModel !== undefined) {
      if (modelNum === null) modelNum = f[cModel];
      else if (f[cModel] !== modelNum) break; // first model only
    }

    const ch = cChain !== undefined ? strip(f[cChain]) : 'A';
    if (!out.has(ch)) out.set(ch, { coords: [], seq: [], last: null });
    const e = out.get(ch);
    if (cSeq !== undefined) {
      const key = cIns !== undefined ? `${f[cSeq]}|${f[cIns]}` : f[cSeq];
      if (key === e.last) continue; // one CA per residue
      e.last = key;
    }
    e.coords.push([
      parseFloat(f[cols.Cartn_x]),
      parseFloat(f[cols.Cartn_y]),
      parseFloat(f[cols.Cartn_z]),
    ]);
    e.seq.push(cSeq !== undefined ? parseInt(f[cSeq], 10) : e.coords.length);
  }
  for (const e of out.values()) delete e.last;
  return out;
}

// _pdbx_struct_mod_residue: modified component -> the standard residue it
// replaces. Present in most modern mmCIF files; the dictionary covers the rest.
function parseModResCif(text) {
  const out = new Map();
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*loop_/.test(lines[i])) continue;
    const names = [];
    let j = i + 1;
    while (j < lines.length && /^\s*_/.test(lines[j])) { names.push(lines[j].trim()); j++; }
    if (!names.length || !names[0].startsWith('_pdbx_struct_mod_residue.')) continue;
    const col = {};
    names.forEach((n, k) => { col[n.slice('_pdbx_struct_mod_residue.'.length)] = k; });
    for (; j < lines.length; j++) {
      if (!lines[j].trim()) continue;
      if (/^\s*(#|loop_|data_|_)/.test(lines[j])) break;
      const f = splitCIFRow(lines[j]);
      const from = col.label_comp_id !== undefined ? strip(f[col.label_comp_id]) : null;
      const to = col.parent_comp_id !== undefined ? strip(f[col.parent_comp_id]) : null;
      if (from && to && STANDARD_AMINO_ACIDS.has(to)) out.set(from, to);
    }
    break;
  }
  return out;
}

function strip(v) {
  return v && v.length > 1 && v[0] === '"' ? v.slice(1, -1) : v;
}

function splitCIFRow(line) {
  const out = [];
  let i = 0;
  while (i < line.length) {
    while (i < line.length && /\s/.test(line[i])) i++;
    if (i >= line.length) break;
    const q = line[i];
    if (q === '"' || q === "'") {
      let j = i + 1;
      while (j < line.length && line[j] !== q) j++;
      out.push(line.slice(i + 1, j));
      i = j + 1;
    } else {
      let j = i;
      while (j < line.length && !/\s/.test(line[j])) j++;
      out.push(line.slice(i, j));
      i = j;
    }
  }
  return out;
}

export function parseStructure(text, format = 'guess') {
  return guessFormat(text, format) === 'mmcif' ? parseCIF(text) : parsePDB(text);
}

function guessFormat(text, format) {
  if (format && format !== 'guess') return format;
  return /^\s*data_/m.test(text) || /_atom_site\./.test(text) ? 'mmcif' : 'pdb';
}

/** Every chain, whichever format this is. */
export function parseStructureChains(text, format = 'guess') {
  return guessFormat(text, format) === 'mmcif' ? parseCIFChains(text) : parsePDBChains(text);
}

// Whitespace-separated "x y z" per line (progres "coords" format).
export function parseCoordsTxt(text) {
  return text.split('\n')
    .filter((l) => l.trim())
    .map((l) => l.trim().split(/\s+/).map(Number));
}

// --- graph ------------------------------------------------------------------

/**
 * Build the CIRPIN input graph from Cα coordinates.
 *
 * The neighbour list deliberately INCLUDES the self index. In the reference,
 * EGNN.forward clears the diagonal of adj_mat but then sets ranking[self] =
 * -1, and nbhd_mask keeps everything with ranking <= 0 — so node i does
 * receive a self message (rel_dist 0, feats_j == feats_i). See
 * train_CIRPIN.py:292-176.
 *
 * The `num_nearest` top-k padding in the reference is a no-op: padded slots
 * are non-contacts with squared distance > 100, so nbhd_mask zeroes them
 * before the sum. A plain sparse neighbour list is therefore equivalent.
 *
 * @param {number[][]} coords - (n, 3) Cα coordinates
 * @returns {{n: number, coords: Float64Array, x: Float32Array,
 *            nbrOffsets: Int32Array, nbrIndices: Int32Array, nbrDist2: Float64Array}}
 */
export function coordsToGraph(coords) {
  const n = coords.length;
  if (n === 0) {
    throw new Error('no Cα coordinates found — check the file contains protein residues');
  }
  if (n < 4) {
    throw new Error(`need at least 4 Cα atoms, got ${n}`);
  }

  const xyz = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    if (coords[i].length !== 3) throw new Error('coords must be (n, 3)');
    xyz[i * 3] = coords[i][0];
    xyz[i * 3 + 1] = coords[i][1];
    xyz[i * 3 + 2] = coords[i][2];
  }

  // Contacts within CONTACT_DIST, self included (contacts matrix has a true
  // diagonal in the reference, and degrees counts it).
  const cut2 = CONTACT_DIST * CONTACT_DIST;
  const degrees = new Int32Array(n);
  const counts = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const xi = xyz[i * 3]; const yi = xyz[i * 3 + 1]; const zi = xyz[i * 3 + 2];
    for (let j = i; j < n; j++) {
      const dx = xi - xyz[j * 3];
      const dy = yi - xyz[j * 3 + 1];
      const dz = zi - xyz[j * 3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= cut2) {
        if (i === j) { counts[i]++; degrees[i]++; } else {
          counts[i]++; counts[j]++; degrees[i]++; degrees[j]++;
        }
      }
    }
  }

  const nbrOffsets = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) nbrOffsets[i + 1] = nbrOffsets[i] + counts[i];
  const nEdges = nbrOffsets[n];
  const nbrIndices = new Int32Array(nEdges);
  const nbrDist2 = new Float64Array(nEdges);
  const fill = nbrOffsets.slice(0, n);

  for (let i = 0; i < n; i++) {
    const xi = xyz[i * 3]; const yi = xyz[i * 3 + 1]; const zi = xyz[i * 3 + 2];
    for (let j = i; j < n; j++) {
      const dx = xi - xyz[j * 3];
      const dy = yi - xyz[j * 3 + 1];
      const dz = zi - xyz[j * 3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= cut2) {
        nbrIndices[fill[i]] = j; nbrDist2[fill[i]] = d2; fill[i]++;
        if (i !== j) { nbrIndices[fill[j]] = i; nbrDist2[fill[j]] = d2; fill[j]++; }
      }
    }
  }

  // Node features: [norm_degree, term_start, term_end, tau, pos_embed(64)]
  const x = new Float32Array(n * N_FEATURES);
  let maxDeg = 0;
  for (let i = 0; i < n; i++) if (degrees[i] > maxDeg) maxDeg = degrees[i];
  for (let i = 0; i < n; i++) x[i * N_FEATURES] = degrees[i] / maxDeg;
  x[0 * N_FEATURES + 1] = 1.0;
  x[(n - 1) * N_FEATURES + 2] = 1.0;

  // tau: torsion over 4 consecutive Cα, assigned to the second; padded
  // [0.0, taus/pi..., 0.0, 0.0] (progres.py:453-467).
  for (let i = 0; i + 3 < n; i++) {
    const ab = sub(xyz, i + 1, i);
    const bc = sub(xyz, i + 2, i + 1);
    const cd = sub(xyz, i + 3, i + 2);
    const abbc = cross(ab, bc);
    const bccd = cross(bc, cd);
    const nbc = norm3(bc); // F.normalize: divide by max(||v||, 1e-12)
    const y = dot3(cross(abbc, bccd), nbc);
    const xx = dot3(abbc, bccd);
    x[(i + 1) * N_FEATURES + 3] = Math.atan2(y, xx) / Math.PI;
  }

  // Sinusoidal positional encoding over 1..n (progres.py:63-72).
  const half = POS_EMBED_DIM / 2;
  const invFreq = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    invFreq[k] = 1.0 / (POS_EMBED_FREQ_INV ** ((2 * k) / POS_EMBED_DIM));
  }
  for (let i = 0; i < n; i++) {
    const base = i * N_FEATURES + 4;
    for (let k = 0; k < half; k++) {
      const a = (i + 1) * invFreq[k];
      x[base + k] = Math.sin(a);
      x[base + half + k] = Math.cos(a);
    }
  }

  return { n, coords: xyz, x, nbrOffsets, nbrIndices, nbrDist2, nEdges };
}

function sub(p, a, b) {
  return [p[a * 3] - p[b * 3], p[a * 3 + 1] - p[b * 3 + 1], p[a * 3 + 2] - p[b * 3 + 2]];
}
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(v) {
  const m = Math.max(Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-12);
  return [v[0] / m, v[1] / m, v[2] / m];
}
