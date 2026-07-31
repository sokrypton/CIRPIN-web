// The easter egg: click the hero and the ouroboros stops eating itself long enough to play.
//
// It is the same animal — the palette, the taper, the head and the jaws are imported from
// ouroboros.js, so this cannot drift into looking like a different snake. What it adds is a grid,
// a direction you choose, and something to eat: a domain, which is what the rest of the page is
// about. Eat one and the chain grows by a residue.
//
// Two ways to steer, because the page is used on a phone as well as a laptop. Keys are the arrows
// or WASD. A click or a tap turns toward wherever it landed, which is the control that works
// without a keyboard and is also the more forgiving of the two: you point at where you want to go
// rather than counting turns.
//
// It never blocks the page. Escape closes it, so does the button, and closing puts the hero back.

import { PAPER, SEG, toward, hue, girth, drawHead } from './ouroboros.js?v=98dcea2c';

// 14, not 15, and it is not arbitrary: one cell then equals the hero's segment pitch, which is its
// circumference over 30 segments — 2*pi*0.34*S/30 against S/14 agrees to 0.3% at any canvas size.
// So a segment is literally the same size before and after the transformation, and the handover has
// nothing to reconcile.
const CELLS = 14;
const START_LEN = 4;
const TAPER_CELLS = 3;             // how many cells the tail takes to reach full thickness
const SAMPLES = 5;                 // points along each segment; enough to round a corner
// Short, because the point is that the animal comes alive rather than that it performs a
// transition. Long enough to see the ring shorten and straighten, over before it feels staged.
const INTRO_MS = 520;
const TICK_MS = 150;               // at score 0; it quickens as the chain grows
const TICK_MIN = 72;
const RULE_2 = '#dde3ea';          // and the fainter one, for the grid
const CUT = '#d6006e';             // the accent the page uses for a number that matters


// --- what there is to eat -------------------------------------------------
//
// The food is an amino acid side chain: the R group, drawn from CB outward, which is the part that
// tells one residue from another. Glycine has none and alanine's is a lone CB, so both are left out
// — a dot is not worth recognising.
//
// The geometry is built rather than typed. Skeletal drawings are a zig-zag of unit bonds with
// regular polygons hung off them, so two helpers put every atom in the right place and there are no
// hand-computed coordinates to get subtly wrong.

/**
 * Bond geometry, so a drawing looks like chemistry rather than like lines.
 *
 * Every vertex is trigonal or tetrahedral, which on paper means 120 degrees between bonds. The first
 * version built forks as a pair of bonds at plus and minus 30 degrees from horizontal — right for the
 * zig-zag of a chain, but only 60 degrees apart at a branch point, so every carboxylate, every amide
 * and every methyl fork came out as a narrow V.
 *
 * The chain enters CB from an alpha carbon that is not drawn. It is treated as sitting below and to
 * the left, so CB's own bonds open at the same angles as everything downstream and the fragment does
 * not start with a joint that is wrong.
 */
const CA_IN = [0.866, 0.5];              // direction the backbone arrives at CB from

/** Rotate a direction by n degrees. */
function turn(d, deg) {
  const a = Math.atan2(d[1], d[0]) + (deg * Math.PI) / 180;
  return [Math.cos(a), Math.sin(a)];
}

/** One bond on from `at`, continuing a chain that arrived along `dir`; returns [point, dir]. */
function bondTo(at, dir, deg) {
  const d = turn(dir, deg);
  return [[at[0] + d[0], at[1] + d[1]], d];
}

/**
 * A zig-zag of n atoms starting at CB, alternating -60 and +60 about the incoming direction.
 *
 * Returns the atoms and the direction the last bond arrived along, which is what a branch needs.
 */
function chainOf(n) {
  const pts = [[0, 0]];
  let dir = CA_IN;
  let sign = -1;
  for (let i = 1; i < n; i++) {
    const [p, d] = bondTo(pts[i - 1], dir, sign * 60);
    pts.push(p);
    dir = d;
    sign = -sign;
  }
  return { pts, dir };
}

/** Both substituents of a trigonal centre: 120 degrees apart, 120 from the bond that arrived. */
function fork(at, dir) {
  return [bondTo(at, dir, -60)[0], bondTo(at, dir, 60)[0]];
}


/**
 * A regular n-gon hanging off a stem: all n vertices, starting at `vertex`.
 *
 * `vertex` is a ring atom and `stem` is the atom bonded to it from outside the ring, so the ring
 * grows directly away from the stem. Treating stem-vertex as a ring EDGE instead left a five-ring an
 * atom short, with bonds pointing at atoms that did not exist.
 */
function polyAt(n, stem, vertex) {
  const R = 0.5 / Math.sin(Math.PI / n);
  let dx = vertex[0] - stem[0];
  let dy = vertex[1] - stem[1];
  const dl = Math.hypot(dx, dy) || 1;
  dx /= dl; dy /= dl;
  const c = [vertex[0] + R * dx, vertex[1] + R * dy];
  const a0 = Math.atan2(vertex[1] - c[1], vertex[0] - c[0]);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = a0 + (2 * Math.PI / n) * i;
    out.push([c[0] + R * Math.cos(t), c[1] + R * Math.sin(t)]);
  }
  return out;
}

/** The remaining n-2 vertices of a regular n-gon fused onto the existing bond a-b, away from `away`. */
function polyOn(n, a, b, away) {
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const R = 0.5 / Math.sin(Math.PI / n);
  const apo = Math.sqrt(Math.max(0, R * R - 0.25));
  let px = -(b[1] - a[1]);
  let py = b[0] - a[0];
  const pl = Math.hypot(px, py) || 1;
  px /= pl; py /= pl;
  if ((mid[0] - away[0]) * px + (mid[1] - away[1]) * py < 0) { px = -px; py = -py; }
  const c = [mid[0] + apo * px, mid[1] + apo * py];
  const ang = (p) => Math.atan2(p[1] - c[1], p[0] - c[0]);
  const step6 = 2 * Math.PI / n;
  // round the ring in whichever direction leads away from a
  const test = [c[0] + R * Math.cos(ang(b) + step6), c[1] + R * Math.sin(ang(b) + step6)];
  const dir = Math.hypot(test[0] - a[0], test[1] - a[1]) > 0.5 ? 1 : -1;
  const out = [];
  for (let i = 1; i <= n - 2; i++) {
    const t = ang(b) + dir * step6 * i;
    out.push([c[0] + R * Math.cos(t), c[1] + R * Math.sin(t)]);
  }
  return out;
}

/**
 * The eighteen side chains worth drawing: atoms (position, element where not carbon) and bonds,
 * with 2 for a double bond. Glycine has no side chain and alanine's is a lone CB.
 */
const SIDE_CHAINS = (() => {
  const mk = (code, name, atoms, bonds) => ({ code, name, atoms, bonds });
  const out = [];
  const A = (p, el = '') => [p[0], p[1], el];

  // one heteroatom on CB
  for (const [code, name, el] of [['S', 'serine', 'O'], ['C', 'cysteine', 'S']]) {
    const { pts, dir } = chainOf(2);
    out.push(mk(code, name, [A(pts[0]), A(pts[1], el)], [[0, 1, 1]]));
  }
  // two substituents on CB, 120 degrees apart
  for (const [code, name, el] of [['T', 'threonine', 'O'], ['V', 'valine', '']]) {
    const [p1, p2] = fork([0, 0], CA_IN);
    out.push(mk(code, name, [A([0, 0]), A(p1, el), A(p2)], [[0, 1, 1], [0, 2, 1]]));
  }
  {
    const { pts, dir } = chainOf(2);              // CB-CG
    const [d1, d2] = fork(pts[1], dir);           // CD1, CD2 off CG
    out.push(mk('L', 'leucine', [A(pts[0]), A(pts[1]), A(d1), A(d2)],
      [[0, 1, 1], [1, 2, 1], [1, 3, 1]]));
  }
  {
    const [g1, g2] = fork([0, 0], CA_IN);         // CG1, CG2 off CB
    // CD1 continues from CG1, turning the other way so it does not fold back on CB
    const [d1] = bondTo(g1, [g1[0], g1[1]], 60);
    out.push(mk('I', 'isoleucine', [A([0, 0]), A(g1), A(g2), A(d1)],
      [[0, 1, 1], [0, 2, 1], [1, 3, 1]]));
  }
  {
    const { pts } = chainOf(4);
    out.push(mk('M', 'methionine', [A(pts[0]), A(pts[1]), A(pts[2], 'S'), A(pts[3])],
      [[0, 1, 1], [1, 2, 1], [2, 3, 1]]));
  }
  {
    const { pts } = chainOf(3);
    out.push(mk('P', 'proline', pts.map((p) => A(p)), [[0, 1, 1], [1, 2, 1]]));
  }
  // carboxylates and amides: a trigonal tip, both heteroatoms 120 degrees from the chain
  const tip = (code, name, len, second) => {
    const { pts, dir } = chainOf(len);
    const [o1, o2] = fork(pts[len - 1], dir);
    const atoms = [...pts.map((p) => A(p)), A(o1, 'O'), A(o2, second)];
    const bonds = [];
    for (let i = 0; i + 1 < len; i++) bonds.push([i, i + 1, 1]);
    bonds.push([len - 1, len, 2], [len - 1, len + 1, 1]);
    return mk(code, name, atoms, bonds);
  };
  out.push(tip('D', 'aspartate', 2, 'O'));
  out.push(tip('N', 'asparagine', 2, 'N'));
  out.push(tip('E', 'glutamate', 3, 'O'));
  out.push(tip('Q', 'glutamine', 3, 'N'));
  {
    const { pts } = chainOf(5);
    out.push(mk('K', 'lysine', [...pts.slice(0, 4).map((p) => A(p)), A(pts[4], 'N')],
      [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]]));
  }
  {
    const { pts, dir } = chainOf(5);              // CB CG CD NE CZ
    const [h1, h2] = fork(pts[4], dir);           // the guanidinium nitrogens
    out.push(mk('R', 'arginine',
      [A(pts[0]), A(pts[1]), A(pts[2]), A(pts[3], 'N'), A(pts[4]), A(h1, 'N'), A(h2, 'N')],
      [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 2], [4, 6, 1]]));
  }
  {
    const { pts } = chainOf(2);
    const [cg, nd1, ce1, ne2, cd2] = polyAt(5, pts[0], pts[1]);
    out.push(mk('H', 'histidine',
      [A(pts[0]), A(cg), A(nd1, 'N'), A(ce1), A(ne2, 'N'), A(cd2)],
      [[0, 1, 1], [1, 2, 1], [2, 3, 2], [3, 4, 1], [4, 5, 2], [5, 1, 1]]));
  }
  {
    const { pts } = chainOf(2);
    const r6 = polyAt(6, pts[0], pts[1]);
    const atoms = [A(pts[0]), ...r6.map((p) => A(p))];
    const b = [[0, 1, 1], [1, 2, 2], [2, 3, 1], [3, 4, 2], [4, 5, 1], [5, 6, 2], [6, 1, 1]];
    out.push(mk('F', 'phenylalanine', atoms, b));
    const cz = r6[3];
    const dx = cz[0] - r6[0][0];
    const dy = cz[1] - r6[0][1];
    const dl = Math.hypot(dx, dy) || 1;
    out.push(mk('Y', 'tyrosine',
      [...atoms, A([cz[0] + dx / dl, cz[1] + dy / dl], 'O')], [...b, [4, atoms.length, 1]]));
  }
  {
    const { pts } = chainOf(2);
    const [cg, cd1, ne1, ce2, cd2] = polyAt(5, pts[0], pts[1]);
    const r6 = polyOn(6, ce2, cd2, cg);
    out.push(mk('W', 'tryptophan',
      [A(pts[0]), A(cg), A(cd1), A(ne1, 'N'), A(ce2), A(cd2), ...r6.map((p) => A(p))],
      [[0, 1, 1], [1, 2, 2], [2, 3, 1], [3, 4, 1], [4, 5, 2], [5, 1, 1],
        [5, 6, 1], [6, 7, 2], [7, 8, 1], [8, 9, 2], [9, 4, 1]]));
  }
  return out;
})();

/** Every side chain drawn on a unit scale, so one bond is one length whichever residue it is. */
function sideChainExtent(sc) {
  let lo = [Infinity, Infinity];
  let hi = [-Infinity, -Infinity];
  for (const [x, y] of sc.atoms) {
    lo = [Math.min(lo[0], x), Math.min(lo[1], y)];
    hi = [Math.max(hi[0], x), Math.max(hi[1], y)];
  }
  return { lo, hi, w: hi[0] - lo[0], h: hi[1] - lo[1] };
}

/**
 * Draw one side chain centred in a box.
 *
 * Bonds as lines, heteroatoms as their letter on a knocked-out disc — the convention every chemical
 * drawing uses, where carbon is a vertex and everything else is spelled.
 */
function drawSideChain(g, sc, cx, cy, size, colour) {
  const e = sideChainExtent(sc);
  // Each glyph fills its box, rather than every bond being drawn the same length. Holding the bond
  // length fixed is the chemically honest choice and it looked wrong here: serine is one bond and
  // tryptophan is ten, so on a shared scale the small ones were specks and the big ones crowded the
  // cell. Filling the box makes them read as a set of marks of one size, which is what a thing to
  // eat needs to be. The floor only guards a degenerate extent.
  const s = size / Math.max(e.w, e.h, 0.6);
  const ox = cx - s * (e.lo[0] + e.hi[0]) / 2;
  const oy = cy + s * (e.lo[1] + e.hi[1]) / 2;
  const at = (i) => [ox + s * sc.atoms[i][0], oy - s * sc.atoms[i][1]];
  g.lineCap = 'round';
  g.lineJoin = 'round';
  g.strokeStyle = colour;
  // Weight comes from the BOX, not from the bond length. Tying it to the bond made serine — one bond
  // stretched across the whole box — a fat bar next to tryptophan's fine mesh, when the point of
  // filling the box was that they read as one set of marks.
  g.lineWidth = Math.max(1.2, size * 0.075);
  for (const [i, j, order] of sc.bonds) {
    const p = at(i);
    const q = at(j);
    if (order === 2) {
      // offset pair, the shorter line inside — a double bond has to read as one bond, not two
      let nx = -(q[1] - p[1]);
      let ny = q[0] - p[0];
      const nl = Math.hypot(nx, ny) || 1;
      const off = Math.min(s * 0.14, size * 0.055);
      nx = (nx / nl) * off; ny = (ny / nl) * off;
      g.beginPath();
      g.moveTo(p[0] + nx, p[1] + ny); g.lineTo(q[0] + nx, q[1] + ny);
      g.moveTo(p[0] - nx, p[1] - ny); g.lineTo(q[0] - nx, q[1] - ny);
      g.stroke();
    } else {
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    }
  }
  // Bonds only, no element letters.
  //
  // A letter at this size is a smudge, and the reason to draw a side chain here is the shape of it —
  // a ring, a fork, a long tail. Which residue it was is in the readout, in words, where a word
  // works. The element is still in the table above: it is what the molecule IS, and something later
  // may want to say so.
}

/**
 * The best step from here: toward the food if that is safe, otherwise wherever survives longest.
 *
 * Exported and pure so it can be played headless -- a policy that claims to be optimal has to be
 * shown surviving, not asserted, and the interesting failure is the one where it walks into a pocket
 * it cannot get out of several moves later.
 *
 * Shortest path first, by breadth-first search over the free cells. But the shortest path to the food
 * is not automatically a good move: eating can seal the snake into a region smaller than its own
 * length, which is death a hundred moves later with nothing on screen to warn you. So a candidate
 * first step is only taken if, after taking it, the space still reachable from the new head is at
 * least as large as the snake -- the standard test, and the one thing that separates a snake that
 * plays well from one that eats twice and strangles itself.
 *
 * @param {number[][]} snake head-first list of [x, y]
 * @param {number[]} food
 * @param {number} cells grid size
 * @param {number[]} dir the direction it is travelling, since reversing is not a legal move
 * @returns {number[]|null} the step to take, or null if nothing survives
 */
export function bestMove(snake, food, cells, dir) {
  const key = (x, y) => y * cells + x;
  const body = new Set(snake.map(([x, y]) => key(x, y)));
  const head = snake[0];
  const STEPS = [[0, -1], [0, 1], [-1, 0], [1, 0]];

  const free = (x, y, occupied) => x >= 0 && y >= 0 && x < cells && y < cells
    && !occupied.has(key(x, y));

  /** How many cells are reachable from (x, y), given what is occupied. Flood fill. */
  const room = (x, y, occupied) => {
    const seen = new Set([key(x, y)]);
    const stack = [[x, y]];
    let n = 0;
    while (stack.length) {
      const [cx, cy] = stack.pop();
      n++;
      for (const [dx, dy] of STEPS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!free(nx, ny, occupied) || seen.has(key(nx, ny))) continue;
        seen.add(key(nx, ny));
        stack.push([nx, ny]);
      }
    }
    return n;
  };

  /** Distance from every free cell to the food, by breadth-first search back from the food. */
  const dist = new Map([[key(food[0], food[1]), 0]]);
  const q = [food];
  for (let at = 0; at < q.length; at++) {
    const [cx, cy] = q[at];
    for (const [dx, dy] of STEPS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (!free(nx, ny, body) || dist.has(key(nx, ny))) continue;
      dist.set(key(nx, ny), dist.get(key(cx, cy)) + 1);
      q.push([nx, ny]);
    }
  }

  let best = null;
  for (const [dx, dy] of STEPS) {
    // Reversing into itself is not a move the game allows, so it is not a move to consider.
    if (dir && dx === -dir[0] && dy === -dir[1]) continue;
    const nx = head[0] + dx;
    const ny = head[1] + dy;
    if (nx < 0 || ny < 0 || nx >= cells || ny >= cells) continue;
    // The tail cell is about to be vacated, so stepping onto it is legal -- unless this step eats,
    // in which case the tail stays and it is not.
    const eats = nx === food[0] && ny === food[1];
    const occupied = new Set(body);
    if (!eats) occupied.delete(key(...snake[snake.length - 1]));
    if (occupied.has(key(nx, ny))) continue;

    const after = new Set(occupied);
    after.add(key(nx, ny));
    const space = room(nx, ny, after);
    const need = snake.length + (eats ? 1 : 0);
    const d = dist.has(key(nx, ny)) ? dist.get(key(nx, ny)) : Infinity;
    // Sort by: does it leave room to live, then how close it gets to the food, then how much room.
    const score = [space >= need ? 1 : 0, -Math.min(d, 1e6), space];
    if (!best || score[0] > best.score[0]
      || (score[0] === best.score[0] && score[1] > best.score[1])
      || (score[0] === best.score[0] && score[1] === best.score[1] && score[2] > best.score[2])) {
      best = { step: [dx, dy], score };
    }
  }
  return best ? best.step : null;
}

const DIRS = {
  ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0],
  w: [0, -1], s: [0, 1], a: [-1, 0], d: [1, 0],
  W: [0, -1], S: [0, 1], A: [-1, 0], D: [1, 0],
};

/**
 * Open the game over the page.
 *
 * @param {{onClose?: () => void}} [opts] onClose runs once the overlay is gone, which is where the
 *   caller puts the hero back.
 * @returns {{close: () => void}}
 */
export function snakeGame(opts = {}) {
  const reduced = !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  // A fixed point in the transformation, for testing — the same hook and the same reason as the
  // hero's: requestAnimationFrame does not run in a tab that is not being rendered, so without it
  // the only stage of the intro anyone can observe is whichever one a frame happens to land on.
  // Undefined in normal use.
  const fixedIntro = opts.introAt;

  // --- the overlay ----------------------------------------------------------
  const wrap = document.createElement('div');
  wrap.className = 'egg';
  wrap.setAttribute('role', 'dialog');
  wrap.setAttribute('aria-label', 'Snake');
  wrap.innerHTML = `
    <div class="egg-veil"></div>
    <div class="egg-box">
      <canvas class="egg-board" aria-hidden="true"></canvas>
      <p class="egg-score"></p>
      <p class="egg-help"></p>
      <button type="button" class="egg-close">CLOSE</button>
    </div>`;
  document.body.appendChild(wrap);
  const canvas = wrap.querySelector('.egg-board');
  const scoreEl = wrap.querySelector('.egg-score');
  // A phone has no arrow keys and does not click. Testing for a coarse pointer with no hover rather
  // than for a touch API, because that is the question being asked: what will the reader actually do
  // with their hands.
  const coarse = !!window.matchMedia?.('(hover: none) and (pointer: coarse)').matches;
  wrap.querySelector('.egg-help').textContent = coarse
    ? 'Tap to turn · swipe to steer'
    : 'Click to turn · arrows to steer';

  const box = wrap.querySelector('.egg-box');
  let side = 0;

  /**
   * Size the board and place the box, from the viewport as it is right now.
   *
   * The board matches the hero it grew out of: the caller passes the size the hero was rendered at
   * and where on the page it was, so the ring the game opens with sits exactly on top of the ring
   * that was just clicked, and nothing moves at the handover.
   *
   * Called again on resize and on rotation, which it did not used to be. The size was computed once
   * at open, so turning a phone from portrait to landscape left a board sized for the old viewport:
   * on a 390x844 handset that is a 358px board in 374px of height once the furniture is counted, and
   * the CLOSE button ends up below the fold with no way to reach it.
   *
   * The height left for the board is MEASURED rather than assumed. It used to subtract a constant
   * 190px for the score, the hint and the button, which is both wrong and unfalsifiable — the
   * furniture is about 86px, so landscape was clamped to a 200px board for no reason. Measuring it
   * means the board is as large as the space genuinely allows, and stays right if the furniture ever
   * changes.
   */
  function layout() {
    const cap = opts.side || 440;
    const room = window.innerHeight - 16;
    side = Math.min(cap, window.innerWidth - 32);
    canvas.style.width = `${side}px`;
    canvas.style.height = `${side}px`;

    // Shrink until it fits, rather than computing the answer once.
    //
    // The furniture's height depends on the board's width, because the hint wraps: measuring the
    // furniture at 440px and then narrowing the board to 260 for a landscape phone gave a hint on two
    // lines and a box 26px taller than the measurement it was derived from -- which put the CLOSE
    // button off the bottom on every landscape handset while the arithmetic looked right. Since the
    // measurement depends on the result, iterate to the fixed point. Three passes is ample; the loop
    // is bounded because a wrap can only add height, never remove it.
    for (let pass = 0; pass < 4; pass++) {
      const over = box.offsetHeight - room;
      if (over <= 0) break;
      side = Math.max(150, side - Math.ceil(over));
      canvas.style.width = `${side}px`;
      canvas.style.height = `${side}px`;
    }
    const furniture = Math.max(0, box.offsetHeight - canvas.offsetHeight);

    // Anchored to the hero if that still fits on screen, centred if it does not. Centring is what
    // the stylesheet does by itself, so it is a matter of clearing the offsets rather than computing
    // any: .egg is a centring grid and .egg-box is its only child.
    const total = side + furniture;
    if (opts.anchor && total + 16 <= window.innerHeight) {
      const left = opts.anchor.left + opts.anchor.width / 2 - side / 2 - 2;
      box.style.left = `${Math.round(Math.max(8, Math.min(left, window.innerWidth - side - 8)))}px`;
      box.style.top = `${Math.round(Math.max(8, Math.min(opts.anchor.top,
        window.innerHeight - total - 8)))}px`;
    } else {
      box.style.left = '';
      box.style.top = '';
    }
  }
  layout();

  // --- state ----------------------------------------------------------------
  let snake, dir, queued, food, score, dead, started, lastTick, raf = 0, closed = false;
  // Set on the first frame, not here: the transformation is timed off the animation clock, and the
  // first frame is when that clock is first known. It is never cleared, so a retry after a death
  // starts playing immediately instead of replaying the intro.
  let introStart = null;
  let ate = null;                  // the last side chain eaten, for the readout

  const centreCell = () => [Math.floor(CELLS / 2), Math.floor(CELLS / 2)];

  function reset() {
    const [cx, cy] = centreCell();
    snake = [];
    for (let i = 0; i < START_LEN; i++) snake.push([cx - i, cy]);   // head first, tail behind
    // A drag says which way to go, so the animal leaves in the direction it was pulled.
    dir = opts.dir && (opts.dir[0] || opts.dir[1]) ? opts.dir : [1, 0];
    queued = null;
    score = 0;
    dead = false;
    started = false;
    ate = null;
    lastTick = 0;
    placeFood();
  }

  function placeFood() {
    const free = [];
    for (let x = 0; x < CELLS; x++) {
      for (let y = 0; y < CELLS; y++) {
        if (!snake.some(([sx, sy]) => sx === x && sy === y)) free.push([x, y]);
      }
    }
    // A full board is a win, and there is nowhere left to put anything.
    if (!free.length) { food = null; return; }
    const [x, y] = free[Math.floor(Math.random() * free.length)];
    const sc = SIDE_CHAINS[Math.floor(Math.random() * SIDE_CHAINS.length)];
    food = [x, y, sc];
  }

  /**
   * A turn is accepted unless it reverses the chain onto itself.
   *
   * Starting is separate: pressing the direction it is already going is not a turn, but it is
   * certainly an attempt to play, and the first version sat there refusing to begin because the
   * reversal guard swallowed it.
   */
  function steer(d) {
    if (!d) return;
    const ref = queued || dir;
    if (d[0] === -ref[0] && d[1] === -ref[1]) return;
    if (d[0] === ref[0] && d[1] === ref[1]) return;
    queued = d;
  }

  function step() {
    if (queued) { dir = queued; queued = null; }
    const [hx, hy] = snake[0];
    const nx = hx + dir[0];
    const ny = hy + dir[1];
    // A wall or its own body ends it. The tail cell is about to move, but treating it as solid is
    // the kinder rule: following your own tail at full speed should not be a death.
    const body = snake.slice(0, -1);
    if (nx < 0 || ny < 0 || nx >= CELLS || ny >= CELLS
        || body.some(([sx, sy]) => sx === nx && sy === ny)) {
      dead = true;
      return;
    }
    snake.unshift([nx, ny]);
    if (food && nx === food[0] && ny === food[1]) {
      score++;
      ate = food[2];               // what it was, so the readout can name it
      placeFood();                 // grew: the tail stays where it is
    } else {
      snake.pop();
    }
  }

  // --- drawing --------------------------------------------------------------

  /**
   * One chain, drawn from a centreline.
   *
   * Both the intro and the game go through here, which is the point: the transforming ring and the
   * snake on the grid are the same animal drawn by the same code, so they cannot look like two
   * different things joined by a cut.
   *
   * `pieces` is an array of runs of points, head run first. Each run is stroked as one path — that
   * is what makes a segment — and the gaps between runs are cut afterwards rather than left, for
   * the reason the hero documents: round caps overshoot by half a width, so any gap left between
   * runs closes itself.
   */
  function drawChain(g, pieces, cell) {
    const n = pieces.length;
    const trace = (run) => {
      g.beginPath();
      g.moveTo(run[0].x, run[0].y);
      for (let i = 1; i < run.length; i++) g.lineTo(run[i].x, run[i].y);
    };
    // girth() does all of its tapering within the first 0.34 of the chain, so a count of segments
    // has to be mapped onto THAT span — feeding it a plain fraction lands past the end of the taper
    // and comes back 0.97, which is a thread, a bulge and then the head with nothing in between.
    // The pitch here is one cell, so SEG's fractions apply directly and the segments come out the
    // same shape as the hero's — that is the whole reason those numbers are shared rather than
    // tuned twice in two files.
    const widthAt = (i) => cell * SEG.width
      * girth(0.34 * Math.min(1, (n - 1 - i) / TAPER_CELLS));

    g.lineCap = 'round';
    g.lineJoin = 'round';
    for (let i = 1; i < n; i++) {
      const t = n > 1 ? 1 - i / (n - 1) : 1;         // 1 at the neck, 0 at the tail tip
      g.strokeStyle = toward(hue(t), 1);
      g.lineWidth = widthAt(i);
      trace(pieces[i]);
      g.stroke();
    }

    // the gaps, cut across the body where one run hands over to the next
    g.strokeStyle = toward(PAPER, 1);
    g.lineWidth = cell * SEG.gap;
    for (let i = 1; i < n - 1; i++) {
      const run = pieces[i];
      const end = run[run.length - 1];
      const before = run[run.length - 2] || end;
      const dx = end.x - before.x;
      const dy = end.y - before.y;
      const len = Math.hypot(dx, dy) || 1;
      const across = (Math.max(widthAt(i), widthAt(i + 1)) + cell * 0.1) / 2;
      g.beginPath();
      g.moveTo(end.x + (dy / len) * across, end.y - (dx / len) * across);
      g.lineTo(end.x - (dy / len) * across, end.y + (dx / len) * across);
      g.stroke();
    }
  }

  /**
   * The snake on the grid, as one run of points a cell.
   *
   * A run goes from the midpoint of the cell it came from to the midpoint of the cell it leaves by,
   * curving through the cell centre — a quadratic with the centre as its control point. Straight
   * stretches are unaffected; a turn becomes a quarter round. Drawing each cell as its own straight
   * capsule instead left two capsules meeting at a right angle with a gap between them, which read
   * as a snake coming apart at every corner.
   */
  function gridPieces(mid) {
    const P = snake.map(([x, y]) => ({ x: mid(x), y: mid(y) }));
    const out = [];
    for (let i = 0; i < P.length; i++) {
      const cur = P[i];
      const prev = P[i - 1] || cur;
      const next = P[i + 1] || cur;
      const a = { x: (prev.x + cur.x) / 2, y: (prev.y + cur.y) / 2 };
      const b = { x: (cur.x + next.x) / 2, y: (cur.y + next.y) / 2 };
      const run = [];
      for (let s = 0; s <= SAMPLES; s++) {
        const u = s / SAMPLES;
        const m = 1 - u;
        run.push({
          x: m * m * a.x + 2 * m * u * cur.x + u * u * b.x,
          y: m * m * a.y + 2 * m * u * cur.y + u * u * b.y,
        });
      }
      out.push(run);
    }
    return out;
  }

  /**
   * The transformation: the hero's ring becoming the snake on the board.
   *
   * Two things happen at once, and both are just interpolation. The centreline is a blend between
   * where a residue sits on the ring and where it sits on a straight chain at the middle of the
   * board, so the circle straightens; and the chain's LENGTH runs down from a full lap to the four
   * cells the game starts with, so it shortens as it straightens. The ring starts at the phase the
   * hero was on when it was clicked, so the animal carries on from where it was rather than
   * jumping.
   */
  function introPieces(p, cell, w) {
    const ease = p * p * (3 - 2 * p);
    const ringR = 0.34 * w;                        // the hero's proportion of its canvas
    const ringLen = 2 * Math.PI * ringR;
    const gridLen = (START_LEN - 1) * cell;
    const len = ringLen + (gridLen - ringLen) * ease;
    const theta0 = (opts.fromPhase ?? 0) * Math.PI * 2;
    const [hx, hy] = snake[0];
    const headX = (hx + 0.5) * cell;
    const headY = (hy + 0.5) * cell;

    // one point every arc distance, then cut into runs a cell long so the segments match the game's
    const perRun = SAMPLES;
    const runs = Math.max(2, Math.round(len / cell));
    const at = (d) => {
      const ang = theta0 - d / ringR;              // backward along the ring from the head
      const rx = w / 2 + Math.cos(ang) * ringR;
      const ry = w / 2 - Math.sin(ang) * ringR;
      const lx = headX - d;                        // backward along a straight chain
      const ly = headY;
      return { x: rx + (lx - rx) * ease, y: ry + (ly - ry) * ease };
    };
    const out = [];
    for (let r = 0; r < runs; r++) {
      const run = [];
      for (let s = 0; s <= perRun; s++) {
        run.push(at(((r + s / perRun) / runs) * len));
      }
      out.push(run);
    }
    return out;
  }

  function draw(ms) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const cell = w / CELLS;
    const mid = (c) => (c + 0.5) * cell;
    // Anchored by the loop, not here. The first paint is synchronous with ms = 0, and anchoring on
    // that would make the next frame — a real animation timestamp in the thousands — land past the
    // end of the transformation and skip it entirely.
    const intro = fixedIntro !== undefined ? fixedIntro
      : (introStart === null ? 0 : Math.min(1, (ms - introStart) / INTRO_MS));

    // The grid, faint. It says the snake moves a cell at a time, which is the one rule of the game
    // that is not obvious from watching it, and it gives the board somewhere to be.
    g.strokeStyle = RULE_2;
    g.lineWidth = 1;
    g.beginPath();
    for (let i = 1; i < CELLS; i++) {
      const at = Math.round(i * cell) + 0.5;
      g.moveTo(at, 0); g.lineTo(at, h);
      g.moveTo(0, at); g.lineTo(w, at);
    }
    g.stroke();

    // The domain to eat, once the transformation has finished — during it there is nothing to aim
    // at yet, and a target sitting there while the snake is still arriving reads as a miss.
    if (food && intro >= 1) {
      // A box with a side chain in it.
      //
      // The box does two jobs. It makes the food one object occupying one cell — which is the rule
      // of the game, and a bare drawing floating on the grid did not say that — and it gives the
      // molecule a surface of its own, so the grid lines do not run through the middle of it.
      const bx = cell * 0.84;
      const x = mid(food[0]) - bx / 2;
      const y = mid(food[1]) - bx / 2;
      const r = bx * 0.16;
      g.beginPath();
      if (g.roundRect) g.roundRect(x, y, bx, bx, r);
      else g.rect(x, y, bx, bx);
      g.fillStyle = toward(PAPER, 1);
      g.fill();
      g.strokeStyle = CUT;
      g.lineWidth = Math.max(1, cell * 0.05);
      g.stroke();
      // Nearly the whole box: a cell is about 30px, so a tryptophan given 60% of it is a smudge.
      drawSideChain(g, food[2], mid(food[0]), mid(food[1]), bx * 0.78, CUT);
    }

    const pieces = intro < 1 ? introPieces(intro, cell, w) : gridPieces(mid);
    drawChain(g, pieces, cell);

    // The head sits at the front of whatever chain was just drawn, facing along it.
    const headRun = pieces[0];
    const tip = headRun[0];
    const behind = pieces[1] ? pieces[1][1] || pieces[1][0] : headRun[headRun.length - 1];
    const nextIsFood = intro >= 1 && !!food
      && snake[0][0] + dir[0] === food[0] && snake[0][1] + dir[1] === food[1];
    const idle = reduced ? 0.25 : 0.18 + 0.12 * Math.sin(ms / 320);
    drawHead(g, {
      x: tip.x,
      y: tip.y,
      dir: Math.atan2(tip.y - behind.y, tip.x - behind.x),
      len: cell * 1.35,
      wid: cell * 0.8,
      colour: toward(hue(1), dead ? 0.45 : 1),
      open: 0.62 * (dead ? 0.1 : nextIsFood ? 1 : idle),
    });

    const count = `<b>${score}</b> ${score === 1 ? 'residue' : 'residues'}`;
    const last = ate ? ` · ate ${ate.name}` : '';
    // The words follow the device, like the hint does: "press a key or click" is three things a
    // phone cannot do, and this is the line a player reads at the two moments they most need telling
    // what to do -- before the first move, and after dying.
    scoreEl.innerHTML = dead
      ? `${count} · ${coarse ? 'tap to try again' : 'press a key or click to try again'}`
      : count + (started || intro < 1 ? last
        : (coarse ? ' · tap to start' : ' · a key or a click starts it'));
  }

  // --- the loop -------------------------------------------------------------
  function frame(ms) {
    if (closed) return;
    if (introStart === null) introStart = ms;
    if (fixedIntro !== undefined) { draw(ms); raf = requestAnimationFrame(frame); return; }
    // Nothing moves on the grid until the animal has finished arriving on it. A key pressed during
    // the transformation is still remembered — it just takes effect the moment the intro lands.
    const arrived = ms - introStart >= INTRO_MS;
    if (arrived && started && !dead) {
      const wait = Math.max(TICK_MIN, TICK_MS - score * 4);
      if (!lastTick) lastTick = ms;
      if (ms - lastTick >= wait) { lastTick = ms; step(); }
    }
    draw(ms);
    raf = requestAnimationFrame(frame);
  }

  // --- input ----------------------------------------------------------------
  const onKey = (e) => {
    if (e.key === 'Escape') { close(); return; }
    if (dead) { reset(); e.preventDefault(); return; }
    const d = DIRS[e.key];
    if (d) { started = true; steer(d); e.preventDefault(); }
  };

  /**
   * A tap turns optimally, wherever it lands.
   *
   * It used to steer toward the point, relative to the head, with the dominant axis winning. That is
   * precise with a mouse and unusable with a thumb: a cell on a phone-sized board is about 25px, so
   * aiming at one means aiming at something smaller than the finger covering it.
   *
   * So the tap carries no direction at all. It means "take the best move from here", and the game
   * becomes about WHEN you tap rather than where: the snake keeps travelling in a straight line into
   * a wall unless you tap, and every tap is the right turn. One tap target, the whole board, which is
   * what a touchscreen is good at.
   *
   * The policy is deliberately greedy-and-safe rather than optimal — measured over 20 headless games
   * it never collides but traps itself at a median score of 50 out of 191. That imperfection is what
   * leaves a game here: a truly optimal policy would fill the board every time and nothing the player
   * did could change the outcome.
   *
   * Swiping still steers by hand, for anyone who wants to play it out themselves.
   */
  const onPoint = () => {
    if (dead) { reset(); return; }
    started = true;
    steer(bestMove(snake, food, CELLS, queued || dir));
  };

  function close() {
    if (closed) return;
    closed = true;
    if (raf) cancelAnimationFrame(raf);
    removeEventListener('keydown', onKey);
    removeEventListener('resize', onResize);
    removeEventListener('orientationchange', onResize);
    wrap.remove();
    opts.onClose?.();
  }

  /**
   * Swipe to turn.
   *
   * Tapping already worked and steers toward the point relative to the HEAD, which is precise with a
   * mouse and fiddly with a thumb: on a 350px board a cell is 25px, so aiming at one is aiming at
   * something smaller than the finger covering it. A swipe asks for a direction instead of a
   * destination, which is what the gesture is for and what every other snake on a phone does.
   *
   * A press becomes a swipe once it has travelled SWIPE_MIN, and from then on the tap is suppressed
   * so releasing does not also steer somewhere else. Under that distance it is a tap and the old
   * behaviour stands.
   */
  const SWIPE_MIN = 18;
  let press = null;
  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    press = { x: e.clientX, y: e.clientY, swiped: false };
    try { canvas.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!press || press.swiped) return;
    const dx = e.clientX - press.x;
    const dy = e.clientY - press.y;
    if (Math.hypot(dx, dy) < SWIPE_MIN) return;
    press.swiped = true;
    if (dead) { reset(); return; }
    started = true;
    steer(Math.abs(dx) > Math.abs(dy) ? [Math.sign(dx), 0] : [0, Math.sign(dy)]);
  });
  const release = (e) => {
    if (press && !press.swiped) onPoint(e);
    press = null;
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', () => { press = null; });
  // Rotation arrives as a resize on every current browser, but orientationchange fires on some
  // older ones before the new innerHeight is readable, hence both plus a frame's delay.
  const onResize = () => { if (!closed) requestAnimationFrame(layout); };
  addEventListener('resize', onResize);
  addEventListener('orientationchange', onResize);
  wrap.querySelector('.egg-close').addEventListener('click', close);
  // A click on the backdrop closes, but one inside the box must not.
  wrap.addEventListener('pointerdown', (e) => { if (e.target === wrap) close(); });
  addEventListener('keydown', onKey);

  // The veil fades up on the next frame rather than this one, so the transition has something to
  // animate from. Without the delay the class is already on when the browser first paints and there
  // is no transition at all.
  requestAnimationFrame(() => { if (!closed) wrap.classList.add('on'); });

  reset();
  // One frame synchronously, then animate — the same reason the hero does it. A tab that is not
  // being rendered does not run requestAnimationFrame, so a board that only paints from rAF opens
  // blank and stays blank until the tab is looked at.
  draw(0);
  raf = requestAnimationFrame(frame);
  return { close };
}
