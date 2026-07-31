// The hero: a protein eating its own tail.
//
// An ouroboros is not decoration here, it is the thesis. A circular permutation is a protein whose
// chain has been cut somewhere else and rejoined — the same fold, a different choice of where the
// sequence begins. Draw a backbone as a closed ring and that is what you have: the C terminus
// arriving at the N terminus, the two ends interchangeable.
//
// So the animation is the argument, and what moves matters. The RING is static — it is the fold, and
// the fold is the thing that does not change. What travels is the head, and with it the colour seam
// where red meets blue: the point at which the chain is cut and rejoined. Same atoms, same shape, a
// different starting residue, going round and round. That is a circular permutation, and a viewer
// gets it in a few seconds without reading anything.
//
// An earlier version rotated the whole animal instead. It looked similar and said something weaker —
// a spinning object rather than a fixed structure being re-read.
//
// The first attempt wound the backbone into a tight 24-turn helix, reasoning that visible secondary
// structure would make it read as protein. It read as a telephone cord. The creature has to come
// first — a tapering body, a head, a bite — and the protein comes from the colour ramp and the
// residue ticks along the back, which do not fight the silhouette.

const N = 300;                 // residues along the body
// (The body's extent is set by GAP, below, next to the geometry it belongs to.)
const R = 1.0;                 // ring radius
// The body is a circle, exactly.
//
// It used to be tilted half a radian out of plane with a five-lobed in-plane wave and a three-lobed
// weave, all of which were there to keep it from looking like a compass circle. Measured against the
// projected radius, that came to 22% out of round — the tilt alone squashes the vertical by
// cos(0.5), and it reads as an ellipse rather than as depth.
//
// None of it was load-bearing. The weave was justified as making the head pass over the tail, but
// the head is drawn last and carries its own lift, so it already does; and a planar circle never
// crosses itself, so there is nothing for depth ordering to resolve. What makes the shape an animal
// rather than a ring is the taper, the hue ramp and the head — not a wobble in the silhouette.
const HEAD_LIFT = -0.07;       // negative is toward the viewer: the head holds the tail in front


/** Fixed positions around the ring. The geometry never changes; only what flows along it does. */
function shape() {
  const pts = [];
  for (let i = 0; i < N; i++) {
    const u = i / N;                        // 0..1 round the ring, and it closes
    const a = u * Math.PI * 2;
    pts.push({ u, a });
  }
  return pts;
}

const SHAPE = shape();
// How far short of a full lap the body stops.
//
// The tail has to end INSIDE the mouth, or the animal is not eating anything. The jaw tips reach
// 0.020 of a lap ahead of the head — that ratio is fixed, since head length and circumference both
// scale with the canvas — so anything above that leaves the tail short of the teeth with a gap
// between. At 0.03 there was a visible break, and the head looked like it was chasing its tail
// rather than holding it. At 0.008 the tip sits about 40% of the way in from the tips, so the tail
// runs up to the mouth and disappears behind the jaws, which are drawn last.
const GAP = 0.008;




// --- the look, shared with the game in snakegame.js ------------------------
//
// The two snakes have to be recognisably the same animal, so whatever both of them draw is defined
// once, here, and exported. The hero owns the ring geometry; the game owns its grid.

export const PAPER = [237, 240, 244];

/** Blend a colour toward the page. */
export function toward(rgb, f) {
  const m = Math.min(1, Math.max(0, f));
  return `rgb(${Math.round(rgb[0] * m + PAPER[0] * (1 - m))},`
    + `${Math.round(rgb[1] * m + PAPER[1] * (1 - m))},`
    + `${Math.round(rgb[2] * m + PAPER[2] * (1 - m))})`;
}

/**
 * N-to-C ramp: blue at the start of the chain, red at the end.
 *
 * The same convention the structure viewer uses, which is the point — the animal is coloured like a
 * protein, so the hero is making a claim about proteins rather than being a mascot.
 */
export function hue(t) {
  const h = (240 - 240 * Math.min(1, Math.max(0, t))) / 60;
  const c = 0.66;
  const x = c * (1 - Math.abs((h % 2) - 1));
  const [r, g, b] = h < 1 ? [c, x, 0] : h < 2 ? [x, c, 0] : h < 3 ? [0, c, x]
    : h < 4 ? [0, x, c] : h < 5 ? [x, 0, c] : [c, 0, x];
  const m = 0.26;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** Thickness along the chain: tail tip thin, fullest two thirds back, a neck before the head. */
export function girth(u) {
  const smooth = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  // The floor is 0.34, not 0.16: at 0.16 the last few points of the tail were a two-pixel thread
  // that faded out before it reached the mouth, so the bite looked like a near miss. A snake's tail
  // is thin, not invisible.
  return (0.34 + 0.66 * smooth(0, 0.34, u)) * (1 - 0.28 * smooth(0.82, 1, u));
}

/**
 * The proportions of a segment, as fractions of the pitch from one segment to the next.
 *
 * Both snakes are built from segments and they have to look like the same creature, so the shape of
 * a segment is defined once rather than tuned twice. Measured off the hero, whose ring gave the
 * proportions everyone has been looking at: a pitch of 12 degrees at radius 204 is 42.7px, carrying
 * a tube 26.5px wide with a 5.7px gap cut out of it.
 */
export const SEG = {
  width: 26.5 / 42.7,      // tube width as a fraction of the pitch
  gap: 5.7 / 42.7,         // and the gap cut between one segment and the next
};

/**
 * The head, facing dir, hinged at the back of the skull.
 *
 * Two earlier heads were hand-built from quadratic curves and both looked wrong in ways that were
 * hard to name — a shape assembled from control points has too many things to get subtly wrong. So
 * this is one ellipse cut along its long axis, each half swung about a hinge at the rear. At zero
 * opening the halves close back into exactly the ellipse that read correctly, which means the
 * animation cannot break the silhouette.
 *
 * The caller positions it. Whatever the head is meant to be biting must already be on the canvas,
 * because the jaws are opaque and close over it — that occlusion is what makes a bite read as a
 * bite. Nothing dark is drawn inside the mouth: the jaws open onto whatever is being eaten.
 *
 * @param {CanvasRenderingContext2D} g
 * @param {{x: number, y: number, dir: number, len: number, wid: number, colour: string,
 *   open: number}} o open is the total gape in radians.
 */
export function drawHead(g, o) {
  const { len, wid, open } = o;
  g.save();
  g.translate(o.x, o.y);
  g.rotate(o.dir);
  const jaw = (sign) => {
    g.save();
    g.translate(-len * 0.5, 0);
    g.rotate(sign * open / 2);
    g.translate(len * 0.5, 0);
    g.fillStyle = o.colour;
    g.beginPath();
    g.ellipse(0, 0, len * 0.5, wid * 0.5, 0,
      sign < 0 ? Math.PI : 0, sign < 0 ? Math.PI * 2 : Math.PI);
    g.closePath();
    g.fill();
    if (sign < 0) {
      // the eye rides on the upper jaw, so it lifts with the gape
      g.fillStyle = 'rgba(22,32,46,.9)';
      g.beginPath();
      g.arc(len * 0.1, -wid * 0.22, Math.max(1.2, wid * 0.18), 0, 6.2832);
      g.fill();
    }
    g.restore();
  };
  jaw(1);                                    // lower
  jaw(-1);                                   // upper
  g.restore();
}

/**
 * Draw and animate the hero.
 *
 * @param {HTMLCanvasElement} canvas
 * @returns {{stop: () => void}} stop it once a real structure is on screen; no reason to spend a
 *   frame every 16ms on an animation nobody is looking at.
 */
export function ouroboros(canvas, opts = {}) {
  const reduced = !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  // A fixed phase, for testing. requestAnimationFrame does not run in a hidden tab, so without this
  // the only frame ever observable is progress 0 — which is no way to check that the head tracks the
  // ring as it travels. Undefined in normal use.
  const fixed = opts.at;
  // Likewise for the gape: the chomp is driven by ms, so a fixed-phase render always shows a shut
  // mouth. Tests need to be able to ask for an open one.
  const fixedGape = opts.gape;
  let raf = 0;
  let stopped = false;
  let lastHead = null;

  function frame(ms) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    // Zero while the hero is still display:none. Keep asking rather than giving up.
    if (!w || !h) { if (!stopped) raf = requestAnimationFrame(frame); return; }
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    // How far round the head has eaten. One lap every 11 seconds: slow enough to look deliberate
    // rather than frantic, fast enough that the motion is obvious without waiting.
    const progress = fixed !== undefined ? fixed : (reduced ? 0.25 : (ms / 11000) % 1);
    const scale = 0.34 * Math.min(w, h);
    const focal = 5.0;

    // One projection, used for the ring, the head, and for probing the ring's tangent.
    //
    // There is no rotation left, so a point's distance from the centre is rad * scale * k, and the
    // body — which passes outward and lift of zero — is a circle of exactly scale pixels. Depth
    // survives only as k, which the head uses to sit forward of the tail it is holding.
    const at = (a, outward, lift) => {
      const rad = R + outward;
      const k = focal / (focal + lift);
      return {
        x: w / 2 + rad * Math.cos(a) * scale * k,
        y: h / 2 - rad * Math.sin(a) * scale * k,
        z: lift,
        k,
      };
    };

    // Positions are static; only the flow along them moves.
    //
    // For each point, s is how far BEHIND the head it lies, in turns. That makes the chain
    // coordinate 1 - s: 1 at the head, 0 at the tail tip. Girth and colour both read from it, so the
    // taper and the blue-to-red ramp travel round a circle that itself never moves.
    const P = SHAPE.map(({ u, a }) => {
      let s2 = (progress - u) % 1;
      if (s2 < 0) s2 += 1;
      const chain = 1 - s2 / (1 - GAP);      // >1 means inside the gap the head occupies
      const p = at(a, 0, 0);
      return { ...p, u, a, chain, t: Math.min(1, Math.max(0, chain)), hidden: chain < 0 };
    });

    // Drawn in CHUNKS, each haloed and then stroked before the next one starts.
    //
    // The body has to be built from runs rather than one path, because the girth tapers and a
    // stroke has one width. Haloing each SEGMENT individually was the original bug: a 13px tube
    // rendered as a 1px thread, every segment eaten by its neighbour's band of page colour.
    // Grouping into 30 runs fixed that, and left something worth keeping — each run's halo takes
    // a couple of pixels out of the end of the one before it, so the body reads as a row of
    // linked capsules rather than as a smooth hose. That is the look, not a defect: it says
    // residues, which is what the animal is made of.
    //
    // So the passes are deliberately NOT separated. Drawing every halo before every body would
    // close the seams and produce one continuous tube — tried, and it loses the segmentation.
    // Run boundaries are fixed to RING positions, and a run is additionally cut at the seam.
    //
    // Two things have to be true at once. The capsules must not move: they are fixed in place and
    // the colour flows through them, so cutting at fixed multiples of `per` in ring index is what
    // keeps the animal still. Ordering the points by chain instead put the boundaries at fixed
    // positions along the chain, which travels — and the whole snake appeared to rotate.
    //
    // But ring index and chain position agree everywhere except at the seam, where the chain
    // restarts: the point behind the head is t=0.98 and the next one round is the tail tip at
    // t=0.00. A run takes its colour and width from its midpoint, so a run containing that came out
    // full red and thin across the join — one link visibly out of order, at the phases where a
    // boundary happened to land on the seam.
    //
    // So the fixed cuts stay, and the seam adds one more. That extra cut moves with the head, and
    // it falls where the head is sitting, which is the one place a seam cannot be seen.
    // The ring has to CLOSE. P holds 300 distinct points, so the segment from the last one back to
    // the first is a real part of the circle, and stopping at the last point left it undrawn: a
    // wider gap and a short final capsule at exactly 3 o'clock, where the array happens to begin.
    // That is an artefact of where the loop starts, and it sat in one place on screen forever.
    // Repeating the first point at the end closes it, and makes all 30 runs the same length.
    const ring = P.concat([P[0]]);
    const CHUNKS = 30;
    // The pitch from one segment to the next, which is what SEG's fractions are relative to.
    const pitch = (2 * Math.PI * scale) / CHUNKS;
    const chunks = [];
    const per = Math.ceil((ring.length - 1) / CHUNKS);
    const push = (from, to) => {
      // from is null when the previous point was already hidden; a single point strokes nothing
      if (from === null || to <= from) return;
      chunks.push({ from, to, mid: ring[Math.floor((from + to) / 2)] });
    };
    for (let c0 = 0; c0 < ring.length - 1; c0 += per) {
      const c1 = Math.min(ring.length - 1, c0 + per);
      let s = null;
      for (let i = c0; i <= c1; i++) {
        // hidden: the arc the head occupies. jump: the chain restarting between i-1 and i.
        const jump = s !== null && Math.abs(ring[i].chain - ring[i - 1].chain) > 0.5;
        if (ring[i].hidden) { push(s, i - 1); s = null; continue; }
        if (jump) { push(s, i - 1); s = i; continue; }
        if (s === null) s = i;
      }
      if (s !== null) push(s, c1);
    }

    const unit = scale / 100;
    g.lineCap = 'round';
    g.lineJoin = 'round';
    for (const ch of chunks) {
      const wid = pitch * SEG.width * girth(ch.mid.t) * ch.mid.k;
      const trace = () => {
        g.beginPath();
        g.moveTo(ring[ch.from].x, ring[ch.from].y);
        for (let i = ch.from + 1; i <= ch.to; i++) g.lineTo(ring[i].x, ring[i].y);
      };
      // No halo. It existed so a crossing would read as one part passing another, and a planar
      // ring has no crossings; all it did here was eat its neighbours by different amounts.
      //
      // Full strength. The old 0.72-1.0 range shaded by depth, and on a flat ring that became a
      // constant 0.86 — a wash over the whole body with nothing behind it to justify it.
      g.strokeStyle = toward(hue(ch.mid.t), 1);
      g.lineWidth = wid;
      trace(); g.stroke();

      // Residue ticks: faint rungs across the back — the protein showing through the animal, and
      // the same trick as scales.
      g.strokeStyle = toward(hue(ch.mid.t), 0.4);
      g.lineWidth = Math.max(0.6, unit * 0.7);
      for (let i = ch.from; i < ch.to; i += 5) {
        const p = ring[i];
        const q = ring[i + 1];
        const dx = q.x - p.x;
        const dy = q.y - p.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len;
        const ny = dx / len;
        g.beginPath();
        g.moveTo(p.x - nx * wid * 0.4, p.y - ny * wid * 0.4);
        g.lineTo(p.x + nx * wid * 0.4, p.y + ny * wid * 0.4);
        g.stroke();
      }
    }

    // The gaps between capsules, cut once the body is whole.
    //
    // They used to be a side effect of draw order: each run's halo scrubbed the trailing end of the
    // run before it. That cannot be uniform on a closed ring — run 0 adjoins run 29, which is drawn
    // last, so run 0 alone was bitten at both ends and came out short and washed toward the page,
    // parked at 3 o'clock because that is where the array starts. Round line caps overshoot any gap
    // left between runs by half a tube width, so leaving a hole does not work either.
    //
    // Cutting them afterwards makes every gap identical by construction, and there is no first or
    // last run for the pattern to break on.
    g.strokeStyle = toward(PAPER, 1);
    g.lineWidth = pitch * SEG.gap;
    for (let b = 0; b < ring.length - 1; b += per) {
      const p = ring[b];
      const q = ring[b + 1];
      if (p.hidden || q.hidden) continue;          // inside the arc the head covers
      const dx = q.x - p.x;
      const dy = q.y - p.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      const across = (pitch * SEG.width * girth(p.t) * p.k + 4.5 * unit) / 2;
      g.beginPath();
      g.moveTo(p.x - nx * across, p.y - ny * across);
      g.lineTo(p.x + nx * across, p.y + ny * across);
      g.stroke();
    }

    // --- the head, last, so it holds the tail rather than hiding behind it ---
    // The head is at the phase angle, lifted just clear of the ring so it reads as being on top of
    // the seam rather than part of it.
    const kHead = focal / (focal + HEAD_LIFT);
    // The head is JOINED to the body at the back of the skull, not centred over the end of it.
    //
    // Centred, the jaws hinge behind the body's newest residue, so the rear half of the mouth
    // opening was filled with red body — which read as a red tongue once the dark interior was
    // taken out. Advancing the head by its own half-length puts the hinge exactly where the body
    // ends: the neck runs up to the join and stops, and the only thing left inside the mouth is the
    // tail. Half of the head's length as a fraction of the circumference, which is scale-free
    // because head size and circumference both scale with the canvas.
    const headAhead = 0.125 * kHead / (2 * Math.PI);
    const headA = (progress + headAhead) * Math.PI * 2;
    // Centred on the body across the tube, and forward of it only in size.
    //
    // Passing the lift through the projection moved the head as well as scaling it, because in this
    // projection k multiplies the radius — so the head sat 19% of a tube width outside the
    // centreline, and looked stuck on askew. Its position now comes from the centreline itself and
    // the lift survives only as the factor its size is drawn at.
    const head = { ...at(headA, 0, 0), z: HEAD_LIFT, k: kHead, a: headA, t: 1 };
    // Facing = the ring's tangent, from TWO points on the undisplaced ring.
    //
    // Getting this right took three attempts. Neck-to-head folded the head's own displacement into
    // the direction. Then probing ahead on the ring but subtracting the displaced head was no better:
    // the head's lift puts it ~10px off the ring while a 0.05 rad tangent step spans only ~5px, so
    // the offset dominated and the facing wobbled inside a 60 degree band instead of sweeping the
    // full circle. Both samples have to come from the same curve.
    const onRing = at(head.a, 0, 0);
    const ahead = at(head.a + 0.05, 0, 0);
    const dir = Math.atan2(ahead.y - onRing.y, ahead.x - onRing.x);
    // the tail tip is the last point still drawn, just ahead of the head
    // the tail tip is the lowest-chain point still drawn, just ahead of the head
    const tailPt = P.reduce((best, p) => (!p.hidden && (!best || p.chain < best.chain)
      ? p : best), null) || P[0];
    lastHead = { x: head.x, y: head.y, dir, progress, tailX: tailPt.x, tailY: tailPt.y, open: 0 };
    const nearH = Math.min(1, Math.max(0, 1 - (head.z + 1.3) / 2.6));
    const hl = 25 * unit * head.k;          // length
    const hw = 14 * unit * head.k;          // width
    const col = toward(hue(head.t), 0.78 + 0.22 * nearH);

    // The mouth opens and closes, and the jaws are the ellipse split in two.
    //
    // Two earlier heads were hand-built from quadratic curves and both looked wrong in ways that
    // were hard to name — a shape assembled from control points has too many things to get subtly
    // wrong. So the working ellipse is kept and simply cut along its long axis, each half hinged at
    // the back of the skull. At zero opening the two halves close back into exactly the ellipse that
    // read correctly, which means the animation cannot break the silhouette.
    const bite = fixedGape !== undefined ? fixedGape
      : (reduced ? 0.35 : Math.sin(Math.PI * ((ms / 1500) % 1)) ** 3);
    const open = 0.62 * bite;                 // radians, total gape
    lastHead.open = open;

    // No dark inside the mouth. It was there to give the gape something to open onto, back when the
    // tail stopped short of the teeth and an open mouth would otherwise have framed bare page. Now
    // the tail runs in as far as the throat, so what the jaws open onto is the tail itself — which
    // is the thing worth showing, and it says "eating" better than a dark wedge did.

    drawHead(g, { x: head.x, y: head.y, dir, len: hl, wid: hw, colour: col, open });

    if (!reduced && !stopped && ms && fixed === undefined) raf = requestAnimationFrame(frame);
  }

  // One frame synchronously, then animate. requestAnimationFrame does not run while a tab is
  // unrendered, so a hero that only paints from rAF is blank until the tab is looked at.
  frame(0);
  if (!reduced && fixed === undefined) raf = requestAnimationFrame(frame);

  return {
    /** Where the head is and which way it faces, from the last frame drawn. */
    headInfo: () => lastHead,
    stop() {
      stopped = true;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    },
  };
}
