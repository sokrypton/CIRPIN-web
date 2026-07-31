// The atlas: 15,176 SCOPe domains laid out in 3D, and what happens when you click one.
//
// Kept as its own page rather than folded into the search app, but no longer its own
// implementation of anything: src/coords.js for the coordinate store, src/ted.js for the PCA basis
// and the code scan, and src/trace3d.js for both the cartoon in the side panel and the drag that
// turns it. What is left here is the scatter — the layout, the depth ordering, the crop, the colour
// channels — which is the only part that is genuinely this page's own.
//
// The layout is precomputed by tools/build_atlas.py: UMAP over the same 32-d PCA codes the search
// scans, under the same cosine metric, so the map cannot disagree with the ranking about what is
// close. The browser downloads 121 KB and does no dimensionality reduction.
//
// What the map is and is not, measured on sampled domains rather than assumed. The third
// dimension costs 2 bytes a row and buys back some of what a projection loses:
//
//                                       2D        3D
//   top-1 neighbour preserved .......  14.5%     19.5%
//   top-10 within the layout top-10 .  45.8%     50.1%
//   top-25 within the top-25 ........  57.4%     60.0%
//
// Still: even in 3D, four fifths of domains do not have their true nearest neighbour nearest in
// the layout. But the nearest point on screen averages 0.914 cosine — above the 0.9 the search
// itself treats as the same fold — so screen neighbours are genuinely related proteins that are
// simply not the best matches. The map is a navigator, not a ranking. Clicking a node lists
// neighbours found by scanning the embeddings, never by screen distance, and the caption says so.
//
// Rotation is not decoration here: a still 3D scatter is less legible than a 2D one, because
// depth has to be inferred. Motion is what makes the structure readable, so the view spins gently
// until touched — and stops entirely under prefers-reduced-motion, where a static view is the
// honest fallback rather than an unusable one.

// SCOPe's seven classes, in SCOPe's own order, and never cycled — an eighth class would need an
// eighth colour, not a reused one.
//
// Okabe-Ito, minus its pale yellow, which is too light to see on paper this colour. Checked with
// tools/palette_check.mjs rather than by eye: every one of the 21 pairs clears an OKLab separation
// of 15 for normal vision and 8 under both deuteranope and protanope simulation, and every colour
// clears a 1.6 contrast ratio against the background. Two classes merging under colour blindness
// would put "all beta" and "small proteins" in one visual bucket with nothing to reveal it.
import { bestView, fillZoom } from './orient.js?v=98dcea2c';
import { drawTraces, fitOf, makeCamera, orbit, prep, mul3, rotX, rotY, spectrumRgb }
  from './trace3d.js?v=98dcea2c';

export const CLASS_COLOURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
  '#E69F00', '#56B4E9', '#333333'];
const NO_CLASS = '#9aa4b0';
const UNKNOWN = 0xFFFF;

// The disagreement ramp and its quantile breaks are kept, unused, for the interpolation work: the
// delta channel is still in the data file, and a sequential ramp that has been checked against the
// background is worth not having to re-derive. Nothing imports them today.
export const DELTA_RAMP = ['#fbe6f0', '#f0b9d5', '#e08cba', '#c85f9c', '#a5347b', '#750d55'];

/** Decode the atlas file: layout coordinates, colour channels, and SCOPe labels. */
export function loadAtlas(buf, meta) {
  const n = meta.records;
  const dv = new DataView(buf);
  const W = 16;                 // bytes a row: x, y, z, delta, nres, fold, class, cpCount
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  const delta = new Float32Array(n);
  const nres = new Uint16Array(n);
  const fold = new Uint16Array(n);
  const cls = new Uint16Array(n);
  const cp = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    // x, y, z are int16 over a single shared scale, so the layout's aspect ratio survives
    x[i] = dv.getInt16(i * W, true);
    y[i] = dv.getInt16(i * W + 2, true);
    z[i] = dv.getInt16(i * W + 4, true);
    delta[i] = dv.getUint16(i * W + 6, true) / 65535;
    nres[i] = dv.getUint16(i * W + 8, true);
    fold[i] = dv.getUint16(i * W + 10, true);
    cls[i] = dv.getUint16(i * W + 12, true);
    cp[i] = dv.getUint16(i * W + 14, true);
  }
  const components = meta.layout?.components ?? (z.some((v) => v !== 0) ? 3 : 2);
  return { n, x, y, z, delta, nres, fold, cls, cp, components };
}

// Point radius at the default zoom. Small on purpose: 15,176 points overlap heavily at full
// extent, and a larger dot there turns the cloud into a blob.
const R_POINT = 1.7;
const R_HIT = 3.4;
// Zoomed in, a fixed radius is the wrong choice for the opposite reason. The points spread apart
// but stay 1.7px, so a close view is a few faint specks on a wide empty field — which reads as the
// map fading out rather than as magnification. Growing the radius with the square root of the zoom
// keeps the ink per screen area roughly constant: dense at full extent, legible up close.
const R_ZOOM_MAX = 5.5;
const pointRadius = (zoom) => Math.min(R_ZOOM_MAX, R_POINT * Math.sqrt(Math.max(1, zoom)));

// Bounds on the zoom.
//
// The cap was 14x, chosen when nothing was cropped and a deep zoom just pushed the cloud off the
// canvas. It was far too tight for this layout, and that is measurable rather than a matter of
// taste: UMAP collapses each tight family into a knot a few hundred layout units across inside a
// cloud of radius 32,000, so the distance from a domain to its 12th nearest neighbour is 213 units
// at the median. Filling the frame with that neighbourhood needs about 71x. At 14x the whole
// neighbourhood of 95% of domains fits inside a single mark — which is why every line drawn to a
// same-fold partner had zero length and the map looked as though it had drawn nothing.
//
// 250x resolves the tightest tenth (216x). Past that the int16 quantisation of the layout starts to
// show as a lattice, which is the real floor: one quantisation step is about 2px at 250x, so there
// is nothing further to magnify.
const ZOOM_MIN = 0.35;
const ZOOM_MAX = 250;
// Where a selection's k-th nearest neighbour should land, as a fraction of the half-frame, when the
// zoom is chosen automatically. k = 12 because that is the length of the neighbour table beside the
// map: the view that opens should be the view those rows are talking about.
const AUTO_K = 12;
const AUTO_FILL = 0.4;

// The crop radius at zoom 1, as a multiple of the cloud radius.
//
// Zooming into a 3D cloud without cropping is not flying in, it is pressing your face against the
// outside of it: the material between the camera and the point of interest stays drawn and hides
// exactly what you zoomed in to see.
//
// The crop is a SPHERE about the selection, not a plane in front of it. A near plane only removes
// what happens to be between you and the point at the current angle, so the neighbourhood you are
// looking at stays buried in the rest of the cloud from every other direction — and then rotating
// changes which material disappears, which reads as the map rearranging itself rather than as a
// view of one place. Cropping equally on all sides means the selection sits in a ball of its own
// surroundings that survives rotation unchanged.
//
// It tightens as 1/zoom: at zoom 1 the ball is larger than the cloud, so the default view is
// untouched and nothing is hidden until you ask to go closer. Culled rather than faded, because a
// faint point still occludes the one behind it and half-transparent clutter is harder to read than
// absence.
const CROP_AT_1 = 2.5;

// Where a line starts being drawn: the cutoff the search itself treats as the same fold, so a line
// on the map means the same thing as a hit in the results table. It is the DEFAULT rather than a
// constant now — the slider in the header moves it — because the interesting question is what
// happens either side of 0.9, and a fixed threshold cannot be interrogated.
export const LINK_CUTOFF = 0.9;
// Every partner above the cutoff is drawn, with no cap. A cap of ten looked tidier and was a lie
// about the data: a domain in a large family has dozens of partners CIRPIN calls the same fold, and
// showing ten of them made a crowded neighbourhood indistinguishable from a sparse one. If the
// result is a dense fan, that density IS the finding, and the slider is how you thin it.
const DEPTH_BINS = 96;      // painter's algorithm by bucket, so no per-frame sort

/**
 * An orbiting scatter over a fixed point set.
 *
 * Canvas 2D rather than WebGL: 15,176 points is well inside what it handles at 60 fps once the
 * cloud is drawn as rects rather than arcs, and it keeps the view consistent with the rest of the
 * app and free of dependencies. A tiled AlphaFold atlas would need WebGL; this does not.
 *
 * Depth ordering is a bucket sort into 96 bins, not a comparison sort. At this point count a full
 * sort every frame is affordable but pointless — bins are O(n) and the error is invisible, since
 * two points in the same bin are within a pixel of each other in depth.
 */
export function makeScatter(canvas, atlas, opts = {}) {
  // Whether to animate at all. Read once here because three things consult it: the idle spin, the
  // zoom-to-click, and anything added later.
  const reduced = !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const is3D = atlas.components === 3;
  // The same camera and the same gesture as the structure viewer, from src/trace3d.js.
  //
  // This view had its own, built on a yaw and a pitch with the pitch clamped to +-1.5 radians to
  // stop it flipping through vertical. A clamp is felt as the drag going dead: keep pulling
  // downward and the cloud stops turning while the mouse is still moving. A matrix accumulated per
  // drag has no poles to clamp, so the gesture never runs out of travel — and it is now literally
  // the same code that turns a protein on the search page, which is the other half of the point.
  const cam = makeCamera(mul3(rotX(-0.35), rotY(0.6)));
  // pivot is the world point rotation happens about, and it is what makes clicking useful: with
  // rotation fixed on the layout's origin, centring a point and then dragging swings it straight
  // back off screen, so a selection could be examined only from the one angle it happened to
  // arrive at. Moving the pivot to the clicked point turns a drag into an orbit around it.
  const state = { hover: -1, focus: -1, links: [], pivot: [0, 0, 0] };

  // extent of the layout, for the initial fit and the perspective divide
  let radius = 1;
  for (let i = 0; i < atlas.n; i++) {
    const r = Math.hypot(atlas.x[i], atlas.y[i], atlas.z[i]);
    if (r > radius) radius = r;
  }

  // projected screen positions from the last frame; hit-testing reads these rather than
  // re-projecting, and they are what makes picking correct under rotation
  const px = new Float32Array(atlas.n);
  const py = new Float32Array(atlas.n);
  const pd = new Float32Array(atlas.n);
  // distance from the selection, in layout units — what the crop tests
  const pr = new Float32Array(atlas.n);
  // the perspective factor each point was drawn with, which is also its size multiplier
  const pk = new Float32Array(atlas.n);
  let cropR = Infinity;
  const bins = new Int32Array(DEPTH_BINS + 1);
  const order = new Int32Array(atlas.n);

  let base = 1;
  // Scale only. It is called on resize, where resetting the camera as well would throw away
  // wherever the user had navigated to.
  function fit() {
    base = 0.42 * Math.min(canvas.clientWidth, canvas.clientHeight) / radius;
  }

  /** Back to the whole cloud, seen from the middle. */
  function home() {
    state.pivot = [0, 0, 0];
    cam.tx = 0;
    cam.ty = 0;
    cam.zoom = 1;
    fit();
  }

  // rgb for each class, parsed once
  const CLASS_RGB = CLASS_COLOURS.map((h) => [1, 3, 5].map((k) => parseInt(h[k] + h[k + 1], 16)));
  const NO_RGB = [1, 3, 5].map((k) => parseInt(NO_CLASS[k] + NO_CLASS[k + 1], 16));

  /**
   * The colour a point carries, with nothing added.
   *
   * Depth used to fade it. On this map colour is DATA — a class is a category with a legend beside
   * it — and a point dimmed for being at the back is a category you cannot name. Depth is carried by
   * size and by the painter's ordering instead, neither of which touches the value.
   */
  function shade(i) {
    const c = atlas.cls[i] === UNKNOWN ? NO_RGB : (CLASS_RGB[atlas.cls[i]] ?? NO_RGB);
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function project() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const s = base * cam.zoom;
    const M = cam.rot;
    // A mild perspective: enough for depth to read, not enough to distort the layout.
    //
    // The camera sits at +focal on the view axis looking back toward the origin, so LARGER z2 is
    // nearer. That sign was the other way round, and it was the bug behind a rotation that looked
    // wrong: this view and the structure viewer read the same rotation matrix — they share one drag
    // now — but disagreed about which side of the cloud faced the camera, so an identical drag
    // turned the two canvases in mirrored directions. On a protein you notice immediately; on a
    // point cloud, where depth is ambiguous and either reading is geometrically possible, it just
    // feels as though the far side is following the mouse. Everything downstream of z2 — the
    // painter's order, the point size, the pick's front-most preference — follows this sign.
    const focal = 3.2 * radius;
    const [px0, py0, pz0] = state.pivot;
    // the ball of surroundings kept around the selection; measured before rotation, so turning the
    // view cannot change which points are in it
    cropR = radius * (CROP_AT_1 / Math.max(1, cam.zoom));
    for (let i = 0; i < atlas.n; i++) {
      // relative to the pivot, so rotation turns the view about it rather than about the origin
      const x = atlas.x[i] - px0;
      const y = atlas.y[i] - py0;
      const z = atlas.z[i] - pz0;
      const x1 = M[0] * x + M[1] * y + M[2] * z;
      const y1 = M[3] * x + M[4] * y + M[5] * z;
      const z2 = M[6] * x + M[7] * y + M[8] * z;
      // focal - z2, not focal + z2: positive view-space z is TOWARD the viewer here, which is the
      // convention src/trace3d.js uses (pe = 1 / (1.9 - vz * 0.55)). See the note on `focal`.
      const k = focal / (focal - z2);
      px[i] = w / 2 + cam.tx + s * x1 * k;
      py[i] = h / 2 + cam.ty - s * y1 * k;
      pd[i] = z2;
      pr[i] = Math.hypot(x, y, z);
      pk[i] = k;
    }
  }

  /**
   * Bucket the points far-to-near, so nearer ones are drawn last.
   *
   * With the colours no longer faded by depth this is the only thing that says which point is in
   * front: the nearer one is painted over the farther one, opaquely, exactly as it would occlude it
   * in life. Bucketed rather than sorted because at this point count a comparison sort every frame
   * is affordable but pointless — two points in one bin are within a pixel of each other in depth.
   */
  function depthOrder() {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < atlas.n; i++) {
      if (pd[i] < lo) lo = pd[i];
      if (pd[i] > hi) hi = pd[i];
    }
    const span = Math.max(hi - lo, 1e-6);
    bins.fill(0);
    const b = new Int32Array(atlas.n);
    for (let i = 0; i < atlas.n; i++) {
      // far (small z2) first, so the nearer point is painted over it
      const q = Math.min(DEPTH_BINS - 1,
        Math.floor(((pd[i] - lo) / span) * (DEPTH_BINS - 1) + 0.5));
      b[i] = q;
      bins[q + 1]++;
    }
    for (let q = 0; q < DEPTH_BINS; q++) bins[q + 1] += bins[q];
    const at = bins.slice();
    for (let i = 0; i < atlas.n; i++) order[at[b[i]]++] = i;
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    project();
    depthOrder();
    const r = pointRadius(cam.zoom);
    // Every point in its own colour, the selection and its partners included.
    //
    // The focus and its neighbours used to be skipped here and redrawn as flat teal and red discs,
    // which overwrote the one thing a reader came to the map for: a neighbour repainted teal no
    // longer says which class it belongs to or how large its disagreement is, so selecting a domain
    // destroyed the values of exactly the points being compared. They are ordinary points now; what
    // marks them is drawn around them, not instead of them.
    for (let k = 0; k < atlas.n; k++) {
      const i = order[k];
      if (pr[i] > cropR) continue;                   // outside the ball around the selection
      const x = px[i];
      const y = py[i];
      if (x < -4 || y < -4 || x > w + 4 || y > h + 4) continue;
      g.fillStyle = shade(i);
      // Size carries depth, the way the structure renderer's stroke width does.
      //
      // It has to carry it, because nothing else is left to: colour is data here and no longer
      // fades, so a flat point size made a near cluster — spread apart by the perspective's 1.37x
      // expansion — read as sparser and therefore SMALLER than a far cluster compressed to 0.81x,
      // which is the opposite of what the geometry says. The same factor that moves a point is now
      // also its radius, so near reads as near. Floored just under a pixel so the back of the cloud
      // stays visible and clickable rather than vanishing.
      const rr = Math.max(0.75, r * pk[i]);
      // rects, not arcs: at under two pixels they are indistinguishable and about three times
      // cheaper, which is what keeps a rotating 15,176-point cloud smooth
      g.fillRect(x - rr, y - rr, rr * 2, rr * 2);
    }

    // Lines to the hits a search would actually return, and nothing else.
    //
    // A line is the whole of the annotation: it says which points are partners and, by its width,
    // how strong each claim is — 0.90 thin, 1.00 thick. Only pairs at or above the CIRPIN cutoff
    // are drawn, so what is connected is exactly what the search would call the same fold. A line
    // whose far end is outside the crop is still drawn to where that point would be: dropping it
    // would say "no such partner" when the truth is "that partner is further away than you are
    // currently looking".
    if (state.focus >= 0 && state.links.length) {
      const fx = px[state.focus];
      const fy = py[state.focus];
      g.lineCap = 'round';
      for (const [sim, j] of state.links) {
        const t = Math.min(1, Math.max(0, (sim - LINK_CUTOFF) / (1 - LINK_CUTOFF)));
        g.strokeStyle = `rgba(13,125,140,${0.35 + 0.5 * t})`;
        g.lineWidth = 1 + 1.8 * t;
        g.beginPath();
        g.moveTo(fx, fy);
        g.lineTo(px[j], py[j]);
        g.stroke();
      }
    }

    // The selection: an outline around the point, leaving the point itself alone.
    //
    // A box, because the marks are boxes. A ring around a square mark reads as a second, different
    // kind of object sitting on top of the point rather than as that point being picked out — and at
    // this size the mismatch between the two shapes is most of what you see. The outline is offset
    // outward far enough that the mark's own colour still shows inside it.
    if (state.focus >= 0) {
      const f = state.focus;
      const rr = Math.max(R_HIT, Math.max(0.75, r * pk[f]) + 2.5);
      g.strokeStyle = '#c8102e';
      g.lineWidth = 1.6;
      g.strokeRect(px[f] - rr, py[f] - rr, rr * 2, rr * 2);
    }
  }

  /** Nearest projected point to a canvas position, preferring the one in front. */
  function pick(mx, my) {
    project();
    let best = -1;
    // the target grows with the drawn point, so a big zoomed dot is not harder to hit than a small
    // one at full extent
    const reach = Math.max(10, pointRadius(cam.zoom) * 3);
    let bd = reach * reach;
    let bz = -Infinity;
    for (let i = 0; i < atlas.n; i++) {
      if (pr[i] > cropR) continue;                   // not drawn, so not clickable
      const dx = px[i] - mx;
      const dy = py[i] - my;
      const d = dx * dx + dy * dy;
      // among near-ties prefer the one in FRONT — the larger pd, matching the sign above. Clicking
      // a dense patch should select the point facing you, not the one hidden behind it.
      if (d < bd || (d < reach * reach && best >= 0 && pd[i] > bz + 1e-6 && d < bd + 25)) {
        if (d < bd || pd[i] > bz) { bd = Math.min(bd, d); bz = pd[i]; best = i; }
      }
    }
    return best;
  }

  let spinning = is3D && !reduced;
  let raf = 0;
  function loop() {
    if (spinning) {
      // About the screen's vertical axis, applied to the left of whatever rotation the view already
      // holds — so the idle spin composes with a drag instead of fighting it.
      cam.rot = mul3(rotY(0.0022), cam.rot);
      draw();
      raf = requestAnimationFrame(loop);
    } else {
      raf = 0;
    }
  }
  function stopSpin() {
    if (!spinning) return;
    spinning = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (opts.onSpin) opts.onSpin(false);
  }

  // Rotation, pan, zoom and the click: the structure viewer's gesture, unchanged.
  //
  // Two things this fixes beyond the pole clamp. The old version listened for mousedown on the
  // canvas and mouseup on the window, with mousemove on the canvas only — so a drag that left the
  // canvas stopped turning the cloud while the button was still down, and came back to life
  // wherever the pointer re-entered, which is a jump. Pointer capture keeps the moves coming. And
  // the click is orbit's own click-versus-drag test rather than a separate click listener, so
  // finishing a rotation over a point no longer also selects it.
  orbit(canvas, cam, draw, {
    zoomMin: ZOOM_MIN,
    zoomMax: ZOOM_MAX,
    gain: 0.006,
    // Shift-drag pans; a 2D layout has nothing to rotate, so it pans with any drag.
    pan: true,
    panOnly: !is3D,
    onFirstDrag: stopSpin,
    onClick(e) {
      if (!opts.onPick) return;
      // Prefer what the tooltip was showing over a fresh pick. On a spinning view the two can
      // disagree: the hover happened at one rotation and the click lands at another, so a fresh
      // pick can find nothing where the user plainly saw a point. Clicking what was named is both
      // more predictable and immune to how the animation happened to be phased.
      const i = state.hover >= 0 ? state.hover : pick(e.offsetX, e.offsetY);
      if (i >= 0) {
        opts.onPick(i);
        if (opts.centreOnPick !== false) centreOn(i);
      }
    },
    // Back to the fitted view — with a bounded zoom you can no longer get lost, but you can still
    // end up somewhere unhelpful.
    onReset: home,
  });

  // Hover is this view's own: orbit knows about dragging, not about what is under the pointer.
  canvas.addEventListener('pointermove', (e) => {
    if (e.buttons) return;                 // mid-drag; the tooltip would chase the rotation
    const i = pick(e.offsetX, e.offsetY);
    if (i !== state.hover) {
      state.hover = i;
      if (opts.onHover) opts.onHover(i, e.offsetX, e.offsetY);
    }
  });

  fit();
  if (spinning) raf = requestAnimationFrame(loop);
  /**
   * Bring a point to the middle of the view, zooming in if it is not already close.
   *
   * Clicking a point in a cloud of 15,176 answers "which one is that" but leaves you looking at
   * the same wide field, so the answer arrives with no context around it. Centring and zooming
   * puts the point among its actual neighbours, which is where the interesting question is — the
   * teal marks are its embedding neighbours, and whether they are nearby is the thing worth
   * seeing.
   *
   * Animated, because a jump loses the relationship between where the point was and where it went
   * — the one thing motion is genuinely better at than a cut. Instant under prefers-reduced-motion,
   * where the destination matters more than the journey.
   */
  function autoZoom(i) {
    // The distance to the selection's AUTO_K-th nearest domain, by partial selection over the whole
    // layout — one O(n) pass, about a millisecond, and it has to be per point because the local
    // scale varies by more than a hundredfold across this map.
    const best = new Float64Array(AUTO_K).fill(Infinity);
    for (let j = 0; j < atlas.n; j++) {
      if (j === i) continue;
      const d = Math.hypot(atlas.x[j] - atlas.x[i], atlas.y[j] - atlas.y[i], atlas.z[j] - atlas.z[i]);
      if (d >= best[AUTO_K - 1]) continue;
      let k = AUTO_K - 1;
      while (k > 0 && best[k - 1] > d) { best[k] = best[k - 1]; k--; }
      best[k] = d;
    }
    const d = best[AUTO_K - 1];
    if (!Number.isFinite(d) || d <= 0) return ZOOM_MAX;
    // base = 0.42 * minSide / radius px per unit, so a distance d lands at base * zoom * d px; ask
    // for AUTO_FILL of the half-frame, which is 0.5 * minSide.
    return Math.min(ZOOM_MAX, Math.max(1, (AUTO_FILL * 0.5 * radius) / (0.42 * d)));
  }

  /**
   * @param {number} i row to centre on
   * @param {number} [target] zoom to reach; by default it is derived from the local density, since a
   *        single number cannot serve a layout whose neighbourhood scale spans two orders of
   *        magnitude — 3.2x was the old default and it left 95% of neighbourhoods inside one mark.
   */
  function centreOn(i, target) {
    const want = target ?? autoZoom(i);
    const zoomTo = Math.min(ZOOM_MAX, Math.max(cam.zoom, want));
    const to = [atlas.x[i], atlas.y[i], atlas.z[i]];
    const from = state.pivot.slice();
    const fromTx = cam.tx;
    const fromTy = cam.ty;
    const fromZoom = cam.zoom;

    // Landing on the pivot with no translation is what puts the point dead centre, and keeps it
    // there through any later rotation.
    const apply = (e) => {
      for (let k = 0; k < 3; k++) state.pivot[k] = from[k] + (to[k] - from[k]) * e;
      cam.tx = fromTx * (1 - e);
      cam.ty = fromTy * (1 - e);
      cam.zoom = fromZoom + (zoomTo - fromZoom) * e;
      draw();
    };

    if (reduced) { apply(1); return; }
    const t0 = performance.now();
    const DUR = 320;
    const step = () => {
      const u = Math.min(1, (performance.now() - t0) / DUR);
      // ease in-out, so the move reads as one gesture rather than a jump with a tail
      apply(u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u));
      if (u < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  return {
    /**
     * Last-frame geometry, for checking what the crop is actually removing.
     *
     * It reports the crop as a fraction of the cloud and the depth range on both sides of it,
     * because the failure mode of a crop is invisible: material quietly missing looks exactly like
     * material that was never there. keptPdRange straddling zero is the evidence that the ball is
     * centred on the selection rather than sitting in front of or behind it.
     */
    stats(row) {
      let vis = 0, culled = 0;
      let visMin = Infinity, visMax = -Infinity, cutMin = Infinity, cutMax = -Infinity;
      for (let i = 0; i < atlas.n; i++) {
        if (pr[i] > cropR) {
          culled++;
          if (pd[i] < cutMin) cutMin = pd[i];
          if (pd[i] > cutMax) cutMax = pd[i];
        } else {
          vis++;
          if (pd[i] < visMin) visMin = pd[i];
          if (pd[i] > visMax) visMax = pd[i];
        }
      }
      // The frontmost and backmost points, with where each was drawn and how big.
      //
      // This is the check for a depth sign that has been flipped, which is otherwise very hard to
      // see: front.k > 1 > back.k says the perspective agrees that front is nearer, and watching
      // front.sx during a rightward drag says whether the near side follows the mouse — it must, and
      // when the sign was wrong it did the opposite while looking perfectly plausible.
      let fi = -1, bi = -1;
      for (let i = 0; i < atlas.n; i++) {
        if (pr[i] > cropR) continue;
        if (fi < 0 || pd[i] > pd[fi]) fi = i;
        if (bi < 0 || pd[i] < pd[bi]) bi = i;
      }
      const at = (i) => (i < 0 ? null
        : { i, sx: px[i], sy: py[i], pd: pd[i], k: pk[i], r: pointRadius(cam.zoom) * pk[i] });
      return { cropR, cropFracOfCloud: cropR / radius, zoom: cam.zoom, radius,
        visible: vis, culled, keptPdRange: [visMin, visMax], culledPdRange: [cutMin, cutMax],
        front: at(fi), back: at(bi),
        // one named row, so a drag can be checked against a FIXED point: the frontmost point is a
        // different domain before and after a turn, and comparing those two says nothing
        row: row === undefined ? null : at(row),
        focusPd: state.focus >= 0 ? pd[state.focus] : null };
    },
    draw,
    fit,
    home,
    is3D,
    centreOn,
    spinning: () => spinning,
    toggleSpin() {
      if (!is3D) return false;
      spinning = !spinning;
      if (spinning && !raf) raf = requestAnimationFrame(loop);
      if (!spinning && raf) { cancelAnimationFrame(raf); raf = 0; }
      return spinning;
    },
    /**
     * Select a point and connect it to its partners.
     *
     * `links` are [similarity, index] pairs and every one of them is drawn. The cutoff is applied
     * by the caller, which owns the slider — filtering again here would mean two places could
     * disagree about which cutoff is in force, and the one on screen would win silently.
     */
    setFocus(i, links) {
      state.focus = i;
      state.links = links || [];
      draw();
    },
  };
}

/**
 * The k nearest domains in embedding space, by scanning the codes.
 *
 * Not by screen distance, which is the whole point: only 14.5% of domains have their true nearest
 * neighbour nearest on the map. 15,176 dot products over 32 dimensions is about 5 ms, so there is
 * no reason to approximate.
 */
export function neighbours(codes, dims, n, row, k) {
  const q = codes.subarray(row * dims, row * dims + dims);
  const best = [];
  for (let i = 0; i < n; i++) {
    if (i === row) continue;
    let s = 0;
    const off = i * dims;
    for (let d = 0; d < dims; d++) s += q[d] * codes[off + d];
    if (best.length < k) {
      best.push([s, i]);
      if (best.length === k) best.sort((a, b) => b[0] - a[0]);
    } else if (s > best[k - 1][0]) {
      best[k - 1] = [s, i];
      for (let j = k - 1; j > 0 && best[j][0] > best[j - 1][0]; j--) {
        const t = best[j]; best[j] = best[j - 1]; best[j - 1] = t;
      }
    }
  }
  if (best.length < k) best.sort((a, b) => b[0] - a[0]);
  return best;
}

/**
 * EVERY domain at or above a similarity, not the k best of them.
 *
 * The map draws a line to each, so this is the difference between "here are some partners" and
 * "here is the whole neighbourhood": a domain in a large family has dozens above 0.9, and a top-k
 * list of them makes a crowded neighbourhood look exactly like a sparse one. The count it returns
 * is itself worth reading — it is how many domains CIRPIN would call this fold at the cutoff on
 * screen.
 *
 * Sorted strongest first, so the caller can take the head for a table and draw the rest as lines
 * without scanning twice. One full pass over 15,176 x 32 is about 5 ms, which is cheap enough to
 * redo on every drag of the slider.
 */
export function partnersAbove(codes, dims, n, row, cutoff) {
  const q = codes.subarray(row * dims, row * dims + dims);
  const out = [];
  for (let i = 0; i < n; i++) {
    if (i === row) continue;
    let s = 0;
    const off = i * dims;
    for (let d = 0; d < dims; d++) s += q[d] * codes[off + d];
    if (s >= cutoff) out.push([s, i]);
  }
  out.sort((a, b) => b[0] - a[0]);
  return out;
}

/**
 * The structure panel: the search page's own cartoon renderer, turned by the same gesture.
 *
 * It used to be a polyline projected onto the structure's two widest axes — enough to recognise a
 * fold, and it said so. But the app already has a renderer that draws helices as ribbons and loops
 * as tubes, chooses the view rather than accepting whichever way the file was deposited, and can be
 * dragged; a flat trace beside it in the same product is two answers to one question. Both now come
 * from src/trace3d.js, so an improvement to the cartoon reaches this panel without being ported.
 *
 * Coloured N to C rather than by domain, which is the colouring this panel needs: on a circular
 * permutation the hues come out of order in space, and that is the thing worth seeing here.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{makeSec: function, smoothSec: function}} ss the secondary-structure assignment, passed
 *        in so this module does not pull the whole TM-align port in behind it
 */
export function makeTraceView(canvas, ss) {
  const reduced = !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const cam = makeCamera();
  let state = null;

  // Square, and as wide as the panel allows. drawTraces fits the structure into a square inside the
  // canvas, so a 190px-tall canvas in a 300px-wide panel drew the protein at 190px with a third of
  // the width left empty on either side. Sizing the square off the width instead spends the space
  // that was already there — a fold is easier to recognise at 300px than at 190.
  const side = () => Math.min(canvas.clientWidth || 190, 300);

  function draw() {
    if (!state) { prep(canvas, side()); return; }
    drawTraces(canvas, [{ coords: state.flat, sec: state.sec, colourAt: state.colourAt }],
      { rot: cam.rot, zoom: cam.zoom, maxH: side(), inset: 6 });
  }

  // The view is chosen per structure, not carried over: unlike the search page — where the query and
  // the hit are the same fold seen twice and a shared orientation is the point — consecutive
  // selections here are unrelated proteins, and each deserves its own widest face.
  function show(coords) {
    if (!coords || coords.length < 6) { state = null; draw(); return; }
    const flat = Float64Array.from(coords);
    const n = flat.length / 3;
    const lut = [];
    for (let i = 0; i < n; i++) lut.push(spectrumRgb(n < 2 ? 0 : i / (n - 1)));
    state = {
      flat,
      sec: ss.smoothSec(ss.makeSec(flat, n)),
      colourAt: (i) => ({ rgb: lut[i], dim: 1 }),
    };
    cam.rot = bestView(flat, false);
    cam.zoom = fillZoom(flat, cam.rot, fitOf([flat]).r);
    cam.tx = 0;
    cam.ty = 0;
    draw();
  }

  orbit(canvas, cam, draw, {
    zoomMin: 0.5,
    zoomMax: 8,
    onReset() {
      if (!state) return;
      cam.rot = bestView(state.flat, false);
      cam.zoom = fillZoom(state.flat, cam.rot, fitOf([state.flat]).r);
    },
  });

  return { show, draw, reduced };
}
