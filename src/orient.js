// Which way round to show a structure.
//
// A fold has no up, so something has to choose, and the choice matters: a random orientation hides
// a third of the structure behind itself and wastes the box it is drawn in. The widest two axes of
// the Cα cloud are the view that shows the most of it, which is why the atlas already drew its
// thumbnails that way — this is that code, lifted out so both it and the main viewer use one copy.
//
// The axes come from the covariance matrix's eigenvectors. It is 3x3, so a few Jacobi sweeps are
// exact enough and there is no dependency to pull in.

/**
 * The principal axes of a Cα cloud, widest first.
 *
 * @param {ArrayLike<number>} flat xyz, xyz, ...
 * @returns {{axes: number[][], centre: number[], spread: number[]}} axes[0] is the widest
 *   direction, axes[2] the thinnest; spread is the standard deviation along each.
 */
export function principalAxes(flat) {
  const n = Math.floor(flat.length / 3);
  const m = [0, 0, 0];
  if (!n) return { axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], centre: m, spread: [0, 0, 0] };
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) m[d] += flat[i * 3 + d] / n;

  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const v = [flat[i * 3] - m[0], flat[i * 3 + 1] - m[1], flat[i * 3 + 2] - m[2]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += v[a] * v[b];
  }
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 12; sweep++) {
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(C[p][q]) < 1e-12) continue;
        const th = 0.5 * Math.atan2(2 * C[p][q], C[q][q] - C[p][p]);
        const c = Math.cos(th);
        const s = Math.sin(th);
        for (let k = 0; k < 3; k++) {
          const a = C[k][p];
          const b = C[k][q];
          C[k][p] = c * a - s * b;
          C[k][q] = s * a + c * b;
        }
        for (let k = 0; k < 3; k++) {
          const a = C[p][k];
          const b = C[q][k];
          C[p][k] = c * a - s * b;
          C[q][k] = s * a + c * b;
          const va = V[k][p];
          const vb = V[k][q];
          V[k][p] = c * va - s * vb;
          V[k][q] = s * va + c * vb;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((a, b) => C[b][b] - C[a][a]);
  const axes = order.map((o) => [V[0][o], V[1][o], V[2][o]]);
  const spread = order.map((o) => Math.sqrt(Math.max(0, C[o][o]) / n));

  // Right-handed, so the third axis is depth rather than a mirror image. A reflected structure is
  // a different molecule, and a viewer that silently shows one is worse than one that picks badly.
  const cross = [
    axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
    axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
    axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
  ];
  if (cross[0] * axes[2][0] + cross[1] * axes[2][1] + cross[2] * axes[2][2] < 0) {
    axes[2] = axes[2].map((v) => -v);
  }
  return { axes, centre: m, spread };
}

/**
 * A rotation that puts the cloud's widest axes across the box it is drawn in.
 *
 * `portrait` swaps which axis goes across: the side-by-side panels are tall and narrow, so the
 * widest direction should run down them, while the single viewer is square-to-landscape and wants it
 * across. Returned row-major, the layout the viewer's projection already uses — its rows are the
 * screen x, y and depth directions.
 *
 * @param {ArrayLike<number>} flat xyz, xyz, ...
 * @param {boolean} [portrait]
 */
export function bestView(flat, portrait = false) {
  const { axes } = principalAxes(flat);
  const [a0, a1, a2] = axes;
  const rows = portrait ? [a1, a0, a2] : [a0, a1, a2];
  // Swapping two rows flips the handedness, so the depth row flips back with it.
  const sign = portrait ? -1 : 1;
  return [
    rows[0][0], rows[0][1], rows[0][2],
    rows[1][0], rows[1][1], rows[1][2],
    sign * rows[2][0], sign * rows[2][1], sign * rows[2][2],
  ];
}

/**
 * How much to scale so the oriented cloud fills its box.
 *
 * The viewer normalises onto the bounding SPHERE, which is the radius of the structure in its
 * longest direction whatever way it is turned. That is what a spinnable view needs — nothing ever
 * clips — but it means a flat or elongated fold, which is most of them, sits inside a sphere far
 * larger than its silhouette and comes up around half the size of its box.
 *
 * The silhouette is computed through the viewer's own projection, perspective included, rather than
 * estimated: the first attempt ignored the perspective divide and left the structure at two thirds
 * of the box. Screen offset is `v * R * pe(z)` and only R carries the zoom, so the extent scales
 * exactly with it and the fit is a division rather than a search.
 *
 * @param {ArrayLike<number>} flat xyz, xyz, ...
 * @param {number[]} rot row-major, as returned by bestView
 * @param {number} radius the sphere radius the viewer normalises by
 * @param {number} [margin] fraction of the half-box to leave empty at the tightest edge
 * @param {number} [cap] never magnify past this, so nothing can blow up absurdly
 */
export function fillZoom(flat, rot, radius, margin = 0.06, cap = 3) {
  const n = Math.floor(flat.length / 3);
  if (!n || !radius) return 1;
  const c = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) c[d] += flat[i * 3 + d] / n;
  // PE_MAX is the perspective factor at z = 1, the value the viewer divides R by, so working in
  // units of it makes the box exactly one half-extent across.
  const PE_MAX = 1 / (1.9 - 0.55);
  let h = 0;
  for (let i = 0; i < n; i++) {
    const v = [flat[i * 3] - c[0], flat[i * 3 + 1] - c[1], flat[i * 3 + 2] - c[2]];
    const x = (rot[0] * v[0] + rot[1] * v[1] + rot[2] * v[2]) / radius;
    const y = (rot[3] * v[0] + rot[4] * v[1] + rot[5] * v[2]) / radius;
    const z = (rot[6] * v[0] + rot[7] * v[1] + rot[8] * v[2]) / radius;
    const pe = 1 / (1.9 - z * 0.55) / PE_MAX;
    h = Math.max(h, Math.abs(x * pe), Math.abs(y * pe));
  }
  return Math.max(1, Math.min(cap, (1 - margin) / Math.max(h, 1e-6)));
}
