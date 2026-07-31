// Reading a file, whole or in part, and the one rule that keeps the two apart.
//
// THE BUG THIS MODULE EXISTS TO PREVENT. An HTTP cache is allowed to satisfy a byte-range request
// out of a complete cached entry: whole body, status 200. Slicing that is correct and the note used
// to say the only cost was transferring more bytes than asked for. The cost is 1,600x. A cluster
// read wants 27 KB; a cached 200 hands over the entire 45 MB shard. One AlphaFold query reported
// 99 MB against the 0.8 MB it needs, and it had been doing so for weeks.
//
// Measured on the live site: a ranged fetch of ted-cirpin-000.bin returned 200 with 45,000,000
// bytes, and the same request with cache: 'no-store' returned 206 with 100. Three forced
// edge-cache misses all answered 206, so the CDN honours Range — the complete entry was in the
// BROWSER's cache.
//
// A complete entry can only be there if something fetched the file whole. Something did: before the
// clustered index existed, the loader downloaded every code shard and the whole id table, by
// design. Those entries then answered every later range read. Hence the invariant this module
// enforces rather than documents:
//
//     A FILE IS EITHER RANGE-READ OR FETCHED WHOLE. NEVER BOTH.
//
// Both halves live here for the same reason: the previous arrangement had three near-identical
// range readers in three files, each carrying its own copy of the lesson, and only one of them had
// the fix.

const ranged = new Set();
const entire = new Set();

/** What each URL has been used for, for the invariant's error messages and for tests. */
export function fetchModes() {
  return { ranged: [...ranged], entire: [...entire] };
}

/** Forget the history. Tests only — a real page never wants this. */
export function resetFetchModes() {
  ranged.clear();
  entire.clear();
}

function claim(url, mode) {
  const other = mode === 'ranged' ? entire : ranged;
  if (other.has(url)) {
    throw new Error(`${url} has already been fetched ${mode === 'ranged' ? 'whole' : 'by range'} `
      + `and is now being fetched ${mode === 'ranged' ? 'by range' : 'whole'}. One or the other: a `
      + 'complete cache entry lets a range request be answered 200 with the whole body, which is '
      + 'how a 27 KB read became 45 MB.');
  }
  (mode === 'ranged' ? ranged : entire).add(url);
}

/**
 * One byte range.
 *
 * cache: 'no-store' keeps the local cache out of the way entirely, so the range is asked of the
 * origin and answered as a range. Nothing is lost by not caching these — they are small, every
 * origin this ships to honours them, and whatever gets re-read repeatedly is held in memory by the
 * caller.
 *
 * A 200 is still handled, because it is a legal answer and the bytes are all there. But it now
 * warns, because it means something upstream is ignoring the header and the transfer is orders of
 * magnitude larger than the read.
 *
 * @param {string} url
 * @param {number} start first byte, inclusive
 * @param {number} end last byte, inclusive
 * @param {(n: number) => void} [onBytes] called with the bytes actually used
 */
export async function fetchRange(url, start, end, onBytes) {
  claim(url, 'ranged');
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` }, cache: 'no-store' });
  if (r.status !== 206 && r.status !== 200) {
    throw new Error(`${url}: range request answered ${r.status}`);
  }
  let b = new Uint8Array(await r.arrayBuffer());
  if (r.status === 200) {
    if (b.byteLength < end + 1) {
      throw new Error(`${url}: answered 200 with ${b.byteLength} bytes, short of ${end + 1}`);
    }
    console.warn(`range request for ${end + 1 - start} bytes of ${url} was answered 200 with `
      + `${b.byteLength} bytes; the cache or the origin is ignoring Range`);
    b = b.subarray(start, end + 1);
  }
  // After the slice, so the figure is bytes USED. Counting the whole body made a cached 200 look
  // like a download even when nothing crossed the network, which is the other half of why the
  // number in the corner was alarming.
  if (onBytes) onBytes(b.byteLength);
  return b;
}

/**
 * A whole file, streamed so a progress bar can reflect real bytes rather than guess.
 *
 * @param {string} url
 * @param {(n: number) => void} [onBytes] every chunk, and the total at the end
 * @param {(n: number, got: number, total: number) => void} [onProgress]
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchWhole(url, onBytes, onProgress) {
  claim(url, 'entire');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  if (!onProgress || !r.body) {
    const b = await r.arrayBuffer();
    if (onBytes) onBytes(b.byteLength);
    return b;
  }
  const total = Number(r.headers.get('content-length')) || 0;
  const chunks = [];
  let got = 0;
  const reader = r.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(value.length, got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  if (onBytes) onBytes(got);
  return out.buffer;
}

/**
 * A whole JSON file.
 *
 * Counted like everything else. It was not, which is a small part of why the byte readout could be
 * trusted only as a lower bound.
 */
export async function fetchJSONWhole(url, onBytes) {
  claim(url, 'entire');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const text = await r.text();
  if (onBytes) onBytes(text.length);
  return JSON.parse(text);
}
