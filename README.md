# CIRPIN-web

The published site for [CIRPIN](https://github.com/sokrypton/CIRPIN) — circular-permutation-invariant
protein structure search, running entirely in the browser.

**This repository is generated. Do not edit it.** Every file here is assembled from the CIRPIN
repository by `web/tools/deploy.py`, which is where the source lives and where changes belong.
Editing a file here means the next deploy silently reverts it.

```
cd CIRPIN
python3 web/tools/deploy.py ../CIRPIN-web
cd ../CIRPIN-web && git add -A && git commit -m "deploy" && git push
```

## Why this is a separate repository

The site is data, and the data is rebuildable. The AlphaFold coordinate store alone is ~600 MB,
and regenerating it rewrites every byte — so keeping it here means CIRPIN's own history never
absorbs it, and this repository's history can be discarded whenever that becomes convenient.

It is also the only site: there is one deployment, at one URL, so there is never a question of
which one is live.

## What a visitor downloads

Nothing is uploaded, and no server does any work — the search, the neural network and the
structural alignment all run in the page. A query costs about 7 MB, of which 5.6 MB is the two
models. The databases are read by HTTP range request against a clustered index, so searching
3.5 million AlphaFold domains transfers roughly a megabyte rather than the 277 MB the index
occupies here.

## Layout

```
index.html app.js worker.js src/     the application
data/cirpin.*  data/progres.*        the two models, 1,063,680 parameters each
data/*.wasm                          SIMD kernels; the JS implementations remain the reference
data/coords-codebook.bin             the coordinate codebook, shared by every database
data/db/       data/coords.*         SCOPe40: 15,176 domains, index and coordinates
data/ted/                            AlphaFold TED: 3,466,144 domains, index and coordinates
```

Both databases store coordinates in one format, read by one decoder — see `src/coords.js`.

## Licence and citation

CIRPIN: Kolodziej, Abulnaga & Ovchinnikov, 2025. Progres: Greener & Jamali, 2025.
Structures come from the [RCSB PDB](https://www.rcsb.org/), [SCOPe](https://scop.berkeley.edu/)
and the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/).
