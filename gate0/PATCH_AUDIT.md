# x265_prism_residual_v1: first-build audit

**Status: best-effort patch, not yet verified against an actual 3.5 checkout or
compiled on Ubuntu.** The implementation workspace could not reach Bitbucket.
Line numbers are approximate; context and the named function are the anchors.
CONFIG v1.3 sections E-F0, F and G-C2 are binding. A successful patch application
or compilation is not evidence that offsets affect residual QP.

## Observation site and QP data flow

The proposed hook is `source/encoder/frameencoder.cpp`,
`FrameEncoder::getEncodedPicture(NALList&)`, immediately **after `m_done.wait()`**
and before handing `m_frame` back to `Encoder`. It calls the new header's
`x265_prism_residual_v1(*m_frame, *m_param)`. This observes the completed frame's
`FrameData::getPicCTU()` array. It is deliberately downstream of the QP decision;
it does not alter that decision or the bitstream.

The source reviewer must follow this chain in the fetched tag, confirming exact
names and order rather than accepting this anticipated layout as verified:

1. Input `x265_picture::quantOffsets` is copied into frame/lowres-owned storage.
   `source/common/lowres.cpp` owns/initializes lowres and AQ storage. In x265
   releases the actual AQ calculation is usually in
   `source/encoder/slicetype.cpp`, `LookaheadTLD::calcAdaptiveQuantFrame()`;
   confirm where the 3.5 branch applies AQ strength and adds quantOffsets to
   `qpAqOffset` / `qpCuTreeOffset`. Lowres allocation alone is not the QP decision.
2. `source/encoder/analysis.cpp`, `Analysis::calculateQpforCuSize()` and
   `Analysis::compressCTU()` (and their intra/inter paths) consume those offsets,
   round/clip the QP, and initialize candidate CU QPs. Check the exact Clip3
   expression and the 8-bit legal bounds; a requested offset is not necessarily
   the QP that reaches residual coding. Do not introduce a new clipping rule.
3. Track winning modes through `CUData::copyToPic()` and DQP handling, including
   `Analysis::checkDQP()` / `CUData::setQPSubCUs()` wherever those functions live
   in this tag. CUs without residual can inherit a predictor QP. RD candidates
   can be discarded; VBV can cause whole rows to be encoded again.
4. Confirm `m_done` is signaled only after final analysis, entropy coding and
   any VBV row restart have completed. Confirm committed `m_qp` and `m_cbf`
   arrays are still alive, unchanged by the snapshot, and not reset/recycled
   until after `getEncodedPicture()` returns. This is what makes the chosen
   observation final and post-AQ, rather than another trial QP.

Alternative sites: lowres/AQ logging is too early; `calculateQpforCuSize()` sees
trial values without final residual/skip information; `checkDQP()` can run on
losing modes; entropy coding can run in both analysis and final coding; an
unqualified `compressCTU()` hook can duplicate rows on VBV restart. A hook in
`Encoder::encode()` after `getEncodedPicture()` is a possible alternative if
3.5's completion boundary differs, but must run even when `pic_out == NULL`.
The encoder's per-frame **average** QP CSV is never a substitute.

## Trace semantics to verify

- Exactly one row per final leaf CU, not per candidate, CTU average, PU or TU.
  `m_numPartitions >> (2 * m_cuDepth[part])` must describe the leaf's contiguous
  Z-scan span. Verify these member names and minimum-partition semantics in 3.5.
- `m_cuPelX/Y + g_zscanToPelX/Y[part]` are luma pixel coordinates. Padding
  partitions are skipped before reading uninitialized depth. Check edge CTUs
  at 1920x1080 and confirm the visible 30x17 CTU domain, including partial
  bottom CTUs; exclude origins at y >= 1080, not the entire last CTU row.
- `m_qp[part]` is the luma QP actually used for the winning residual. Verify
  agreement with an independent decoder's CU/QG syntax inspection on a small
  intra and a reordered GOP stream. Stored QP inference in zero-CBF CUs is not
  evidence of map effect.
- `has_residual` ORs the luma and chroma CBF bitmasks across **all** minimum
  partitions/TUs within the leaf. Test split TUs, chroma-only residual, non-skip
  zero-residual and skip CUs. `isSkipped(part)` must be the final CU skip flag.
  PCM/transquant-bypass are outside this stack's pinned default settings.
- `frame.m_pts` must come from input `x265_picture::pts`, not POC, DTS, or encode
  order. Test nonconsecutive PTS (e.g. 100, 110, ...), B-frame reordering and
  delayed flush. The stock harness uses PTS 0..9; a separate reviewer probe is
  required to prove the nonconsecutive case.
- With `PRISM_X265_QP_TRACE` unset/empty: a getenv/branch per completed frame,
  no file I/O, traversal, or lock. With it set: one pthread mutex protects the
  whole frame, append/header creation, fflush and fclose. The header is included
  from exactly one translation unit. The environment must stay unchanged until
  every active encoder drains. Each sequential encode gets a fresh trace path;
  append mode is for frame accumulation, not for reusing old evidence.
- Run a separate multi-frame-thread/WPP probe and check unique (PTS,x,y) rows,
  one header, no interleaving inside rows and flush visibility before process
  exit. Normal GATE 0 locks frame-threads=1/pools=none/wpp=0 for reproducibility.
  Any instrumentation error in the native log fails the Python worker.

## First Kaggle build/review procedure

1. Record tag commits from `stack.json` and `build_log.txt`; compare x265 3.5
   with stock (same commit), and HM with HM-16.2. These tags are bootstrap
   references; archive/review the resolved SHAs and pins before later reuse.
2. Review every patch hunk in the actual source. `git apply` is tried first;
   `patch -p1 --forward --batch --verbose --fuzz=0` is the fallback. File names,
   failed hunk numbers/line ranges and rejected text go into `build_log.txt`
   and `patch-rejects.diff`. Even a fallback success with shifted lines needs
   source review. No script silently fabricates replacement instrumentation.
3. On failure, inspect the preserved partial tree/rejects and adjust the patch
   against the real source. Restore or move aside that generated checkout
   deliberately, then rebuild. The builder refuses unexpected tracked edits.
   Do not repeatedly apply a changed patch over a partially patched tree.
4. Verify the code above, compile, then run quick mode. It performs two
   ten-frame encodes and both stock decoders for each stream. `SMOKE_PASS` is
   only connectivity/effect evidence at QP 32/GOP 10; it never opens training.
5. Run the full 60-encode / 600-input-frame matrix (five QPs, two GOP modes,
   six controls/maps). Require independent FFmpeg HEVC and HM decode success,
   stock == patched/map-0 bit identity, same-binary/no-map == map-0, and signed
   residual-QP effects for +/-6 and checkerboard on every frame. Inspect native
   logs, bit counts and edge coordinates. Review version/info-SEI differences
   if bit identity fails; do not dismiss the failed gate or rewrite bitstreams.
6. Archive the exact patch, audit result, compiler/build log, commits, library
   and shim hashes, decoder hash, JSON reports, residual traces and bit CSVs.
   A fresh `BuildPin` measures an artifact; it is not a prior council-reviewed
   pin. The runner explicitly records `patch_review_required: true`.

## Known unresolved risks

Stock x265 can disable AQ in CQP or bypass offsets on particular paths. Verify
`Encoder::configure()` and the AQ path in 3.5. This patch does **not** change RC,
enable a hidden alternative mode, or force AQ back on: ignored offsets must
produce FAIL, even when both builds and both decoders work. A later functional
encoder patch needs its own source audit and zero-map/stock identity proof.

HM-16.2's historical make layout/targets and modern GCC compatibility still
need a real Ubuntu build. Only decoder libraries and the release decoder are
built, with warning errors disabled and the original C++98 dialect. Verify HM
without `-o` leaves reconstruction disabled; FFmpeg uses the null muxer. Raw YUV
must stay in memory. Network/apt restrictions, upstream tag movement or missing
repositories must remain nonzero build failures and a SKIP report.

## Council review 2026-11-01 (v2)

- v1 patch used `ctu->m_depth[part]`; x265 3.5 member is `m_cuDepth[part]` (C++ compile error in frameencoder.cpp, caught by GATE 0 quick v1 on Kaggle). Fixed per council verdict 20261101_021216 (CX-1/CX-3/ZC-2/ZC-3).
- NASM on Kaggle is 2.16.01 (sufficient); asm messages in v1 log were warnings, not errors. Assembly stays ON by default; ENABLE_ASSEMBLY=OFF only as recorded fallback (identical flags stock+patched; F0-scope only, never F1 latency evidence).

## Content provenance (council esc#4, 20261101_034554)

- `synthetic_frames` regenerates independent per-frame seeded noise (SeedSequence(0).spawn(10), PCG64). `synthetic_content_sha256` is recorded in gate0.json; any rerun whose hash differs must FAIL. The zero-map run in quick v4 remains valid evidence of skip-dominant behavior, not of map effect.
