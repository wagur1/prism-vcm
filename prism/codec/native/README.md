# Pinned x265 shim and GATE 0

Build locally with CMake, explicitly setting `X265_INCLUDE_DIR` and `X265_LIBRARY`
to the matching header/library. The shim checks struct sizes and binds the
header-versioned 8-bit C API. Set `PRISM_X265_SHIM` and `PRISM_X265_LIBRARY` to
absolute paths. Record `X265Library().pin` (both binary SHA256s, version, ABI,
picture size); review and freeze this JSON before running the spike. The stock
and experiment builds must use identical preset/AQ/cutree/SAO/QG/GOP settings.
Never load a shim compiled against a different header. No analysis-reuse path.

The required realized-QP instrument is **x265_prism_residual_v1**. A pinned source
build must add this instrumentation at the final residual coding-unit QP decision,
after AQ/offset clipping, and emit CSV to `PRISM_X265_QP_TRACE`:

```
frame,x,y,qp,has_residual,skip
0,0,0,32,1,0
```

`frame` is input PTS (not encode order); `x,y` are luma pixel coordinates. Include
final CU QP, nonzero coded residual and skip flags. Use a thread-safe writer. The
exact insertion site must be audited against the pinned source; this repo does
not invent traces or claim an unbuilt instrumentation patch. Stock x265 CSV
provides per-frame `Bits`, **not** positional QP evidence. Missing traces fail the
spike. CU partition changes are matched only at shared residual block positions.

Run `python scripts/spike_gate0.py --pin build-pin.json --reference-command-json
'["TAppDecoderStatic", "-b", "{bitstream}"]'` with an installed HEVC reference
decoder configured to discard reconstruction. FFmpeg is the other decoder. Raw
YUV never goes to disk. Every condition uses ten frames, across intra and GOP,
all five primary CQP values, maps zero/+6/-6/checkerboard. Reports retain whole
bitstream bytes, frame bits, trace evidence, and zero-map/stock identity. A 15%
bit change alone cannot pass the gate.

For an offsets patch, provide `--stock-shim`, `--stock-library`, `--stock-pin` for
the unpatched stock build. The same-binary no-map and map-zero comparison is also
required and always performed by the harness; report
any difference to stock. Training must reject SKIP/FAIL. No local codec build,
source tree, trace instrumentation or decoder is bundled or downloaded.
