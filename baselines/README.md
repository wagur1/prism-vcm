| ID | Codec | Delta-QP | Delta-x |
|---|---|---|---|
| M1 | Stock anchor | Off | Off |
| M2 | Pinned x265 quantOffsets | On | Off |
| M3 | Pinned x265 | Off | On, after verification |
| M4 | Pinned x265 quantOffsets | On | On, after verification |

Keep preset, GOP, AQ, cutree, SAO, and QG identical. If patched, additionally
report the same-binary/map-zero anchor and its difference to stock. Zero-map
bit identity is a benchmark prerequisite. CQP is primary only after GATE 0 PASS.
CRF/ABR/VBV and zero-sum ablations are reported separately.
