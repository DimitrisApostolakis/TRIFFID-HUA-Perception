# UDC tracker attribution

Derived from theodouk11/udc_tracker, revision
`11188042c8b487374a75b609424998df3f041522`, `tracker_inference.py`.
Original license: Apache-2.0, reproduced in UDC_LICENSE.
The upstream repository derives its detector framework from Intellindust-AI-Lab/FT-FSOD.
Only tracking primitives are included here; no detector weights or framework.

Modifications: detector/CLI/domain constants removed; settings and IDs made
instance-local; per-frame birth suppression updated; NumPy-containing gallery
entries removed by identity rather than array equality. The adapter adds
1-Hz elapsed-time handling, class groups and current-observation-only output.
HSV appearance recovery remains heuristic and is not learned ReID.

2026-11-06: apply IoU admissibility before Hungarian assignment and retain the
exact high/low detection index for each current match. Wrappers preserve the
associated role evidence and appearance rather than rematching filtered boxes.

Original downloaded source SHA256: `f9010c0a0690aee9d163a3dcd0583d99ca89d37ec20a65d1e1278662a8350247`.
