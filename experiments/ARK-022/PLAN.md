# ARK-022 — DORMANT CAPABILITY RETENTION (design draft; DESIGN_READY)

**OBJECTIVE:** can a learned capability survive 250/500/1000/2000/4000-update replay-free
gaps, and does the answer depend on the protection history that preceded dormancy
(plastic / static-replay / Guardian)?

**DESIGN:** identical replay-free dormancy arms for all protection histories; SEALED A
evaluation only at the frozen gap endpoints (no peeking-driven adaptation); real-text
continuation continues during the gap.

**METRIC:** sealed A robust-min at each endpoint; survival curves per history.

**CHANGES OUR MIND IF:** protection history has zero effect on dormancy survival — replay
buys nothing durable (report it); strong history effect — protection builds durable
structure, motivating ARK-030 (memory compression).

**STATUS:** DESIGN_READY. Run after ARK-021 (its PRESERVED/RECONSTRUCTED classification
determines interpretation). Reuses V4 parents/tasks/checkpoint machinery unchanged.
