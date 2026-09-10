# ARK-019 V3 PRE-EXECUTION HORIZON AMENDMENT

**Status:** FROZEN BEFORE V3 IMPLEMENTATION / EXECUTION.

The mandatory continuation horizon is **1000 updates per arm**, not 1200. All other V3 arms, seeds, thresholds, replay doses, cap semantics and success criteria remain unchanged.

Reason: the actual ARK-018 ~21M subject costs roughly the same order of compute per mixed update as the completed real-text campaign. Preserving the scientifically more important design properties — 4 matched sets, both independent parents, all 4 causal arms, frequent CONTROL/SEALED measurement and exact-resume machinery — is higher value than spending 20% more horizon on each arm and risking a Colab wall failure. Historical ARK-018 binding acquisition on the relevant SCIENCE_ONLY substrate qualified around 300–400 updates, so 1000 updates still leaves a substantial post-acquisition continuation interval while retaining the full multi-seed causal structure.

This change is prospective and outcome-blind: no ARK-019 V3 model update has run. Runtime calibration must still fail closed if the complete fixed 1000-update design cannot conservatively fit the 175-minute campaign wall. The runtime resolver may not shorten the experiment further.