# CYR-GPU-009 POSTRUN AUDIT — INPUT TO CYR-GPU-010

Returned bundle SHA256: `dc15f14d3bc81551b1f0b00285faa4b23c9e68f1341405377959a7aba108f216`.

Environment: Tesla T4, CUDA, torch 2.11.0+cu128. Runtime: 5,192.566 seconds (86.54 minutes). Resolved proxy: TINY, 1,647,104 parameters.

Each parent consumed 2,000,000 actual real tokens and 15,632 optimizer updates.

Seed 707 train-probe complete exact: 0.9375 at ~400k tokens, then 1.0 through the remainder. DEV_CONTROLLER: 0, 0, 0, 0, 0.0104167 at 2M.

Seed 808 train-probe complete exact: 1.0 at 400k/800k/1.2M, 0.9375 at 1.6M, 1.0 at 2M. DEV_CONTROLLER remained 0 throughout.

Neither seed reached G90. No HIGH/LOW continuation arm executed. Official verdict: `INCONCLUSIVE_NO_COMPLETE_MATCHED_PAIR`.

Scientific interpretation used for CYR-GPU-010: the immediate bottleneck is capability formation/generalization at TINY, not retention.
