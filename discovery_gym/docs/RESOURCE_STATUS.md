# Resource Status

Captured on 2026-09-25 in the isolated worktree before any future model workload. Values are host observations, not qualifications.

| Resource | Observed value | Scientific status |
|---|---|---|
| CPU | AMD Ryzen 7 170, 8 cores / 16 logical processors | available for bounded CPU checks; no model training run |
| RAM | 16,452,849,664 bytes (about 15.3 GiB) | host capacity only |
| C: disk | 193,588,322,304 bytes free (about 180.3 GiB) | host capacity only |
| Integrated GPU | AMD Radeon graphics, 512 MiB reported | not used or qualified |
| Discrete GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 4 GiB reported | present; no CUDA, driver, runtime, or workload qualification |

DG0 used only deterministic standard-library Python and created a small JSON evidence directory. No model weights, datasets, downloads, optimizer updates, GPU kernels, TPU sessions, or long jobs were used. A future DG1 estimate must measure the selected model's memory, step time, episode throughput, storage, and failure/recovery overhead on this host. Hardware qualification, local implementation evidence, and scientific support remain separate statuses.
