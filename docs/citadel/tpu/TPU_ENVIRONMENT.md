# TPU_ENVIRONMENT.md

Notebook-startup probe. Fail-closed. Never hard-code the device generation:
the current expectation is Kaggle free-TPU `v5e-8`, but the probe records
whatever is actually present and aborts clearly when unsupported.

## Required recorded fields (`TPU_ENVIRONMENT.json`)

```text
tpu_present: bool
tpu_generation: string (as reported, e.g. "v5e", "v5p", "v4", "v3", "unknown")
xla_devices: int (e.g. xm.xrt_world_size())
per_device_memory: string-or-null (if observable)
torch_version: string
torch_xla_version: string
xla_runtime: string (e.g. "PJRT" + backend string)
host_ram: string
local_disk_free: string
kaggle_session_limits: string (as displayed by the environment)
probe_utc: string
probe_pass: bool
```

## Fail-closed rules

1. If `tpu_present == false` or `xla_devices < 1` → abort with
   `ABORT_NO_TPU`. No CPU/GPU fallback run may be recorded as a TPU receipt.
2. If `torch_xla` import fails or version is unpinned/unknown → abort with
   `ABORT_XLA_VERSION`. Record attempted versions.
3. If expected 8 devices are absent (e.g. single-device VM) → single-device
   milestones (T0/T1) may proceed; the 8-device milestone (T2) is BLOCKED,
   not downgraded.
4. The environment receipt is a prerequisite field inside every later TPU
   receipt (`TPU_ONE_UPDATE.json`, `TPU_RESUME.json`, `TPU_8_DEVICE.json`,
   `TPU_CALCULATOR_CHECKPOINT.json`, `TPU_THROUGHPUT.json`). A TPU receipt
   without an embedded environment block is invalid.

## Budget note

Probe cost: seconds. No model, no training. This is rung 0 of the compute
ladder and gates everything below.
