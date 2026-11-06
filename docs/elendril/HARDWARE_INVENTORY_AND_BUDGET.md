# Hardware inventory and budget

Local Windows inventory exposed one NVIDIA GeForce RTX 4050 Laptop GPU, driver 591.62, with 6,141 MiB VRAM. The idle sample showed 5,920 MiB free, 44 °C and 0% utilization. Python 3.11.15 had PyTorch 2.14.0+cpu, CUDA unavailable, and zero framework-visible accelerators. There was no configured SLURM/PBS allocation or scheduler command. Access to the reported 40–50 GPUs remains unverified. No additional hosts were contacted. The CPU is an eight-core, sixteen-thread AMD Ryzen 7 170. About 6.8 GiB of physical RAM and 203 GiB of C-drive space were available at initial admission.

Local training ran with one intra-op thread, serial arms within each registered campaign, a 2 GiB available-RAM launch floor, a 1 GiB runtime floor and a 180-second process cap. The separately launched replay, diagnostics and repair jobs stayed small. Process-tree memory and wall time for the primary campaign are in the [resource receipt](../../artifacts/elendril/eld-dev-004/resources.jsonl). Local GPU hours and GPU energy use were zero. CPU energy was unmeasured.

## Owner-run accelerator lane

The [Colab T4 notebook](../../notebooks/ELENDRIL_004T4_COLAB.ipynb) pins the experiment source and CUDA wheel. It requires a single T4 and 8,000 MiB free at admission, runs a CPU/CUDA loss parity canary and an exact same-device restart canary, and keeps the larger campaign behind an explicit switch. The expanded four-layer profile uses four arms, three seeds and 1,536 updates per arm. A 75-minute session deadline and 15-minute segment ceiling bound execution. Each 384-update boundary has a local checkpoint and a verified Drive mirror. Use one publisher per run ID. A mismatch between checkpoint and evidence heads stops for reconciliation.

Notebook JSON and cell syntax passed locally. The expanded CPU profile ran and rescored a two-step canary. CUDA and Drive remain owner-run gates. Colab hardware availability is determined at runtime. [Colab FAQ](https://research.google.com/colaboratory/faq.html) · [Official PyTorch wheel catalogue](https://pytorch.org/get-started/previous-versions/).

If a separately authorized multi-GPU allocation becomes accessible, record device UUIDs, load, free memory, host identity and checkpoint storage before scheduling. Qualify one device first, then run independent seeded arms on disjoint devices. Multi-device independent arms are separate replications. Distributed single-model training needs a separate protocol and authorization.
