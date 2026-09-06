# First executable BRAMASTRA results

Date: 2026-09-06. These are local development experiments on randomly initialized models. They are not AGI, independent-transfer, or learned self-diagnosis results.

## What was built

`bramastra_lab/` implements a dense causal Transformer with tied byte embeddings, RoPE, RMSNorm, and SwiGLU; deterministic paired binding worlds; explicit answer/EOS supervision; strict free generation; bounded training; raw evidence receipts; and full local continuation checks. The program commits an experiment manifest before training and uses a transparent fixed rule to nominate the next research action.

The executed profile has **117,312 parameters**, two layers, width 64, and a 32-token context. This is a cheap instrument derived from the blueprint, not the full B0 or B1 campaign. No pretrained weights were used. CPU FP32, a constant 3e-4 AdamW learning rate, 600 updates per arm, and batches of 16 were held fixed.

The target system is described in [RESEARCH_LOOP.md](RESEARCH_LOOP.md). The distinction matters: Codex chose these experiments; the small core learned answer behavior; the controller did not learn an experiment policy.

## 1. Teaching stopping repairs the complete-answer contract

Each comparison held initial parameters, examples, sampled batches, update count, and the answer-loss coefficient fixed. The intervention adds EOS supervision. The loss is `mean answer CE + terminal_weight * mean EOS CE`; weights are zero and one. This is deliberately different from renormalizing a shared token loss, which would also change the answer-loss coefficient.

| Seed | Training worlds / queries | Without EOS: correct answer prefix | Without EOS: complete answers | With EOS: complete answers |
|---|---:|---:|---:|---:|
| 601 | 16 / 32 | 32/32 | 0/32 | 32/32 |
| 602 | 16 / 32 | 32/32 | 0/32 | 32/32 |

Both no-EOS arms hit the generation limit on all training queries. Both EOS arms stopped correctly on all training queries. This establishes the effect on this development instrument across two initializations and datasets. It does not prove that adding EOS would solve Citadel arithmetic; that remains a separate experiment with its own task and model.

Sources: [seed 601](../../artifacts/bramastra/terminal_dev_seed601/result.json), [seed 602](../../artifacts/bramastra/terminal_dev_seed602/result.json). Their neighboring arm JSON files retain every prediction and stop reason.

## 2. Learning the training set did not establish transfer

| EOS-supervised run | Fresh-world complete answers | Fresh worlds with both query variants correct | Changed-rendering complete answers |
|---|---:|---:|---:|
| Seed 601, 16 training worlds | 22/128 (17.2%) | 1/64 | 0/128 |
| Seed 602, 16 training worlds | 33/128 (25.8%) | 2/64 | 0/128 |

The model satisfies the tiny-set learnability check and has substantial unresolved generalization failures. The rendering shift changes syntax and order on fresh worlds; it does not isolate a single causal factor. Invalid UTF-8 outputs on this shift are reported as failures, not discarded.

## 3. More training variety improved an aggregate while leaving query control unearned

An exploratory follow-up increased the training pool from 16 to 256 worlds with the same seed, model, update count, batch size, and optimizer. The EOS arm reached 261/512 on its training queries and **62/128 (48.4%) on fresh-world queries**.

However, it answered both query variants correctly in **0/64 fresh worlds**. In **62/64 worlds**, it returned the same answer despite the changed query. Explicit policies that always copy the first or last fact's value each achieve **64/128 (50%)** on this balanced task while also scoring zero on both-correct pairs.

The 48.4% score therefore does not establish useful query-sensitive selection. The deterministic 50% baseline is a measured policy result, not an assigned chance rate for free text. The data-diversity follow-up was chosen after inspecting development results, and is not a preregistered confirmatory comparison; exclusion of overlapping training worlds can also change the evaluation cohort. No significance claim is made for its cross-run difference.

Source: [diversity experiment](../../artifacts/bramastra/binding_diversity_dev_seed601/result.json). [Recomputed analysis](../../artifacts/bramastra/analysis.json) binds the raw arm receipts and independently parses the compact prompts to check the truth and query-blind baselines.

## Engineering evidence and cost

All six arms changed real parameters, populated optimizer moments, completed the same 600-update budget, and passed exact local continuation for model parameters, optimizer state, and sampler choice. The continuation probe's extra update is excluded from final evaluation and reported separately. Full states exist locally; Git contains the manifests, source snapshots, datasets, raw predictions, and results, not the `.pt` checkpoints.

The three experiment loops reported approximately **103.5 seconds total** on local CPU, including their model/evaluation/continuation work inside the measured loop. Shell startup, dependency imports, source capture, unit tests, and documentation are outside that timer. No Kaggle TPU allocation was consumed.

The focused checks cover masking, encoding, supervision alignment, complete-answer scoring, query-marker parsing, continuation integrity, and recomputation of the committed receipts. The code supports local CPU and optional CUDA; TPU execution, remote recovery, mixed precision, and multi-device performance remain unverified and outside this implementation.

## Next research decision

Keep explicit terminal supervision. Investigate query-sensitive selection at the current scale before adding model capacity or a learned discovery controller. The next comparison should isolate one change, such as preserving complete query-swap groups in each minibatch, against the current sampler at the same data and update budget. Require both-correct fresh-world gains beyond query-blind controls; keep rendering transfer separate.

That is a proposed next experiment, not an inferred solution. The current contribution is an executable core and evidence chain that can learn, fail visibly, and inform the next decision without confusing formatting, memorization, aggregate accuracy, and general intelligence.
