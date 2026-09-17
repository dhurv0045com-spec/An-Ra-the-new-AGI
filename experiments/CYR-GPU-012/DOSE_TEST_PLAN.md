# DOSE-001 — development dose sensitivity

Registered before execution. Assistant-selected bounded next step, not a user
specified protocol. Two-arm 600-update idea and mechanism probe NOT executed.
This is NOT the curriculum-vs-control causal experiment; no such claim allowed.

Inputs: original CYR012 FINAL model SHA256
94e82920eaf03089630b85b9061bd0fa680b629c0bcd526e58bf301712127a99;
exact prior one-step FIXTURE.json SHA256
e726d2854bacd1fd21feac7e6e8022e5b48d3dd99161a172797bc70deecac4e4.
64 fixed training examples repeated in identical order each step; 100 development
holdout examples excluded by canonical pair from training and original partitions.
Fixture already consumed: not fresh or sealed. Parent checkpoint remains immutable.

Primary lane: 200 FP32 Adam steps, lr1e-5, betas(.9,.999),eps1e-8,weight_decay0,
clip norm1, fresh optimizer. Same frozen model/tokenizer/render and answer+EOS
mean CE as ONE-STEP-001. Model seed3301, execution seed8121. No sampling.

Measure at 0,50,100,150,200: teacher-forced whole-answer accuracy including EOS,
raw token arrays, answer+EOS loss on holdout and training64. Also free greedy exact
with EOS on holdout100 at 0 and200, using frozen _generate_texts, cap8,batch32.
Free greedy at these endpoints is descriptive validation of deployable behavior,
not a replacement for the original conditional rule below. Save FINAL candidate
model-only and reload to assert exact state equality and matching final greedy rows.

Conditional second lane: ONLY if all five teacher-forced counts ==27/100,
run one lr1e-4 lane, 200 steps, same parent and fresh optimizer with identical other
settings. Otherwise record second lane NOT_RUN. No further LR or dose trials.
Any invalid/incomplete lane blocks second lane. Initial count must equal prior27.
No selection or production promotion. Final-step deltas reported literally;
intermediate maximum distinguished from final. No significance/AGI/mechanism claim.
A lack of improvement at this dose does not reject learning at other doses.

Resource guard: local RTX4050, 2 CPU threads; 3GiB CUDA allocation cap; preflight
RAM>=3GiB,GPU free>=4GiB,temp<85C. Check at every25 steps and evaluation boundary;
abort RAM<2GiB,temp>=85C,total wall>600s. Terminal cap600s; sequential lanes with
verified CUDA release. No parallel compute. Preserve failures and partial traces.
Hash this plan, runner, imported frozen source, parent, fixture before training;
verify source/parent after. Output outside repo dose01; refuse existing directory.
Scope: local exploratory experiment, no commits/pushes, original verdict unchanged.
