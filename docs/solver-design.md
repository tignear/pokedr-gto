# Solver Design

The previous solver is intentionally discarded. The replacement targets a dense,
GPU-portable CFR layout:

- fixed public tree IDs
- dense private range indices
- dense `infoset x private_bucket x action` regret and strategy arrays
- public chance sampling or public chance batching, not private external sampling
- observed action likelihood computed for the whole range
- no dynamic `HashMap` node allocation in the hot solver path

Detailed postflop GPU CFR math and the showdown reuse boundary are documented
in [postflop-gpu-cfr.md](postflop-gpu-cfr.md).

Research anchors:

- Juho Kim, "GPU-Accelerated Counterfactual Regret Minimization", arXiv:2408.14778.
- Juho Kim and Tuomas Sandholm, "Parallelizing Counterfactual Regret Minimization", arXiv:2605.14277.
- Boning Li and Longbo Huang, "Real-Time Parallel Counterfactual Regret Minimization", arXiv:2605.19928.

Implementation order:

1. CPU reference dense regret-matching and CFR update.
2. Dense public-tree and range tensor model.
3. Full-range public-chance CFR traversal.
4. Belief update from average strategy for every private bucket.
5. GPU backend for regret, strategy, belief, and batched leaf evaluation.

Postflop action abstraction:

- Build a small canonical action set per node instead of taking the raw union of
  every plausible sizing.
- Check/call/fold are forced when legal.
- Standard bet/raise sizes are street-aware and intentionally sparse.
- Observed opponent sizings are forced into the abstraction, but nearby standard
  sizes are replaced instead of added. This avoids splitting strategy frequency
  across near-equivalent actions.
- Near all-in sizings are absorbed into all-in.
- Aggressive actions are capped; observed sizings and all-in have higher priority
  than generic standard sizes.

Acceptance targets:

- Correctness first: CPU reference and GPU dense kernels must match within `1e-5`
  only on small deterministic fixtures. Larger runs should rely on invariants,
  checkpoint metrics, and performance regressions instead of full CPU comparison.
- GPU correctness tests run through a `harness = false` smoke binary so Dozen/wgpu
  resources stay on the main thread.
- Runtime target: end-to-end agent action selection should reach `0.1s/hand` on
  the benchmark machine before arena strength is treated as meaningful.
- Strength target: after runtime is under control, evaluate against fixed weak
  baselines and target `+1 BB/hand` mean over a statistically meaningful sample.
