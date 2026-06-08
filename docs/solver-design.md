# Solver Design

The active solver is the node-local full-range public-tree CFR implementation:

- public tree shape comes from `pokedr-agent::build_flop_tree`;
- exact representative public chance nodes are expanded inside
  `NodeLocalCfrSolver`;
- each node owns its regret and strategy-sum rows in action-major order;
- terminal values are exact and use the optimized fold, river, all-in, and
  prefix-blocker paths from `node_cfr` and `terminal_cfv`;
- exploitability is computed by exact profile and best-response traversals over
  the same node-local tree.

The public CLI exposes one solving path: `solve-flop`, backed by
`NodeLocalCfrSolver`. Old arena, three-phase, storage-layout, and parallel-plan
experiments are kept under the crate-internal `legacy` namespace for tests and
historical comparison only. They should not be wired back into user-facing CLI
flags unless they are promoted as the single active solver.

CLI configuration:

- `build-tree`, `solve-flop`, and `board-isomorphism` accept one or more
  `--config` TOML files.
- Config files are merged field-by-field in command-line order, so later files
  override earlier files without replacing entire sections.
- Explicit CLI arguments override all config files.
- The supported sections are `[spot]`, `[tree]`, `[solver]`, `[output]`, and
  `[logging]`; see [solver-config.example.toml](solver-config.example.toml).
- Solver logs use `tracing` and can be controlled with `--log-level` or
  `[logging].level`.

Range input:

- `full` still means all 1326 private combos.
- Exact combos such as `AhAd` and comma-separated combo lists are supported.
- Traditional poker range tokens are supported: `TT+`, `99+:0.7`, `ATo+`,
  `A2s+`, `KQs`, `AK`, etc.
- A `:weight` suffix applies only to newly introduced combos. If tokens overlap,
  the first token keeps ownership of that combo. For example `TT+,88+:0.7`
  keeps TT+ at weight `1.0` and adds only 88-99 at weight `0.7`.

The replacement target remains a layout that can later be made GPU-portable, but
the current correctness and performance baseline is the node-local CPU solver,
not the old dense phase solver:

- fixed public tree IDs;
- dense private range indices;
- exact public chance isomorphism;
- dense action-major regret and strategy rows per decision node;
- no user-facing choice among multiple CFR engines.

Detailed postflop GPU CFR math and the showdown reuse boundary are documented in
[postflop-gpu-cfr.md](postflop-gpu-cfr.md).

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
