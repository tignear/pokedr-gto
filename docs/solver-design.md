# Solver Design

The previous solver is intentionally discarded. The replacement targets a dense,
GPU-portable CFR layout:

- fixed public tree IDs
- dense private range indices
- dense `infoset x private_bucket x action` regret and strategy arrays
- public chance sampling or public chance batching, not private external sampling
- observed action likelihood computed for the whole range
- no dynamic `HashMap` node allocation in the hot solver path

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
