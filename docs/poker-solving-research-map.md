# Poker Solving Research Map

This note maps the public research relevant to building `pokedr-gto` into
implementation decisions. It is intentionally broader than the current code:
full-game solving, subgame solving, abstraction, CFR variants, pruning,
parallelism, and approximate value methods all affect the design.

## Core Observation

No-limit Hold'em has been studied heavily, but public results rarely describe a
simple exact "solve the whole no-limit tree naively" implementation. Strong
systems combine several ideas:

- a CFR-family solver;
- action and card abstraction;
- public-card isomorphism;
- pruning and/or sampling;
- subgame solving or depth-limited lookahead;
- highly engineered traversal/storage;
- sometimes learned value functions.

For this project, the most important lesson is that memory blowup is expected
unless the game is represented schematically. A full-game solver should share
action skeletons and isomorphic chance structure, while streaming or chunking
regret/strategy state that cannot stay resident.

## CFR Family

Classic CFR is the baseline theoretical framework. CFR+ made poker-scale tabular
solving practical enough to solve heads-up limit Hold'em, and is still the
right baseline for tests.

Discounted CFR is the most relevant exact variant for current work. Brown and
Sandholm's DCFR paper introduces regret and strategy discounting schemes that
outperform CFR+ in their tested large games and are compatible with important
pruning methods.

Hyperparameter schedules are worth treating as a production knob, not a toy.
Recent work on HS scheduling reports large speedups by dynamically changing the
discounting hyperparameter in DCFR/PCFR+ style algorithms. This does not remove
the need for fast traversal, but it changes the number of iterations needed to
reach a target exploitability.

Implementation implication:

- Keep CFR+ as a correctness baseline.
- Use DCFR+/scheduled DCFR+ as the main exact solver candidate.
- Report quality as root exploitability in `bb/100`, not iteration count.
- Treat PDCFR+/PCFR+ as a measured variant, not a default assumption.

Relevant sources:

- Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
  https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information
- Tammelin, "Solving Large Imperfect Information Games Using CFR+"
  https://arxiv.org/abs/1407.5042
- Brown and Sandholm, "Solving Imperfect-Information Games via Discounted
  Regret Minimization" https://arxiv.org/abs/1809.04040
- Zhang, McAleer, and Sandholm, "Faster Game Solving via Hyperparameter
  Schedules" https://arxiv.org/abs/2404.09097

## Full Game vs Subgame Solving

Libratus-style systems do not simply solve an isolated postflop spot without
constraints. The safe/nested subgame-solving literature exists because an
imperfect-information subgame generally depends on unreached parts of the full
game. Solving a local subgame incorrectly can create an exploitable strategy
even if the local subgame looks strong.

DeepStack-style systems go the other way: they continually re-solve a
depth-limited lookahead and replace the rest of the game with a learned value
function. That can be fast, but correctness depends on the value function and
continual-resolving construction.

Implementation implication:

- If we want exactness first, full-game CFR is the cleanest correctness model.
- If full-game memory is impossible, use safe boundary values, not scalar
  equity leaves.
- For real-time off-tree actions, safe/nested subgame solving is the relevant
  research line.

Relevant sources:

- Brown and Sandholm, "Safe and Nested Subgame Solving for
  Imperfect-Information Games" https://arxiv.org/abs/1705.02955
- Moravcik et al., "DeepStack: Expert-Level Artificial Intelligence in
  Heads-Up No-Limit Poker" https://www.science.org/doi/10.1126/science.aam6960
- Brown and Sandholm, "Superhuman AI for Heads-Up No-Limit Poker: Libratus
  Beats Top Professionals" https://www.science.org/doi/10.1126/science.aao1733
- Zhang and Sandholm, "Subgame Solving without Common Knowledge"
  https://arxiv.org/abs/2106.06068

## Abstraction

No-limit action spaces are too large without bet-size abstraction. Public
research treats action abstraction as a core problem, not as a UI detail.
Recent work even learns action abstractions because fixed abstractions can be
strategically wrong.

Card abstraction is more nuanced. For exact postflop or full-game work,
private combos must remain concrete because blockers matter. However, public
chance should be isomorphic, and user-facing ranges should usually be
rank-class/suit-symmetric so isomorphism remains effective.

Implementation implication:

- Reject or isolate suit-asymmetric user ranges in normal exact-solver configs.
- Use exact private combos internally.
- Use suit-isomorphic public chance classes for flop, turn, and river.
- Do not treat "legal actions + standard sizes" as final; action abstraction
  needs diagnostics and possibly automatic refinement.

Relevant sources:

- Hawkin, Holte, and Szafron, "Automated Action Abstraction of Imperfect
  Information Extensive-Form Games" https://ojs.aaai.org/index.php/AAAI/article/view/7880
- Li, Fang, and Huang, "RL-CFR: Improving Action Abstraction for Imperfect
  Information Extensive-Form Games with Reinforcement Learning"
  https://arxiv.org/abs/2403.04344
- Fu et al., "No-Regret Strategy Solving in Imperfect-Information Games via
  Pre-Trained Embedding" https://arxiv.org/abs/2511.12083

## Pruning and Sampling

Regret-based pruning is one of the strongest exact-CFR ideas for reducing work.
The important caveat is that CFR+ style clipping destroys some negative-regret
information needed for exact pruning bounds. A pruning implementation must keep
enough regret state to know when an action can resume.

Sampling methods reduce per-iteration work but add variance and usually change
the convergence profile. They are worth studying for full-game preflop scale,
but the exact full-runout path should remain the validation baseline.

Implementation implication:

- Exact RBP belongs behind a DCFR-style regret store that preserves negative
  regrets or equivalent resume information.
- Public chance sampling is a possible scale lever, but it should be compared
  against exact isomorphic chance, not silently replace it.
- Approximate sampling/pruning must report exploitability against the exact
  game when feasible.

Relevant sources:

- Brown and Sandholm, "Regret-Based Pruning in Extensive-Form Games"
  https://papers.neurips.cc/paper/5910-regret-based-pruning-in-extensive-form-games
- Brown and Sandholm, "Reduced Space and Faster Convergence in
  Imperfect-Information Games via Regret-Based Pruning"
  https://arxiv.org/abs/1609.03234
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive
  Games" https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html

## Parallel and GPU Solving

The public GPU/parallel CFR literature does not imply that every CFR operation
should be moved to the GPU. The useful decomposition is by information set,
tree node/layer, chance board, and batched leaf evaluation. Recent parallel CFR
work explicitly frames real-time solving as a pipeline problem, not a single
kernel problem.

Implementation implication:

- Separate steady-state iteration time from setup/build time.
- Keep terminal CFV, reach propagation, backup, and regret update separately
  profiled.
- Use batching only when it preserves exact reach-dependent values.
- Streaming state from disk is acceptable if resident memory is impossible, but
  it does not reduce terminal work by itself.

Relevant source:

- Li and Huang, "Real-Time Parallel Counterfactual Regret Minimization"
  https://arxiv.org/abs/2605.19928

## Commercial Solver Clues

Commercial solvers generally do not publish their full implementation details.
Their public docs and benchmark posts still reveal useful constraints:

- Practical quality is usually discussed in exploitability or Nash distance,
  not just iterations.
- Tree configuration is a first-class input.
- Commercial "AI" solves often combine exact solving with predictive or
  learned approximations.
- Exact river solving is easier than exact full-tree solving; river and turn
  subgames are commonly special-cased.

Implementation implication:

- The viewer/config should expose tree parameters and target exploitability.
- Exact full-runout solving should remain available for validation.
- A fast approximate path is likely necessary for interactive use, but it
  should be built after exact invariants are stable.

Relevant public sources:

- PioSOLVER technical details https://piosolver.com/docs/technical_details/
- PioSOLVER numbers/exploitability docs
  https://piosolver.com/docs/viewer/numbers_in_piosolver/
- GTO Wizard, "How Solvers Work" https://blog.gtowizard.com/how-solvers-work/
- GTO Wizard AI benchmarks https://blog.gtowizard.com/gto-wizard-ai-benchmarks/

## What This Means For `pokedr-gto`

The current full-game planner showing river-dominated state blowup is consistent
with the literature: no-limit river raise trees are expensive unless the solver
uses tight abstraction, schematic storage, pruning, or decomposition.

The next implementation direction should be:

1. Keep the semantic target as full-game CFR.
2. Store the game schematically:
   - preflop skeleton;
   - postflop action skeleton;
   - public-card isomorphism representatives;
   - combo permutation maps.
3. Stream `(preflop boundary, flop class)` regret/strategy chunks if resident
   state does not fit.
4. Share static terminal board tables and action skeletons across chunks.
5. Add action-tree diagnostics that identify river raise-chain explosion before
   running CFR.
6. Add exact exploitability tests on small full-game trees before scaling.

The most likely sources of large speedups are not micro-optimizations. They are:

- better action abstraction for high-SPR small-pot river branches;
- exact public-card isomorphism throughout the full game;
- regret-based pruning with correct negative-regret/resume state;
- scheduled DCFR+/PCFR+ iteration reduction;
- streaming/schematic storage to avoid impossible resident memory;
- possible future safe-solving or learned-value boundary once the exact path is
  trustworthy.
