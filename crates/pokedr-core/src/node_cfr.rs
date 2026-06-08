use crate::cards::{Board, Card};
use crate::isomorphism::{
    all_suit_permutations, next_card_isomorphism, private_combo_permutation_indices,
};
use crate::range::{ComboWeight, RangeSpec};
use crate::terminal_cfv::{
    PreparedTerminalBoard, terminal_side_values_prefix_blocker_sorted_board_targets_into,
};
use crate::tree::{ActionKind, Player, PublicNodeKind, PublicTree, Street, TerminalReason};
use crate::{RealCfrAverageStrategy, RealCfrConfig, RealCfrExploitability, RealCfrVariant};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLocalCfrSummary {
    pub iterations: u32,
    pub states: usize,
    pub decision_states: usize,
    pub action_slots: usize,
    pub terminal_evals: usize,
    pub elapsed_ms: f64,
    pub oop_update_pass_value: f32,
    pub ip_update_pass_value: f32,
    pub storage_gib: f64,
    pub scratch_allocations: usize,
    pub terminal_cache_hits: usize,
    pub terminal_cache_misses: usize,
    pub terminal_calls: usize,
    pub terminal_ns: u64,
    pub fold_calls: usize,
    pub fold_ns: u64,
    pub showdown_calls: usize,
    pub showdown_ns: u64,
    pub showdown_only_calls: usize,
    pub showdown_only_ns: u64,
    pub allin_calls: usize,
    pub allin_ns: u64,
    pub allin_flop_calls: usize,
    pub allin_flop_ns: u64,
    pub allin_turn_calls: usize,
    pub allin_turn_ns: u64,
    pub allin_river_calls: usize,
    pub allin_river_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeLocalCfrIterationSummary {
    pub iteration: u32,
    pub terminal_evals: usize,
    pub elapsed_ms: f64,
    pub oop_update_pass_value: f32,
    pub ip_update_pass_value: f32,
}

#[derive(Debug, Clone)]
pub struct NodeLocalSolutionSnapshot {
    pub iterations: u32,
    pub oop_combos: Vec<ComboWeight>,
    pub ip_combos: Vec<ComboWeight>,
    pub nodes: Vec<NodeLocalSolutionNode>,
}

#[derive(Debug, Clone)]
pub struct NodeLocalSolutionNode {
    pub id: usize,
    pub public_node: usize,
    pub board: Board,
    pub street: Street,
    pub pot: u32,
    pub player: Player,
    pub kind: NodeLocalSolutionNodeKind,
    pub children: Vec<usize>,
    pub actions: Vec<ActionKind>,
    pub strategy: Option<NodeLocalStrategySnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLocalSolutionNodeKind {
    Decision,
    Chance,
    Terminal { reason: TerminalReason },
}

#[derive(Debug, Clone)]
pub struct NodeLocalStrategySnapshot {
    pub player: Player,
    pub combos: usize,
    pub actions: usize,
    pub action_major: Vec<f32>,
}

#[derive(Debug)]
pub struct NodeLocalCfrSolver {
    tree: PublicTree,
    oop_range: RangeSpec,
    ip_range: RangeSpec,
    oop_combos: Vec<ComboWeight>,
    ip_combos: Vec<ComboWeight>,
    nodes: Vec<NodeLocalNodeCell>,
    node_by_key: BTreeMap<(usize, u64), usize>,
    terminal_cache_index_by_key: BTreeMap<u64, usize>,
    terminal_cache: Vec<NodeLocalTerminalCache>,
    fold_cache_index_by_key: BTreeMap<u64, usize>,
    fold_cache: Vec<NodeLocalFoldCache>,
    allin_oracle_index_by_key: BTreeMap<u64, usize>,
    allin_oracles: Vec<NodeLocalAllInOracle>,
    combo_permutations: BTreeMap<u8, ComboPermutationMaps>,
    oop_same_ip_combo_indices: Vec<Option<usize>>,
    ip_same_oop_combo_indices: Vec<Option<usize>>,
    completed_iterations: u32,
    action_slots: usize,
    decision_states: usize,
    profile_enabled: bool,
    profile: NodeLocalProfile,
    scratch_pools: Vec<Mutex<Vec<NodeLocalScratch>>>,
    terminal_side_cache_enabled: bool,
}

#[derive(Debug, Default)]
struct NodeLocalProfile {
    scratch_allocations: AtomicUsize,
    terminal_cache_hits: AtomicUsize,
    terminal_cache_misses: AtomicUsize,
    terminal_calls: AtomicUsize,
    terminal_ns: AtomicU64,
    fold_calls: AtomicUsize,
    fold_ns: AtomicU64,
    showdown_calls: AtomicUsize,
    showdown_ns: AtomicU64,
    showdown_only_calls: AtomicUsize,
    showdown_only_ns: AtomicU64,
    allin_calls: AtomicUsize,
    allin_ns: AtomicU64,
    allin_flop_calls: AtomicUsize,
    allin_flop_ns: AtomicU64,
    allin_turn_calls: AtomicUsize,
    allin_turn_ns: AtomicU64,
    allin_river_calls: AtomicUsize,
    allin_river_ns: AtomicU64,
}

impl NodeLocalProfile {
    fn reset(&self) {
        self.scratch_allocations.store(0, Ordering::Relaxed);
        self.terminal_cache_hits.store(0, Ordering::Relaxed);
        self.terminal_cache_misses.store(0, Ordering::Relaxed);
        self.terminal_calls.store(0, Ordering::Relaxed);
        self.terminal_ns.store(0, Ordering::Relaxed);
        self.fold_calls.store(0, Ordering::Relaxed);
        self.fold_ns.store(0, Ordering::Relaxed);
        self.showdown_calls.store(0, Ordering::Relaxed);
        self.showdown_ns.store(0, Ordering::Relaxed);
        self.showdown_only_calls.store(0, Ordering::Relaxed);
        self.showdown_only_ns.store(0, Ordering::Relaxed);
        self.allin_calls.store(0, Ordering::Relaxed);
        self.allin_ns.store(0, Ordering::Relaxed);
        self.allin_flop_calls.store(0, Ordering::Relaxed);
        self.allin_flop_ns.store(0, Ordering::Relaxed);
        self.allin_turn_calls.store(0, Ordering::Relaxed);
        self.allin_turn_ns.store(0, Ordering::Relaxed);
        self.allin_river_calls.store(0, Ordering::Relaxed);
        self.allin_river_ns.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug)]
#[repr(transparent)]
struct NodeLocalNodeCell(UnsafeCell<NodeLocalNode>);

unsafe impl Send for NodeLocalNodeCell {}
unsafe impl Sync for NodeLocalNodeCell {}

impl NodeLocalNodeCell {
    fn new(node: NodeLocalNode) -> Self {
        Self(UnsafeCell::new(node))
    }

    fn get(&self) -> &NodeLocalNode {
        unsafe { &*self.0.get() }
    }

    fn get_mut(&self) -> &mut NodeLocalNode {
        unsafe { &mut *self.0.get() }
    }
}

#[derive(Debug)]
struct NodeLocalNode {
    public_node: usize,
    board: Board,
    pot: u32,
    kind: NodeLocalKind,
    children: Vec<usize>,
    chance_concrete_events: usize,
    chance_permutation_codes: Vec<Vec<u8>>,
    terminal_cache_indices: Vec<usize>,
    fold_cache_index: Option<usize>,
    allin_oracle_index: Option<usize>,
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeLocalKind {
    Terminal {
        reason: TerminalReason,
        folding_player: Player,
    },
    Chance,
    Decision {
        player: Player,
        actions: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeLocalEvaluationMode {
    Profile,
    BestResponse,
}

#[derive(Debug, Clone)]
struct ComboPermutationMaps {
    oop_source_to_target: Vec<usize>,
    ip_source_to_target: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct PreparedComboTarget {
    range_index: usize,
    board_index: u16,
}

#[derive(Debug, Clone, Copy)]
struct PreparedRiverTarget {
    strength: u64,
    range_index: u16,
    first_card: u8,
    second_card: u8,
}

#[derive(Debug, Clone, Copy)]
struct PreparedLiveTarget {
    range_index: u16,
    first_card: u8,
    second_card: u8,
}

#[derive(Debug)]
struct NodeLocalScratch {
    strategies: Vec<f32>,
    denominators: Vec<f32>,
    child_values: Vec<f32>,
    action_values: Vec<f32>,
    next_oop: Vec<f32>,
    next_ip: Vec<f32>,
    terminal_oop_live: Vec<f32>,
    terminal_ip_live: Vec<f32>,
    terminal_values: Vec<f32>,
    terminal_counts: Vec<f32>,
    side_cache: NodeLocalTerminalSideCache,
}

#[derive(Debug, Clone)]
struct NodeLocalTerminalCache {
    prepared: PreparedTerminalBoard,
    oop_targets: Vec<PreparedComboTarget>,
    ip_targets: Vec<PreparedComboTarget>,
    oop_targets_sorted: Vec<PreparedComboTarget>,
    ip_targets_sorted: Vec<PreparedComboTarget>,
    oop_river_targets_sorted: Vec<PreparedRiverTarget>,
    ip_river_targets_sorted: Vec<PreparedRiverTarget>,
    oop_board_targets_sorted: Vec<u16>,
    ip_board_targets_sorted: Vec<u16>,
}

#[derive(Debug, Clone)]
struct NodeLocalFoldCache {
    oop_live_targets: Vec<PreparedLiveTarget>,
    ip_live_targets: Vec<PreparedLiveTarget>,
}

#[derive(Debug, Clone)]
struct NodeLocalAllInOracle {
    oop_payoffs: Vec<f32>,
    ip_payoffs: Vec<f32>,
    oop_live_indices: Vec<usize>,
    ip_live_indices: Vec<usize>,
    oop_divisor: f32,
    ip_divisor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeLocalTerminalSide {
    Oop,
    Ip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeLocalTerminalSideCacheKey {
    cache_index: usize,
    side: NodeLocalTerminalSide,
    reach_hash: u64,
}

#[derive(Debug, Clone)]
struct NodeLocalTerminalSideCacheEntry {
    signature: Vec<(u16, u32)>,
    values: Vec<f32>,
}

#[derive(Debug, Default)]
struct NodeLocalTerminalSideCache {
    entries: HashMap<NodeLocalTerminalSideCacheKey, Vec<NodeLocalTerminalSideCacheEntry>>,
}

struct PooledNodeLocalScratch<'a> {
    solver: &'a NodeLocalCfrSolver,
    scratch: Option<NodeLocalScratch>,
}

impl<'a> PooledNodeLocalScratch<'a> {
    fn new(solver: &'a NodeLocalCfrSolver) -> Self {
        Self {
            solver,
            scratch: Some(solver.take_worker_scratch()),
        }
    }
}

impl Deref for PooledNodeLocalScratch<'_> {
    type Target = NodeLocalScratch;

    fn deref(&self) -> &Self::Target {
        self.scratch
            .as_ref()
            .expect("pooled node-local scratch was already returned")
    }
}

impl DerefMut for PooledNodeLocalScratch<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch
            .as_mut()
            .expect("pooled node-local scratch was already returned")
    }
}

impl Drop for PooledNodeLocalScratch<'_> {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.solver.return_worker_scratch(scratch);
        }
    }
}

struct NodeLocalChanceAccumulator<'a> {
    out: Vec<f32>,
    child_values: Vec<f32>,
    scratch: Option<PooledNodeLocalScratch<'a>>,
    terminal_evals: usize,
    error: Option<String>,
}

impl<'a> NodeLocalChanceAccumulator<'a> {
    fn new(solver: &'a NodeLocalCfrSolver, value_len: usize) -> Self {
        Self {
            out: vec![0.0; value_len],
            child_values: vec![0.0; value_len],
            scratch: Some(PooledNodeLocalScratch::new(solver)),
            terminal_evals: 0,
            error: None,
        }
    }

    fn new_empty(value_len: usize) -> Self {
        Self {
            out: vec![0.0; value_len],
            child_values: Vec::new(),
            scratch: None,
            terminal_evals: 0,
            error: None,
        }
    }
}

impl NodeLocalCfrSolver {
    pub fn new(
        tree: PublicTree,
        oop_range: RangeSpec,
        ip_range: RangeSpec,
    ) -> Result<Self, String> {
        let oop_combos = oop_range.combos().to_vec();
        let ip_combos = ip_range.combos().to_vec();
        let combo_permutations = all_suit_permutations()
            .into_iter()
            .filter_map(|permutation| {
                let oop_source_to_target =
                    private_combo_permutation_indices(&oop_combos, permutation)?;
                let ip_source_to_target =
                    private_combo_permutation_indices(&ip_combos, permutation)?;
                Some((
                    permutation.code(),
                    ComboPermutationMaps {
                        oop_source_to_target,
                        ip_source_to_target,
                    },
                ))
            })
            .collect::<BTreeMap<_, _>>();
        let oop_same_ip_combo_indices = same_combo_indices(&oop_combos, &ip_combos);
        let ip_same_oop_combo_indices = same_combo_indices(&ip_combos, &oop_combos);
        let terminal_boards = unordered_river_boards_from_flop(&tree.spot.board)?;
        let mut terminal_cache_index_by_key = BTreeMap::new();
        let mut terminal_cache = Vec::with_capacity(terminal_boards.len());
        for board in &terminal_boards {
            let prepared = PreparedTerminalBoard::new(board)?;
            let oop_targets = prepared_combo_targets(&prepared, &oop_combos);
            let ip_targets = prepared_combo_targets(&prepared, &ip_combos);
            let mut oop_targets_sorted = oop_targets.clone();
            let mut ip_targets_sorted = ip_targets.clone();
            sort_combo_targets_by_strength(&prepared, &mut oop_targets_sorted);
            sort_combo_targets_by_strength(&prepared, &mut ip_targets_sorted);
            let oop_river_targets_sorted = prepared_river_targets(&prepared, &oop_targets_sorted);
            let ip_river_targets_sorted = prepared_river_targets(&prepared, &ip_targets_sorted);
            let mut oop_board_targets_sorted = prepared_board_targets(&oop_targets);
            let mut ip_board_targets_sorted = prepared_board_targets(&ip_targets);
            prepared.sort_indices_by_strength(&mut oop_board_targets_sorted);
            prepared.sort_indices_by_strength(&mut ip_board_targets_sorted);
            terminal_cache_index_by_key.insert(unordered_board_key(board), terminal_cache.len());
            terminal_cache.push(NodeLocalTerminalCache {
                prepared,
                oop_targets,
                ip_targets,
                oop_targets_sorted,
                ip_targets_sorted,
                oop_river_targets_sorted,
                ip_river_targets_sorted,
                oop_board_targets_sorted,
                ip_board_targets_sorted,
            });
        }
        let mut solver = Self {
            tree,
            oop_range,
            ip_range,
            oop_combos,
            ip_combos,
            nodes: Vec::new(),
            node_by_key: BTreeMap::new(),
            terminal_cache_index_by_key,
            terminal_cache,
            fold_cache_index_by_key: BTreeMap::new(),
            fold_cache: Vec::new(),
            allin_oracle_index_by_key: BTreeMap::new(),
            allin_oracles: Vec::new(),
            combo_permutations,
            oop_same_ip_combo_indices,
            ip_same_oop_combo_indices,
            completed_iterations: 0,
            action_slots: 0,
            decision_states: 0,
            profile_enabled: std::env::var_os("POKEDR_NODE_CFR_PROFILE").is_some(),
            profile: NodeLocalProfile::default(),
            scratch_pools: (0..rayon::current_num_threads())
                .map(|_| Mutex::new(Vec::new()))
                .collect(),
            terminal_side_cache_enabled: std::env::var_os("POKEDR_NODE_CFR_TERMINAL_SIDE_CACHE")
                .is_some_and(|value| value == "1"),
        };
        let board = solver.tree.spot.board.clone();
        solver.collect_node(0, &board)?;
        Ok(solver)
    }

    pub fn summary(&self) -> NodeLocalCfrSummary {
        let mut board_cards = 0usize;
        let mut public_nodes = 0usize;
        for node in &self.nodes {
            let node = node.get();
            board_cards += node.board.cards().len();
            public_nodes = public_nodes.wrapping_add(node.public_node);
        }
        std::hint::black_box((board_cards, public_nodes));
        NodeLocalCfrSummary {
            iterations: self.completed_iterations,
            states: self.nodes.len(),
            decision_states: self.decision_states,
            action_slots: self.action_slots,
            terminal_evals: 0,
            elapsed_ms: 0.0,
            oop_update_pass_value: 0.0,
            ip_update_pass_value: 0.0,
            storage_gib: self.storage_gib(),
            scratch_allocations: self.profile.scratch_allocations.load(Ordering::Relaxed),
            terminal_cache_hits: self.profile.terminal_cache_hits.load(Ordering::Relaxed),
            terminal_cache_misses: self.profile.terminal_cache_misses.load(Ordering::Relaxed),
            terminal_calls: self.profile.terminal_calls.load(Ordering::Relaxed),
            terminal_ns: self.profile.terminal_ns.load(Ordering::Relaxed),
            fold_calls: self.profile.fold_calls.load(Ordering::Relaxed),
            fold_ns: self.profile.fold_ns.load(Ordering::Relaxed),
            showdown_calls: self.profile.showdown_calls.load(Ordering::Relaxed),
            showdown_ns: self.profile.showdown_ns.load(Ordering::Relaxed),
            showdown_only_calls: self.profile.showdown_only_calls.load(Ordering::Relaxed),
            showdown_only_ns: self.profile.showdown_only_ns.load(Ordering::Relaxed),
            allin_calls: self.profile.allin_calls.load(Ordering::Relaxed),
            allin_ns: self.profile.allin_ns.load(Ordering::Relaxed),
            allin_flop_calls: self.profile.allin_flop_calls.load(Ordering::Relaxed),
            allin_flop_ns: self.profile.allin_flop_ns.load(Ordering::Relaxed),
            allin_turn_calls: self.profile.allin_turn_calls.load(Ordering::Relaxed),
            allin_turn_ns: self.profile.allin_turn_ns.load(Ordering::Relaxed),
            allin_river_calls: self.profile.allin_river_calls.load(Ordering::Relaxed),
            allin_river_ns: self.profile.allin_river_ns.load(Ordering::Relaxed),
        }
    }

    pub fn storage_gib(&self) -> f64 {
        self.action_slots as f64 * 2.0 * std::mem::size_of::<f32>() as f64
            / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn solution_snapshot(&self) -> NodeLocalSolutionSnapshot {
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for (id, node_cell) in self.nodes.iter().enumerate() {
            let node = node_cell.get();
            let public_node = &self.tree.nodes[node.public_node];
            let (kind, actions, strategy) = match &node.kind {
                NodeLocalKind::Decision { player, actions } => {
                    let public_actions = match &public_node.kind {
                        PublicNodeKind::Decision { actions, .. } => actions.clone(),
                        _ => Vec::new(),
                    };
                    let combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let mut action_major = Vec::new();
                    if *actions == 1 {
                        action_major.resize(combos, 1.0);
                    } else {
                        let mut denominators = Vec::new();
                        average_strategies_action_major_into(
                            &node.strategy_sum,
                            combos,
                            *actions,
                            &mut action_major,
                            &mut denominators,
                        );
                    }
                    (
                        NodeLocalSolutionNodeKind::Decision,
                        public_actions,
                        Some(NodeLocalStrategySnapshot {
                            player: *player,
                            combos,
                            actions: *actions,
                            action_major,
                        }),
                    )
                }
                NodeLocalKind::Chance => (NodeLocalSolutionNodeKind::Chance, Vec::new(), None),
                NodeLocalKind::Terminal { reason, .. } => (
                    NodeLocalSolutionNodeKind::Terminal { reason: *reason },
                    Vec::new(),
                    None,
                ),
            };
            nodes.push(NodeLocalSolutionNode {
                id,
                public_node: node.public_node,
                board: node.board.clone(),
                street: public_node.state.street,
                pot: node.pot,
                player: public_node.state.player,
                kind,
                children: node.children.clone(),
                actions,
                strategy,
            });
        }
        NodeLocalSolutionSnapshot {
            iterations: self.completed_iterations,
            oop_combos: self.oop_combos.clone(),
            ip_combos: self.ip_combos.clone(),
            nodes,
        }
    }

    fn worker_scratch_pool_index(&self) -> usize {
        if self.scratch_pools.is_empty() {
            return 0;
        }
        rayon::current_thread_index().unwrap_or(0) % self.scratch_pools.len()
    }

    fn take_worker_scratch(&self) -> NodeLocalScratch {
        let pool_index = self.worker_scratch_pool_index();
        if let Some(scratch) = self.scratch_pools[pool_index]
            .lock()
            .expect("node-local scratch pool lock was poisoned")
            .pop()
        {
            scratch
        } else {
            NodeLocalScratch::new(self)
        }
    }

    fn return_worker_scratch(&self, scratch: NodeLocalScratch) {
        let pool_index = self.worker_scratch_pool_index();
        self.scratch_pools[pool_index]
            .lock()
            .expect("node-local scratch pool lock was poisoned")
            .push(scratch);
    }

    fn clear_worker_terminal_side_caches(&self) {
        for pool in &self.scratch_pools {
            let mut scratches = pool
                .lock()
                .expect("node-local scratch pool lock was poisoned");
            for scratch in scratches.iter_mut() {
                scratch.side_cache.entries.clear();
            }
        }
    }

    pub fn run_with_progress(
        &mut self,
        config: RealCfrConfig,
        mut progress: impl FnMut(NodeLocalCfrIterationSummary),
    ) -> Result<NodeLocalCfrSummary, String> {
        if self.profile_enabled {
            self.profile.reset();
        }
        let started_all = Instant::now();
        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();
        let oop_root_reach = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let ip_root_reach = self
            .ip_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let mut root_oop = vec![0.0; self.oop_combos.len()];
        let mut root_ip = vec![0.0; self.ip_combos.len()];
        let mut root_oop_scratch = NodeLocalScratch::new(self);
        let mut root_ip_scratch = NodeLocalScratch::new(self);
        let mut terminal_evals = 0usize;
        for iteration in 1..=config.iterations {
            let started = Instant::now();
            self.completed_iterations += 1;
            self.clear_worker_terminal_side_caches();
            root_oop_scratch.side_cache.entries.clear();
            root_ip_scratch.side_cache.entries.clear();
            terminal_evals = 0;
            let average_weight = self.completed_iterations as f32;
            terminal_evals += self.traverse_update_side_into(
                0,
                Player::Oop,
                &oop_root_reach,
                &ip_root_reach,
                average_weight,
                config,
                &mut root_oop,
                &mut root_oop_scratch,
            )?;
            terminal_evals += self.traverse_update_side_into(
                0,
                Player::Ip,
                &oop_root_reach,
                &ip_root_reach,
                average_weight,
                config,
                &mut root_ip,
                &mut root_ip_scratch,
            )?;
            progress(NodeLocalCfrIterationSummary {
                iteration,
                terminal_evals,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                oop_update_pass_value: weighted_average(
                    &root_oop,
                    &self.oop_combos,
                    oop_weight,
                    ip_weight,
                ),
                ip_update_pass_value: weighted_average(
                    &root_ip,
                    &self.ip_combos,
                    ip_weight,
                    oop_weight,
                ),
            });
        }
        Ok(NodeLocalCfrSummary {
            iterations: self.completed_iterations,
            states: self.nodes.len(),
            decision_states: self.decision_states,
            action_slots: self.action_slots,
            terminal_evals,
            elapsed_ms: started_all.elapsed().as_secs_f64() * 1000.0,
            oop_update_pass_value: weighted_average(
                &root_oop,
                &self.oop_combos,
                oop_weight,
                ip_weight,
            ),
            ip_update_pass_value: weighted_average(
                &root_ip,
                &self.ip_combos,
                ip_weight,
                oop_weight,
            ),
            storage_gib: self.storage_gib(),
            scratch_allocations: self.profile.scratch_allocations.load(Ordering::Relaxed),
            terminal_cache_hits: self.profile.terminal_cache_hits.load(Ordering::Relaxed),
            terminal_cache_misses: self.profile.terminal_cache_misses.load(Ordering::Relaxed),
            terminal_calls: self.profile.terminal_calls.load(Ordering::Relaxed),
            terminal_ns: self.profile.terminal_ns.load(Ordering::Relaxed),
            fold_calls: self.profile.fold_calls.load(Ordering::Relaxed),
            fold_ns: self.profile.fold_ns.load(Ordering::Relaxed),
            showdown_calls: self.profile.showdown_calls.load(Ordering::Relaxed),
            showdown_ns: self.profile.showdown_ns.load(Ordering::Relaxed),
            showdown_only_calls: self.profile.showdown_only_calls.load(Ordering::Relaxed),
            showdown_only_ns: self.profile.showdown_only_ns.load(Ordering::Relaxed),
            allin_calls: self.profile.allin_calls.load(Ordering::Relaxed),
            allin_ns: self.profile.allin_ns.load(Ordering::Relaxed),
            allin_flop_calls: self.profile.allin_flop_calls.load(Ordering::Relaxed),
            allin_flop_ns: self.profile.allin_flop_ns.load(Ordering::Relaxed),
            allin_turn_calls: self.profile.allin_turn_calls.load(Ordering::Relaxed),
            allin_turn_ns: self.profile.allin_turn_ns.load(Ordering::Relaxed),
            allin_river_calls: self.profile.allin_river_calls.load(Ordering::Relaxed),
            allin_river_ns: self.profile.allin_river_ns.load(Ordering::Relaxed),
        })
    }

    pub fn exploitability(&self, _threads: usize) -> Result<RealCfrExploitability, String> {
        let oop_root_reach = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let ip_root_reach = self
            .ip_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let mut scratch = NodeLocalScratch::new(self);
        let mut profile_oop = vec![0.0; self.oop_combos.len()];
        let mut profile_ip = vec![0.0; self.ip_combos.len()];
        let mut oop_br = vec![0.0; self.oop_combos.len()];
        let mut ip_br = vec![0.0; self.ip_combos.len()];

        self.evaluate_side_into(
            0,
            Player::Oop,
            &ip_root_reach,
            NodeLocalEvaluationMode::Profile,
            &mut profile_oop,
            &mut scratch,
        )?;
        scratch.side_cache.entries.clear();
        self.evaluate_side_into(
            0,
            Player::Ip,
            &oop_root_reach,
            NodeLocalEvaluationMode::Profile,
            &mut profile_ip,
            &mut scratch,
        )?;
        scratch.side_cache.entries.clear();
        self.evaluate_side_into(
            0,
            Player::Oop,
            &ip_root_reach,
            NodeLocalEvaluationMode::BestResponse,
            &mut oop_br,
            &mut scratch,
        )?;
        scratch.side_cache.entries.clear();
        self.evaluate_side_into(
            0,
            Player::Ip,
            &oop_root_reach,
            NodeLocalEvaluationMode::BestResponse,
            &mut ip_br,
            &mut scratch,
        )?;

        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();
        let profile_oop_value =
            weighted_average(&profile_oop, &self.oop_combos, oop_weight, ip_weight);
        let profile_ip_value =
            weighted_average(&profile_ip, &self.ip_combos, ip_weight, oop_weight);
        let oop_best_response_value =
            weighted_average(&oop_br, &self.oop_combos, oop_weight, ip_weight);
        let ip_best_response_value =
            weighted_average(&ip_br, &self.ip_combos, ip_weight, oop_weight);
        let oop_gain = (oop_best_response_value - profile_oop_value).max(0.0);
        let ip_gain = (ip_best_response_value - profile_ip_value).max(0.0);
        let nash_conv_chips = oop_gain + ip_gain;
        let exploitability_chips = nash_conv_chips * 0.5;
        Ok(RealCfrExploitability {
            profile_oop_value,
            profile_ip_value,
            oop_best_response_value,
            ip_best_response_value,
            oop_gain,
            ip_gain,
            nash_conv_chips,
            exploitability_chips,
            exploitability_bb_per_100: exploitability_chips,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn traverse_update_side_into(
        &self,
        node_index: usize,
        update_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
        average_weight: f32,
        config: RealCfrConfig,
        out: &mut [f32],
        scratch: &mut NodeLocalScratch,
    ) -> Result<usize, String> {
        let node = self.nodes[node_index].get();
        match node.kind {
            NodeLocalKind::Terminal {
                reason,
                folding_player,
            } => {
                if self.profile_enabled {
                    self.profile.terminal_calls.fetch_add(1, Ordering::Relaxed);
                    let started = Instant::now();
                    let result = self.terminal_side_into(
                        node_index,
                        update_player,
                        reason,
                        folding_player,
                        oop_reach,
                        ip_reach,
                        out,
                        scratch,
                    );
                    self.profile.terminal_ns.fetch_add(
                        started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                        Ordering::Relaxed,
                    );
                    result
                } else {
                    self.terminal_side_into(
                        node_index,
                        update_player,
                        reason,
                        folding_player,
                        oop_reach,
                        ip_reach,
                        out,
                        scratch,
                    )
                }
            }
            NodeLocalKind::Chance => {
                out.fill(0.0);
                let chance_weight = 1.0 / node.chance_concrete_events as f32;
                let target_len = out.len();
                if should_parallel_node(node, target_len) {
                    let result = node
                        .children
                        .par_iter()
                        .copied()
                        .enumerate()
                        .fold(
                            || NodeLocalChanceAccumulator::new(self, target_len),
                            |mut accumulator, (action_index, child)| {
                                if accumulator.error.is_some() {
                                    return accumulator;
                                }
                                accumulator.child_values.resize(target_len, 0.0);
                                let Some(local_scratch) = accumulator.scratch.as_mut() else {
                                    accumulator.error =
                                        Some("chance accumulator scratch is missing".to_string());
                                    return accumulator;
                                };
                                match self.traverse_update_side_into(
                                    child,
                                    update_player,
                                    oop_reach,
                                    ip_reach,
                                    average_weight,
                                    config,
                                    &mut accumulator.child_values,
                                    local_scratch,
                                ) {
                                    Ok(terminal_evals) => {
                                        accumulator.terminal_evals += terminal_evals;
                                    }
                                    Err(error) => {
                                        accumulator.error = Some(error);
                                        return accumulator;
                                    }
                                }
                                for code in &node.chance_permutation_codes[action_index] {
                                    let Some(maps) = self.combo_permutations.get(code) else {
                                        accumulator.error = Some(
                                            "chance isomorphism permutation is missing combo maps"
                                                .to_string(),
                                        );
                                        return accumulator;
                                    };
                                    let source_to_target = match update_player {
                                        Player::Oop => &maps.oop_source_to_target,
                                        Player::Ip => &maps.ip_source_to_target,
                                    };
                                    add_permuted_scaled_slice(
                                        &mut accumulator.out,
                                        &accumulator.child_values,
                                        source_to_target,
                                        chance_weight,
                                    );
                                }
                                accumulator
                            },
                        )
                        .reduce(
                            || NodeLocalChanceAccumulator::new_empty(target_len),
                            |mut left, right| {
                                if left.error.is_none() {
                                    left.error = right.error;
                                }
                                left.terminal_evals += right.terminal_evals;
                                add_slice(&mut left.out, &right.out);
                                left
                            },
                        );
                    if let Some(error) = result.error {
                        return Err(error);
                    }
                    out.copy_from_slice(&result.out);
                    return Ok(result.terminal_evals);
                }
                scratch.child_values.resize(target_len, 0.0);
                let mut terminal_evals = 0usize;
                for (action_index, child) in node.children.iter().copied().enumerate() {
                    let mut child_values = std::mem::take(&mut scratch.child_values);
                    terminal_evals += self.traverse_update_side_into(
                        child,
                        update_player,
                        oop_reach,
                        ip_reach,
                        average_weight,
                        config,
                        &mut child_values,
                        scratch,
                    )?;
                    for code in &node.chance_permutation_codes[action_index] {
                        let maps = self.combo_permutations.get(code).ok_or_else(|| {
                            "chance isomorphism permutation is missing combo maps".to_string()
                        })?;
                        let source_to_target = match update_player {
                            Player::Oop => &maps.oop_source_to_target,
                            Player::Ip => &maps.ip_source_to_target,
                        };
                        add_permuted_scaled_slice(
                            out,
                            &child_values,
                            source_to_target,
                            chance_weight,
                        );
                    }
                    scratch.child_values = child_values;
                }
                Ok(terminal_evals)
            }
            NodeLocalKind::Decision { player, actions } => {
                if actions == 1 {
                    return self.traverse_update_side_into(
                        node.children[0],
                        update_player,
                        oop_reach,
                        ip_reach,
                        average_weight,
                        config,
                        out,
                        scratch,
                    );
                }
                let acting_combos = match player {
                    Player::Oop => self.oop_combos.len(),
                    Player::Ip => self.ip_combos.len(),
                };
                let target_len = out.len();
                let mut strategies = std::mem::take(&mut scratch.strategies);
                let mut denominators = std::mem::take(&mut scratch.denominators);
                let node_ref = self.nodes[node_index].get();
                current_strategies_action_major_into(
                    &node_ref.regrets,
                    acting_combos,
                    actions,
                    &mut strategies,
                    &mut denominators,
                );
                if player != update_player {
                    out.fill(0.0);
                    if should_parallel_node(node, target_len) {
                        let results = match player {
                            Player::Oop => (0..actions)
                                .into_par_iter()
                                .map_init(
                                    || PooledNodeLocalScratch::new(self),
                                    |local_scratch, action| {
                                        let mut child_values = vec![0.0; target_len];
                                        let mut next_oop = vec![0.0; oop_reach.len()];
                                        strategy_reach_action_major_into(
                                            &mut next_oop,
                                            oop_reach,
                                            &strategies,
                                            acting_combos,
                                            actions,
                                            action,
                                        );
                                        let terminal_evals = self.traverse_update_side_into(
                                            node.children[action],
                                            update_player,
                                            &next_oop,
                                            ip_reach,
                                            average_weight,
                                            config,
                                            &mut child_values,
                                            local_scratch,
                                        )?;
                                        Ok::<_, String>((child_values, terminal_evals))
                                    },
                                )
                                .collect::<Result<Vec<_>, _>>(),
                            Player::Ip => (0..actions)
                                .into_par_iter()
                                .map_init(
                                    || PooledNodeLocalScratch::new(self),
                                    |local_scratch, action| {
                                        let mut child_values = vec![0.0; target_len];
                                        let mut next_ip = vec![0.0; ip_reach.len()];
                                        strategy_reach_action_major_into(
                                            &mut next_ip,
                                            ip_reach,
                                            &strategies,
                                            acting_combos,
                                            actions,
                                            action,
                                        );
                                        let terminal_evals = self.traverse_update_side_into(
                                            node.children[action],
                                            update_player,
                                            oop_reach,
                                            &next_ip,
                                            average_weight,
                                            config,
                                            &mut child_values,
                                            local_scratch,
                                        )?;
                                        Ok::<_, String>((child_values, terminal_evals))
                                    },
                                )
                                .collect::<Result<Vec<_>, _>>(),
                        }?;
                        let mut terminal_evals = 0usize;
                        for (child_values, child_terminal_evals) in results {
                            terminal_evals += child_terminal_evals;
                            add_slice(out, &child_values);
                        }
                        scratch.strategies = strategies;
                        scratch.denominators = denominators;
                        return Ok(terminal_evals);
                    }
                    let mut terminal_evals = 0usize;
                    let mut child_values = std::mem::take(&mut scratch.child_values);
                    child_values.resize(target_len, 0.0);
                    match player {
                        Player::Oop => {
                            let mut next_oop = std::mem::take(&mut scratch.next_oop);
                            next_oop.resize(oop_reach.len(), 0.0);
                            for action in 0..actions {
                                let child = self.nodes[node_index].get().children[action];
                                strategy_reach_action_major_into(
                                    &mut next_oop,
                                    oop_reach,
                                    &strategies,
                                    acting_combos,
                                    actions,
                                    action,
                                );
                                terminal_evals += self.traverse_update_side_into(
                                    child,
                                    update_player,
                                    &next_oop,
                                    ip_reach,
                                    average_weight,
                                    config,
                                    &mut child_values,
                                    scratch,
                                )?;
                                add_slice(out, &child_values);
                            }
                            scratch.next_oop = next_oop;
                        }
                        Player::Ip => {
                            let mut next_ip = std::mem::take(&mut scratch.next_ip);
                            next_ip.resize(ip_reach.len(), 0.0);
                            for action in 0..actions {
                                let child = self.nodes[node_index].get().children[action];
                                strategy_reach_action_major_into(
                                    &mut next_ip,
                                    ip_reach,
                                    &strategies,
                                    acting_combos,
                                    actions,
                                    action,
                                );
                                terminal_evals += self.traverse_update_side_into(
                                    child,
                                    update_player,
                                    oop_reach,
                                    &next_ip,
                                    average_weight,
                                    config,
                                    &mut child_values,
                                    scratch,
                                )?;
                                add_slice(out, &child_values);
                            }
                            scratch.next_ip = next_ip;
                        }
                    }
                    scratch.child_values = child_values;
                    scratch.strategies = strategies;
                    scratch.denominators = denominators;
                    return Ok(terminal_evals);
                }

                let mut action_values = std::mem::take(&mut scratch.action_values);
                action_values.resize(actions * target_len, 0.0);
                let terminal_evals = if should_parallel_node(node, target_len) {
                    match player {
                        Player::Oop => action_values
                            .par_chunks_mut(target_len)
                            .enumerate()
                            .map_init(
                                || PooledNodeLocalScratch::new(self),
                                |local_scratch, (action, action_out)| {
                                    let next_oop = if config.average_strategy
                                        == RealCfrAverageStrategy::ReachWeighted
                                    {
                                        let mut next_oop = vec![0.0; oop_reach.len()];
                                        strategy_reach_action_major_into(
                                            &mut next_oop,
                                            oop_reach,
                                            &strategies,
                                            acting_combos,
                                            actions,
                                            action,
                                        );
                                        next_oop
                                    } else {
                                        Vec::new()
                                    };
                                    let child_oop = if config.average_strategy
                                        == RealCfrAverageStrategy::ReachWeighted
                                    {
                                        &next_oop
                                    } else {
                                        oop_reach
                                    };
                                    self.traverse_update_side_into(
                                        node.children[action],
                                        update_player,
                                        child_oop,
                                        ip_reach,
                                        average_weight,
                                        config,
                                        action_out,
                                        local_scratch,
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum(),
                        Player::Ip => action_values
                            .par_chunks_mut(target_len)
                            .enumerate()
                            .map_init(
                                || PooledNodeLocalScratch::new(self),
                                |local_scratch, (action, action_out)| {
                                    let next_ip = if config.average_strategy
                                        == RealCfrAverageStrategy::ReachWeighted
                                    {
                                        let mut next_ip = vec![0.0; ip_reach.len()];
                                        strategy_reach_action_major_into(
                                            &mut next_ip,
                                            ip_reach,
                                            &strategies,
                                            acting_combos,
                                            actions,
                                            action,
                                        );
                                        next_ip
                                    } else {
                                        Vec::new()
                                    };
                                    let child_ip = if config.average_strategy
                                        == RealCfrAverageStrategy::ReachWeighted
                                    {
                                        &next_ip
                                    } else {
                                        ip_reach
                                    };
                                    self.traverse_update_side_into(
                                        node.children[action],
                                        update_player,
                                        oop_reach,
                                        child_ip,
                                        average_weight,
                                        config,
                                        action_out,
                                        local_scratch,
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum(),
                    }
                } else {
                    let mut terminal_evals = 0usize;
                    match player {
                        Player::Oop => {
                            let mut next_oop = std::mem::take(&mut scratch.next_oop);
                            next_oop.resize(oop_reach.len(), 0.0);
                            for action in 0..actions {
                                let child = self.nodes[node_index].get().children[action];
                                let child_oop = if config.average_strategy
                                    == RealCfrAverageStrategy::ReachWeighted
                                {
                                    strategy_reach_action_major_into(
                                        &mut next_oop,
                                        oop_reach,
                                        &strategies,
                                        acting_combos,
                                        actions,
                                        action,
                                    );
                                    &next_oop
                                } else {
                                    oop_reach
                                };
                                terminal_evals += self.traverse_update_side_into(
                                    child,
                                    update_player,
                                    child_oop,
                                    ip_reach,
                                    average_weight,
                                    config,
                                    &mut action_values
                                        [action * target_len..(action + 1) * target_len],
                                    scratch,
                                )?;
                            }
                            scratch.next_oop = next_oop;
                        }
                        Player::Ip => {
                            let mut next_ip = std::mem::take(&mut scratch.next_ip);
                            next_ip.resize(ip_reach.len(), 0.0);
                            for action in 0..actions {
                                let child = self.nodes[node_index].get().children[action];
                                let child_ip = if config.average_strategy
                                    == RealCfrAverageStrategy::ReachWeighted
                                {
                                    strategy_reach_action_major_into(
                                        &mut next_ip,
                                        ip_reach,
                                        &strategies,
                                        acting_combos,
                                        actions,
                                        action,
                                    );
                                    &next_ip
                                } else {
                                    ip_reach
                                };
                                terminal_evals += self.traverse_update_side_into(
                                    child,
                                    update_player,
                                    oop_reach,
                                    child_ip,
                                    average_weight,
                                    config,
                                    &mut action_values
                                        [action * target_len..(action + 1) * target_len],
                                    scratch,
                                )?;
                            }
                            scratch.next_ip = next_ip;
                        }
                    }
                    terminal_evals
                };
                combine_acting_action_major_values(out, &action_values, &strategies, actions);
                let own_reach = match player {
                    Player::Oop => oop_reach,
                    Player::Ip => ip_reach,
                };
                let factors = NodeLocalUpdateFactors::new(
                    config.variant,
                    average_weight,
                    config.average_strategy,
                );
                let node_mut = self.nodes[node_index].get_mut();
                apply_node_local_updates_action_major(
                    &mut node_mut.regrets,
                    &mut node_mut.strategy_sum,
                    &action_values,
                    out,
                    &strategies,
                    own_reach,
                    acting_combos,
                    actions,
                    &factors,
                );
                scratch.action_values = action_values;
                scratch.strategies = strategies;
                scratch.denominators = denominators;
                Ok(terminal_evals)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_side_into(
        &self,
        node_index: usize,
        value_player: Player,
        opponent_reach: &[f32],
        mode: NodeLocalEvaluationMode,
        out: &mut [f32],
        scratch: &mut NodeLocalScratch,
    ) -> Result<usize, String> {
        let node = self.nodes[node_index].get();
        match node.kind {
            NodeLocalKind::Terminal {
                reason,
                folding_player,
            } => {
                let empty_own: &[f32] = &[];
                let (oop_reach, ip_reach) = match value_player {
                    Player::Oop => (empty_own, opponent_reach),
                    Player::Ip => (opponent_reach, empty_own),
                };
                self.terminal_side_into(
                    node_index,
                    value_player,
                    reason,
                    folding_player,
                    oop_reach,
                    ip_reach,
                    out,
                    scratch,
                )
            }
            NodeLocalKind::Chance => {
                out.fill(0.0);
                let chance_weight = 1.0 / node.chance_concrete_events as f32;
                let target_len = out.len();
                if should_parallel_node(node, target_len) {
                    let result = node
                        .children
                        .par_iter()
                        .copied()
                        .enumerate()
                        .fold(
                            || NodeLocalChanceAccumulator::new(self, target_len),
                            |mut accumulator, (action_index, child)| {
                                if accumulator.error.is_some() {
                                    return accumulator;
                                }
                                accumulator.child_values.resize(target_len, 0.0);
                                let Some(local_scratch) = accumulator.scratch.as_mut() else {
                                    accumulator.error =
                                        Some("chance accumulator scratch is missing".to_string());
                                    return accumulator;
                                };
                                match self.evaluate_side_into(
                                    child,
                                    value_player,
                                    opponent_reach,
                                    mode,
                                    &mut accumulator.child_values,
                                    local_scratch,
                                ) {
                                    Ok(terminal_evals) => {
                                        accumulator.terminal_evals += terminal_evals;
                                    }
                                    Err(error) => {
                                        accumulator.error = Some(error);
                                        return accumulator;
                                    }
                                }
                                for code in &node.chance_permutation_codes[action_index] {
                                    let Some(maps) = self.combo_permutations.get(code) else {
                                        accumulator.error = Some(
                                            "chance isomorphism permutation is missing combo maps"
                                                .to_string(),
                                        );
                                        return accumulator;
                                    };
                                    let source_to_target = match value_player {
                                        Player::Oop => &maps.oop_source_to_target,
                                        Player::Ip => &maps.ip_source_to_target,
                                    };
                                    add_permuted_scaled_slice(
                                        &mut accumulator.out,
                                        &accumulator.child_values,
                                        source_to_target,
                                        chance_weight,
                                    );
                                }
                                accumulator
                            },
                        )
                        .reduce(
                            || NodeLocalChanceAccumulator::new_empty(target_len),
                            |mut left, right| {
                                if left.error.is_none() {
                                    left.error = right.error;
                                }
                                left.terminal_evals += right.terminal_evals;
                                add_slice(&mut left.out, &right.out);
                                left
                            },
                        );
                    if let Some(error) = result.error {
                        return Err(error);
                    }
                    out.copy_from_slice(&result.out);
                    return Ok(result.terminal_evals);
                }
                let mut child_values = std::mem::take(&mut scratch.child_values);
                child_values.resize(out.len(), 0.0);
                let mut terminal_evals = 0usize;
                for (action_index, child) in node.children.iter().copied().enumerate() {
                    terminal_evals += self.evaluate_side_into(
                        child,
                        value_player,
                        opponent_reach,
                        mode,
                        &mut child_values,
                        scratch,
                    )?;
                    for code in &node.chance_permutation_codes[action_index] {
                        let maps = self.combo_permutations.get(code).ok_or_else(|| {
                            "chance isomorphism permutation is missing combo maps".to_string()
                        })?;
                        let source_to_target = match value_player {
                            Player::Oop => &maps.oop_source_to_target,
                            Player::Ip => &maps.ip_source_to_target,
                        };
                        add_permuted_scaled_slice(
                            out,
                            &child_values,
                            source_to_target,
                            chance_weight,
                        );
                    }
                }
                scratch.child_values = child_values;
                Ok(terminal_evals)
            }
            NodeLocalKind::Decision { player, actions } => {
                if actions == 1 {
                    return self.evaluate_side_into(
                        node.children[0],
                        value_player,
                        opponent_reach,
                        mode,
                        out,
                        scratch,
                    );
                }
                let acting_combos = match player {
                    Player::Oop => self.oop_combos.len(),
                    Player::Ip => self.ip_combos.len(),
                };
                let mut strategies = std::mem::take(&mut scratch.strategies);
                let mut denominators = std::mem::take(&mut scratch.denominators);
                average_strategies_action_major_into(
                    &node.strategy_sum,
                    acting_combos,
                    actions,
                    &mut strategies,
                    &mut denominators,
                );

                let target_len = out.len();
                let mut action_values = std::mem::take(&mut scratch.action_values);
                action_values.resize(actions * target_len, 0.0);
                if player == value_player {
                    let terminal_evals = if should_parallel_node(node, target_len) {
                        action_values
                            .par_chunks_mut(target_len)
                            .enumerate()
                            .map_init(
                                || PooledNodeLocalScratch::new(self),
                                |local_scratch, (action, action_out)| {
                                    self.evaluate_side_into(
                                        node.children[action],
                                        value_player,
                                        opponent_reach,
                                        mode,
                                        action_out,
                                        local_scratch,
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum()
                    } else {
                        let mut terminal_evals = 0usize;
                        for action in 0..actions {
                            terminal_evals += self.evaluate_side_into(
                                node.children[action],
                                value_player,
                                opponent_reach,
                                mode,
                                &mut action_values[action * target_len..(action + 1) * target_len],
                                scratch,
                            )?;
                        }
                        terminal_evals
                    };
                    match mode {
                        NodeLocalEvaluationMode::Profile => combine_acting_action_major_values(
                            out,
                            &action_values,
                            &strategies,
                            actions,
                        ),
                        NodeLocalEvaluationMode::BestResponse => {
                            combine_best_response_action_major_values(out, &action_values, actions)
                        }
                    }
                    scratch.action_values = action_values;
                    scratch.strategies = strategies;
                    scratch.denominators = denominators;
                    return Ok(terminal_evals);
                }

                let terminal_evals = if should_parallel_node(node, target_len) {
                    let results = (0..actions)
                        .into_par_iter()
                        .map_init(
                            || PooledNodeLocalScratch::new(self),
                            |local_scratch, action| {
                                let mut child_values = vec![0.0; target_len];
                                let mut next_opponent = vec![0.0; opponent_reach.len()];
                                strategy_reach_action_major_into(
                                    &mut next_opponent,
                                    opponent_reach,
                                    &strategies,
                                    acting_combos,
                                    actions,
                                    action,
                                );
                                let terminal_evals = self.evaluate_side_into(
                                    node.children[action],
                                    value_player,
                                    &next_opponent,
                                    mode,
                                    &mut child_values,
                                    local_scratch,
                                )?;
                                Ok::<_, String>((action, child_values, terminal_evals))
                            },
                        )
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut terminal_evals = 0usize;
                    for (action, child_values, child_terminal_evals) in results {
                        terminal_evals += child_terminal_evals;
                        action_values[action * target_len..(action + 1) * target_len]
                            .copy_from_slice(&child_values);
                    }
                    terminal_evals
                } else {
                    let mut next_opponent = match player {
                        Player::Oop => std::mem::take(&mut scratch.next_oop),
                        Player::Ip => std::mem::take(&mut scratch.next_ip),
                    };
                    next_opponent.resize(opponent_reach.len(), 0.0);
                    let mut terminal_evals = 0usize;
                    for action in 0..actions {
                        strategy_reach_action_major_into(
                            &mut next_opponent,
                            opponent_reach,
                            &strategies,
                            acting_combos,
                            actions,
                            action,
                        );
                        terminal_evals += self.evaluate_side_into(
                            node.children[action],
                            value_player,
                            &next_opponent,
                            mode,
                            &mut action_values[action * target_len..(action + 1) * target_len],
                            scratch,
                        )?;
                    }
                    match player {
                        Player::Oop => scratch.next_oop = next_opponent,
                        Player::Ip => scratch.next_ip = next_opponent,
                    }
                    terminal_evals
                };
                combine_nonacting_action_major_values(out, &action_values, actions);
                scratch.action_values = action_values;
                scratch.strategies = strategies;
                scratch.denominators = denominators;
                Ok(terminal_evals)
            }
        }
    }

    fn collect_node(&mut self, public_node: usize, board: &Board) -> Result<usize, String> {
        let key = (public_node, ordered_board_key(board));
        if let Some(index) = self.node_by_key.get(&key) {
            return Ok(*index);
        }

        let index = self.nodes.len();
        self.node_by_key.insert(key, index);
        self.nodes.push(NodeLocalNodeCell::new(NodeLocalNode {
            public_node,
            board: board.clone(),
            pot: 0,
            kind: NodeLocalKind::Terminal {
                reason: TerminalReason::Showdown,
                folding_player: Player::Oop,
            },
            children: Vec::new(),
            chance_concrete_events: 0,
            chance_permutation_codes: Vec::new(),
            terminal_cache_indices: Vec::new(),
            fold_cache_index: None,
            allin_oracle_index: None,
            regrets: Vec::new(),
            strategy_sum: Vec::new(),
        }));

        let public = self
            .tree
            .nodes
            .get(public_node)
            .ok_or_else(|| "public node index is out of bounds".to_string())?
            .clone();

        let kind;
        let mut children = Vec::new();
        let mut chance_concrete_events = 0usize;
        let mut chance_permutation_codes = Vec::new();
        let mut terminal_cache_indices = Vec::new();
        let mut fold_cache_index = None;
        let mut allin_oracle_index = None;
        let mut regrets = Vec::new();
        let mut strategy_sum = Vec::new();

        match public.kind {
            PublicNodeKind::Terminal { reason } => {
                kind = NodeLocalKind::Terminal {
                    reason,
                    folding_player: public.state.player,
                };
                match reason {
                    TerminalReason::Fold => {
                        fold_cache_index = Some(self.fold_cache_index(board));
                    }
                    TerminalReason::Showdown => {
                        for terminal_board in terminal_boards(board)? {
                            let index = self
                                .terminal_cache_index_by_key
                                .get(&unordered_board_key(&terminal_board))
                                .copied()
                                .ok_or_else(|| "terminal board is outside cache".to_string())?;
                            terminal_cache_indices.push(index);
                        }
                    }
                    TerminalReason::AllIn => {
                        allin_oracle_index = self.allin_oracle_index(board)?;
                        if allin_oracle_index.is_none() {
                            for terminal_board in terminal_boards(board)? {
                                let index = self
                                    .terminal_cache_index_by_key
                                    .get(&unordered_board_key(&terminal_board))
                                    .copied()
                                    .ok_or_else(|| "terminal board is outside cache".to_string())?;
                                terminal_cache_indices.push(index);
                            }
                        }
                    }
                }
            }
            PublicNodeKind::Chance(_) => {
                kind = NodeLocalKind::Chance;
                let Some(child) = public.children.first().copied() else {
                    return Ok(index);
                };
                let chance = next_card_isomorphism(board, &self.oop_range, &self.ip_range);
                chance_concrete_events = chance.concrete_events;
                for class in chance.classes {
                    let card = *class
                        .representative
                        .first()
                        .ok_or_else(|| "chance class has no representative".to_string())?;
                    children.push(self.collect_node(child, &board.push(card)?)?);
                    chance_permutation_codes.push(
                        class
                            .members
                            .iter()
                            .map(|member| member.permutation_to_representative.code())
                            .collect(),
                    );
                }
            }
            PublicNodeKind::Decision { player, actions } => {
                kind = NodeLocalKind::Decision {
                    player,
                    actions: actions.len(),
                };
                for child in public.children {
                    children.push(self.collect_node(child, board)?);
                }
                if actions.len() > 1 {
                    let combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let slots = combos * actions.len();
                    regrets.resize(slots, 0.0);
                    strategy_sum.resize(slots, 0.0);
                    self.action_slots += slots;
                    self.decision_states += 1;
                }
            }
        }

        let node = self.nodes[index].get_mut();
        node.pot = public.state.pot;
        node.kind = kind;
        node.children = children;
        node.chance_concrete_events = chance_concrete_events;
        node.chance_permutation_codes = chance_permutation_codes;
        node.terminal_cache_indices = terminal_cache_indices;
        node.fold_cache_index = fold_cache_index;
        node.allin_oracle_index = allin_oracle_index;
        node.regrets = regrets;
        node.strategy_sum = strategy_sum;
        Ok(index)
    }

    fn fold_cache_index(&mut self, board: &Board) -> usize {
        let key = ordered_board_key(board);
        if let Some(index) = self.fold_cache_index_by_key.get(&key) {
            return *index;
        }
        let index = self.fold_cache.len();
        self.fold_cache_index_by_key.insert(key, index);
        self.fold_cache.push(NodeLocalFoldCache {
            oop_live_targets: live_combo_targets(&self.oop_combos, board),
            ip_live_targets: live_combo_targets(&self.ip_combos, board),
        });
        index
    }

    fn allin_oracle_index(&mut self, board: &Board) -> Result<Option<usize>, String> {
        if board.cards().len() != 3 {
            return Ok(None);
        }
        let key = ordered_board_key(board);
        if let Some(index) = self.allin_oracle_index_by_key.get(&key) {
            return Ok(Some(*index));
        }
        let cells = self.oop_combos.len() * self.ip_combos.len();
        let bytes = cells * 2 * std::mem::size_of::<f32>();
        let limit_mib = std::env::var("POKEDR_NODE_CFR_ALLIN_ORACLE_LIMIT_MIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if limit_mib == 0 {
            return Ok(None);
        }
        if bytes > limit_mib * 1024 * 1024 {
            return Ok(None);
        }
        let index = self.allin_oracles.len();
        let oracle = build_allin_oracle(board, &self.oop_combos, &self.ip_combos)?;
        self.allin_oracle_index_by_key.insert(key, index);
        self.allin_oracles.push(oracle);
        Ok(Some(index))
    }

    #[allow(clippy::too_many_arguments)]
    fn terminal_side_into(
        &self,
        node_index: usize,
        update_player: Player,
        reason: TerminalReason,
        folding_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
        out: &mut [f32],
        scratch: &mut NodeLocalScratch,
    ) -> Result<usize, String> {
        let node = self.nodes[node_index].get();
        match reason {
            TerminalReason::Fold => {
                if self.profile_enabled {
                    self.profile.fold_calls.fetch_add(1, Ordering::Relaxed);
                }
                let started = self.profile_enabled.then(Instant::now);
                self.fold_side_into(
                    node,
                    node.pot,
                    update_player,
                    folding_player,
                    oop_reach,
                    ip_reach,
                    out,
                );
                if let Some(started) = started {
                    self.profile.fold_ns.fetch_add(
                        started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                        Ordering::Relaxed,
                    );
                }
                Ok(0)
            }
            TerminalReason::Showdown | TerminalReason::AllIn => {
                if self.profile_enabled {
                    self.profile.showdown_calls.fetch_add(1, Ordering::Relaxed);
                    match reason {
                        TerminalReason::Showdown => {
                            self.profile
                                .showdown_only_calls
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        TerminalReason::AllIn => {
                            self.profile.allin_calls.fetch_add(1, Ordering::Relaxed);
                            match node.board.cards().len() {
                                3 => {
                                    self.profile
                                        .allin_flop_calls
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                                4 => {
                                    self.profile
                                        .allin_turn_calls
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                                5 => {
                                    self.profile
                                        .allin_river_calls
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                                _ => {}
                            }
                        }
                        TerminalReason::Fold => {}
                    }
                }
                let started = self.profile_enabled.then(Instant::now);
                let result = if matches!(reason, TerminalReason::AllIn) {
                    if let Some(oracle_index) = node.allin_oracle_index {
                        self.allin_oracle_side_into(
                            oracle_index,
                            node.pot,
                            update_player,
                            oop_reach,
                            ip_reach,
                            out,
                        );
                        Ok(0)
                    } else {
                        self.showdown_side_into(
                            node_index,
                            node.pot,
                            update_player,
                            oop_reach,
                            ip_reach,
                            out,
                            scratch,
                        )
                    }
                } else {
                    self.showdown_side_into(
                        node_index,
                        node.pot,
                        update_player,
                        oop_reach,
                        ip_reach,
                        out,
                        scratch,
                    )
                };
                if let Some(started) = started {
                    let elapsed_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                    self.profile
                        .showdown_ns
                        .fetch_add(elapsed_ns, Ordering::Relaxed);
                    match reason {
                        TerminalReason::Showdown => {
                            self.profile
                                .showdown_only_ns
                                .fetch_add(elapsed_ns, Ordering::Relaxed);
                        }
                        TerminalReason::AllIn => {
                            self.profile
                                .allin_ns
                                .fetch_add(elapsed_ns, Ordering::Relaxed);
                            match node.board.cards().len() {
                                3 => {
                                    self.profile
                                        .allin_flop_ns
                                        .fetch_add(elapsed_ns, Ordering::Relaxed);
                                }
                                4 => {
                                    self.profile
                                        .allin_turn_ns
                                        .fetch_add(elapsed_ns, Ordering::Relaxed);
                                }
                                5 => {
                                    self.profile
                                        .allin_river_ns
                                        .fetch_add(elapsed_ns, Ordering::Relaxed);
                                }
                                _ => {}
                            }
                        }
                        TerminalReason::Fold => {}
                    }
                }
                result
            }
        }
    }

    fn allin_oracle_side_into(
        &self,
        oracle_index: usize,
        pot: u32,
        update_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
        out: &mut [f32],
    ) {
        out.fill(0.0);
        let oracle = &self.allin_oracles[oracle_index];
        let pot = pot as f32;
        match update_player {
            Player::Oop => {
                allin_oracle_matrix_vector_into(
                    &oracle.oop_payoffs,
                    self.ip_combos.len(),
                    ip_reach,
                    &oracle.oop_live_indices,
                    &oracle.ip_live_indices,
                    pot / oracle.oop_divisor,
                    out,
                );
            }
            Player::Ip => {
                allin_oracle_matrix_vector_into(
                    &oracle.ip_payoffs,
                    self.oop_combos.len(),
                    oop_reach,
                    &oracle.ip_live_indices,
                    &oracle.oop_live_indices,
                    pot / oracle.ip_divisor,
                    out,
                );
            }
        }
    }

    fn fold_side_into(
        &self,
        node: &NodeLocalNode,
        pot: u32,
        update_player: Player,
        folding_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
        out: &mut [f32],
    ) {
        let pot = pot as f32;
        let fold_cache = node
            .fold_cache_index
            .and_then(|index| self.fold_cache.get(index))
            .expect("fold terminal node is missing fold cache");
        let sign = if folding_player == update_player {
            -pot
        } else {
            pot
        };
        match update_player {
            Player::Oop => opponent_weights_for_fast_into(
                ip_reach,
                &self.oop_same_ip_combo_indices,
                &fold_cache.oop_live_targets,
                &fold_cache.ip_live_targets,
                sign,
                out,
            ),
            Player::Ip => opponent_weights_for_fast_into(
                oop_reach,
                &self.ip_same_oop_combo_indices,
                &fold_cache.ip_live_targets,
                &fold_cache.oop_live_targets,
                sign,
                out,
            ),
        }
    }

    fn showdown_side_into(
        &self,
        node_index: usize,
        pot: u32,
        update_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
        out: &mut [f32],
        scratch: &mut NodeLocalScratch,
    ) -> Result<usize, String> {
        out.fill(0.0);
        let node = self.nodes[node_index].get();
        let cache_indices = &node.terminal_cache_indices;
        let pot = pot as f32;
        if self.terminal_side_cache_enabled {
            scratch.terminal_counts.resize(out.len(), 0.0);
            scratch.terminal_counts.fill(0.0);
            for cache_index in cache_indices {
                let cache = &self.terminal_cache[*cache_index];
                let prepared = &cache.prepared;
                let (opponent_reach, own_targets, board_targets) = match update_player {
                    Player::Oop => {
                        scratch
                            .terminal_ip_live
                            .resize(prepared.combos().len(), 0.0);
                        reach_on_targets_into(
                            &cache.ip_targets,
                            ip_reach,
                            &mut scratch.terminal_ip_live,
                        );
                        (
                            &scratch.terminal_ip_live,
                            &cache.oop_targets,
                            &cache.oop_board_targets_sorted,
                        )
                    }
                    Player::Ip => {
                        scratch
                            .terminal_oop_live
                            .resize(prepared.combos().len(), 0.0);
                        reach_on_targets_into(
                            &cache.oop_targets,
                            oop_reach,
                            &mut scratch.terminal_oop_live,
                        );
                        (
                            &scratch.terminal_oop_live,
                            &cache.ip_targets,
                            &cache.ip_board_targets_sorted,
                        )
                    }
                };
                terminal_side_cached_values(
                    &mut scratch.side_cache,
                    *cache_index,
                    match update_player {
                        Player::Oop => NodeLocalTerminalSide::Oop,
                        Player::Ip => NodeLocalTerminalSide::Ip,
                    },
                    prepared,
                    opponent_reach,
                    board_targets,
                    &mut scratch.terminal_values,
                    self.profile_enabled.then_some(&self.profile),
                )?;
                for target in own_targets {
                    out[target.range_index] +=
                        scratch.terminal_values[target.board_index as usize] * pot;
                    scratch.terminal_counts[target.range_index] += 1.0;
                }
            }
            for (value, count) in out.iter_mut().zip(&scratch.terminal_counts) {
                if *count > 0.0 {
                    *value /= *count;
                }
            }
        } else {
            if node.board.cards().len() == 5 {
                let cache_index = cache_indices
                    .first()
                    .copied()
                    .ok_or_else(|| "river terminal node is missing terminal cache".to_string())?;
                let cache = &self.terminal_cache[cache_index];
                match update_player {
                    Player::Oop => terminal_side_river_targets_sorted_accumulate(
                        &cache.ip_river_targets_sorted,
                        ip_reach,
                        &cache.oop_river_targets_sorted,
                        pot,
                        out,
                    ),
                    Player::Ip => terminal_side_river_targets_sorted_accumulate(
                        &cache.oop_river_targets_sorted,
                        oop_reach,
                        &cache.ip_river_targets_sorted,
                        pot,
                        out,
                    ),
                }
            } else {
                for cache_index in cache_indices {
                    let cache = &self.terminal_cache[*cache_index];
                    let prepared = &cache.prepared;
                    match update_player {
                        Player::Oop => terminal_side_range_targets_sorted_accumulate(
                            prepared,
                            &cache.ip_targets_sorted,
                            ip_reach,
                            &cache.oop_targets_sorted,
                            pot,
                            out,
                        ),
                        Player::Ip => terminal_side_range_targets_sorted_accumulate(
                            prepared,
                            &cache.oop_targets_sorted,
                            oop_reach,
                            &cache.ip_targets_sorted,
                            pot,
                            out,
                        ),
                    }
                }
            }
            let divisor = terminal_runout_count_for_live_combo(node.board.cards().len());
            for value in out {
                *value /= divisor;
            }
        }
        Ok(cache_indices.len())
    }
}

fn ordered_board_key(board: &Board) -> u64 {
    board
        .cards()
        .iter()
        .fold(0u64, |key, card| (key << 6) | card.index() as u64)
}

fn should_parallel_node(node: &NodeLocalNode, value_len: usize) -> bool {
    node.board.cards().len() < 5 && node.children.len() > 1 && value_len >= 16
}

impl NodeLocalScratch {
    fn new(solver: &NodeLocalCfrSolver) -> Self {
        if solver.profile_enabled {
            solver
                .profile
                .scratch_allocations
                .fetch_add(1, Ordering::Relaxed);
        }
        let prepared = &solver.terminal_cache[0].prepared;
        Self {
            strategies: Vec::new(),
            denominators: Vec::new(),
            child_values: Vec::new(),
            action_values: Vec::new(),
            next_oop: Vec::new(),
            next_ip: Vec::new(),
            terminal_oop_live: vec![0.0; prepared.combos().len()],
            terminal_ip_live: vec![0.0; prepared.combos().len()],
            terminal_values: vec![0.0; prepared.combos().len()],
            terminal_counts: Vec::new(),
            side_cache: NodeLocalTerminalSideCache::default(),
        }
    }
}

fn current_strategies_action_major_into(
    regrets: &[f32],
    combos: usize,
    actions: usize,
    strategies: &mut Vec<f32>,
    denominators: &mut Vec<f32>,
) {
    strategies.resize(regrets.len(), 0.0);
    denominators.resize(combos, 0.0);
    denominators.fill(0.0);
    for (strategy, regret) in strategies.iter_mut().zip(regrets) {
        *strategy = regret.max(0.0);
    }
    for action in 0..actions {
        for (denominator, strategy) in denominators
            .iter_mut()
            .zip(&strategies[action * combos..(action + 1) * combos])
        {
            *denominator += *strategy;
        }
    }
    let uniform = 1.0 / actions as f32;
    for action in 0..actions {
        for (strategy, denominator) in strategies[action * combos..(action + 1) * combos]
            .iter_mut()
            .zip(denominators.iter().copied())
        {
            *strategy = if denominator > 0.0 {
                *strategy / denominator
            } else {
                uniform
            };
        }
    }
}

fn average_strategies_action_major_into(
    strategy_sum: &[f32],
    combos: usize,
    actions: usize,
    strategies: &mut Vec<f32>,
    denominators: &mut Vec<f32>,
) {
    strategies.resize(strategy_sum.len(), 0.0);
    denominators.resize(combos, 0.0);
    denominators.fill(0.0);
    strategies.copy_from_slice(strategy_sum);
    for action in 0..actions {
        let row = &strategies[action * combos..(action + 1) * combos];
        for (denominator, value) in denominators.iter_mut().zip(row) {
            *denominator += *value;
        }
    }
    let uniform = 1.0 / actions as f32;
    for action in 0..actions {
        let row = &mut strategies[action * combos..(action + 1) * combos];
        for (strategy, denominator) in row.iter_mut().zip(denominators.iter().copied()) {
            if denominator > 0.0 {
                *strategy /= denominator;
            } else {
                *strategy = uniform;
            }
        }
    }
}

fn strategy_reach_action_major_into(
    out: &mut [f32],
    reach: &[f32],
    strategies: &[f32],
    combos: usize,
    _actions: usize,
    action: usize,
) {
    let row = &strategies[action * combos..(action + 1) * combos];
    for ((out, reach), strategy) in out.iter_mut().zip(reach).zip(row) {
        *out = *reach * *strategy;
    }
}

fn combine_acting_action_major_values(
    out: &mut [f32],
    action_values: &[f32],
    strategies: &[f32],
    actions: usize,
) {
    out.fill(0.0);
    let combos = out.len();
    for action in 0..actions {
        let value_row = &action_values[action * combos..(action + 1) * combos];
        let strategy_row = &strategies[action * combos..(action + 1) * combos];
        for ((out, value), strategy) in out.iter_mut().zip(value_row).zip(strategy_row) {
            *out += *value * *strategy;
        }
    }
}

fn combine_best_response_action_major_values(
    out: &mut [f32],
    action_values: &[f32],
    actions: usize,
) {
    let combos = out.len();
    for combo in 0..combos {
        let mut best = f32::NEG_INFINITY;
        for action in 0..actions {
            best = best.max(action_values[action * combos + combo]);
        }
        out[combo] = best;
    }
}

fn combine_nonacting_action_major_values(out: &mut [f32], action_values: &[f32], actions: usize) {
    let combos = out.len();
    for combo in 0..combos {
        let mut value = 0.0;
        for action in 0..actions {
            value += action_values[action * combos + combo];
        }
        out[combo] = value;
    }
}

fn add_slice(out: &mut [f32], input: &[f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out += *input;
    }
}

fn add_permuted_scaled_slice(
    out: &mut [f32],
    input: &[f32],
    source_to_target: &[usize],
    scale: f32,
) {
    for (source, target) in source_to_target.iter().copied().enumerate() {
        out[source] += input[target] * scale;
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeLocalUpdateFactors {
    variant: RealCfrVariant,
    average_strategy: RealCfrAverageStrategy,
    positive_regret_discount: f32,
    negative_regret_discount: f32,
    strategy_discount: f32,
    iteration_weight: f32,
}

impl NodeLocalUpdateFactors {
    fn new(
        variant: RealCfrVariant,
        iteration_weight: f32,
        average_strategy: RealCfrAverageStrategy,
    ) -> Self {
        let (positive_regret_discount, negative_regret_discount, strategy_discount) = match variant
        {
            RealCfrVariant::CfrPlus => (1.0, 1.0, 1.0),
            RealCfrVariant::Dcfr { alpha, beta, gamma } => (
                dcfr_discount_for_exponent(iteration_weight, alpha),
                dcfr_discount_for_exponent(iteration_weight, beta),
                dcfr_strategy_discount(iteration_weight, gamma),
            ),
            RealCfrVariant::DcfrPlus { alpha, gamma } => (
                dcfr_discount_for_exponent(iteration_weight, alpha),
                dcfr_discount_for_exponent(iteration_weight, 0.0),
                dcfr_strategy_discount(iteration_weight, gamma),
            ),
        };
        Self {
            variant,
            average_strategy,
            positive_regret_discount,
            negative_regret_discount,
            strategy_discount,
            iteration_weight,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_node_local_updates_action_major(
    regrets: &mut [f32],
    strategy_sum: &mut [f32],
    action_values: &[f32],
    node_values: &[f32],
    strategies: &[f32],
    own_reach: &[f32],
    combos: usize,
    actions: usize,
    factors: &NodeLocalUpdateFactors,
) {
    match factors.variant {
        RealCfrVariant::CfrPlus => {
            for action in 0..actions {
                let offset = action * combos;
                let regret_row = &mut regrets[offset..offset + combos];
                let strategy_sum_row = &mut strategy_sum[offset..offset + combos];
                let action_value_row = &action_values[offset..offset + combos];
                let strategy_row = &strategies[offset..offset + combos];
                match factors.average_strategy {
                    RealCfrAverageStrategy::ReachWeighted => {
                        for (
                            (((regret, strategy_sum), &action_value), &node_value),
                            (&strategy, &reach),
                        ) in regret_row
                            .iter_mut()
                            .zip(strategy_sum_row)
                            .zip(action_value_row)
                            .zip(node_values)
                            .zip(strategy_row.iter().zip(own_reach))
                        {
                            *regret = (*regret + action_value - node_value).max(0.0);
                            *strategy_sum += factors.iteration_weight * reach * strategy;
                        }
                    }
                    RealCfrAverageStrategy::Local => {
                        for ((((regret, strategy_sum), &action_value), &node_value), &strategy) in
                            regret_row
                                .iter_mut()
                                .zip(strategy_sum_row)
                                .zip(action_value_row)
                                .zip(node_values)
                                .zip(strategy_row)
                        {
                            *regret = (*regret + action_value - node_value).max(0.0);
                            *strategy_sum += factors.iteration_weight * strategy;
                        }
                    }
                }
            }
        }
        RealCfrVariant::Dcfr { .. } => {
            for action in 0..actions {
                let offset = action * combos;
                let regret_row = &mut regrets[offset..offset + combos];
                let strategy_sum_row = &mut strategy_sum[offset..offset + combos];
                let action_value_row = &action_values[offset..offset + combos];
                let strategy_row = &strategies[offset..offset + combos];
                match factors.average_strategy {
                    RealCfrAverageStrategy::ReachWeighted => {
                        for (
                            (((regret, strategy_sum), &action_value), &node_value),
                            (&strategy, &reach),
                        ) in regret_row
                            .iter_mut()
                            .zip(strategy_sum_row)
                            .zip(action_value_row)
                            .zip(node_values)
                            .zip(strategy_row.iter().zip(own_reach))
                        {
                            let regret_discount = if *regret >= 0.0 {
                                factors.positive_regret_discount
                            } else {
                                factors.negative_regret_discount
                            };
                            *regret = *regret * regret_discount + action_value - node_value;
                            *strategy_sum =
                                *strategy_sum * factors.strategy_discount + reach * strategy;
                        }
                    }
                    RealCfrAverageStrategy::Local => {
                        for ((((regret, strategy_sum), &action_value), &node_value), &strategy) in
                            regret_row
                                .iter_mut()
                                .zip(strategy_sum_row)
                                .zip(action_value_row)
                                .zip(node_values)
                                .zip(strategy_row)
                        {
                            let regret_discount = if *regret >= 0.0 {
                                factors.positive_regret_discount
                            } else {
                                factors.negative_regret_discount
                            };
                            *regret = *regret * regret_discount + action_value - node_value;
                            *strategy_sum = *strategy_sum * factors.strategy_discount + strategy;
                        }
                    }
                }
            }
        }
        RealCfrVariant::DcfrPlus { .. } => {
            for action in 0..actions {
                let offset = action * combos;
                let regret_row = &mut regrets[offset..offset + combos];
                let strategy_sum_row = &mut strategy_sum[offset..offset + combos];
                let action_value_row = &action_values[offset..offset + combos];
                let strategy_row = &strategies[offset..offset + combos];
                match factors.average_strategy {
                    RealCfrAverageStrategy::ReachWeighted => {
                        for (
                            (((regret, strategy_sum), &action_value), &node_value),
                            (&strategy, &reach),
                        ) in regret_row
                            .iter_mut()
                            .zip(strategy_sum_row)
                            .zip(action_value_row)
                            .zip(node_values)
                            .zip(strategy_row.iter().zip(own_reach))
                        {
                            *regret = (*regret * factors.positive_regret_discount + action_value
                                - node_value)
                                .max(0.0);
                            *strategy_sum =
                                *strategy_sum * factors.strategy_discount + reach * strategy;
                        }
                    }
                    RealCfrAverageStrategy::Local => {
                        for ((((regret, strategy_sum), &action_value), &node_value), &strategy) in
                            regret_row
                                .iter_mut()
                                .zip(strategy_sum_row)
                                .zip(action_value_row)
                                .zip(node_values)
                                .zip(strategy_row)
                        {
                            *regret = (*regret * factors.positive_regret_discount + action_value
                                - node_value)
                                .max(0.0);
                            *strategy_sum = *strategy_sum * factors.strategy_discount + strategy;
                        }
                    }
                }
            }
        }
    }
}

fn dcfr_discount_for_exponent(iteration: f32, exponent: f32) -> f32 {
    let powered = iteration.powf(exponent.max(0.0));
    powered / (powered + 1.0)
}

fn dcfr_strategy_discount(iteration: f32, gamma: f32) -> f32 {
    (iteration / (iteration + 1.0)).powf(gamma.max(0.0))
}

fn opponent_weights_for_fast_into(
    opponent_reach: &[f32],
    same_combo_indices: &[Option<usize>],
    own_live_targets: &[PreparedLiveTarget],
    opponent_live_targets: &[PreparedLiveTarget],
    scale: f32,
    out: &mut [f32],
) {
    let mut total = 0.0f32;
    let mut card_totals = [0.0f32; 52];
    for opponent in opponent_live_targets {
        let reach = opponent_reach[opponent.range_index as usize];
        if reach == 0.0 {
            continue;
        }
        total += reach;
        card_totals[opponent.first_card as usize] += reach;
        card_totals[opponent.second_card as usize] += reach;
    }
    out.fill(0.0);
    for own in own_live_targets {
        let own_index = own.range_index as usize;
        let same_reach = same_combo_indices[own_index]
            .map(|index| opponent_reach[index])
            .unwrap_or(0.0);
        out[own_index] =
            (total - card_totals[own.first_card as usize] - card_totals[own.second_card as usize]
                + same_reach)
                * scale;
    }
}

fn same_combo_indices(
    own_combos: &[ComboWeight],
    opponent_combos: &[ComboWeight],
) -> Vec<Option<usize>> {
    let opponent_by_key = opponent_combos
        .iter()
        .enumerate()
        .map(|(index, combo)| (combo_key(combo.first, combo.second), index))
        .collect::<BTreeMap<_, _>>();
    own_combos
        .iter()
        .map(|combo| {
            opponent_by_key
                .get(&combo_key(combo.first, combo.second))
                .copied()
        })
        .collect()
}

fn combo_key(first: Card, second: Card) -> u64 {
    (1u64 << first.index()) | (1u64 << second.index())
}

fn prepared_combo_targets(
    prepared: &PreparedTerminalBoard,
    combos: &[ComboWeight],
) -> Vec<PreparedComboTarget> {
    combos
        .iter()
        .enumerate()
        .filter_map(|(range_index, combo)| {
            prepared
                .combo_index(combo.first, combo.second)
                .map(|board_index| PreparedComboTarget {
                    range_index,
                    board_index: board_index as u16,
                })
        })
        .collect()
}

fn live_combo_indices(combos: &[ComboWeight], board: &Board) -> Vec<usize> {
    combos
        .iter()
        .enumerate()
        .filter_map(|(index, combo)| {
            (!board.contains(combo.first) && !board.contains(combo.second)).then_some(index)
        })
        .collect()
}

fn live_combo_targets(combos: &[ComboWeight], board: &Board) -> Vec<PreparedLiveTarget> {
    combos
        .iter()
        .enumerate()
        .filter_map(|(index, combo)| {
            if board.contains(combo.first) || board.contains(combo.second) {
                return None;
            }
            Some(PreparedLiveTarget {
                range_index: index
                    .try_into()
                    .expect("range has more than u16::MAX private combos"),
                first_card: combo
                    .first
                    .index()
                    .try_into()
                    .expect("card index does not fit in u8"),
                second_card: combo
                    .second
                    .index()
                    .try_into()
                    .expect("card index does not fit in u8"),
            })
        })
        .collect()
}

fn prepared_board_targets(targets: &[PreparedComboTarget]) -> Vec<u16> {
    targets.iter().map(|target| target.board_index).collect()
}

fn sort_combo_targets_by_strength(
    prepared: &PreparedTerminalBoard,
    targets: &mut [PreparedComboTarget],
) {
    targets.sort_unstable_by_key(|target| prepared.strength(target.board_index as usize));
}

fn prepared_river_targets(
    prepared: &PreparedTerminalBoard,
    targets_sorted: &[PreparedComboTarget],
) -> Vec<PreparedRiverTarget> {
    targets_sorted
        .iter()
        .map(|target| {
            let combo = prepared.combo(target.board_index as usize);
            PreparedRiverTarget {
                strength: prepared.strength(target.board_index as usize),
                range_index: target
                    .range_index
                    .try_into()
                    .expect("range has more than u16::MAX private combos"),
                first_card: combo
                    .first
                    .index()
                    .try_into()
                    .expect("card index does not fit in u8"),
                second_card: combo
                    .second
                    .index()
                    .try_into()
                    .expect("card index does not fit in u8"),
            }
        })
        .collect()
}

fn reach_on_targets_into(targets: &[PreparedComboTarget], reach: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for target in targets {
        out[target.board_index as usize] = reach[target.range_index];
    }
}

fn terminal_side_range_targets_sorted_accumulate(
    prepared: &PreparedTerminalBoard,
    opponent_targets_sorted: &[PreparedComboTarget],
    opponent_reach: &[f32],
    own_targets_sorted: &[PreparedComboTarget],
    pot: f32,
    out: &mut [f32],
) {
    let mut reach_sum = 0.0f32;
    let mut card_sums = [0.0f32; 52];
    let mut opponent_cursor = 0usize;
    for own_target in own_targets_sorted {
        let own_board_index = own_target.board_index as usize;
        let own_strength = prepared.strength(own_board_index);
        while opponent_cursor < opponent_targets_sorted.len() {
            let opponent_target = opponent_targets_sorted[opponent_cursor];
            let opponent_board_index = opponent_target.board_index as usize;
            if prepared.strength(opponent_board_index) >= own_strength {
                break;
            }
            add_target_reach_to_card_sums(
                prepared,
                opponent_target,
                opponent_reach,
                &mut reach_sum,
                &mut card_sums,
            );
            opponent_cursor += 1;
        }
        let own_combo = prepared.combo(own_board_index);
        out[own_target.range_index] +=
            non_blocked_target_reach(own_combo, reach_sum, &card_sums) * pot;
    }

    reach_sum = 0.0;
    card_sums = [0.0f32; 52];
    opponent_cursor = opponent_targets_sorted.len();
    for own_target in own_targets_sorted.iter().rev() {
        let own_board_index = own_target.board_index as usize;
        let own_strength = prepared.strength(own_board_index);
        while opponent_cursor > 0 {
            let opponent_target = opponent_targets_sorted[opponent_cursor - 1];
            let opponent_board_index = opponent_target.board_index as usize;
            if prepared.strength(opponent_board_index) <= own_strength {
                break;
            }
            add_target_reach_to_card_sums(
                prepared,
                opponent_target,
                opponent_reach,
                &mut reach_sum,
                &mut card_sums,
            );
            opponent_cursor -= 1;
        }
        let own_combo = prepared.combo(own_board_index);
        out[own_target.range_index] -=
            non_blocked_target_reach(own_combo, reach_sum, &card_sums) * pot;
    }
}

fn terminal_side_river_targets_sorted_accumulate(
    opponent_targets_sorted: &[PreparedRiverTarget],
    opponent_reach: &[f32],
    own_targets_sorted: &[PreparedRiverTarget],
    pot: f32,
    out: &mut [f32],
) {
    let mut reach_sum = 0.0f32;
    let mut card_sums = [0.0f32; 52];
    let mut opponent_cursor = 0usize;
    for own_target in own_targets_sorted {
        while opponent_cursor < opponent_targets_sorted.len() {
            let opponent_target = opponent_targets_sorted[opponent_cursor];
            if opponent_target.strength >= own_target.strength {
                break;
            }
            add_river_target_reach_to_card_sums(
                opponent_target,
                opponent_reach,
                &mut reach_sum,
                &mut card_sums,
            );
            opponent_cursor += 1;
        }
        out[own_target.range_index as usize] +=
            non_blocked_river_target_reach(*own_target, reach_sum, &card_sums) * pot;
    }

    reach_sum = 0.0;
    card_sums = [0.0f32; 52];
    opponent_cursor = opponent_targets_sorted.len();
    for own_target in own_targets_sorted.iter().rev() {
        while opponent_cursor > 0 {
            let opponent_target = opponent_targets_sorted[opponent_cursor - 1];
            if opponent_target.strength <= own_target.strength {
                break;
            }
            add_river_target_reach_to_card_sums(
                opponent_target,
                opponent_reach,
                &mut reach_sum,
                &mut card_sums,
            );
            opponent_cursor -= 1;
        }
        out[own_target.range_index as usize] -=
            non_blocked_river_target_reach(*own_target, reach_sum, &card_sums) * pot;
    }
}

fn build_allin_oracle(
    board: &Board,
    oop_combos: &[ComboWeight],
    ip_combos: &[ComboWeight],
) -> Result<NodeLocalAllInOracle, String> {
    let oop_live_indices = live_combo_indices(oop_combos, board);
    let ip_live_indices = live_combo_indices(ip_combos, board);
    let oop_cols = ip_combos.len();
    let ip_cols = oop_combos.len();
    let mut oop_payoffs = vec![0.0f32; oop_combos.len() * oop_cols];
    let mut ip_payoffs = vec![0.0f32; ip_combos.len() * ip_cols];
    for terminal_board in terminal_boards(board)? {
        let prepared = PreparedTerminalBoard::new(&terminal_board)?;
        let oop_targets = prepared_combo_targets(&prepared, oop_combos);
        let ip_targets = prepared_combo_targets(&prepared, ip_combos);
        for oop_target in &oop_targets {
            let oop_combo = prepared.combo(oop_target.board_index as usize);
            let oop_strength = prepared.strength(oop_target.board_index as usize);
            for ip_target in &ip_targets {
                let ip_combo = prepared.combo(ip_target.board_index as usize);
                if combos_overlap(oop_combo, ip_combo) {
                    continue;
                }
                let ip_strength = prepared.strength(ip_target.board_index as usize);
                let oop_outcome = if oop_strength > ip_strength {
                    1.0
                } else if oop_strength < ip_strength {
                    -1.0
                } else {
                    0.0
                };
                oop_payoffs[oop_target.range_index * oop_cols + ip_target.range_index] +=
                    oop_outcome;
                ip_payoffs[ip_target.range_index * ip_cols + oop_target.range_index] -= oop_outcome;
            }
        }
    }
    let divisor = terminal_runout_count_for_live_combo(board.cards().len());
    Ok(NodeLocalAllInOracle {
        oop_payoffs,
        ip_payoffs,
        oop_live_indices,
        ip_live_indices,
        oop_divisor: divisor,
        ip_divisor: divisor,
    })
}

fn allin_oracle_matrix_vector_into(
    payoffs: &[f32],
    cols: usize,
    opponent_reach: &[f32],
    own_live_indices: &[usize],
    opponent_live_indices: &[usize],
    scale: f32,
    out: &mut [f32],
) {
    for own_index in own_live_indices.iter().copied() {
        let row = &payoffs[own_index * cols..(own_index + 1) * cols];
        let mut value = 0.0f32;
        for opponent_index in opponent_live_indices.iter().copied() {
            value += row[opponent_index] * opponent_reach[opponent_index];
        }
        out[own_index] = value * scale;
    }
}

fn combos_overlap(
    first: crate::terminal_cfv::PrivateCombo,
    second: crate::terminal_cfv::PrivateCombo,
) -> bool {
    first.first == second.first
        || first.first == second.second
        || first.second == second.first
        || first.second == second.second
}

fn add_target_reach_to_card_sums(
    prepared: &PreparedTerminalBoard,
    target: PreparedComboTarget,
    reach: &[f32],
    reach_sum: &mut f32,
    card_sums: &mut [f32; 52],
) {
    let value = reach[target.range_index as usize];
    if value == 0.0 {
        return;
    }
    let combo = prepared.combo(target.board_index as usize);
    *reach_sum += value;
    card_sums[combo.first.index()] += value;
    card_sums[combo.second.index()] += value;
}

fn add_river_target_reach_to_card_sums(
    target: PreparedRiverTarget,
    reach: &[f32],
    reach_sum: &mut f32,
    card_sums: &mut [f32; 52],
) {
    let value = reach[target.range_index as usize];
    if value == 0.0 {
        return;
    }
    *reach_sum += value;
    card_sums[target.first_card as usize] += value;
    card_sums[target.second_card as usize] += value;
}

fn non_blocked_target_reach(
    combo: crate::terminal_cfv::PrivateCombo,
    reach_sum: f32,
    card_sums: &[f32; 52],
) -> f32 {
    reach_sum - card_sums[combo.first.index()] - card_sums[combo.second.index()]
}

fn non_blocked_river_target_reach(
    target: PreparedRiverTarget,
    reach_sum: f32,
    card_sums: &[f32; 52],
) -> f32 {
    reach_sum - card_sums[target.first_card as usize] - card_sums[target.second_card as usize]
}

fn terminal_runout_count_for_live_combo(board_cards: usize) -> f32 {
    match board_cards {
        3 => 47.0 * 46.0 / 2.0,
        4 => 46.0,
        5 => 1.0,
        _ => 1.0,
    }
}

fn terminal_side_cached_values(
    cache: &mut NodeLocalTerminalSideCache,
    cache_index: usize,
    side: NodeLocalTerminalSide,
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    targets_sorted_by_strength: &[u16],
    scratch_values: &mut Vec<f32>,
    profile: Option<&NodeLocalProfile>,
) -> Result<(), String> {
    let signature = reach_signature(opponent_reach);
    let reach_hash = signature
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, (index, value)| {
            mix_hash(mix_hash(hash, *index as u64), *value as u64)
        });
    let key = NodeLocalTerminalSideCacheKey {
        cache_index,
        side,
        reach_hash,
    };
    if let Some(entries) = cache.entries.get(&key) {
        for entry in entries {
            if entry.signature == signature {
                if let Some(profile) = profile {
                    profile.terminal_cache_hits.fetch_add(1, Ordering::Relaxed);
                }
                scratch_values.clone_from(&entry.values);
                return Ok(());
            }
        }
    }
    if let Some(profile) = profile {
        profile
            .terminal_cache_misses
            .fetch_add(1, Ordering::Relaxed);
    }
    scratch_values.resize(prepared.combos().len(), 0.0);
    terminal_side_values_prefix_blocker_sorted_board_targets_into(
        prepared,
        opponent_reach,
        targets_sorted_by_strength,
        scratch_values,
    )?;
    let entries = cache.entries.entry(key).or_default();
    entries.push(NodeLocalTerminalSideCacheEntry {
        signature,
        values: scratch_values.clone(),
    });
    Ok(())
}

fn reach_signature(reach: &[f32]) -> Vec<(u16, u32)> {
    reach
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0.0).then_some((index as u16, value.to_bits())))
        .collect()
}

fn mix_hash(hash: u64, value: u64) -> u64 {
    hash.wrapping_mul(0x100000001b3).wrapping_add(value)
}

fn terminal_boards(board: &Board) -> Result<Vec<Board>, String> {
    match board.cards().len() {
        5 => Ok(vec![board.clone()]),
        4 => {
            let deck = board.remaining_deck();
            let mut boards = Vec::with_capacity(deck.len());
            for river in deck {
                boards.push(board.push(river)?);
            }
            Ok(boards)
        }
        3 => {
            let deck = board.remaining_deck();
            let mut boards = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
            for turn in 0..deck.len() {
                for river in turn + 1..deck.len() {
                    boards.push(board.push(deck[turn])?.push(deck[river])?);
                }
            }
            Ok(boards)
        }
        other => Err(format!("terminal board has invalid length {other}")),
    }
}

fn unordered_river_boards_from_flop(flop: &Board) -> Result<Vec<Board>, String> {
    if flop.cards().len() != 3 {
        return Err("node-local CFR solver must start from a flop board".to_string());
    }
    let deck = flop.remaining_deck();
    let mut boards = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
    for turn in 0..deck.len() {
        for river in turn + 1..deck.len() {
            boards.push(flop.push(deck[turn])?.push(deck[river])?);
        }
    }
    Ok(boards)
}

fn unordered_board_key(board: &Board) -> u64 {
    board
        .cards()
        .iter()
        .fold(0u64, |key, card| key | (1u64 << card.index()))
}

fn weighted_average(
    values: &[f32],
    combos: &[ComboWeight],
    own_total_weight: f32,
    opponent_total_weight: f32,
) -> f32 {
    if own_total_weight <= 0.0 || opponent_total_weight <= 0.0 {
        return 0.0;
    }
    values
        .iter()
        .zip(combos)
        .map(|(value, combo)| *value * combo.weight)
        .sum::<f32>()
        / (own_total_weight * opponent_total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RealCfrAverageStrategy;
    use crate::legacy::real_cfr::ArenaAlternatingCfrSolver;
    use crate::tree::{
        ActionAbstraction, ChanceExpansion, RaisePolicy, Spot, StreetTemplate, TreeBuilder,
        TreeTemplate,
    };
    use std::str::FromStr;

    #[test]
    fn node_local_cfr_allocates_same_slots_as_arena_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let oop_range = RangeSpec::from_str("AcAd,KcKd").unwrap();
        let ip_range = RangeSpec::from_str("QcQd,JcJd").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: tiny_checkdown_abstraction(),
            chance_expansion: ChanceExpansion::Enumerate,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 200,
            effective_stack: 900,
            oop_range: oop_range.clone(),
            ip_range: ip_range.clone(),
            first_player: Player::Oop,
        })
        .unwrap();
        let arena =
            ArenaAlternatingCfrSolver::new(tree.clone(), oop_range.clone(), ip_range.clone())
                .unwrap();
        let node = NodeLocalCfrSolver::new(tree, oop_range, ip_range).unwrap();
        assert_eq!(node.summary().states, arena.state_count());
        assert_eq!(node.summary().action_slots, arena.regret_len());
    }

    #[test]
    fn node_local_cfr_runs_one_iteration_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let oop_range = RangeSpec::from_str("AcAd,KcKd").unwrap();
        let ip_range = RangeSpec::from_str("QcQd,JcJd").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: tiny_checkdown_abstraction(),
            chance_expansion: ChanceExpansion::Enumerate,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 200,
            effective_stack: 900,
            oop_range: oop_range.clone(),
            ip_range: ip_range.clone(),
            first_player: Player::Oop,
        })
        .unwrap();
        let mut solver = NodeLocalCfrSolver::new(tree, oop_range, ip_range).unwrap();
        let summary = solver
            .run_with_progress(
                RealCfrConfig {
                    iterations: 1,
                    variant: RealCfrVariant::CfrPlus,
                    average_strategy: RealCfrAverageStrategy::ReachWeighted,
                },
                |_| {},
            )
            .unwrap();
        assert_eq!(summary.iterations, 1);
        assert!(summary.terminal_evals > 0);
        assert!(summary.oop_update_pass_value.is_finite());
        assert!(summary.ip_update_pass_value.is_finite());
    }

    #[test]
    fn node_local_exploitability_runs_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let oop_range = RangeSpec::from_str("AcAd,KcKd").unwrap();
        let ip_range = RangeSpec::from_str("QcQd,JcJd").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: tiny_checkdown_abstraction(),
            chance_expansion: ChanceExpansion::Enumerate,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 200,
            effective_stack: 900,
            oop_range: oop_range.clone(),
            ip_range: ip_range.clone(),
            first_player: Player::Oop,
        })
        .unwrap();
        let mut solver = NodeLocalCfrSolver::new(tree, oop_range, ip_range).unwrap();
        solver
            .run_with_progress(
                RealCfrConfig {
                    iterations: 1,
                    variant: RealCfrVariant::CfrPlus,
                    average_strategy: RealCfrAverageStrategy::ReachWeighted,
                },
                |_| {},
            )
            .unwrap();
        let exploitability = solver.exploitability(1).unwrap();
        assert!(exploitability.profile_oop_value.is_finite());
        assert!(exploitability.profile_ip_value.is_finite());
        assert!(exploitability.exploitability_bb_per_100.is_finite());
        assert!(exploitability.oop_gain >= 0.0);
        assert!(exploitability.ip_gain >= 0.0);
    }

    #[test]
    fn river_fast_path_matches_sorted_terminal_path() {
        let board = Board::from_str("As7h2cTd9d").unwrap();
        let prepared = PreparedTerminalBoard::new(&board).unwrap();
        let oop_range = RangeSpec::from_str("AcAd,KcKd,QcQd,JcJd,TcTh").unwrap();
        let ip_range = RangeSpec::from_str("AhKh,QhJh,9c9h,8c8h,7c7d").unwrap();
        let mut oop_targets = prepared_combo_targets(&prepared, oop_range.combos());
        let mut ip_targets = prepared_combo_targets(&prepared, ip_range.combos());
        sort_combo_targets_by_strength(&prepared, &mut oop_targets);
        sort_combo_targets_by_strength(&prepared, &mut ip_targets);
        let oop_river_targets = prepared_river_targets(&prepared, &oop_targets);
        let ip_river_targets = prepared_river_targets(&prepared, &ip_targets);
        let oop_reach = oop_range
            .combos()
            .iter()
            .enumerate()
            .map(|(index, combo)| combo.weight * (index as f32 + 1.0))
            .collect::<Vec<_>>();
        let ip_reach = ip_range
            .combos()
            .iter()
            .enumerate()
            .map(|(index, combo)| combo.weight * (index as f32 + 0.5))
            .collect::<Vec<_>>();
        let mut generic_oop = vec![0.0; oop_range.combos().len()];
        let mut fast_oop = vec![0.0; oop_range.combos().len()];
        terminal_side_range_targets_sorted_accumulate(
            &prepared,
            &ip_targets,
            &ip_reach,
            &oop_targets,
            200.0,
            &mut generic_oop,
        );
        terminal_side_river_targets_sorted_accumulate(
            &ip_river_targets,
            &ip_reach,
            &oop_river_targets,
            200.0,
            &mut fast_oop,
        );
        assert_eq!(generic_oop, fast_oop);

        let mut generic_ip = vec![0.0; ip_range.combos().len()];
        let mut fast_ip = vec![0.0; ip_range.combos().len()];
        terminal_side_range_targets_sorted_accumulate(
            &prepared,
            &oop_targets,
            &oop_reach,
            &ip_targets,
            200.0,
            &mut generic_ip,
        );
        terminal_side_river_targets_sorted_accumulate(
            &oop_river_targets,
            &oop_reach,
            &ip_river_targets,
            200.0,
            &mut fast_ip,
        );
        assert_eq!(generic_ip, fast_ip);
    }

    fn tiny_checkdown_abstraction() -> ActionAbstraction {
        ActionAbstraction {
            min_bet: 2,
            flop: StreetTemplate {
                first_bet_sizes: Vec::new(),
                donk_bet_sizes: Vec::new(),
            },
            turn: StreetTemplate {
                first_bet_sizes: Vec::new(),
                donk_bet_sizes: Vec::new(),
            },
            river: StreetTemplate {
                first_bet_sizes: Vec::new(),
                donk_bet_sizes: Vec::new(),
            },
            raise: RaisePolicy {
                raise_multiplier: 2.0,
                raise_sizes: Vec::new(),
                max_raises_per_street: 0,
                shove_spr_threshold: 0.0,
                shove_commit_fraction: 1.0,
                add_all_in_threshold: 0.0,
                force_all_in_threshold: 1.0,
                merging_threshold: 0.0,
            },
        }
    }
}
