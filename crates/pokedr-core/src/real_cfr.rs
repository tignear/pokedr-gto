use crate::cards::{Board, Card};
use crate::isomorphism::{
    all_suit_permutations, fixed_flop_future_board_isomorphism, next_card_isomorphism,
    private_combo_permutation_indices, terminal_board_isomorphism,
};
use crate::range::{ComboWeight, RangeSpec};
use crate::terminal_cfv::{
    PreparedTerminalBoard, TerminalCfvScratch, terminal_cfv_prefix_blocker_board_targets_into,
    terminal_cfv_prefix_blocker_sorted_board_targets_into, terminal_cfv_sparse_board_targets_into,
    terminal_side_values_prefix_blocker_sorted_board_targets_into,
    terminal_side_values_sparse_board_targets_into,
};
use crate::tree::{Player, PublicNodeKind, PublicTree, TerminalReason};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrConfig {
    pub iterations: u32,
    pub variant: RealCfrVariant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealCfrVariant {
    CfrPlus,
    Dcfr { alpha: f32, beta: f32, gamma: f32 },
    DcfrPlus { alpha: f32, gamma: f32 },
}

impl Default for RealCfrVariant {
    fn default() -> Self {
        Self::CfrPlus
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrSummary {
    pub iterations: u32,
    pub decision_nodes: usize,
    pub action_slots: usize,
    pub terminal_evals: usize,
    pub root_oop_value: f32,
    pub root_ip_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrIterationSummary {
    pub iteration: u32,
    pub terminal_evals: usize,
    pub elapsed_ms: f64,
    pub root_oop_value: f32,
    pub root_ip_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrPhaseIterationSummary {
    pub iteration: u32,
    pub terminal_evals: usize,
    pub reach_ms: f64,
    pub terminal_ms: f64,
    pub backup_ms: f64,
    pub root_oop_value: f32,
    pub root_ip_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrPhaseSummary {
    pub iterations: u32,
    pub states: usize,
    pub decision_nodes: usize,
    pub action_slots: usize,
    pub terminal_evals: usize,
    pub reach_ms: f64,
    pub terminal_ms: f64,
    pub backup_ms: f64,
    pub root_oop_value: f32,
    pub root_ip_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrExploitability {
    pub profile_oop_value: f32,
    pub profile_ip_value: f32,
    pub oop_best_response_value: f32,
    pub ip_best_response_value: f32,
    pub oop_gain: f32,
    pub ip_gain: f32,
    pub nash_conv_chips: f32,
    pub exploitability_chips: f32,
    pub exploitability_bb_per_100: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalBoardPhaseSummary {
    pub terminal_evals: usize,
    pub threads: usize,
    pub elapsed_ms: f64,
    pub checksum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerminalEvalBreakdown {
    pub fold_terminals: usize,
    pub showdown_terminals: usize,
    pub all_in_terminals: usize,
    pub river_showdown_evals: usize,
    pub flop_all_in_runout_evals: usize,
    pub turn_all_in_runout_evals: usize,
    pub river_all_in_evals: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalBoardLocality {
    pub tasks: usize,
    pub unique_boards: usize,
    pub current_order_runs: usize,
    pub average_run_len: f64,
    pub max_run_len: usize,
    pub min_tasks_per_board: usize,
    pub max_tasks_per_board: usize,
    pub average_tasks_per_board: f64,
    pub board_major_task_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalBoardReuseReport {
    pub state_board_pairs: usize,
    pub unique_boards: usize,
    pub average_state_board_pairs_per_board: f64,
    pub min_state_board_pairs_per_board: usize,
    pub max_state_board_pairs_per_board: usize,
    pub average_unique_terminal_states_per_board: f64,
    pub average_oop_unique_reaches_per_board: f64,
    pub average_ip_unique_reaches_per_board: f64,
    pub average_pair_unique_reaches_per_board: f64,
    pub average_oop_value_side_reuse_factor: f64,
    pub average_ip_value_side_reuse_factor: f64,
    pub rows: Vec<TerminalBoardReuseRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TerminalBoardReuseRow {
    pub board_index: usize,
    pub board: Board,
    pub state_board_pairs: usize,
    pub unique_terminal_states: usize,
    pub pot_buckets: usize,
    pub oop_unique_reaches: usize,
    pub ip_unique_reaches: usize,
    pub pair_unique_reaches: usize,
    pub oop_value_side_reuse_factor: f64,
    pub ip_value_side_reuse_factor: f64,
    pub average_oop_nonzero: f64,
    pub average_ip_nonzero: f64,
    pub max_oop_nonzero: usize,
    pub max_ip_nonzero: usize,
}

#[derive(Debug, Clone)]
pub struct RealCfrSolver {
    tree: PublicTree,
    oop_range: RangeSpec,
    ip_range: RangeSpec,
    oop_combos: Vec<ComboWeight>,
    ip_combos: Vec<ComboWeight>,
    infosets: Vec<Option<RealInfoset>>,
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
    completed_iterations: u32,
    flop_board: Board,
    turn_index_by_key: BTreeMap<u64, usize>,
    river_index_by_key: BTreeMap<u64, usize>,
    terminal_cache_index_by_key: BTreeMap<u64, usize>,
    terminal_cache: Vec<TerminalEvalCache>,
    combo_permutations: BTreeMap<u8, ComboPermutationMaps>,
    oop_same_ip_combo_indices: Vec<Option<usize>>,
    ip_same_oop_combo_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct RealInfoset {
    player: Player,
    board_count: usize,
    actions: usize,
    slots_start: usize,
    slots_len: usize,
}

#[derive(Debug, Clone)]
struct Values {
    oop: Vec<f32>,
    ip: Vec<f32>,
    terminal_evals: usize,
}

#[derive(Debug, Clone)]
struct TerminalEvalCache {
    board: Board,
    prepared: PreparedTerminalBoard,
    oop_combo_indices: Vec<Option<usize>>,
    ip_combo_indices: Vec<Option<usize>>,
    oop_targets: Vec<PreparedComboTarget>,
    ip_targets: Vec<PreparedComboTarget>,
    oop_board_targets: Vec<u16>,
    ip_board_targets: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedComboTarget {
    range_index: usize,
    board_index: u16,
}

#[derive(Debug, Clone)]
struct TerminalBoardTask {
    terminal_node: usize,
    board: Board,
    cache_index: usize,
}

const TERMINAL_SPARSE_NONZERO_LIMIT: usize = 64;

#[derive(Debug, Clone)]
struct TerminalAccumulator {
    values: Values,
    oop_counts: Vec<f32>,
    ip_counts: Vec<f32>,
}

struct RecursiveTerminalScratch {
    cfv: TerminalCfvScratch,
    oop_live: Vec<f32>,
    ip_live: Vec<f32>,
    oop_nonzero: Vec<u16>,
    ip_nonzero: Vec<u16>,
    accumulator: TerminalAccumulator,
    vectors: Vec<Vec<f32>>,
}

impl RecursiveTerminalScratch {
    fn take_vec(&mut self, len: usize) -> Vec<f32> {
        match self.vectors.pop() {
            Some(mut values) => {
                values.resize(len, 0.0);
                values.fill(0.0);
                values
            }
            None => vec![0.0; len],
        }
    }

    fn release_vec(&mut self, values: Vec<f32>) {
        self.vectors.push(values);
    }
}

#[derive(Debug, Default)]
struct RecursiveCfrProfile {
    enabled: bool,
    terminal_calls: u64,
    fold_calls: u64,
    showdown_calls: u64,
    chance_calls: u64,
    chance_cards: u64,
    decision_calls: u64,
    strategy_builds: u64,
    reach_scratch_writes: u64,
    values_zero: u64,
}

impl RecursiveCfrProfile {
    fn reset_counts(&mut self) {
        self.terminal_calls = 0;
        self.fold_calls = 0;
        self.showdown_calls = 0;
        self.chance_calls = 0;
        self.chance_cards = 0;
        self.decision_calls = 0;
        self.strategy_builds = 0;
        self.reach_scratch_writes = 0;
        self.values_zero = 0;
    }
}

#[derive(Debug, Clone, Default)]
struct TerminalWorkerProfile {
    worker_index: usize,
    tasks: usize,
    terminal_states: usize,
    fold_states: usize,
    sparse_tasks: usize,
    prefix_tasks: usize,
    zero_reach_tasks: usize,
    side_cache_hits: usize,
    side_cache_misses: usize,
    oop_nonzero_sum: usize,
    ip_nonzero_sum: usize,
    oop_nonzero_max: usize,
    ip_nonzero_max: usize,
    output_states: usize,
    board_expand_ms: f64,
    fold_ms: f64,
    reach_map_ms: f64,
    cfv_ms: f64,
    accumulator_ms: f64,
    elapsed_ms: f64,
    side_cache_keys: Vec<TerminalSideCacheKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TerminalSideValue {
    OopValue,
    IpValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TerminalSideCacheKey {
    cache_index: usize,
    side: TerminalSideValue,
    reach_hash: u64,
}

#[derive(Debug, Clone)]
struct TerminalSideCacheEntry {
    signature: Vec<(u16, u32)>,
    values: Arc<[f32]>,
}

#[derive(Debug, Default)]
struct TerminalSideValueCache {
    entries: HashMap<TerminalSideCacheKey, Vec<TerminalSideCacheEntry>>,
    hits: usize,
    misses: usize,
}

#[derive(Debug, Default)]
struct TerminalBoardReuseStats {
    tasks: usize,
    terminal_states: BTreeSet<usize>,
    pots: BTreeSet<u32>,
    oop_reaches: BTreeSet<u64>,
    ip_reaches: BTreeSet<u64>,
    pair_reaches: BTreeSet<u64>,
    oop_nonzero_sum: usize,
    ip_nonzero_sum: usize,
    oop_nonzero_max: usize,
    ip_nonzero_max: usize,
}

impl TerminalBoardReuseStats {
    fn add(
        &mut self,
        state_index: usize,
        pot: u32,
        oop_hash: u64,
        ip_hash: u64,
        oop_nonzero: usize,
        ip_nonzero: usize,
    ) {
        self.tasks += 1;
        self.terminal_states.insert(state_index);
        self.pots.insert(pot);
        self.oop_reaches.insert(oop_hash);
        self.ip_reaches.insert(ip_hash);
        self.pair_reaches
            .insert(mix_hash(mix_hash(0xcbf29ce484222325, oop_hash), ip_hash));
        self.oop_nonzero_sum += oop_nonzero;
        self.ip_nonzero_sum += ip_nonzero;
        self.oop_nonzero_max = self.oop_nonzero_max.max(oop_nonzero);
        self.ip_nonzero_max = self.ip_nonzero_max.max(ip_nonzero);
    }
}

#[derive(Debug, Clone)]
struct PhaseState {
    node_id: usize,
    board: Board,
    board_slot: usize,
    children: Vec<usize>,
    chance_member_permutation_codes: Vec<Vec<u8>>,
    chance_concrete_events: usize,
    terminal_cache_indices: Vec<usize>,
    terminal_cache_refs: Vec<TerminalCacheRef>,
}

#[derive(Debug, Clone)]
struct TerminalCacheRef {
    cache_index: usize,
    member_permutation_codes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ComboPermutationMaps {
    oop_source_to_target: Vec<usize>,
    ip_source_to_target: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackupRun {
    start: usize,
    end: usize,
    slot_start: usize,
    slot_end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackupChunk {
    state_start: usize,
    state_end: usize,
    slot_start: usize,
    slot_end: usize,
}

struct BackupDecisionJob<'a> {
    state_start: usize,
    slot_start: usize,
    values: &'a mut [Values],
    regrets: &'a mut [f32],
    strategy_sum: &'a mut [f32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BackupLevelPlan {
    levels: Vec<Vec<BackupRun>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvaluationMode {
    Profile,
    OopBestResponse,
    IpBestResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrategySource {
    Current,
    Average,
}

impl RealCfrSolver {
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
        let flop_board = tree.spot.board.clone();
        let (turn_boards, river_boards) =
            representative_ordered_future_boards(&flop_board, &oop_range, &ip_range)?;
        let turn_index_by_key = turn_boards
            .iter()
            .enumerate()
            .map(|(index, board)| (ordered_board_key(board), index))
            .collect::<BTreeMap<_, _>>();
        let river_index_by_key = river_boards
            .iter()
            .enumerate()
            .map(|(index, board)| (ordered_board_key(board), index))
            .collect::<BTreeMap<_, _>>();
        let terminal_boards = unordered_river_boards_from_flop(&flop_board)?;
        let mut terminal_cache_index_by_key = BTreeMap::new();
        let mut terminal_cache = Vec::with_capacity(terminal_boards.len());
        for board in &terminal_boards {
            let prepared = PreparedTerminalBoard::new(board)?;
            let oop_combo_indices = prepared_combo_indices(&prepared, &oop_combos);
            let ip_combo_indices = prepared_combo_indices(&prepared, &ip_combos);
            let oop_targets = prepared_combo_targets(&oop_combo_indices);
            let ip_targets = prepared_combo_targets(&ip_combo_indices);
            let mut oop_board_targets = prepared_board_targets(&oop_targets);
            let mut ip_board_targets = prepared_board_targets(&ip_targets);
            prepared.sort_indices_by_strength(&mut oop_board_targets);
            prepared.sort_indices_by_strength(&mut ip_board_targets);
            terminal_cache_index_by_key.insert(unordered_board_key(board), terminal_cache.len());
            terminal_cache.push(TerminalEvalCache {
                board: board.clone(),
                oop_combo_indices,
                ip_combo_indices,
                oop_targets,
                ip_targets,
                oop_board_targets,
                ip_board_targets,
                prepared,
            });
        }
        let mut infosets = vec![None; tree.nodes.len()];
        let mut total_action_slots = 0usize;
        for node in &tree.nodes {
            let PublicNodeKind::Decision { player, actions } = &node.kind else {
                continue;
            };
            if actions.len() <= 1 {
                continue;
            }
            let combos = match player {
                Player::Oop => oop_combos.len(),
                Player::Ip => ip_combos.len(),
            };
            let board_count = match node.state.board.cards().len() {
                3 => 1,
                4 => turn_index_by_key.len(),
                5 => river_index_by_key.len(),
                other => return Err(format!("invalid public board length {other}")),
            };
            let slots_len = board_count * combos * actions.len();
            infosets[node.id] = Some(RealInfoset {
                player: *player,
                board_count,
                actions: actions.len(),
                slots_start: total_action_slots,
                slots_len,
            });
            total_action_slots += slots_len;
        }
        Ok(Self {
            tree,
            oop_range,
            ip_range,
            oop_combos,
            ip_combos,
            infosets,
            regrets: vec![0.0; total_action_slots],
            strategy_sum: vec![0.0; total_action_slots],
            completed_iterations: 0,
            flop_board,
            turn_index_by_key,
            river_index_by_key,
            terminal_cache_index_by_key,
            terminal_cache,
            combo_permutations,
            oop_same_ip_combo_indices,
            ip_same_oop_combo_indices,
        })
    }

    pub fn regret_len(&self) -> usize {
        self.regrets.len()
    }

    pub fn strategy_sum_len(&self) -> usize {
        self.strategy_sum.len()
    }

    pub fn storage_gib(&self) -> f64 {
        (self.regrets.len() + self.strategy_sum.len()) as f64 * std::mem::size_of::<f32>() as f64
            / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn run(&mut self, config: RealCfrConfig) -> Result<RealCfrSummary, String> {
        self.run_with_progress(config, |_| {})
    }

    pub fn run_with_progress(
        &mut self,
        config: RealCfrConfig,
        mut progress: impl FnMut(RealCfrIterationSummary),
    ) -> Result<RealCfrSummary, String> {
        let mut root_oop = vec![0.0; self.oop_combos.len()];
        let mut root_ip = vec![0.0; self.ip_combos.len()];
        let mut root_terminal_evals = 0usize;
        let scratch_source = self
            .terminal_cache
            .first()
            .ok_or_else(|| "terminal board cache is empty".to_string())?;
        let terminal_combos = scratch_source.prepared.combos().len();
        let mut terminal_scratch = RecursiveTerminalScratch {
            cfv: TerminalCfvScratch::new(&scratch_source.prepared),
            oop_live: vec![0.0; terminal_combos],
            ip_live: vec![0.0; terminal_combos],
            oop_nonzero: Vec::new(),
            ip_nonzero: Vec::new(),
            accumulator: TerminalAccumulator::zero(self.oop_combos.len(), self.ip_combos.len()),
            vectors: Vec::new(),
        };
        let mut terminal_ref_cache = BTreeMap::new();
        let mut side_cache = TerminalSideValueCache::default();
        let mut recursive_profile = RecursiveCfrProfile {
            enabled: std::env::var_os("POKEDR_REAL_CFR_RECURSIVE_PROFILE").is_some(),
            ..RecursiveCfrProfile::default()
        };
        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();
        for iteration in 1..=config.iterations {
            let started = std::time::Instant::now();
            let oop_reach = self
                .oop_combos
                .iter()
                .map(|combo| combo.weight)
                .collect::<Vec<_>>();
            let ip_reach = self
                .ip_combos
                .iter()
                .map(|combo| combo.weight)
                .collect::<Vec<_>>();
            let board = self.flop_board.clone();
            self.completed_iterations += 1;
            let average_weight = self.completed_iterations as f32;
            root_terminal_evals = self.traverse_slices_into(
                &mut root_oop,
                &mut root_ip,
                0,
                &board,
                &oop_reach,
                &ip_reach,
                average_weight,
                config.variant,
                &mut terminal_scratch,
                &mut terminal_ref_cache,
                &mut side_cache,
                &mut recursive_profile,
            )?;
            if recursive_profile.enabled {
                eprintln!(
                    "real_cfr_recursive_profile iteration={} terminal_calls={} fold_calls={} showdown_calls={} chance_calls={} chance_cards={} decision_calls={} strategy_builds={} reach_scratch_writes={} values_zero={}",
                    iteration,
                    recursive_profile.terminal_calls,
                    recursive_profile.fold_calls,
                    recursive_profile.showdown_calls,
                    recursive_profile.chance_calls,
                    recursive_profile.chance_cards,
                    recursive_profile.decision_calls,
                    recursive_profile.strategy_builds,
                    recursive_profile.reach_scratch_writes,
                    recursive_profile.values_zero,
                );
                recursive_profile.reset_counts();
            }
            progress(RealCfrIterationSummary {
                iteration,
                terminal_evals: root_terminal_evals,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                root_oop_value: weighted_average(
                    &root_oop,
                    &self.oop_combos,
                    oop_weight,
                    ip_weight,
                ),
                root_ip_value: weighted_average(&root_ip, &self.ip_combos, ip_weight, oop_weight),
            });
        }
        let root_oop_value = weighted_average(&root_oop, &self.oop_combos, oop_weight, ip_weight);
        let root_ip_value = weighted_average(&root_ip, &self.ip_combos, ip_weight, oop_weight);
        let action_slots = self
            .infosets
            .iter()
            .filter_map(Option::as_ref)
            .map(|infoset| infoset.slots_len)
            .sum();
        let decision_nodes = self
            .infosets
            .iter()
            .filter(|infoset| infoset.is_some())
            .count();
        Ok(RealCfrSummary {
            iterations: config.iterations,
            decision_nodes,
            action_slots,
            terminal_evals: root_terminal_evals,
            root_oop_value,
            root_ip_value,
        })
    }

    pub fn run_three_phase(
        &mut self,
        config: RealCfrConfig,
        threads: usize,
        mut progress: impl FnMut(RealCfrPhaseIterationSummary),
    ) -> Result<RealCfrPhaseSummary, String> {
        self.run_three_phase_with_terminal_side_cache(
            config,
            threads,
            terminal_side_cache_enabled(),
            &mut progress,
        )
    }

    fn run_three_phase_with_terminal_side_cache(
        &mut self,
        config: RealCfrConfig,
        threads: usize,
        use_side_cache: bool,
        mut progress: impl FnMut(RealCfrPhaseIterationSummary),
    ) -> Result<RealCfrPhaseSummary, String> {
        let states = self.collect_phase_states()?;
        let backup_plan = backup_level_plan(
            &self.tree,
            &self.infosets,
            self.oop_combos.len(),
            self.ip_combos.len(),
            &states,
        );
        let mut root = Values::zero(self.oop_combos.len(), self.ip_combos.len());
        let mut values =
            vec![Values::zero(self.oop_combos.len(), self.ip_combos.len()); states.len()];
        let mut oop_reaches = vec![vec![0.0f32; self.oop_combos.len()]; states.len()];
        let mut ip_reaches = vec![vec![0.0f32; self.ip_combos.len()]; states.len()];
        let mut last_terminal_evals = 0usize;
        let mut total_reach_ms = 0.0;
        let mut total_terminal_ms = 0.0;
        let mut total_backup_ms = 0.0;
        let profile_start_iteration = real_cfr_profile_start_iteration();
        let profile_reach_requested = std::env::var_os("POKEDR_REAL_CFR_REACH_PROFILE").is_some();
        let profile_terminal_requested =
            std::env::var_os("POKEDR_REAL_CFR_TERMINAL_PROFILE").is_some();
        let profile_side_cache_keys_requested =
            std::env::var_os("POKEDR_REAL_CFR_SIDE_CACHE_KEY_PROFILE").is_some();
        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();

        for iteration in 1..=config.iterations {
            let global_iteration = self.completed_iterations + 1;
            let profile_this_iteration = global_iteration >= profile_start_iteration;
            let profile_reach = profile_reach_requested && profile_this_iteration;
            let profile_terminal = profile_terminal_requested && profile_this_iteration;
            let profile_side_cache_keys =
                profile_side_cache_keys_requested && profile_this_iteration;
            let reach_started = std::time::Instant::now();
            self.forward_reaches_into_with_profile(
                &states,
                &mut oop_reaches,
                &mut ip_reaches,
                profile_reach,
            )?;
            let reach_ms = reach_started.elapsed().as_secs_f64() * 1000.0;

            let terminal_started = std::time::Instant::now();
            self.terminal_phase_into_with_profile_options(
                &states,
                &oop_reaches,
                &ip_reaches,
                threads,
                &mut values,
                0.0,
                profile_terminal,
                profile_side_cache_keys,
                use_side_cache,
            )?;
            let terminal_ms = terminal_started.elapsed().as_secs_f64() * 1000.0;

            let backup_started = std::time::Instant::now();
            self.completed_iterations += 1;
            let average_weight = self.completed_iterations as f32;
            last_terminal_evals = self.backup_phase(
                &states,
                &backup_plan,
                &oop_reaches,
                &ip_reaches,
                &mut values,
                threads,
                average_weight,
                config.variant,
            )?;
            let backup_ms = backup_started.elapsed().as_secs_f64() * 1000.0;

            root = values[0].clone();
            let root_oop_value =
                weighted_average(&root.oop, &self.oop_combos, oop_weight, ip_weight);
            let root_ip_value = weighted_average(&root.ip, &self.ip_combos, ip_weight, oop_weight);
            total_reach_ms += reach_ms;
            total_terminal_ms += terminal_ms;
            total_backup_ms += backup_ms;
            progress(RealCfrPhaseIterationSummary {
                iteration,
                terminal_evals: last_terminal_evals,
                reach_ms,
                terminal_ms,
                backup_ms,
                root_oop_value,
                root_ip_value,
            });
        }

        let root_oop_value = weighted_average(&root.oop, &self.oop_combos, oop_weight, ip_weight);
        let root_ip_value = weighted_average(&root.ip, &self.ip_combos, ip_weight, oop_weight);
        let action_slots = self
            .infosets
            .iter()
            .filter_map(Option::as_ref)
            .map(|infoset| infoset.slots_len)
            .sum();
        let decision_nodes = self
            .infosets
            .iter()
            .filter(|infoset| infoset.is_some())
            .count();
        Ok(RealCfrPhaseSummary {
            iterations: config.iterations,
            states: states.len(),
            decision_nodes,
            action_slots,
            terminal_evals: last_terminal_evals,
            reach_ms: total_reach_ms,
            terminal_ms: total_terminal_ms,
            backup_ms: total_backup_ms,
            root_oop_value,
            root_ip_value,
        })
    }

    pub fn exploitability(&self, threads: usize) -> Result<RealCfrExploitability, String> {
        let states = self.collect_phase_states()?;
        let profile = self.evaluate_states(&states, EvaluationMode::Profile, threads)?;
        let oop_br = self.evaluate_states(&states, EvaluationMode::OopBestResponse, threads)?;
        let ip_br = self.evaluate_states(&states, EvaluationMode::IpBestResponse, threads)?;
        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();
        let profile_oop_value =
            weighted_average(&profile.oop, &self.oop_combos, oop_weight, ip_weight);
        let profile_ip_value =
            weighted_average(&profile.ip, &self.ip_combos, ip_weight, oop_weight);
        let oop_best_response_value =
            weighted_average(&oop_br.oop, &self.oop_combos, oop_weight, ip_weight);
        let ip_best_response_value =
            weighted_average(&ip_br.ip, &self.ip_combos, ip_weight, oop_weight);
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

    pub fn run_terminal_board_phase(
        &self,
        threads: usize,
    ) -> Result<TerminalBoardPhaseSummary, String> {
        self.run_terminal_board_phase_ordered(threads, false)
    }

    pub fn run_terminal_board_phase_board_major(
        &self,
        threads: usize,
    ) -> Result<TerminalBoardPhaseSummary, String> {
        self.run_terminal_board_phase_ordered(threads, true)
    }

    fn run_terminal_board_phase_ordered(
        &self,
        threads: usize,
        board_major: bool,
    ) -> Result<TerminalBoardPhaseSummary, String> {
        let started = std::time::Instant::now();
        let mut tasks = self.collect_terminal_board_tasks()?;
        if board_major {
            tasks.sort_by_key(|task| (task.cache_index, task.terminal_node));
        }
        let threads = effective_worker_count(threads).min(tasks.len().max(1));
        let oop_reach = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let ip_reach = self
            .ip_combos
            .iter()
            .map(|combo| combo.weight)
            .collect::<Vec<_>>();
        let chunk = tasks.len().div_ceil(threads);
        let checksum = tasks
            .par_chunks(chunk)
            .map(|tasks| -> Result<f64, String> {
                let mut checksum = 0.0f64;
                let scratch_source = self
                    .terminal_cache
                    .first()
                    .ok_or_else(|| "terminal board cache is empty".to_string())?;
                let combos = scratch_source.prepared.combos().len();
                let mut scratch = TerminalCfvScratch::new(&scratch_source.prepared);
                let mut oop_live = vec![0.0f32; combos];
                let mut ip_live = vec![0.0f32; combos];
                for task in tasks {
                    let cache = &self.terminal_cache[task.cache_index];
                    reach_on_prepared_board_targets_into(
                        &cache.oop_targets,
                        &oop_reach,
                        &mut oop_live,
                    );
                    reach_on_prepared_board_targets_into(
                        &cache.ip_targets,
                        &ip_reach,
                        &mut ip_live,
                    );
                    terminal_cfv_prefix_blocker_board_targets_into(
                        &cache.prepared,
                        &oop_live,
                        &ip_live,
                        &cache.oop_board_targets,
                        &cache.ip_board_targets,
                        &mut scratch,
                    )?;
                    checksum += terminal_task_checksum(task, &cache.prepared, &scratch);
                }
                Ok(checksum)
            })
            .try_reduce(|| 0.0f64, |left, right| Ok(left + right))?;
        Ok(TerminalBoardPhaseSummary {
            terminal_evals: tasks.len(),
            threads,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            checksum,
        })
    }

    pub fn terminal_board_locality(&self) -> Result<TerminalBoardLocality, String> {
        let tasks = self.collect_terminal_board_tasks()?;
        let mut counts = vec![0usize; self.terminal_cache.len()];
        let mut current_order_runs = 0usize;
        let mut max_run_len = 0usize;
        let mut previous_cache_index = None;
        let mut current_run_len = 0usize;
        for task in &tasks {
            counts[task.cache_index] += 1;
            if previous_cache_index == Some(task.cache_index) {
                current_run_len += 1;
            } else {
                if current_run_len > 0 {
                    max_run_len = max_run_len.max(current_run_len);
                }
                current_order_runs += 1;
                previous_cache_index = Some(task.cache_index);
                current_run_len = 1;
            }
        }
        max_run_len = max_run_len.max(current_run_len);
        let used_counts = counts
            .into_iter()
            .filter(|count| *count > 0)
            .collect::<Vec<_>>();
        let unique_boards = used_counts.len();
        let min_tasks_per_board = used_counts.iter().copied().min().unwrap_or(0);
        let max_tasks_per_board = used_counts.iter().copied().max().unwrap_or(0);
        let average_tasks_per_board = if unique_boards > 0 {
            tasks.len() as f64 / unique_boards as f64
        } else {
            0.0
        };
        let average_run_len = if current_order_runs > 0 {
            tasks.len() as f64 / current_order_runs as f64
        } else {
            0.0
        };
        Ok(TerminalBoardLocality {
            tasks: tasks.len(),
            unique_boards,
            current_order_runs,
            average_run_len,
            max_run_len,
            min_tasks_per_board,
            max_tasks_per_board,
            average_tasks_per_board,
            board_major_task_bytes: tasks.len() * std::mem::size_of::<TerminalBoardTask>(),
        })
    }

    pub fn terminal_board_reuse_report(&self) -> Result<TerminalBoardReuseReport, String> {
        let states = self.collect_phase_states()?;
        let (oop_reaches, ip_reaches) = self.forward_reaches_for_mode(
            &states,
            EvaluationMode::Profile,
            StrategySource::Current,
        )?;
        let first_cache = self
            .terminal_cache
            .first()
            .ok_or_else(|| "terminal board cache is empty".to_string())?;
        let combos = first_cache.prepared.combos().len();
        let mut oop_live = vec![0.0f32; combos];
        let mut ip_live = vec![0.0f32; combos];
        let mut oop_nonzero = Vec::new();
        let mut ip_nonzero = Vec::new();
        let mut stats = (0..self.terminal_cache.len())
            .map(|_| TerminalBoardReuseStats::default())
            .collect::<Vec<_>>();

        for (state_index, state) in states.iter().enumerate() {
            if state.terminal_cache_indices.is_empty() {
                continue;
            }
            let node = &self.tree.nodes[state.node_id];
            let PublicNodeKind::Terminal {
                reason: TerminalReason::Showdown | TerminalReason::AllIn,
            } = node.kind
            else {
                continue;
            };
            for cache_index in &state.terminal_cache_indices {
                let cache = &self.terminal_cache[*cache_index];
                reach_on_prepared_board_targets_sparse_into(
                    &cache.oop_targets,
                    &oop_reaches[state_index],
                    &mut oop_live,
                    &mut oop_nonzero,
                );
                reach_on_prepared_board_targets_sparse_into(
                    &cache.ip_targets,
                    &ip_reaches[state_index],
                    &mut ip_live,
                    &mut ip_nonzero,
                );
                let oop_hash = hash_sparse_reach(&oop_live, &oop_nonzero);
                let ip_hash = hash_sparse_reach(&ip_live, &ip_nonzero);
                stats[*cache_index].add(
                    state_index,
                    node.state.pot,
                    oop_hash,
                    ip_hash,
                    oop_nonzero.len(),
                    ip_nonzero.len(),
                );
            }
        }

        let mut rows = stats
            .iter()
            .enumerate()
            .filter(|(_, stats)| stats.tasks > 0)
            .map(|(board_index, stats)| {
                let oop_value_side_reuse_factor = if stats.ip_reaches.is_empty() {
                    0.0
                } else {
                    stats.tasks as f64 / stats.ip_reaches.len() as f64
                };
                let ip_value_side_reuse_factor = if stats.oop_reaches.is_empty() {
                    0.0
                } else {
                    stats.tasks as f64 / stats.oop_reaches.len() as f64
                };
                TerminalBoardReuseRow {
                    board_index,
                    board: self.terminal_cache[board_index].board.clone(),
                    state_board_pairs: stats.tasks,
                    unique_terminal_states: stats.terminal_states.len(),
                    pot_buckets: stats.pots.len(),
                    oop_unique_reaches: stats.oop_reaches.len(),
                    ip_unique_reaches: stats.ip_reaches.len(),
                    pair_unique_reaches: stats.pair_reaches.len(),
                    oop_value_side_reuse_factor,
                    ip_value_side_reuse_factor,
                    average_oop_nonzero: stats.oop_nonzero_sum as f64 / stats.tasks as f64,
                    average_ip_nonzero: stats.ip_nonzero_sum as f64 / stats.tasks as f64,
                    max_oop_nonzero: stats.oop_nonzero_max,
                    max_ip_nonzero: stats.ip_nonzero_max,
                }
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .state_board_pairs
                .cmp(&left.state_board_pairs)
                .then_with(|| right.pair_unique_reaches.cmp(&left.pair_unique_reaches))
                .then_with(|| left.board_index.cmp(&right.board_index))
        });

        let unique_boards = rows.len();
        let state_board_pairs = rows.iter().map(|row| row.state_board_pairs).sum::<usize>();
        let unique_terminal_states = rows
            .iter()
            .map(|row| row.unique_terminal_states)
            .sum::<usize>();
        Ok(TerminalBoardReuseReport {
            state_board_pairs,
            unique_boards,
            average_state_board_pairs_per_board: average_usize(state_board_pairs, unique_boards),
            min_state_board_pairs_per_board: rows
                .iter()
                .map(|row| row.state_board_pairs)
                .min()
                .unwrap_or(0),
            max_state_board_pairs_per_board: rows
                .iter()
                .map(|row| row.state_board_pairs)
                .max()
                .unwrap_or(0),
            average_unique_terminal_states_per_board: average_usize(
                unique_terminal_states,
                unique_boards,
            ),
            average_oop_unique_reaches_per_board: average_usize(
                rows.iter().map(|row| row.oop_unique_reaches).sum(),
                unique_boards,
            ),
            average_ip_unique_reaches_per_board: average_usize(
                rows.iter().map(|row| row.ip_unique_reaches).sum(),
                unique_boards,
            ),
            average_pair_unique_reaches_per_board: average_usize(
                rows.iter().map(|row| row.pair_unique_reaches).sum(),
                unique_boards,
            ),
            average_oop_value_side_reuse_factor: average_f64(
                rows.iter().map(|row| row.oop_value_side_reuse_factor).sum(),
                unique_boards,
            ),
            average_ip_value_side_reuse_factor: average_f64(
                rows.iter().map(|row| row.ip_value_side_reuse_factor).sum(),
                unique_boards,
            ),
            rows,
        })
    }

    pub fn terminal_eval_breakdown(&self) -> Result<TerminalEvalBreakdown, String> {
        let mut breakdown = TerminalEvalBreakdown::default();
        self.terminal_eval_breakdown_from(0, &self.flop_board, &mut breakdown)?;
        Ok(breakdown)
    }

    fn collect_phase_states(&self) -> Result<Vec<PhaseState>, String> {
        let mut states = Vec::new();
        let mut index_by_key = BTreeMap::new();
        let mut terminal_ref_cache = BTreeMap::new();
        self.collect_phase_state_from(
            0,
            &self.flop_board,
            &mut states,
            &mut index_by_key,
            &mut terminal_ref_cache,
        )?;
        reorder_phase_states_by_level_and_slot(
            &self.tree,
            &self.infosets,
            self.oop_combos.len(),
            self.ip_combos.len(),
            states,
        )
    }

    fn collect_phase_state_from(
        &self,
        node_id: usize,
        board: &Board,
        states: &mut Vec<PhaseState>,
        index_by_key: &mut BTreeMap<(usize, u64), usize>,
        terminal_ref_cache: &mut BTreeMap<u64, Vec<TerminalCacheRef>>,
    ) -> Result<usize, String> {
        let key = (node_id, ordered_board_key(board));
        if let Some(index) = index_by_key.get(&key) {
            return Ok(*index);
        }
        let state_index = states.len();
        index_by_key.insert(key, state_index);
        let board_slot = self.board_slot(board)?;
        let terminal_cache_indices = self.phase_state_terminal_cache_indices(node_id, board)?;
        let terminal_cache_refs =
            self.phase_state_terminal_cache_refs(node_id, board, terminal_ref_cache)?;
        states.push(PhaseState {
            node_id,
            board: board.clone(),
            board_slot,
            children: Vec::new(),
            chance_member_permutation_codes: Vec::new(),
            chance_concrete_events: 0,
            terminal_cache_indices,
            terminal_cache_refs,
        });

        let node = &self.tree.nodes[node_id];
        let mut children = Vec::new();
        let mut chance_member_permutation_codes = Vec::new();
        let mut chance_concrete_events = 0usize;
        match &node.kind {
            PublicNodeKind::Terminal { .. } => {}
            PublicNodeKind::Chance(_) => {
                let Some(child) = node.children.first().copied() else {
                    states[state_index].children = children;
                    return Ok(state_index);
                };
                let chance_classes = next_card_isomorphism(board, &self.oop_range, &self.ip_range);
                chance_concrete_events = chance_classes.concrete_events;
                for chance_class in chance_classes.classes {
                    let card = *chance_class
                        .representative
                        .first()
                        .ok_or_else(|| "chance class has no representative card".to_string())?;
                    children.push(self.collect_phase_state_from(
                        child,
                        &board.push(card)?,
                        states,
                        index_by_key,
                        terminal_ref_cache,
                    )?);
                    chance_member_permutation_codes.push(
                        chance_class
                            .members
                            .iter()
                            .map(|member| member.permutation_to_representative.code())
                            .collect(),
                    );
                }
            }
            PublicNodeKind::Decision { .. } => {
                for child in &node.children {
                    children.push(self.collect_phase_state_from(
                        *child,
                        board,
                        states,
                        index_by_key,
                        terminal_ref_cache,
                    )?);
                }
            }
        }
        states[state_index].children = children;
        states[state_index].chance_member_permutation_codes = chance_member_permutation_codes;
        states[state_index].chance_concrete_events = chance_concrete_events;
        Ok(state_index)
    }

    fn phase_state_terminal_cache_indices(
        &self,
        node_id: usize,
        board: &Board,
    ) -> Result<Vec<usize>, String> {
        let PublicNodeKind::Terminal { reason } = self.tree.nodes[node_id].kind else {
            return Ok(Vec::new());
        };
        if !matches!(reason, TerminalReason::Showdown | TerminalReason::AllIn) {
            return Ok(Vec::new());
        }
        terminal_boards(board)?
            .into_iter()
            .map(|terminal_board| {
                self.terminal_cache_index_by_key
                    .get(&unordered_board_key(&terminal_board))
                    .copied()
                    .ok_or_else(|| "terminal board is outside the solver board cache".to_string())
            })
            .collect()
    }

    fn phase_state_terminal_cache_refs(
        &self,
        node_id: usize,
        board: &Board,
        terminal_ref_cache: &mut BTreeMap<u64, Vec<TerminalCacheRef>>,
    ) -> Result<Vec<TerminalCacheRef>, String> {
        let PublicNodeKind::Terminal { reason } = self.tree.nodes[node_id].kind else {
            return Ok(Vec::new());
        };
        if !matches!(reason, TerminalReason::Showdown | TerminalReason::AllIn) {
            return Ok(Vec::new());
        }
        let key = ordered_board_key(board);
        if let Some(cached) = terminal_ref_cache.get(&key) {
            return Ok(cached.clone());
        }
        let refs = self.terminal_cache_refs_for_board(board)?;
        terminal_ref_cache.insert(key, refs.clone());
        Ok(refs)
    }

    fn terminal_cache_refs_for_board(
        &self,
        board: &Board,
    ) -> Result<Vec<TerminalCacheRef>, String> {
        if board.cards().len() == 5 {
            let cache_index = self
                .terminal_cache_index_by_key
                .get(&unordered_board_key(board))
                .copied()
                .ok_or_else(|| "terminal board is outside the solver board cache".to_string())?;
            return Ok(vec![TerminalCacheRef {
                cache_index,
                member_permutation_codes: vec![
                    crate::isomorphism::SuitPermutation::identity().code(),
                ],
            }]);
        }
        terminal_board_isomorphism(board, &self.oop_range, &self.ip_range)?
            .into_iter()
            .map(|class| {
                let cache_index = self
                    .terminal_cache_index_by_key
                    .get(&unordered_board_key(&class.representative_board))
                    .copied()
                    .ok_or_else(|| {
                        "terminal board is outside the solver board cache".to_string()
                    })?;
                Ok(TerminalCacheRef {
                    cache_index,
                    member_permutation_codes: class
                        .members
                        .iter()
                        .map(|member| member.permutation_to_representative.code())
                        .collect(),
                })
            })
            .collect()
    }

    fn forward_reaches_into_with_profile(
        &self,
        states: &[PhaseState],
        oop_reaches: &mut [Vec<f32>],
        ip_reaches: &mut [Vec<f32>],
        profile_reach: bool,
    ) -> Result<(), String> {
        self.forward_reaches_for_mode_into(
            states,
            EvaluationMode::Profile,
            StrategySource::Current,
            oop_reaches,
            ip_reaches,
            profile_reach,
        )
    }

    fn forward_reaches_for_mode(
        &self,
        states: &[PhaseState],
        mode: EvaluationMode,
        strategy_source: StrategySource,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        let mut oop_reaches = vec![vec![0.0f32; self.oop_combos.len()]; states.len()];
        let mut ip_reaches = vec![vec![0.0f32; self.ip_combos.len()]; states.len()];
        for (index, combo) in self.oop_combos.iter().enumerate() {
            oop_reaches[0][index] = combo.weight;
        }
        for (index, combo) in self.ip_combos.iter().enumerate() {
            ip_reaches[0][index] = combo.weight;
        }

        for (state_index, state) in states.iter().enumerate() {
            let node = &self.tree.nodes[state.node_id];
            let parent_oop_reach = oop_reaches[state_index].clone();
            let parent_ip_reach = ip_reaches[state_index].clone();
            match &node.kind {
                PublicNodeKind::Terminal { .. } => {}
                PublicNodeKind::Chance(_) => {
                    if state.children.is_empty() {
                        continue;
                    }
                    for child in &state.children {
                        add_reach(&mut oop_reaches[*child], &parent_oop_reach);
                        add_reach(&mut ip_reaches[*child], &parent_ip_reach);
                    }
                }
                PublicNodeKind::Decision { player, actions } => {
                    let actions_len = actions.len();
                    if actions_len == 1 {
                        let child = state.children[0];
                        add_reach(&mut oop_reaches[child], &parent_oop_reach);
                        add_reach(&mut ip_reaches[child], &parent_ip_reach);
                        continue;
                    }
                    let acting_combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let board_slot = state.board_slot;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let infoset = self.infosets[state.node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    let slot_start = infoset.slots_start + row_start;
                    let slot_end = infoset.slots_start + row_end;
                    let strategies = match strategy_source {
                        StrategySource::Current => current_strategies(
                            &self.regrets[slot_start..slot_end],
                            acting_combos,
                            actions_len,
                        ),
                        StrategySource::Average => average_strategies(
                            &self.strategy_sum[slot_start..slot_end],
                            acting_combos,
                            actions_len,
                        ),
                    };
                    for action_index in 0..actions_len {
                        let child = state.children[action_index];
                        match player {
                            Player::Oop => {
                                add_reach(&mut ip_reaches[child], &parent_ip_reach);
                                if mode == EvaluationMode::OopBestResponse {
                                    add_reach(&mut oop_reaches[child], &parent_oop_reach);
                                } else {
                                    add_strategy_reach(
                                        &mut oop_reaches[child],
                                        &parent_oop_reach,
                                        &strategies,
                                        actions_len,
                                        action_index,
                                    );
                                }
                            }
                            Player::Ip => {
                                add_reach(&mut oop_reaches[child], &parent_oop_reach);
                                if mode == EvaluationMode::IpBestResponse {
                                    add_reach(&mut ip_reaches[child], &parent_ip_reach);
                                } else {
                                    add_strategy_reach(
                                        &mut ip_reaches[child],
                                        &parent_ip_reach,
                                        &strategies,
                                        actions_len,
                                        action_index,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((oop_reaches, ip_reaches))
    }

    fn forward_reaches_for_mode_into(
        &self,
        states: &[PhaseState],
        mode: EvaluationMode,
        strategy_source: StrategySource,
        oop_reaches: &mut [Vec<f32>],
        ip_reaches: &mut [Vec<f32>],
        profile_reach: bool,
    ) -> Result<(), String> {
        if oop_reaches.len() != states.len() || ip_reaches.len() != states.len() {
            return Err("reach scratch length does not match state count".to_string());
        }
        let reach_started = if profile_reach {
            Some(Instant::now())
        } else {
            None
        };
        let zero_started = if profile_reach {
            Some(Instant::now())
        } else {
            None
        };
        for reach in oop_reaches.iter_mut() {
            reach.fill(0.0);
        }
        for reach in ip_reaches.iter_mut() {
            reach.fill(0.0);
        }
        let zero_ms = zero_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let root_started = if profile_reach {
            Some(Instant::now())
        } else {
            None
        };
        for (index, combo) in self.oop_combos.iter().enumerate() {
            oop_reaches[0][index] = combo.weight;
        }
        for (index, combo) in self.ip_combos.iter().enumerate() {
            ip_reaches[0][index] = combo.weight;
        }
        let root_ms = root_started
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let mut propagate_ms = 0.0;
        let mut chance_edges = 0usize;
        let mut decision_edges = 0usize;
        for (state_index, state) in states.iter().enumerate() {
            let node = &self.tree.nodes[state.node_id];
            match &node.kind {
                PublicNodeKind::Terminal { .. } => {}
                PublicNodeKind::Chance(_) => {
                    if state.children.is_empty() {
                        continue;
                    }
                    let (parent_oop, child_oop_reaches) =
                        split_reach_state_and_children(oop_reaches, state_index);
                    let (parent_ip, child_ip_reaches) =
                        split_reach_state_and_children(ip_reaches, state_index);
                    let propagate_started = if profile_reach {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    for child in &state.children {
                        add_reach(
                            child_reach_mut(child_oop_reaches, state_index, *child),
                            parent_oop,
                        );
                        add_reach(
                            child_reach_mut(child_ip_reaches, state_index, *child),
                            parent_ip,
                        );
                    }
                    if let Some(propagate_started) = propagate_started {
                        chance_edges += state.children.len();
                        propagate_ms += propagate_started.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                PublicNodeKind::Decision { player, actions } => {
                    let actions_len = actions.len();
                    if actions_len == 1 {
                        let child = state.children[0];
                        let (parent_oop, child_oop_reaches) =
                            split_reach_state_and_children(oop_reaches, state_index);
                        let (parent_ip, child_ip_reaches) =
                            split_reach_state_and_children(ip_reaches, state_index);
                        add_reach(
                            child_reach_mut(child_oop_reaches, state_index, child),
                            parent_oop,
                        );
                        add_reach(
                            child_reach_mut(child_ip_reaches, state_index, child),
                            parent_ip,
                        );
                        if profile_reach {
                            decision_edges += 1;
                        }
                        continue;
                    }
                    let acting_combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let board_slot = state.board_slot;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let infoset = self.infosets[state.node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    let slot_start = infoset.slots_start + row_start;
                    let slot_end = infoset.slots_start + row_end;
                    let (parent_oop, child_oop_reaches) =
                        split_reach_state_and_children(oop_reaches, state_index);
                    let (parent_ip, child_ip_reaches) =
                        split_reach_state_and_children(ip_reaches, state_index);
                    let propagate_started = if profile_reach {
                        Some(Instant::now())
                    } else {
                        None
                    };
                    match player {
                        Player::Oop => {
                            for action_index in 0..actions_len {
                                let child = state.children[action_index];
                                add_reach(
                                    child_reach_mut(child_ip_reaches, state_index, child),
                                    parent_ip,
                                );
                            }
                            if mode == EvaluationMode::OopBestResponse {
                                for action_index in 0..actions_len {
                                    let child = state.children[action_index];
                                    add_reach(
                                        child_reach_mut(child_oop_reaches, state_index, child),
                                        parent_oop,
                                    );
                                }
                            } else {
                                match strategy_source {
                                    StrategySource::Current => add_current_strategy_reaches(
                                        child_oop_reaches,
                                        state_index,
                                        &state.children,
                                        parent_oop,
                                        &self.regrets[slot_start..slot_end],
                                        actions_len,
                                    ),
                                    StrategySource::Average => add_average_strategy_reaches(
                                        child_oop_reaches,
                                        state_index,
                                        &state.children,
                                        parent_oop,
                                        &self.strategy_sum[slot_start..slot_end],
                                        actions_len,
                                    ),
                                }
                            }
                        }
                        Player::Ip => {
                            for action_index in 0..actions_len {
                                let child = state.children[action_index];
                                add_reach(
                                    child_reach_mut(child_oop_reaches, state_index, child),
                                    parent_oop,
                                );
                            }
                            if mode == EvaluationMode::IpBestResponse {
                                for action_index in 0..actions_len {
                                    let child = state.children[action_index];
                                    add_reach(
                                        child_reach_mut(child_ip_reaches, state_index, child),
                                        parent_ip,
                                    );
                                }
                            } else {
                                match strategy_source {
                                    StrategySource::Current => add_current_strategy_reaches(
                                        child_ip_reaches,
                                        state_index,
                                        &state.children,
                                        parent_ip,
                                        &self.regrets[slot_start..slot_end],
                                        actions_len,
                                    ),
                                    StrategySource::Average => add_average_strategy_reaches(
                                        child_ip_reaches,
                                        state_index,
                                        &state.children,
                                        parent_ip,
                                        &self.strategy_sum[slot_start..slot_end],
                                        actions_len,
                                    ),
                                }
                            }
                        }
                    }
                    if let Some(propagate_started) = propagate_started {
                        decision_edges += actions_len;
                        propagate_ms += propagate_started.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            }
        }
        if profile_reach {
            eprintln!(
                "real_cfr_reach_profile states={} chance_edges={} decision_edges={} zero_ms={:.3} root_ms={:.3} propagate_ms={:.3} total_ms={:.3}",
                states.len(),
                chance_edges,
                decision_edges,
                zero_ms,
                root_ms,
                propagate_ms,
                reach_started
                    .expect("reach profile timer must exist")
                    .elapsed()
                    .as_secs_f64()
                    * 1000.0
            );
        }
        Ok(())
    }

    fn evaluate_states(
        &self,
        states: &[PhaseState],
        mode: EvaluationMode,
        threads: usize,
    ) -> Result<Values, String> {
        let (oop_reaches, ip_reaches) =
            self.forward_reaches_for_mode(states, mode, StrategySource::Average)?;
        let mut values = self.terminal_phase(states, &oop_reaches, &ip_reaches, threads)?;
        self.evaluation_backup_phase(states, mode, &mut values)?;
        Ok(values[0].clone())
    }

    fn terminal_phase(
        &self,
        states: &[PhaseState],
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        threads: usize,
    ) -> Result<Vec<Values>, String> {
        let values_alloc_started = Instant::now();
        let mut values =
            vec![Values::zero(self.oop_combos.len(), self.ip_combos.len()); states.len()];
        let values_alloc_ms = values_alloc_started.elapsed().as_secs_f64() * 1000.0;
        self.terminal_phase_into(
            states,
            oop_reaches,
            ip_reaches,
            threads,
            &mut values,
            values_alloc_ms,
        )?;
        Ok(values)
    }

    fn terminal_phase_into(
        &self,
        states: &[PhaseState],
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        threads: usize,
        values: &mut [Values],
        values_alloc_ms: f64,
    ) -> Result<(), String> {
        self.terminal_phase_into_with_profile_options(
            states,
            oop_reaches,
            ip_reaches,
            threads,
            values,
            values_alloc_ms,
            std::env::var_os("POKEDR_REAL_CFR_TERMINAL_PROFILE").is_some(),
            std::env::var_os("POKEDR_REAL_CFR_SIDE_CACHE_KEY_PROFILE").is_some(),
            terminal_side_cache_enabled(),
        )
    }

    fn terminal_phase_into_with_profile_options(
        &self,
        states: &[PhaseState],
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        threads: usize,
        values: &mut [Values],
        values_alloc_ms: f64,
        profile_terminal: bool,
        profile_side_cache_keys: bool,
        use_side_cache: bool,
    ) -> Result<(), String> {
        if values.len() != states.len() {
            return Err("terminal phase scratch length does not match state count".to_string());
        }
        let terminal_start = first_terminal_state_index(states);
        let terminal_len = states.len() - terminal_start;
        if terminal_len == 0 {
            return Ok(());
        }
        let threads = effective_worker_count(threads).min(terminal_len);
        let partitions = terminal_phase_partitions(states, terminal_start, threads);
        let worker_scope_started = Instant::now();
        let mut partition_chunks = Vec::with_capacity(partitions.len());
        let mut remaining_values = &mut values[terminal_start..];
        for (start_index, end_index) in partitions {
            let chunk_len = end_index - start_index;
            let (values_chunk, remaining) = remaining_values.split_at_mut(chunk_len);
            let worker_index = partition_chunks.len();
            partition_chunks.push((worker_index, start_index, values_chunk));
            remaining_values = remaining;
        }
        let profiles = partition_chunks
            .into_par_iter()
            .map(|(worker_index, start_index, values_chunk)| {
                self.terminal_phase_worker(
                    states,
                    oop_reaches,
                    ip_reaches,
                    start_index,
                    worker_index,
                    values_chunk,
                    profile_terminal,
                    profile_side_cache_keys,
                    use_side_cache,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let worker_scope_ms = worker_scope_started.elapsed().as_secs_f64() * 1000.0;
        if profile_terminal {
            let total_tasks = profiles.iter().map(|profile| profile.tasks).sum();
            eprintln!(
                "real_cfr_terminal_stage_profile values_alloc_ms={:.3} worker_scope_ms={:.3}",
                values_alloc_ms, worker_scope_ms,
            );
            print_terminal_worker_profiles(total_tasks, profiles.len(), &profiles);
        }
        if profile_side_cache_keys {
            print_terminal_side_cache_key_profile(&profiles);
        }
        Ok(())
    }

    fn terminal_phase_worker(
        &self,
        states: &[PhaseState],
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        start_index: usize,
        worker_index: usize,
        values_chunk: &mut [Values],
        profile_terminal: bool,
        profile_side_cache_keys: bool,
        use_side_cache: bool,
    ) -> Result<TerminalWorkerProfile, String> {
        let started = Instant::now();
        let scratch_source = self
            .terminal_cache
            .first()
            .ok_or_else(|| "terminal board cache is empty".to_string())?;
        let combos = scratch_source.prepared.combos().len();
        let mut scratch = TerminalCfvScratch::new(&scratch_source.prepared);
        let mut oop_live = vec![0.0f32; combos];
        let mut ip_live = vec![0.0f32; combos];
        let mut oop_nonzero = Vec::new();
        let mut ip_nonzero = Vec::new();
        let zero_oop_values = vec![0.0f32; self.oop_combos.len()];
        let zero_ip_values = vec![0.0f32; self.ip_combos.len()];
        let sparse_nonzero_limit = terminal_sparse_nonzero_limit();
        let mut side_cache = TerminalSideValueCache::default();
        let mut accumulator =
            TerminalAccumulator::zero(self.oop_combos.len(), self.ip_combos.len());
        let mut profile = TerminalWorkerProfile {
            worker_index,
            ..TerminalWorkerProfile::default()
        };
        for (local_index, values_slot) in values_chunk.iter_mut().enumerate() {
            let state_index = start_index + local_index;
            let state = &states[state_index];
            let node = &self.tree.nodes[state.node_id];
            let PublicNodeKind::Terminal { reason } = node.kind else {
                continue;
            };
            match reason {
                TerminalReason::Fold => {
                    profile.terminal_states += 1;
                    profile.fold_states += 1;
                    let fold_started = profile_terminal.then(Instant::now);
                    self.fold_values_into(
                        values_slot,
                        &state.board,
                        node.state.pot,
                        node.state.player,
                        &oop_reaches[state_index],
                        &ip_reaches[state_index],
                    );
                    if let Some(started) = fold_started {
                        profile.fold_ms += started.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                TerminalReason::Showdown | TerminalReason::AllIn => {
                    profile.terminal_states += 1;
                    accumulator.reset();
                    for terminal_ref in &state.terminal_cache_refs {
                        let cache = &self.terminal_cache[terminal_ref.cache_index];
                        let reach_started = profile_terminal.then(Instant::now);
                        reach_on_prepared_board_targets_sparse_into(
                            &cache.oop_targets,
                            &oop_reaches[state_index],
                            &mut oop_live,
                            &mut oop_nonzero,
                        );
                        reach_on_prepared_board_targets_sparse_into(
                            &cache.ip_targets,
                            &ip_reaches[state_index],
                            &mut ip_live,
                            &mut ip_nonzero,
                        );
                        if let Some(started) = reach_started {
                            profile.reach_map_ms += started.elapsed().as_secs_f64() * 1000.0;
                        }
                        profile.tasks += 1;
                        profile.oop_nonzero_sum += oop_nonzero.len();
                        profile.ip_nonzero_sum += ip_nonzero.len();
                        profile.oop_nonzero_max = profile.oop_nonzero_max.max(oop_nonzero.len());
                        profile.ip_nonzero_max = profile.ip_nonzero_max.max(ip_nonzero.len());
                        if oop_nonzero.is_empty() && ip_nonzero.is_empty() {
                            profile.zero_reach_tasks += 1;
                            let accumulator_started = profile_terminal.then(Instant::now);
                            self.add_terminal_ref_zero_values(
                                &mut accumulator,
                                terminal_ref,
                                cache,
                            )?;
                            if let Some(started) = accumulator_started {
                                profile.accumulator_ms += started.elapsed().as_secs_f64() * 1000.0;
                            }
                            continue;
                        }
                        let cfv_started = profile_terminal.then(Instant::now);
                        let use_sparse = oop_nonzero.len() <= sparse_nonzero_limit
                            && ip_nonzero.len() <= sparse_nonzero_limit;
                        if use_side_cache && oop_nonzero.is_empty() {
                            profile.zero_reach_tasks += 1;
                            let hero_values = terminal_side_cached_values(
                                &mut side_cache,
                                cache,
                                terminal_ref.cache_index,
                                TerminalSideValue::OopValue,
                                &ip_live,
                                &ip_nonzero,
                                &cache.oop_targets,
                                &cache.oop_board_targets,
                                self.oop_combos.len(),
                                use_sparse,
                                &mut scratch,
                            )?;
                            self.add_terminal_ref_compact_values(
                                &mut accumulator,
                                terminal_ref,
                                cache,
                                node.state.pot,
                                &hero_values,
                                &zero_ip_values,
                            )?;
                        } else if use_side_cache && ip_nonzero.is_empty() {
                            profile.zero_reach_tasks += 1;
                            let villain_values = terminal_side_cached_values(
                                &mut side_cache,
                                cache,
                                terminal_ref.cache_index,
                                TerminalSideValue::IpValue,
                                &oop_live,
                                &oop_nonzero,
                                &cache.ip_targets,
                                &cache.ip_board_targets,
                                self.ip_combos.len(),
                                use_sparse,
                                &mut scratch,
                            )?;
                            self.add_terminal_ref_compact_values(
                                &mut accumulator,
                                terminal_ref,
                                cache,
                                node.state.pot,
                                &zero_oop_values,
                                &villain_values,
                            )?;
                        } else if use_side_cache {
                            let hero_values = terminal_side_cached_values(
                                &mut side_cache,
                                cache,
                                terminal_ref.cache_index,
                                TerminalSideValue::OopValue,
                                &ip_live,
                                &ip_nonzero,
                                &cache.oop_targets,
                                &cache.oop_board_targets,
                                self.oop_combos.len(),
                                use_sparse,
                                &mut scratch,
                            )?;
                            let villain_values = terminal_side_cached_values(
                                &mut side_cache,
                                cache,
                                terminal_ref.cache_index,
                                TerminalSideValue::IpValue,
                                &oop_live,
                                &oop_nonzero,
                                &cache.ip_targets,
                                &cache.ip_board_targets,
                                self.ip_combos.len(),
                                use_sparse,
                                &mut scratch,
                            )?;
                            self.add_terminal_ref_compact_values(
                                &mut accumulator,
                                terminal_ref,
                                cache,
                                node.state.pot,
                                &hero_values,
                                &villain_values,
                            )?;
                        } else if use_sparse {
                            profile.sparse_tasks += 1;
                            terminal_cfv_sparse_board_targets_into(
                                &cache.prepared,
                                &oop_live,
                                &ip_live,
                                &oop_nonzero,
                                &ip_nonzero,
                                &cache.oop_board_targets,
                                &cache.ip_board_targets,
                                &mut scratch,
                            )?;
                        } else {
                            profile.prefix_tasks += 1;
                            terminal_cfv_prefix_blocker_sorted_board_targets_into(
                                &cache.prepared,
                                &oop_live,
                                &ip_live,
                                &cache.oop_board_targets,
                                &cache.ip_board_targets,
                                &mut scratch,
                            )?;
                        }
                        if let Some(started) = cfv_started {
                            profile.cfv_ms += started.elapsed().as_secs_f64() * 1000.0;
                        }
                        if !use_side_cache {
                            let accumulator_started = profile_terminal.then(Instant::now);
                            self.add_terminal_ref_values(
                                &mut accumulator,
                                terminal_ref,
                                cache,
                                node.state.pot,
                                scratch.hero_values(),
                                scratch.villain_values(),
                            )?;
                            if let Some(started) = accumulator_started {
                                profile.accumulator_ms += started.elapsed().as_secs_f64() * 1000.0;
                            }
                        }
                    }
                    accumulator.finish();
                    values_slot.copy_from(&accumulator.values);
                }
            }
        }
        profile.side_cache_hits = side_cache.hits;
        profile.side_cache_misses = side_cache.misses;
        if profile_side_cache_keys {
            profile.side_cache_keys = side_cache.entries.keys().cloned().collect();
        }
        profile.output_states = values_chunk.len();
        profile.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(profile)
    }

    fn add_terminal_ref_values(
        &self,
        accumulator: &mut TerminalAccumulator,
        terminal_ref: &TerminalCacheRef,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
    ) -> Result<(), String> {
        let identity_code = crate::isomorphism::SuitPermutation::identity().code();
        for permutation_code in &terminal_ref.member_permutation_codes {
            if *permutation_code == identity_code {
                accumulator.add_board_values(cache, pot, hero_values, villain_values);
                continue;
            }
            let maps = self
                .combo_permutations
                .get(permutation_code)
                .ok_or_else(|| {
                    "terminal isomorphism permutation is missing combo maps".to_string()
                })?;
            accumulator.add_board_values_permuted(
                cache,
                pot,
                hero_values,
                villain_values,
                &maps.oop_source_to_target,
                &maps.ip_source_to_target,
            );
        }
        Ok(())
    }

    fn add_terminal_ref_zero_values(
        &self,
        accumulator: &mut TerminalAccumulator,
        terminal_ref: &TerminalCacheRef,
        cache: &TerminalEvalCache,
    ) -> Result<(), String> {
        let identity_code = crate::isomorphism::SuitPermutation::identity().code();
        for permutation_code in &terminal_ref.member_permutation_codes {
            if *permutation_code == identity_code {
                accumulator.add_zero_board_values(cache);
                continue;
            }
            let maps = self
                .combo_permutations
                .get(permutation_code)
                .ok_or_else(|| {
                    "terminal isomorphism permutation is missing combo maps".to_string()
                })?;
            accumulator.add_zero_board_values_permuted(
                cache,
                &maps.oop_source_to_target,
                &maps.ip_source_to_target,
            );
        }
        Ok(())
    }

    fn add_terminal_ref_compact_values(
        &self,
        accumulator: &mut TerminalAccumulator,
        terminal_ref: &TerminalCacheRef,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
    ) -> Result<(), String> {
        let identity_code = crate::isomorphism::SuitPermutation::identity().code();
        for permutation_code in &terminal_ref.member_permutation_codes {
            if *permutation_code == identity_code {
                accumulator.add_compact_board_values(cache, pot, hero_values, villain_values);
                continue;
            }
            let maps = self
                .combo_permutations
                .get(permutation_code)
                .ok_or_else(|| {
                    "terminal isomorphism permutation is missing combo maps".to_string()
                })?;
            accumulator.add_compact_board_values_permuted(
                cache,
                pot,
                hero_values,
                villain_values,
                &maps.oop_source_to_target,
                &maps.ip_source_to_target,
            );
        }
        Ok(())
    }

    fn backup_phase(
        &mut self,
        states: &[PhaseState],
        backup_plan: &BackupLevelPlan,
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        values: &mut [Values],
        threads: usize,
        average_weight: f32,
        variant: RealCfrVariant,
    ) -> Result<usize, String> {
        let update_factors = RealCfrUpdateFactors::new(variant, average_weight);
        let worker_count = effective_worker_count(threads);
        for level in backup_plan.levels.iter().skip(1) {
            for run in level {
                let decision_end =
                    backup_run_decision_prefix_end(&self.tree, &self.infosets, states, run);
                if decision_end > run.start {
                    self.backup_decision_prefix(
                        states,
                        run.start,
                        decision_end,
                        oop_reaches,
                        ip_reaches,
                        values,
                        worker_count,
                        &update_factors,
                    )?;
                }
                for state_index in decision_end..run.end {
                    backup_chance_state(
                        &self.tree,
                        states,
                        values,
                        state_index,
                        &self.combo_permutations,
                    )?;
                }
            }
        }
        Ok(values[0].terminal_evals)
    }

    fn backup_decision_prefix(
        &mut self,
        states: &[PhaseState],
        state_start: usize,
        state_end: usize,
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        values: &mut [Values],
        threads: usize,
        update_factors: &RealCfrUpdateFactors,
    ) -> Result<(), String> {
        let chunks = backup_decision_chunks(
            &self.tree,
            &self.infosets,
            self.oop_combos.len(),
            self.ip_combos.len(),
            states,
            state_start,
            state_end,
            threads,
        );
        if chunks.is_empty() {
            return Ok(());
        }
        let (value_prefix, child_values) = values.split_at_mut(state_end);
        let child_values: &[Values] = child_values;
        let current_values = &mut value_prefix[state_start..state_end];
        let tree = &self.tree;
        let infosets = &self.infosets;
        let oop_combos = self.oop_combos.len();
        let ip_combos = self.ip_combos.len();
        let regrets = self.regrets.as_mut_slice();
        let strategy_sum = self.strategy_sum.as_mut_slice();
        let mut jobs = Vec::with_capacity(chunks.len());
        let mut value_cursor = state_start;
        let mut value_tail = current_values;
        let mut slot_cursor = 0usize;
        let mut regret_tail = regrets;
        let mut strategy_tail = strategy_sum;
        for chunk in chunks {
            let value_skip = chunk.state_start - value_cursor;
            let (_, rest) = value_tail.split_at_mut(value_skip);
            value_tail = rest;
            let value_len = chunk.state_end - chunk.state_start;
            let (value_chunk, rest) = value_tail.split_at_mut(value_len);
            value_tail = rest;
            value_cursor = chunk.state_end;

            let slot_skip = chunk.slot_start - slot_cursor;
            let (_, rest) = regret_tail.split_at_mut(slot_skip);
            regret_tail = rest;
            let (_, rest) = strategy_tail.split_at_mut(slot_skip);
            strategy_tail = rest;
            let slot_len = chunk.slot_end - chunk.slot_start;
            let (regret_chunk, rest) = regret_tail.split_at_mut(slot_len);
            regret_tail = rest;
            let (strategy_chunk, rest) = strategy_tail.split_at_mut(slot_len);
            strategy_tail = rest;
            slot_cursor = chunk.slot_end;

            jobs.push(BackupDecisionJob {
                state_start: chunk.state_start,
                slot_start: chunk.slot_start,
                values: value_chunk,
                regrets: regret_chunk,
                strategy_sum: strategy_chunk,
            });
        }

        jobs.into_par_iter()
            .map(|job| {
                backup_decision_chunk(
                    tree,
                    infosets,
                    oop_combos,
                    ip_combos,
                    states,
                    oop_reaches,
                    ip_reaches,
                    child_values,
                    state_end,
                    job.values,
                    job.state_start,
                    job.regrets,
                    job.strategy_sum,
                    job.slot_start,
                    update_factors,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    fn evaluation_backup_phase(
        &self,
        states: &[PhaseState],
        mode: EvaluationMode,
        values: &mut [Values],
    ) -> Result<(), String> {
        for state_index in (0..states.len()).rev() {
            let state = &states[state_index];
            let node = &self.tree.nodes[state.node_id];
            match &node.kind {
                PublicNodeKind::Terminal { .. } => {}
                PublicNodeKind::Chance(_) => {
                    let mut state_values =
                        Values::zero(self.oop_combos.len(), self.ip_combos.len());
                    for child in &state.children {
                        state_values.add_scaled(&values[*child], 1.0);
                    }
                    values[state_index] = state_values;
                }
                PublicNodeKind::Decision { player, actions } => {
                    let actions_len = actions.len();
                    if actions_len == 1 {
                        values[state_index] = values[state.children[0]].clone();
                        continue;
                    }
                    let acting_combos = match player {
                        Player::Oop => self.oop_combos.len(),
                        Player::Ip => self.ip_combos.len(),
                    };
                    let board_slot = state.board_slot;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let infoset = self.infosets[state.node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    let slot_start = infoset.slots_start + row_start;
                    let slot_end = infoset.slots_start + row_end;
                    let strategies = average_strategies(
                        &self.strategy_sum[slot_start..slot_end],
                        acting_combos,
                        actions_len,
                    );
                    let action_values = state
                        .children
                        .iter()
                        .map(|child| values[*child].clone())
                        .collect::<Vec<_>>();
                    let mut state_values =
                        Values::zero(self.oop_combos.len(), self.ip_combos.len());
                    match (mode, player) {
                        (EvaluationMode::OopBestResponse, Player::Oop) => {
                            combine_best_response_values(
                                &mut state_values.oop,
                                &action_values,
                                actions_len,
                                Player::Oop,
                            );
                            combine_nonacting_values(
                                &mut state_values.ip,
                                &action_values,
                                Player::Ip,
                            );
                        }
                        (EvaluationMode::IpBestResponse, Player::Ip) => {
                            combine_best_response_values(
                                &mut state_values.ip,
                                &action_values,
                                actions_len,
                                Player::Ip,
                            );
                            combine_nonacting_values(
                                &mut state_values.oop,
                                &action_values,
                                Player::Oop,
                            );
                        }
                        (_, Player::Oop) => {
                            combine_acting_values(
                                &mut state_values.oop,
                                &action_values,
                                &strategies,
                                actions_len,
                                Player::Oop,
                            );
                            combine_nonacting_values(
                                &mut state_values.ip,
                                &action_values,
                                Player::Ip,
                            );
                        }
                        (_, Player::Ip) => {
                            combine_acting_values(
                                &mut state_values.ip,
                                &action_values,
                                &strategies,
                                actions_len,
                                Player::Ip,
                            );
                            combine_nonacting_values(
                                &mut state_values.oop,
                                &action_values,
                                Player::Oop,
                            );
                        }
                    }
                    state_values.terminal_evals =
                        action_values.iter().map(|value| value.terminal_evals).sum();
                    values[state_index] = state_values;
                }
            }
        }
        Ok(())
    }

    fn terminal_eval_breakdown_from(
        &self,
        node_id: usize,
        board: &Board,
        breakdown: &mut TerminalEvalBreakdown,
    ) -> Result<(), String> {
        let node = &self.tree.nodes[node_id];
        match &node.kind {
            PublicNodeKind::Terminal { reason } => match reason {
                TerminalReason::Fold => breakdown.fold_terminals += 1,
                TerminalReason::Showdown => {
                    breakdown.showdown_terminals += 1;
                    if board.cards().len() == 5 {
                        breakdown.river_showdown_evals += 1;
                    } else {
                        return Err("showdown terminal before river is invalid".to_string());
                    }
                }
                TerminalReason::AllIn => {
                    breakdown.all_in_terminals += 1;
                    match board.cards().len() {
                        3 => breakdown.flop_all_in_runout_evals += terminal_boards(board)?.len(),
                        4 => breakdown.turn_all_in_runout_evals += terminal_boards(board)?.len(),
                        5 => breakdown.river_all_in_evals += 1,
                        other => {
                            return Err(format!(
                                "all-in terminal has invalid board length {other}"
                            ));
                        }
                    }
                }
            },
            PublicNodeKind::Chance(_) => {
                let Some(child) = node.children.first().copied() else {
                    return Ok(());
                };
                for card in board.remaining_deck() {
                    self.terminal_eval_breakdown_from(child, &board.push(card)?, breakdown)?;
                }
            }
            PublicNodeKind::Decision { .. } => {
                for child in &node.children {
                    self.terminal_eval_breakdown_from(*child, board, breakdown)?;
                }
            }
        }
        Ok(())
    }

    fn collect_terminal_board_tasks(&self) -> Result<Vec<TerminalBoardTask>, String> {
        let mut tasks = Vec::new();
        self.collect_terminal_board_tasks_from(0, &self.flop_board, &mut tasks)?;
        Ok(tasks)
    }

    fn collect_terminal_board_tasks_from(
        &self,
        node_id: usize,
        board: &Board,
        tasks: &mut Vec<TerminalBoardTask>,
    ) -> Result<(), String> {
        let node = &self.tree.nodes[node_id];
        match &node.kind {
            PublicNodeKind::Terminal { reason } => {
                if matches!(reason, TerminalReason::Showdown | TerminalReason::AllIn) {
                    for terminal_board in terminal_boards(board)? {
                        let cache_index = self
                            .terminal_cache_index_by_key
                            .get(&unordered_board_key(&terminal_board))
                            .copied()
                            .ok_or_else(|| {
                                "terminal board is outside the solver board cache".to_string()
                            })?;
                        tasks.push(TerminalBoardTask {
                            terminal_node: node_id,
                            board: terminal_board,
                            cache_index,
                        });
                    }
                }
            }
            PublicNodeKind::Chance(_) => {
                let Some(child) = node.children.first().copied() else {
                    return Ok(());
                };
                for card in board.remaining_deck() {
                    self.collect_terminal_board_tasks_from(child, &board.push(card)?, tasks)?;
                }
            }
            PublicNodeKind::Decision { .. } => {
                for child in &node.children {
                    self.collect_terminal_board_tasks_from(*child, board, tasks)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn traverse_slices_into(
        &mut self,
        out_oop: &mut [f32],
        out_ip: &mut [f32],
        node_id: usize,
        board: &Board,
        oop_reach: &[f32],
        ip_reach: &[f32],
        average_weight: f32,
        variant: RealCfrVariant,
        terminal_scratch: &mut RecursiveTerminalScratch,
        terminal_ref_cache: &mut BTreeMap<u64, Vec<TerminalCacheRef>>,
        side_cache: &mut TerminalSideValueCache,
        profile: &mut RecursiveCfrProfile,
    ) -> Result<usize, String> {
        let node = self.tree.nodes[node_id].clone();
        match node.kind {
            PublicNodeKind::Terminal { reason } => self.terminal_slices_into(
                out_oop,
                out_ip,
                board,
                node.state.pot,
                node.state.player,
                reason,
                oop_reach,
                ip_reach,
                terminal_scratch,
                terminal_ref_cache,
                side_cache,
                profile,
            ),
            PublicNodeKind::Chance(_) => {
                if profile.enabled {
                    profile.chance_calls += 1;
                    profile.values_zero += 1;
                }
                out_oop.fill(0.0);
                out_ip.fill(0.0);
                let Some(child) = node.children.first().copied() else {
                    return Ok(0);
                };
                let chance_classes = next_card_isomorphism(board, &self.oop_range, &self.ip_range);
                let chance_weight = 1.0f32 / chance_classes.concrete_events as f32;
                if profile.enabled {
                    profile.chance_cards += chance_classes.concrete_events as u64;
                }
                let mut child_oop = terminal_scratch.take_vec(out_oop.len());
                let mut child_ip = terminal_scratch.take_vec(out_ip.len());
                let mut terminal_evals = 0usize;
                for chance_class in chance_classes.classes {
                    let card = *chance_class
                        .representative
                        .first()
                        .ok_or_else(|| "chance class has no representative card".to_string())?;
                    let next_board = board.push(card)?;
                    let child_terminal_evals = self.traverse_slices_into(
                        &mut child_oop,
                        &mut child_ip,
                        child,
                        &next_board,
                        oop_reach,
                        ip_reach,
                        average_weight,
                        variant,
                        terminal_scratch,
                        terminal_ref_cache,
                        side_cache,
                        profile,
                    )?;
                    terminal_evals += child_terminal_evals * chance_class.multiplicity;
                    for member in chance_class.members {
                        let permutation_code = member.permutation_to_representative.code();
                        let maps =
                            self.combo_permutations
                                .get(&permutation_code)
                                .ok_or_else(|| {
                                    "chance isomorphism permutation is missing combo maps"
                                        .to_string()
                                })?;
                        add_permuted_scaled_slice(
                            out_oop,
                            &child_oop,
                            &maps.oop_source_to_target,
                            chance_weight,
                        );
                        add_permuted_scaled_slice(
                            out_ip,
                            &child_ip,
                            &maps.ip_source_to_target,
                            chance_weight,
                        );
                    }
                }
                terminal_scratch.release_vec(child_oop);
                terminal_scratch.release_vec(child_ip);
                Ok(terminal_evals)
            }
            PublicNodeKind::Decision { player, actions } => {
                if profile.enabled {
                    profile.decision_calls += 1;
                }
                let actions_len = actions.len();
                if actions_len == 1 {
                    return self.traverse_slices_into(
                        out_oop,
                        out_ip,
                        node.children[0],
                        board,
                        oop_reach,
                        ip_reach,
                        average_weight,
                        variant,
                        terminal_scratch,
                        terminal_ref_cache,
                        side_cache,
                        profile,
                    );
                }
                let acting_combos = match player {
                    Player::Oop => self.oop_combos.len(),
                    Player::Ip => self.ip_combos.len(),
                };
                let board_slot = self.board_slot(board)?;
                let row_len = acting_combos * actions_len;
                let row_start = board_slot * row_len;
                let row_end = row_start + row_len;
                let infoset = self.infosets[node_id]
                    .as_ref()
                    .expect("decision node must have infoset");
                debug_assert_eq!(infoset.player, player);
                debug_assert_eq!(infoset.actions, actions_len);
                debug_assert!(board_slot < infoset.board_count);
                let slot_start = infoset.slots_start + row_start;
                let slot_end = infoset.slots_start + row_end;
                let strategies = current_strategies(
                    &self.regrets[slot_start..slot_end],
                    acting_combos,
                    actions_len,
                );
                if profile.enabled {
                    profile.strategy_builds += 1;
                    profile.values_zero += 1;
                }

                let oop_len = out_oop.len();
                let ip_len = out_ip.len();
                let mut action_oop = terminal_scratch.take_vec(actions_len * oop_len);
                let mut action_ip = terminal_scratch.take_vec(actions_len * ip_len);
                let mut terminal_evals = 0usize;
                match player {
                    Player::Oop => {
                        let mut next_oop = terminal_scratch.take_vec(oop_reach.len());
                        for action_index in 0..actions_len {
                            strategy_reach_into(
                                &mut next_oop,
                                oop_reach,
                                &strategies,
                                actions_len,
                                action_index,
                            );
                            if profile.enabled {
                                profile.reach_scratch_writes += next_oop.len() as u64;
                            }
                            let oop_row = &mut action_oop
                                [action_index * oop_len..(action_index + 1) * oop_len];
                            let ip_row =
                                &mut action_ip[action_index * ip_len..(action_index + 1) * ip_len];
                            terminal_evals += self.traverse_slices_into(
                                oop_row,
                                ip_row,
                                node.children[action_index],
                                board,
                                &next_oop,
                                ip_reach,
                                average_weight,
                                variant,
                                terminal_scratch,
                                terminal_ref_cache,
                                side_cache,
                                profile,
                            )?;
                        }
                        terminal_scratch.release_vec(next_oop);
                    }
                    Player::Ip => {
                        let mut next_ip = terminal_scratch.take_vec(ip_reach.len());
                        for action_index in 0..actions_len {
                            strategy_reach_into(
                                &mut next_ip,
                                ip_reach,
                                &strategies,
                                actions_len,
                                action_index,
                            );
                            if profile.enabled {
                                profile.reach_scratch_writes += next_ip.len() as u64;
                            }
                            let oop_row = &mut action_oop
                                [action_index * oop_len..(action_index + 1) * oop_len];
                            let ip_row =
                                &mut action_ip[action_index * ip_len..(action_index + 1) * ip_len];
                            terminal_evals += self.traverse_slices_into(
                                oop_row,
                                ip_row,
                                node.children[action_index],
                                board,
                                oop_reach,
                                &next_ip,
                                average_weight,
                                variant,
                                terminal_scratch,
                                terminal_ref_cache,
                                side_cache,
                                profile,
                            )?;
                        }
                        terminal_scratch.release_vec(next_ip);
                    }
                }

                match player {
                    Player::Oop => {
                        combine_acting_flat_values(out_oop, &action_oop, &strategies, actions_len);
                        combine_nonacting_flat_values(out_ip, &action_ip, actions_len);
                    }
                    Player::Ip => {
                        combine_acting_flat_values(out_ip, &action_ip, &strategies, actions_len);
                        combine_nonacting_flat_values(out_oop, &action_oop, actions_len);
                    }
                }

                let own_reach = match player {
                    Player::Oop => oop_reach,
                    Player::Ip => ip_reach,
                };
                for combo in 0..acting_combos {
                    let node_value = match player {
                        Player::Oop => out_oop[combo],
                        Player::Ip => out_ip[combo],
                    };
                    for action_index in 0..actions_len {
                        let action_value = match player {
                            Player::Oop => action_oop[action_index * oop_len + combo],
                            Player::Ip => action_ip[action_index * ip_len + combo],
                        };
                        let local_slot = combo * actions_len + action_index;
                        let slot = slot_start + local_slot;
                        apply_real_cfr_update(
                            &mut self.regrets[slot],
                            &mut self.strategy_sum[slot],
                            action_value - node_value,
                            own_reach[combo] * strategies[local_slot],
                            average_weight,
                            variant,
                        );
                    }
                }
                terminal_scratch.release_vec(action_oop);
                terminal_scratch.release_vec(action_ip);
                Ok(terminal_evals)
            }
        }
    }

    fn board_slot(&self, board: &Board) -> Result<usize, String> {
        match board.cards().len() {
            3 => Ok(0),
            4 => self
                .turn_index_by_key
                .get(&ordered_board_key(board))
                .copied()
                .ok_or_else(|| "turn board is outside the solver board index".to_string()),
            5 => self
                .river_index_by_key
                .get(&ordered_board_key(board))
                .copied()
                .ok_or_else(|| "river board is outside the solver board index".to_string()),
            other => Err(format!("invalid public board length {other}")),
        }
    }

    fn fold_values_into(
        &self,
        values: &mut Values,
        board: &Board,
        pot: u32,
        folding_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) {
        let pot = pot as f32;
        opponent_weights_for_fast_into(
            &self.oop_combos,
            &self.ip_combos,
            ip_reach,
            &self.oop_same_ip_combo_indices,
            board,
            &mut values.oop,
        );
        opponent_weights_for_fast_into(
            &self.ip_combos,
            &self.oop_combos,
            oop_reach,
            &self.ip_same_oop_combo_indices,
            board,
            &mut values.ip,
        );
        for value in values.oop.iter_mut() {
            *value = if folding_player == Player::Oop {
                -pot
            } else {
                pot
            } * *value;
        }
        for value in values.ip.iter_mut() {
            *value = if folding_player == Player::Ip {
                -pot
            } else {
                pot
            } * *value;
        }
        values.terminal_evals = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn terminal_slices_into(
        &self,
        out_oop: &mut [f32],
        out_ip: &mut [f32],
        board: &Board,
        pot: u32,
        folding_player: Player,
        reason: TerminalReason,
        oop_reach: &[f32],
        ip_reach: &[f32],
        terminal_scratch: &mut RecursiveTerminalScratch,
        terminal_ref_cache: &mut BTreeMap<u64, Vec<TerminalCacheRef>>,
        side_cache: &mut TerminalSideValueCache,
        profile: &mut RecursiveCfrProfile,
    ) -> Result<usize, String> {
        if profile.enabled {
            profile.terminal_calls += 1;
        }
        match reason {
            TerminalReason::Fold => {
                if profile.enabled {
                    profile.fold_calls += 1;
                }
                self.fold_slices_into(
                    out_oop,
                    out_ip,
                    board,
                    pot,
                    folding_player,
                    oop_reach,
                    ip_reach,
                );
                Ok(0)
            }
            TerminalReason::Showdown | TerminalReason::AllIn => {
                if profile.enabled {
                    profile.showdown_calls += 1;
                }
                let key = ordered_board_key(board);
                let terminal_refs = if let Some(cached) = terminal_ref_cache.get(&key) {
                    cached.clone()
                } else {
                    let refs = self.terminal_cache_refs_for_board(board)?;
                    terminal_ref_cache.insert(key, refs.clone());
                    refs
                };
                let sparse_nonzero_limit = terminal_sparse_nonzero_limit();
                terminal_scratch.accumulator.reset();
                for terminal_ref in &terminal_refs {
                    let cache = &self.terminal_cache[terminal_ref.cache_index];
                    reach_on_prepared_board_targets_sparse_into(
                        &cache.oop_targets,
                        oop_reach,
                        &mut terminal_scratch.oop_live,
                        &mut terminal_scratch.oop_nonzero,
                    );
                    reach_on_prepared_board_targets_sparse_into(
                        &cache.ip_targets,
                        ip_reach,
                        &mut terminal_scratch.ip_live,
                        &mut terminal_scratch.ip_nonzero,
                    );
                    let use_sparse = terminal_scratch.oop_nonzero.len() <= sparse_nonzero_limit
                        && terminal_scratch.ip_nonzero.len() <= sparse_nonzero_limit;
                    let hero_values = terminal_side_cached_values(
                        side_cache,
                        cache,
                        terminal_ref.cache_index,
                        TerminalSideValue::OopValue,
                        &terminal_scratch.ip_live,
                        &terminal_scratch.ip_nonzero,
                        &cache.oop_targets,
                        &cache.oop_board_targets,
                        self.oop_combos.len(),
                        use_sparse,
                        &mut terminal_scratch.cfv,
                    )?;
                    let villain_values = terminal_side_cached_values(
                        side_cache,
                        cache,
                        terminal_ref.cache_index,
                        TerminalSideValue::IpValue,
                        &terminal_scratch.oop_live,
                        &terminal_scratch.oop_nonzero,
                        &cache.ip_targets,
                        &cache.ip_board_targets,
                        self.ip_combos.len(),
                        use_sparse,
                        &mut terminal_scratch.cfv,
                    )?;
                    self.add_terminal_ref_compact_values(
                        &mut terminal_scratch.accumulator,
                        terminal_ref,
                        cache,
                        pot,
                        &hero_values,
                        &villain_values,
                    )?;
                }
                terminal_scratch.accumulator.finish();
                out_oop.copy_from_slice(&terminal_scratch.accumulator.values.oop);
                out_ip.copy_from_slice(&terminal_scratch.accumulator.values.ip);
                Ok(terminal_scratch.accumulator.values.terminal_evals)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_slices_into(
        &self,
        out_oop: &mut [f32],
        out_ip: &mut [f32],
        board: &Board,
        pot: u32,
        folding_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) {
        let pot = pot as f32;
        opponent_weights_for_fast_into(
            &self.oop_combos,
            &self.ip_combos,
            ip_reach,
            &self.oop_same_ip_combo_indices,
            board,
            out_oop,
        );
        opponent_weights_for_fast_into(
            &self.ip_combos,
            &self.oop_combos,
            oop_reach,
            &self.ip_same_oop_combo_indices,
            board,
            out_ip,
        );
        for value in out_oop.iter_mut() {
            *value = if folding_player == Player::Oop {
                -pot
            } else {
                pot
            } * *value;
        }
        for value in out_ip.iter_mut() {
            *value = if folding_player == Player::Ip {
                -pot
            } else {
                pot
            } * *value;
        }
    }
}

impl Values {
    fn zero(oop: usize, ip: usize) -> Self {
        Self {
            oop: vec![0.0; oop],
            ip: vec![0.0; ip],
            terminal_evals: 0,
        }
    }

    fn reset(&mut self) {
        self.oop.fill(0.0);
        self.ip.fill(0.0);
        self.terminal_evals = 0;
    }

    fn copy_from(&mut self, other: &Self) {
        self.oop.copy_from_slice(&other.oop);
        self.ip.copy_from_slice(&other.ip);
        self.terminal_evals = other.terminal_evals;
    }

    fn add_scaled(&mut self, other: &Self, scale: f32) {
        for (left, right) in self.oop.iter_mut().zip(&other.oop) {
            *left += *right * scale;
        }
        for (left, right) in self.ip.iter_mut().zip(&other.ip) {
            *left += *right * scale;
        }
        self.terminal_evals += other.terminal_evals;
    }
}

impl TerminalAccumulator {
    fn zero(oop: usize, ip: usize) -> Self {
        Self {
            values: Values::zero(oop, ip),
            oop_counts: vec![0.0; oop],
            ip_counts: vec![0.0; ip],
        }
    }

    fn reset(&mut self) {
        self.values.oop.fill(0.0);
        self.values.ip.fill(0.0);
        self.values.terminal_evals = 0;
        self.oop_counts.fill(0.0);
        self.ip_counts.fill(0.0);
    }

    fn add_board_values(
        &mut self,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
    ) {
        let pot = pot as f32;
        for target in &cache.oop_targets {
            self.values.oop[target.range_index] += hero_values[target.board_index as usize] * pot;
            self.oop_counts[target.range_index] += 1.0;
        }
        for target in &cache.ip_targets {
            self.values.ip[target.range_index] += villain_values[target.board_index as usize] * pot;
            self.ip_counts[target.range_index] += 1.0;
        }
        self.values.terminal_evals += 1;
    }

    fn add_zero_board_values(&mut self, cache: &TerminalEvalCache) {
        for target in &cache.oop_targets {
            self.oop_counts[target.range_index] += 1.0;
        }
        for target in &cache.ip_targets {
            self.ip_counts[target.range_index] += 1.0;
        }
        self.values.terminal_evals += 1;
    }

    fn add_compact_board_values(
        &mut self,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
    ) {
        let pot = pot as f32;
        for target in &cache.oop_targets {
            self.values.oop[target.range_index] += hero_values[target.range_index] * pot;
            self.oop_counts[target.range_index] += 1.0;
        }
        for target in &cache.ip_targets {
            self.values.ip[target.range_index] += villain_values[target.range_index] * pot;
            self.ip_counts[target.range_index] += 1.0;
        }
        self.values.terminal_evals += 1;
    }

    fn add_board_values_permuted(
        &mut self,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
        oop_source_to_target: &[usize],
        ip_source_to_target: &[usize],
    ) {
        let pot = pot as f32;
        for (source_index, target_index) in oop_source_to_target.iter().enumerate() {
            if let Some(board_index) = cache.oop_combo_indices[*target_index] {
                self.values.oop[source_index] += hero_values[board_index] * pot;
                self.oop_counts[source_index] += 1.0;
            }
        }
        for (source_index, target_index) in ip_source_to_target.iter().enumerate() {
            if let Some(board_index) = cache.ip_combo_indices[*target_index] {
                self.values.ip[source_index] += villain_values[board_index] * pot;
                self.ip_counts[source_index] += 1.0;
            }
        }
        self.values.terminal_evals += 1;
    }

    fn add_zero_board_values_permuted(
        &mut self,
        cache: &TerminalEvalCache,
        oop_source_to_target: &[usize],
        ip_source_to_target: &[usize],
    ) {
        for (source_index, target_index) in oop_source_to_target.iter().enumerate() {
            if cache.oop_combo_indices[*target_index].is_some() {
                self.oop_counts[source_index] += 1.0;
            }
        }
        for (source_index, target_index) in ip_source_to_target.iter().enumerate() {
            if cache.ip_combo_indices[*target_index].is_some() {
                self.ip_counts[source_index] += 1.0;
            }
        }
        self.values.terminal_evals += 1;
    }

    fn add_compact_board_values_permuted(
        &mut self,
        cache: &TerminalEvalCache,
        pot: u32,
        hero_values: &[f32],
        villain_values: &[f32],
        oop_source_to_target: &[usize],
        ip_source_to_target: &[usize],
    ) {
        let pot = pot as f32;
        for (source_index, target_index) in oop_source_to_target.iter().enumerate() {
            if cache.oop_combo_indices[*target_index].is_some() {
                self.values.oop[source_index] += hero_values[*target_index] * pot;
                self.oop_counts[source_index] += 1.0;
            }
        }
        for (source_index, target_index) in ip_source_to_target.iter().enumerate() {
            if cache.ip_combo_indices[*target_index].is_some() {
                self.values.ip[source_index] += villain_values[*target_index] * pot;
                self.ip_counts[source_index] += 1.0;
            }
        }
        self.values.terminal_evals += 1;
    }

    fn finish(&mut self) {
        for (value, count) in self.values.oop.iter_mut().zip(&self.oop_counts) {
            if *count > 0.0 {
                *value /= *count;
            }
        }
        for (value, count) in self.values.ip.iter_mut().zip(&self.ip_counts) {
            if *count > 0.0 {
                *value /= *count;
            }
        }
    }
}

fn current_strategies(regrets: &[f32], combos: usize, actions: usize) -> Vec<f32> {
    let mut strategies = Vec::new();
    current_strategies_into(regrets, combos, actions, &mut strategies);
    strategies
}

fn current_strategies_into(
    regrets: &[f32],
    combos: usize,
    actions: usize,
    strategies: &mut Vec<f32>,
) {
    strategies.resize(regrets.len(), 0.0);
    for combo in 0..combos {
        let row_start = combo * actions;
        let mut positive_sum = 0.0;
        for action in 0..actions {
            let value = regrets[row_start + action];
            if value > 0.0 {
                positive_sum += value;
            }
        }
        if positive_sum > 0.0 {
            for action in 0..actions {
                strategies[row_start + action] =
                    regrets[row_start + action].max(0.0) / positive_sum;
            }
        } else {
            let uniform = 1.0 / actions as f32;
            for action in 0..actions {
                strategies[row_start + action] = uniform;
            }
        }
    }
}

fn apply_real_cfr_update(
    regret: &mut f32,
    strategy_sum: &mut f32,
    regret_delta: f32,
    average_delta: f32,
    iteration_weight: f32,
    variant: RealCfrVariant,
) {
    match variant {
        RealCfrVariant::CfrPlus => {
            *regret = (*regret + regret_delta).max(0.0);
            *strategy_sum += iteration_weight * average_delta;
        }
        RealCfrVariant::Dcfr { alpha, beta, gamma } => {
            let regret_discount = dcfr_regret_discount(*regret, iteration_weight, alpha, beta);
            let strategy_discount = dcfr_strategy_discount(iteration_weight, gamma);
            *regret = *regret * regret_discount + regret_delta;
            *strategy_sum = *strategy_sum * strategy_discount + average_delta;
        }
        RealCfrVariant::DcfrPlus { alpha, gamma } => {
            let regret_discount = dcfr_regret_discount(*regret, iteration_weight, alpha, 0.0);
            let strategy_discount = dcfr_strategy_discount(iteration_weight, gamma);
            *regret = (*regret * regret_discount + regret_delta).max(0.0);
            *strategy_sum = *strategy_sum * strategy_discount + average_delta;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RealCfrUpdateFactors {
    variant: RealCfrVariant,
    positive_regret_discount: f32,
    negative_regret_discount: f32,
    strategy_discount: f32,
    iteration_weight: f32,
}

impl RealCfrUpdateFactors {
    fn new(variant: RealCfrVariant, iteration_weight: f32) -> Self {
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
            positive_regret_discount,
            negative_regret_discount,
            strategy_discount,
            iteration_weight,
        }
    }
}

fn apply_real_cfr_update_with_factors(
    regret: &mut f32,
    strategy_sum: &mut f32,
    regret_delta: f32,
    average_delta: f32,
    factors: &RealCfrUpdateFactors,
) {
    match factors.variant {
        RealCfrVariant::CfrPlus => {
            *regret = (*regret + regret_delta).max(0.0);
            *strategy_sum += factors.iteration_weight * average_delta;
        }
        RealCfrVariant::Dcfr { .. } => {
            let regret_discount = if *regret >= 0.0 {
                factors.positive_regret_discount
            } else {
                factors.negative_regret_discount
            };
            *regret = *regret * regret_discount + regret_delta;
            *strategy_sum = *strategy_sum * factors.strategy_discount + average_delta;
        }
        RealCfrVariant::DcfrPlus { .. } => {
            *regret = (*regret * factors.positive_regret_discount + regret_delta).max(0.0);
            *strategy_sum = *strategy_sum * factors.strategy_discount + average_delta;
        }
    }
}

fn dcfr_regret_discount(regret: f32, iteration: f32, alpha: f32, beta: f32) -> f32 {
    let exponent = if regret >= 0.0 { alpha } else { beta };
    dcfr_discount_for_exponent(iteration, exponent)
}

fn dcfr_discount_for_exponent(iteration: f32, exponent: f32) -> f32 {
    let powered = iteration.powf(exponent.max(0.0));
    powered / (powered + 1.0)
}

fn dcfr_strategy_discount(iteration: f32, gamma: f32) -> f32 {
    (iteration / (iteration + 1.0)).powf(gamma.max(0.0))
}

fn average_strategies(strategy_sum: &[f32], combos: usize, actions: usize) -> Vec<f32> {
    let mut strategies = Vec::new();
    average_strategies_into(strategy_sum, combos, actions, &mut strategies);
    strategies
}

fn average_strategies_into(
    strategy_sum: &[f32],
    combos: usize,
    actions: usize,
    strategies: &mut Vec<f32>,
) {
    strategies.resize(strategy_sum.len(), 0.0);
    for combo in 0..combos {
        let row_start = combo * actions;
        let mut total = 0.0;
        for action in 0..actions {
            total += strategy_sum[row_start + action];
        }
        if total > 0.0 {
            for action in 0..actions {
                strategies[row_start + action] = strategy_sum[row_start + action] / total;
            }
        } else {
            let uniform = 1.0 / actions as f32;
            for action in 0..actions {
                strategies[row_start + action] = uniform;
            }
        }
    }
}

fn strategy_reach_into(
    out: &mut [f32],
    input: &[f32],
    strategies: &[f32],
    actions: usize,
    action: usize,
) {
    for (combo, (out, input)) in out.iter_mut().zip(input).enumerate() {
        *out = *input * strategies[combo * actions + action];
    }
}

fn add_permuted_scaled_slice(
    out: &mut [f32],
    input: &[f32],
    source_to_target: &[usize],
    scale: f32,
) {
    for (source_index, target_index) in source_to_target.iter().copied().enumerate() {
        out[source_index] += input[target_index] * scale;
    }
}

fn add_permuted_scaled_values(
    out: &mut Values,
    input: &Values,
    maps: &ComboPermutationMaps,
    scale: f32,
) {
    add_permuted_scaled_slice(&mut out.oop, &input.oop, &maps.oop_source_to_target, scale);
    add_permuted_scaled_slice(&mut out.ip, &input.ip, &maps.ip_source_to_target, scale);
    out.terminal_evals += input.terminal_evals;
}

fn add_reach(out: &mut [f32], input: &[f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out += *input;
    }
}

fn combine_acting_flat_values(
    out: &mut [f32],
    action_values: &[f32],
    strategies: &[f32],
    actions: usize,
) {
    let combos = out.len();
    for combo in 0..combos {
        let mut value = 0.0f32;
        for action in 0..actions {
            value += strategies[combo * actions + action] * action_values[action * combos + combo];
        }
        out[combo] = value;
    }
}

fn combine_nonacting_flat_values(out: &mut [f32], action_values: &[f32], actions: usize) {
    let combos = out.len();
    for combo in 0..combos {
        let mut value = 0.0f32;
        for action in 0..actions {
            value += action_values[action * combos + combo];
        }
        out[combo] = value;
    }
}

fn add_strategy_reach(
    out: &mut [f32],
    input: &[f32],
    strategies: &[f32],
    actions: usize,
    action: usize,
) {
    for (combo, (out, input)) in out.iter_mut().zip(input).enumerate() {
        *out += *input * strategies[combo * actions + action];
    }
}

fn add_current_strategy_reaches(
    child_reaches: &mut [Vec<f32>],
    state_index: usize,
    children: &[usize],
    input: &[f32],
    regrets: &[f32],
    actions: usize,
) {
    debug_assert_eq!(children.len(), actions);
    for (combo, input_value) in input.iter().copied().enumerate() {
        if input_value == 0.0 {
            continue;
        }
        let row_start = combo * actions;
        let mut positive_sum = 0.0;
        for action in 0..actions {
            let value = regrets[row_start + action];
            if value > 0.0 {
                positive_sum += value;
            }
        }
        if positive_sum > 0.0 {
            let scale = input_value / positive_sum;
            for action in 0..actions {
                let child = children[action] - state_index - 1;
                child_reaches[child][combo] += regrets[row_start + action].max(0.0) * scale;
            }
        } else {
            let value = input_value / actions as f32;
            for child in children {
                child_reaches[*child - state_index - 1][combo] += value;
            }
        }
    }
}

fn add_average_strategy_reaches(
    child_reaches: &mut [Vec<f32>],
    state_index: usize,
    children: &[usize],
    input: &[f32],
    strategy_sum: &[f32],
    actions: usize,
) {
    debug_assert_eq!(children.len(), actions);
    for (combo, input_value) in input.iter().copied().enumerate() {
        if input_value == 0.0 {
            continue;
        }
        let row_start = combo * actions;
        let mut total = 0.0;
        for action in 0..actions {
            total += strategy_sum[row_start + action];
        }
        if total > 0.0 {
            let scale = input_value / total;
            for action in 0..actions {
                let child = children[action] - state_index - 1;
                child_reaches[child][combo] += strategy_sum[row_start + action] * scale;
            }
        } else {
            let value = input_value / actions as f32;
            for child in children {
                child_reaches[*child - state_index - 1][combo] += value;
            }
        }
    }
}

fn split_reach_state_and_children(
    reaches: &mut [Vec<f32>],
    state_index: usize,
) -> (&[f32], &mut [Vec<f32>]) {
    let (parents, children) = reaches.split_at_mut(state_index + 1);
    (&parents[state_index], children)
}

fn child_reach_mut(
    child_reaches: &mut [Vec<f32>],
    state_index: usize,
    child_index: usize,
) -> &mut [f32] {
    debug_assert!(child_index > state_index);
    &mut child_reaches[child_index - state_index - 1]
}

fn reorder_phase_states_by_level_and_slot(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    states: Vec<PhaseState>,
) -> Result<Vec<PhaseState>, String> {
    if states.is_empty() {
        return Ok(states);
    }
    let state_level = phase_state_levels(&states);
    let mut order = (0..states.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        state_level[*right]
            .cmp(&state_level[*left])
            .then_with(|| {
                phase_state_slot_sort_key(tree, infosets, oop_combos, ip_combos, &states[*left])
                    .cmp(&phase_state_slot_sort_key(
                        tree,
                        infosets,
                        oop_combos,
                        ip_combos,
                        &states[*right],
                    ))
            })
            .then_with(|| left.cmp(right))
    });
    if order.first().copied() != Some(0) {
        return Err("phase-state reorder did not keep the root first".to_string());
    }

    let mut old_to_new = vec![0usize; states.len()];
    for (new_index, old_index) in order.iter().copied().enumerate() {
        old_to_new[old_index] = new_index;
    }

    let mut reordered = Vec::with_capacity(states.len());
    for old_index in order {
        let mut state = states[old_index].clone();
        state.children = state
            .children
            .iter()
            .map(|child| old_to_new[*child])
            .collect();
        debug_assert!(state.children.iter().all(|child| *child > reordered.len()));
        reordered.push(state);
    }
    Ok(reordered)
}

fn phase_state_levels(states: &[PhaseState]) -> Vec<usize> {
    let mut state_level = vec![0usize; states.len()];
    for state_index in (0..states.len()).rev() {
        state_level[state_index] = states[state_index]
            .children
            .iter()
            .map(|child| state_level[*child] + 1)
            .max()
            .unwrap_or(0);
    }
    state_level
}

fn phase_state_slot_sort_key(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    state: &PhaseState,
) -> (bool, usize, usize) {
    let Some((slot_start, slot_end)) =
        backup_state_slot_range(tree, infosets, oop_combos, ip_combos, state)
    else {
        return (true, usize::MAX, usize::MAX);
    };
    (false, slot_start, slot_end)
}

fn first_terminal_state_index(states: &[PhaseState]) -> usize {
    states
        .iter()
        .position(|state| state.children.is_empty())
        .unwrap_or(states.len())
}

fn effective_worker_count(requested: usize) -> usize {
    if requested == 0 {
        rayon::current_num_threads()
    } else {
        requested
    }
    .max(1)
}

fn backup_run_decision_prefix_end(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    states: &[PhaseState],
    run: &BackupRun,
) -> usize {
    states[run.start..run.end]
        .iter()
        .position(|state| {
            !matches!(
                tree.nodes[state.node_id].kind,
                PublicNodeKind::Decision { .. }
            ) || infosets[state.node_id].is_none()
        })
        .map_or(run.end, |offset| run.start + offset)
}

fn backup_decision_chunks(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    states: &[PhaseState],
    state_start: usize,
    state_end: usize,
    threads: usize,
) -> Vec<BackupChunk> {
    let state_count = state_end - state_start;
    if state_count == 0 {
        return Vec::new();
    }
    let chunk_count = threads.max(1).min(state_count);
    let chunk_len = state_count.div_ceil(chunk_count);
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut start = state_start;
    while start < state_end {
        let end = (start + chunk_len).min(state_end);
        let run = backup_run_with_slots(tree, infosets, oop_combos, ip_combos, states, start, end);
        debug_assert!(run.slot_start < run.slot_end);
        chunks.push(BackupChunk {
            state_start: start,
            state_end: end,
            slot_start: run.slot_start,
            slot_end: run.slot_end,
        });
        start = end;
    }
    debug_assert!(backup_chunks_are_ordered(&chunks));
    chunks
}

fn backup_chunks_are_ordered(chunks: &[BackupChunk]) -> bool {
    let mut previous_slot_end = 0usize;
    let mut previous_state_end = 0usize;
    for (index, chunk) in chunks.iter().enumerate() {
        if index > 0 {
            if chunk.state_start != previous_state_end || chunk.slot_start < previous_slot_end {
                return false;
            }
        }
        previous_state_end = chunk.state_end;
        previous_slot_end = chunk.slot_end;
    }
    true
}

fn backup_chance_state(
    tree: &PublicTree,
    states: &[PhaseState],
    values: &mut [Values],
    state_index: usize,
    combo_permutations: &BTreeMap<u8, ComboPermutationMaps>,
) -> Result<(), String> {
    let state = &states[state_index];
    match &tree.nodes[state.node_id].kind {
        PublicNodeKind::Chance(_) => {
            if state.chance_concrete_events == 0 {
                return Err("chance state has no concrete events".to_string());
            }
            if state.children.len() != state.chance_member_permutation_codes.len() {
                return Err("chance state member classes do not match children".to_string());
            }
            let chance_weight = 1.0 / state.chance_concrete_events as f32;
            let (state_value, child_values) = split_state_and_children(values, state_index);
            state_value.reset();
            for (child, permutation_codes) in state
                .children
                .iter()
                .zip(&state.chance_member_permutation_codes)
            {
                let child_value = child_value(child_values, state_index, *child);
                for permutation_code in permutation_codes {
                    let maps = combo_permutations.get(permutation_code).ok_or_else(|| {
                        "chance isomorphism permutation is missing combo maps".to_string()
                    })?;
                    add_permuted_scaled_values(state_value, child_value, maps, chance_weight);
                }
            }
            Ok(())
        }
        PublicNodeKind::Terminal { .. } => Ok(()),
        PublicNodeKind::Decision { actions, .. } if actions.len() == 1 => {
            let child = state
                .children
                .first()
                .copied()
                .ok_or_else(|| "single-action decision has no child".to_string())?;
            values[state_index] = values[child].clone();
            Ok(())
        }
        PublicNodeKind::Decision { .. } => {
            Err("multi-action decision state reached chance backup path".to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backup_decision_chunk(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    states: &[PhaseState],
    oop_reaches: &[Vec<f32>],
    ip_reaches: &[Vec<f32>],
    child_values: &[Values],
    child_base: usize,
    current_values: &mut [Values],
    current_base: usize,
    regrets: &mut [f32],
    strategy_sum: &mut [f32],
    slot_base: usize,
    update_factors: &RealCfrUpdateFactors,
) -> Result<(), String> {
    for (local_state_index, state_value) in current_values.iter_mut().enumerate() {
        let state_index = current_base + local_state_index;
        let state = &states[state_index];
        let PublicNodeKind::Decision { player, actions } = &tree.nodes[state.node_id].kind else {
            return Err("non-decision state reached decision backup path".to_string());
        };
        let actions_len = actions.len();
        let acting_combos = match player {
            Player::Oop => oop_combos,
            Player::Ip => ip_combos,
        };
        let board_slot = state.board_slot;
        let row_len = acting_combos * actions_len;
        let row_start = board_slot * row_len;
        match player {
            Player::Oop => combine_nonacting_child_values_from_base(
                &mut state_value.ip,
                child_values,
                child_base,
                &state.children,
                Player::Ip,
            ),
            Player::Ip => combine_nonacting_child_values_from_base(
                &mut state_value.oop,
                child_values,
                child_base,
                &state.children,
                Player::Oop,
            ),
        }
        state_value.terminal_evals = state
            .children
            .iter()
            .map(|child| child_value_from_base(child_values, child_base, *child).terminal_evals)
            .sum();

        let own_reach = match player {
            Player::Oop => &oop_reaches[state_index],
            Player::Ip => &ip_reaches[state_index],
        };
        let infoset = infosets[state.node_id]
            .as_ref()
            .expect("decision node must have infoset");
        let slots_start = infoset.slots_start;
        let mut strategy_probs = [0.0f32; 8];
        for combo in 0..acting_combos {
            let local_row_start = combo * actions_len;
            let row_slot = slots_start + row_start + local_row_start - slot_base;
            fill_strategy_probs(
                &regrets[row_slot..row_slot + actions_len],
                &mut strategy_probs,
            )?;
            let mut node_value = 0.0f32;
            for action_index in 0..actions_len {
                let action_values =
                    child_value_from_base(child_values, child_base, state.children[action_index]);
                let action_value = match player {
                    Player::Oop => action_values.oop[combo],
                    Player::Ip => action_values.ip[combo],
                };
                node_value += strategy_probs[action_index] * action_value;
            }
            match player {
                Player::Oop => state_value.oop[combo] = node_value,
                Player::Ip => state_value.ip[combo] = node_value,
            }
            for action_index in 0..actions_len {
                let action_values =
                    child_value_from_base(child_values, child_base, state.children[action_index]);
                let action_value = match player {
                    Player::Oop => action_values.oop[combo],
                    Player::Ip => action_values.ip[combo],
                };
                let local_slot = row_slot + action_index;
                apply_real_cfr_update_with_factors(
                    &mut regrets[local_slot],
                    &mut strategy_sum[local_slot],
                    action_value - node_value,
                    own_reach[combo] * strategy_probs[action_index],
                    update_factors,
                );
            }
        }
    }
    Ok(())
}

fn child_value_from_base(children: &[Values], child_base: usize, child_index: usize) -> &Values {
    debug_assert!(child_index >= child_base);
    &children[child_index - child_base]
}

fn combine_nonacting_child_values_from_base(
    out: &mut [f32],
    child_values: &[Values],
    child_base: usize,
    children: &[usize],
    player: Player,
) {
    for combo in 0..out.len() {
        let mut value = 0.0f32;
        for child in children {
            value += match player {
                Player::Oop => child_value_from_base(child_values, child_base, *child).oop[combo],
                Player::Ip => child_value_from_base(child_values, child_base, *child).ip[combo],
            };
        }
        out[combo] = value;
    }
}

fn backup_level_plan(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    states: &[PhaseState],
) -> BackupLevelPlan {
    let mut state_level = vec![0usize; states.len()];
    let mut max_level = 0usize;
    for state_index in (0..states.len()).rev() {
        let level = states[state_index]
            .children
            .iter()
            .map(|child| state_level[*child] + 1)
            .max()
            .unwrap_or(0);
        state_level[state_index] = level;
        max_level = max_level.max(level);
    }

    let mut levels = vec![Vec::new(); max_level + 1];
    let mut run_start = 0usize;
    let mut run_level = state_level.first().copied().unwrap_or(0);
    for (state_index, level) in state_level.iter().copied().enumerate().skip(1) {
        if level != run_level {
            levels[run_level].push(backup_run_with_slots(
                tree,
                infosets,
                oop_combos,
                ip_combos,
                states,
                run_start,
                state_index,
            ));
            run_start = state_index;
            run_level = level;
        }
    }
    if !states.is_empty() {
        levels[run_level].push(backup_run_with_slots(
            tree,
            infosets,
            oop_combos,
            ip_combos,
            states,
            run_start,
            states.len(),
        ));
    }
    for level in &mut levels {
        sort_backup_level_runs(level);
        if !backup_level_slots_are_ordered(level) {
            let mut refined = Vec::new();
            for run in level.iter() {
                for state_index in run.start..run.end {
                    refined.push(backup_run_with_slots(
                        tree,
                        infosets,
                        oop_combos,
                        ip_combos,
                        states,
                        state_index,
                        state_index + 1,
                    ));
                }
            }
            sort_backup_level_runs(&mut refined);
            debug_assert!(backup_level_slots_are_ordered(&refined));
            *level = refined;
        }
    }
    BackupLevelPlan { levels }
}

fn sort_backup_level_runs(level: &mut [BackupRun]) {
    level.sort_by_key(|run| {
        (
            run.slot_start == run.slot_end,
            run.slot_start,
            run.slot_end,
            run.start,
        )
    });
}

fn backup_level_slots_are_ordered(level: &[BackupRun]) -> bool {
    let mut previous_end = 0usize;
    let mut have_previous = false;
    for run in level {
        if run.slot_start == run.slot_end {
            continue;
        }
        if have_previous && run.slot_start < previous_end {
            return false;
        }
        previous_end = run.slot_end;
        have_previous = true;
    }
    true
}

fn backup_run_with_slots(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    states: &[PhaseState],
    start: usize,
    end: usize,
) -> BackupRun {
    let mut slot_start = usize::MAX;
    let mut slot_end = 0usize;
    for state in &states[start..end] {
        let Some((start, end)) =
            backup_state_slot_range(tree, infosets, oop_combos, ip_combos, state)
        else {
            continue;
        };
        slot_start = slot_start.min(start);
        slot_end = slot_end.max(end);
    }
    if slot_start == usize::MAX {
        slot_start = 0;
        slot_end = 0;
    }
    BackupRun {
        start,
        end,
        slot_start,
        slot_end,
    }
}

fn backup_state_slot_range(
    tree: &PublicTree,
    infosets: &[Option<RealInfoset>],
    oop_combos: usize,
    ip_combos: usize,
    state: &PhaseState,
) -> Option<(usize, usize)> {
    let PublicNodeKind::Decision { .. } = &tree.nodes[state.node_id].kind else {
        return None;
    };
    let infoset = infosets[state.node_id].as_ref()?;
    let combos = match infoset.player {
        Player::Oop => oop_combos,
        Player::Ip => ip_combos,
    };
    let row_len = combos * infoset.actions;
    let slot_start = infoset.slots_start + state.board_slot * row_len;
    Some((slot_start, slot_start + row_len))
}

fn combine_acting_values(
    out: &mut [f32],
    action_values: &[Values],
    strategies: &[f32],
    actions: usize,
    player: Player,
) {
    for combo in 0..out.len() {
        let mut value = 0.0f32;
        for action in 0..actions {
            value += strategies[combo * actions + action]
                * match player {
                    Player::Oop => action_values[action].oop[combo],
                    Player::Ip => action_values[action].ip[combo],
                };
        }
        out[combo] = value;
    }
}

fn combine_nonacting_values(out: &mut [f32], action_values: &[Values], player: Player) {
    for combo in 0..out.len() {
        out[combo] = action_values
            .iter()
            .map(|values| match player {
                Player::Oop => values.oop[combo],
                Player::Ip => values.ip[combo],
            })
            .sum();
    }
}

fn split_state_and_children(values: &mut [Values], state_index: usize) -> (&mut Values, &[Values]) {
    let (parents, children) = values.split_at_mut(state_index + 1);
    (&mut parents[state_index], children)
}

fn child_value(children: &[Values], state_index: usize, child_index: usize) -> &Values {
    debug_assert!(child_index > state_index);
    &children[child_index - state_index - 1]
}

fn fill_strategy_probs(row: &[f32], out: &mut [f32; 8]) -> Result<(), String> {
    if row.len() > out.len() {
        return Err(format!(
            "real CFR backup supports at most {} actions per node, got {}",
            out.len(),
            row.len()
        ));
    }
    let positive_sum = row
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .sum::<f32>();
    if positive_sum > 0.0 {
        for (index, value) in row.iter().copied().enumerate() {
            out[index] = value.max(0.0) / positive_sum;
        }
    } else {
        let uniform = 1.0 / row.len() as f32;
        for index in 0..row.len() {
            out[index] = uniform;
        }
    }
    Ok(())
}

fn combine_best_response_values(
    out: &mut [f32],
    action_values: &[Values],
    actions: usize,
    player: Player,
) {
    for combo in 0..out.len() {
        let mut best = f32::NEG_INFINITY;
        for action in 0..actions {
            let value = match player {
                Player::Oop => action_values[action].oop[combo],
                Player::Ip => action_values[action].ip[combo],
            };
            best = best.max(value);
        }
        out[combo] = best;
    }
}

#[cfg(test)]
fn opponent_weights_for(
    own_combos: &[ComboWeight],
    opponent_combos: &[ComboWeight],
    opponent_reach: &[f32],
    board: &Board,
) -> Vec<f32> {
    own_combos
        .iter()
        .map(|own| {
            if board.contains(own.first) || board.contains(own.second) {
                return 0.0;
            }
            opponent_combos
                .iter()
                .zip(opponent_reach)
                .filter(|(opponent, reach)| {
                    **reach > 0.0
                        && !board.contains(opponent.first)
                        && !board.contains(opponent.second)
                        && !combos_collide(own, opponent)
                })
                .map(|(_, reach)| *reach)
                .sum()
        })
        .collect()
}

#[cfg(test)]
fn opponent_weights_for_fast(
    own_combos: &[ComboWeight],
    opponent_combos: &[ComboWeight],
    opponent_reach: &[f32],
    same_combo_indices: &[Option<usize>],
    board: &Board,
) -> Vec<f32> {
    let mut out = vec![0.0f32; own_combos.len()];
    opponent_weights_for_fast_into(
        own_combos,
        opponent_combos,
        opponent_reach,
        same_combo_indices,
        board,
        &mut out,
    );
    out
}

fn opponent_weights_for_fast_into(
    own_combos: &[ComboWeight],
    opponent_combos: &[ComboWeight],
    opponent_reach: &[f32],
    same_combo_indices: &[Option<usize>],
    board: &Board,
    out: &mut [f32],
) {
    debug_assert_eq!(opponent_combos.len(), opponent_reach.len());
    debug_assert_eq!(own_combos.len(), same_combo_indices.len());
    debug_assert_eq!(own_combos.len(), out.len());

    let mut total = 0.0f32;
    let mut card_totals = [0.0f32; 52];
    for (combo, reach) in opponent_combos.iter().zip(opponent_reach) {
        if *reach == 0.0 || board.contains(combo.first) || board.contains(combo.second) {
            continue;
        }
        total += *reach;
        card_totals[combo.first.index()] += *reach;
        card_totals[combo.second.index()] += *reach;
    }

    for ((own, same_index), out) in own_combos
        .iter()
        .zip(same_combo_indices)
        .zip(out.iter_mut())
    {
        if board.contains(own.first) || board.contains(own.second) {
            *out = 0.0;
            continue;
        }
        let same_reach = same_index
            .and_then(|index| {
                let opponent = &opponent_combos[index];
                (!board.contains(opponent.first) && !board.contains(opponent.second))
                    .then_some(opponent_reach[index])
            })
            .unwrap_or(0.0);
        *out =
            total - card_totals[own.first.index()] - card_totals[own.second.index()] + same_reach;
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

fn prepared_combo_indices(
    prepared: &PreparedTerminalBoard,
    combos: &[ComboWeight],
) -> Vec<Option<usize>> {
    combos
        .iter()
        .map(|combo| prepared.combo_index(combo.first, combo.second))
        .collect()
}

fn prepared_combo_targets(combo_indices: &[Option<usize>]) -> Vec<PreparedComboTarget> {
    combo_indices
        .iter()
        .enumerate()
        .filter_map(|(range_index, board_index)| {
            board_index.map(|board_index| PreparedComboTarget {
                range_index,
                board_index: board_index as u16,
            })
        })
        .collect()
}

fn prepared_board_targets(targets: &[PreparedComboTarget]) -> Vec<u16> {
    targets.iter().map(|target| target.board_index).collect()
}

fn reach_on_prepared_board_targets_into(
    targets: &[PreparedComboTarget],
    reach: &[f32],
    out: &mut [f32],
) {
    out.fill(0.0);
    for target in targets {
        let reach = reach[target.range_index];
        if reach != 0.0 {
            out[target.board_index as usize] += reach;
        }
    }
}

fn reach_on_prepared_board_targets_sparse_into(
    targets: &[PreparedComboTarget],
    reach: &[f32],
    out: &mut [f32],
    nonzero: &mut Vec<u16>,
) {
    for board_index in nonzero.iter() {
        out[*board_index as usize] = 0.0;
    }
    nonzero.clear();
    for target in targets {
        let reach = reach[target.range_index];
        if reach == 0.0 {
            continue;
        }
        let board_index = target.board_index as usize;
        if out[board_index] == 0.0 {
            nonzero.push(target.board_index);
        }
        out[board_index] += reach;
    }
}

fn terminal_sparse_nonzero_limit() -> usize {
    std::env::var("POKEDR_REAL_CFR_SPARSE_NONZERO_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(TERMINAL_SPARSE_NONZERO_LIMIT)
}

fn real_cfr_profile_start_iteration() -> u32 {
    std::env::var("POKEDR_REAL_CFR_PROFILE_START_ITER")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|iteration| *iteration > 0)
        .unwrap_or(1)
}

fn terminal_phase_partitions(
    states: &[PhaseState],
    terminal_start: usize,
    threads: usize,
) -> Vec<(usize, usize)> {
    let terminal_len = states.len() - terminal_start;
    let threads = threads.max(1).min(terminal_len.max(1));
    if terminal_len == 0 {
        return Vec::new();
    }
    let total_weight = states[terminal_start..]
        .iter()
        .map(terminal_phase_state_weight)
        .sum::<usize>()
        .max(1);
    let target_weight = total_weight.div_ceil(threads);
    let mut partitions = Vec::with_capacity(threads);
    let mut start = terminal_start;
    let mut weight = 0usize;
    for index in terminal_start..states.len() {
        weight += terminal_phase_state_weight(&states[index]);
        let remaining_states = states.len() - index - 1;
        let remaining_partitions = threads.saturating_sub(partitions.len() + 1);
        if weight >= target_weight && remaining_states >= remaining_partitions {
            partitions.push((start, index + 1));
            start = index + 1;
            weight = 0;
            if partitions.len() + 1 == threads {
                break;
            }
        }
    }
    if start < states.len() {
        partitions.push((start, states.len()));
    }
    partitions
}

fn terminal_phase_state_weight(state: &PhaseState) -> usize {
    state.terminal_cache_refs.len().max(1)
}

fn average_usize(sum: usize, count: usize) -> f64 {
    if count > 0 {
        sum as f64 / count as f64
    } else {
        0.0
    }
}

fn average_f64(sum: f64, count: usize) -> f64 {
    if count > 0 { sum / count as f64 } else { 0.0 }
}

fn terminal_side_cache_enabled() -> bool {
    std::env::var("POKEDR_REAL_CFR_TERMINAL_SIDE_CACHE")
        .map(|value| {
            let value = value.trim();
            !(value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off"))
        })
        .unwrap_or(true)
}

fn terminal_side_cached_values(
    cache: &mut TerminalSideValueCache,
    terminal: &TerminalEvalCache,
    cache_index: usize,
    side: TerminalSideValue,
    opponent_reach: &[f32],
    opponent_nonzero: &[u16],
    targets: &[PreparedComboTarget],
    board_targets: &[u16],
    range_len: usize,
    use_sparse: bool,
    _scratch: &mut TerminalCfvScratch,
) -> Result<Arc<[f32]>, String> {
    let reach_hash = hash_sparse_reach(opponent_reach, opponent_nonzero);
    let key = TerminalSideCacheKey {
        cache_index,
        side,
        reach_hash,
    };
    if let Some(entries) = cache.entries.get(&key) {
        if let Some(entry) = entries.iter().find(|entry| {
            sparse_reach_signature_matches(&entry.signature, opponent_reach, opponent_nonzero)
        }) {
            cache.hits += 1;
            return Ok(entry.values.clone());
        }
    }

    cache.misses += 1;
    let combos = terminal.prepared.combos().len();
    let mut board_values = vec![0.0f32; combos];
    if use_sparse {
        terminal_side_values_sparse_board_targets_into(
            &terminal.prepared,
            opponent_reach,
            opponent_nonzero,
            board_targets,
            &mut board_values,
        )?;
    } else {
        terminal_side_values_prefix_blocker_sorted_board_targets_into(
            &terminal.prepared,
            opponent_reach,
            board_targets,
            &mut board_values,
        )?;
    }
    let mut values = vec![0.0f32; range_len];
    for target in targets {
        values[target.range_index] = board_values[target.board_index as usize];
    }
    let values = Arc::<[f32]>::from(values);
    cache
        .entries
        .entry(key)
        .or_default()
        .push(TerminalSideCacheEntry {
            signature: sparse_reach_signature(opponent_reach, opponent_nonzero),
            values: values.clone(),
        });
    Ok(values)
}

fn sparse_reach_signature_matches(
    signature: &[(u16, u32)],
    reach: &[f32],
    nonzero: &[u16],
) -> bool {
    signature.len() == nonzero.len()
        && signature
            .iter()
            .zip(nonzero)
            .all(|((left_index, left_bits), right_index)| {
                left_index == right_index && *left_bits == reach[*right_index as usize].to_bits()
            })
}

fn sparse_reach_signature(reach: &[f32], nonzero: &[u16]) -> Vec<(u16, u32)> {
    nonzero
        .iter()
        .map(|index| (*index, reach[*index as usize].to_bits()))
        .collect()
}

fn hash_sparse_reach(reach: &[f32], nonzero: &[u16]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for board_index in nonzero {
        let board_index = *board_index as usize;
        hash = mix_hash(hash, board_index as u64);
        hash = mix_hash(hash, reach[board_index].to_bits() as u64);
    }
    hash
}

fn mix_hash(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x100000001b3)
}

fn print_terminal_worker_profiles(
    total_tasks: usize,
    threads: usize,
    profiles: &[TerminalWorkerProfile],
) {
    let tasks = profiles.iter().map(|profile| profile.tasks).sum::<usize>();
    let sparse_tasks = profiles
        .iter()
        .map(|profile| profile.sparse_tasks)
        .sum::<usize>();
    let prefix_tasks = profiles
        .iter()
        .map(|profile| profile.prefix_tasks)
        .sum::<usize>();
    let zero_reach_tasks = profiles
        .iter()
        .map(|profile| profile.zero_reach_tasks)
        .sum::<usize>();
    let side_cache_hits = profiles
        .iter()
        .map(|profile| profile.side_cache_hits)
        .sum::<usize>();
    let side_cache_misses = profiles
        .iter()
        .map(|profile| profile.side_cache_misses)
        .sum::<usize>();
    let terminal_states = profiles
        .iter()
        .map(|profile| profile.terminal_states)
        .sum::<usize>();
    let fold_states = profiles
        .iter()
        .map(|profile| profile.fold_states)
        .sum::<usize>();
    let oop_nonzero_sum = profiles
        .iter()
        .map(|profile| profile.oop_nonzero_sum)
        .sum::<usize>();
    let ip_nonzero_sum = profiles
        .iter()
        .map(|profile| profile.ip_nonzero_sum)
        .sum::<usize>();
    let max_elapsed = profiles
        .iter()
        .map(|profile| profile.elapsed_ms)
        .fold(0.0f64, f64::max);
    let min_elapsed = profiles
        .iter()
        .map(|profile| profile.elapsed_ms)
        .fold(f64::INFINITY, f64::min);
    let board_expand_ms = profiles
        .iter()
        .map(|profile| profile.board_expand_ms)
        .sum::<f64>();
    let fold_ms = profiles.iter().map(|profile| profile.fold_ms).sum::<f64>();
    let reach_map_ms = profiles
        .iter()
        .map(|profile| profile.reach_map_ms)
        .sum::<f64>();
    let cfv_ms = profiles.iter().map(|profile| profile.cfv_ms).sum::<f64>();
    let accumulator_ms = profiles
        .iter()
        .map(|profile| profile.accumulator_ms)
        .sum::<f64>();
    let avg_oop_nonzero = if tasks > 0 {
        oop_nonzero_sum as f64 / tasks as f64
    } else {
        0.0
    };
    let avg_ip_nonzero = if tasks > 0 {
        ip_nonzero_sum as f64 / tasks as f64
    } else {
        0.0
    };
    eprintln!(
        "real_cfr_terminal_profile tasks={} accounted_tasks={} threads={} sparse_tasks={} prefix_tasks={} zero_reach_tasks={} side_cache_hits={} side_cache_misses={} avg_oop_nonzero={:.2} avg_ip_nonzero={:.2} min_worker_ms={:.3} max_worker_ms={:.3}",
        total_tasks,
        tasks,
        threads,
        sparse_tasks,
        prefix_tasks,
        zero_reach_tasks,
        side_cache_hits,
        side_cache_misses,
        avg_oop_nonzero,
        avg_ip_nonzero,
        min_elapsed,
        max_elapsed,
    );
    eprintln!(
        "real_cfr_terminal_phase_breakdown terminal_states={} fold_states={} board_expand_ms={:.3} fold_ms={:.3} reach_map_ms={:.3} cfv_ms={:.3} accumulator_ms={:.3}",
        terminal_states,
        fold_states,
        board_expand_ms,
        fold_ms,
        reach_map_ms,
        cfv_ms,
        accumulator_ms,
    );
    for profile in profiles {
        let avg_oop = if profile.tasks > 0 {
            profile.oop_nonzero_sum as f64 / profile.tasks as f64
        } else {
            0.0
        };
        let avg_ip = if profile.tasks > 0 {
            profile.ip_nonzero_sum as f64 / profile.tasks as f64
        } else {
            0.0
        };
        eprintln!(
            "real_cfr_terminal_worker worker={} terminal_states={} fold_states={} tasks={} sparse={} prefix={} zero_reach={} side_cache_hits={} side_cache_misses={} avg_oop_nonzero={:.2} avg_ip_nonzero={:.2} max_oop_nonzero={} max_ip_nonzero={} output_states={} board_expand_ms={:.3} fold_ms={:.3} reach_map_ms={:.3} cfv_ms={:.3} accumulator_ms={:.3} elapsed_ms={:.3}",
            profile.worker_index,
            profile.terminal_states,
            profile.fold_states,
            profile.tasks,
            profile.sparse_tasks,
            profile.prefix_tasks,
            profile.zero_reach_tasks,
            profile.side_cache_hits,
            profile.side_cache_misses,
            avg_oop,
            avg_ip,
            profile.oop_nonzero_max,
            profile.ip_nonzero_max,
            profile.output_states,
            profile.board_expand_ms,
            profile.fold_ms,
            profile.reach_map_ms,
            profile.cfv_ms,
            profile.accumulator_ms,
            profile.elapsed_ms,
        );
    }
}

fn print_terminal_side_cache_key_profile(profiles: &[TerminalWorkerProfile]) {
    let mut global_keys: HashMap<TerminalSideCacheKey, usize> = HashMap::new();
    let mut board_keys: HashMap<usize, HashSet<TerminalSideCacheKey>> = HashMap::new();
    let mut worker_local_keys = 0usize;
    for profile in profiles {
        let local_keys = profile
            .side_cache_keys
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        worker_local_keys += local_keys.len();
        for key in local_keys {
            *global_keys.entry(key.clone()).or_insert(0) += 1;
            board_keys.entry(key.cache_index).or_default().insert(key);
        }
    }
    let unique_keys = global_keys.len();
    let cross_worker_keys = global_keys
        .values()
        .filter(|worker_count| **worker_count > 1)
        .count();
    let cross_worker_extra_touches = global_keys
        .values()
        .map(|worker_count| worker_count.saturating_sub(1))
        .sum::<usize>();
    let max_workers_for_key = global_keys.values().copied().max().unwrap_or(0);
    let mut board_unique_counts = board_keys
        .iter()
        .map(|(cache_index, keys)| (*cache_index, keys.len()))
        .collect::<Vec<_>>();
    board_unique_counts.sort_unstable_by(|left, right| right.1.cmp(&left.1));
    let top_boards = board_unique_counts
        .iter()
        .take(8)
        .map(|(cache_index, count)| format!("{cache_index}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "real_cfr_side_cache_key_profile workers={} worker_local_keys={} unique_keys={} cross_worker_keys={} cross_worker_extra_touches={} max_workers_for_key={} top_boards={}",
        profiles.len(),
        worker_local_keys,
        unique_keys,
        cross_worker_keys,
        cross_worker_extra_touches,
        max_workers_for_key,
        top_boards,
    );
}

fn terminal_task_checksum(
    task: &TerminalBoardTask,
    prepared: &PreparedTerminalBoard,
    scratch: &TerminalCfvScratch,
) -> f64 {
    let board_factor = unordered_board_key(&task.board) as f64 + task.terminal_node as f64 + 1.0;
    let hero = scratch
        .hero_values()
        .iter()
        .take(prepared.combos().len().min(8))
        .enumerate()
        .map(|(index, value)| *value as f64 * (index as f64 + 1.0))
        .sum::<f64>();
    let villain = scratch
        .villain_values()
        .iter()
        .take(prepared.combos().len().min(8))
        .enumerate()
        .map(|(index, value)| *value as f64 * (index as f64 + 17.0))
        .sum::<f64>();
    (hero + villain) * board_factor
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
fn combos_collide(left: &ComboWeight, right: &ComboWeight) -> bool {
    left.first == right.first
        || left.first == right.second
        || left.second == right.first
        || left.second == right.second
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

fn representative_ordered_future_boards(
    flop: &Board,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> Result<(Vec<Board>, Vec<Board>), String> {
    if flop.cards().len() != 3 {
        return Err("real CFR solver must start from a flop board".to_string());
    }
    let report = fixed_flop_future_board_isomorphism(flop, oop_range, ip_range)?;
    let mut turn_boards = Vec::with_capacity(report.turn.classes.len());
    let mut river_boards = Vec::with_capacity(report.ordered_turn_river_representative_events);
    for (turn_class, river_classes) in report
        .turn
        .classes
        .iter()
        .zip(&report.representative_turn_river_classes)
    {
        let turn_card = *turn_class
            .representative
            .first()
            .ok_or_else(|| "turn class has no representative card".to_string())?;
        let turn_board = flop.push(turn_card)?;
        turn_boards.push(turn_board.clone());
        for river_class in &river_classes.classes {
            let river_card = *river_class
                .representative
                .first()
                .ok_or_else(|| "river class has no representative card".to_string())?;
            river_boards.push(turn_board.push(river_card)?);
        }
    }
    Ok((turn_boards, river_boards))
}

fn unordered_river_boards_from_flop(flop: &Board) -> Result<Vec<Board>, String> {
    if flop.cards().len() != 3 {
        return Err("real CFR solver must start from a flop board".to_string());
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
    let mut key = 0u64;
    for card in board.cards() {
        key |= 1u64 << card.index();
    }
    key
}

fn ordered_board_key(board: &Board) -> u64 {
    let mut key = 0u64;
    for card in board.cards() {
        key = key * 52 + card.index() as u64;
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{ActionAbstraction, ChanceExpansion, Spot, TreeBuilder, TreeTemplate};
    use std::str::FromStr;

    #[test]
    fn real_cfr_allocates_isomorphic_board_indexed_state_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::conservative_default(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 650,
            effective_stack: 9700,
            oop_range: RangeSpec::from_str("AsAh,KsKh").unwrap(),
            ip_range: RangeSpec::from_str("QsQh,JsJh").unwrap(),
            first_player: Player::Oop,
        })
        .unwrap();
        let solver = RealCfrSolver::new(
            tree,
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let iso = fixed_flop_future_board_isomorphism(
            &solver.flop_board,
            &RangeSpec::from_str("AsAh,KsKh").unwrap(),
            &RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        assert_eq!(solver.turn_index_by_key.len(), iso.turn.classes.len());
        assert_eq!(
            solver.river_index_by_key.len(),
            iso.ordered_turn_river_representative_events
        );
        let action_slots = solver
            .infosets
            .iter()
            .filter_map(Option::as_ref)
            .map(|infoset| infoset.slots_len)
            .sum::<usize>();
        assert_eq!(action_slots, 7_929_144);
        assert_eq!(solver.terminal_cache.len(), 49 * 48 / 2);
        assert_eq!(solver.terminal_cache_index_by_key.len(), 49 * 48 / 2);
    }

    #[test]
    #[ignore = "exact board-indexed CFR traverses all turn/river runouts and is intentionally slow"]
    fn real_cfr_runs_terminal_cfv_backup_and_stays_zero_sum_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::conservative_default(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 650,
            effective_stack: 9700,
            oop_range: RangeSpec::from_str("AsAh,KsKh").unwrap(),
            ip_range: RangeSpec::from_str("QsQh,JsJh").unwrap(),
            first_player: Player::Oop,
        })
        .unwrap();
        let mut solver = RealCfrSolver::new(
            tree,
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let summary = solver
            .run(RealCfrConfig {
                iterations: 1,
                variant: RealCfrVariant::CfrPlus,
            })
            .unwrap();
        assert_eq!(summary.iterations, 1);
        assert!(summary.decision_nodes > 0);
        assert!(summary.action_slots > 0);
        assert!(summary.terminal_evals > 0);
        assert!(
            (summary.root_oop_value + summary.root_ip_value).abs() < 0.05,
            "{summary:?}"
        );
    }

    #[test]
    #[ignore = "exact board-indexed three-phase CFR traverses all turn/river runouts and is intentionally slow"]
    fn three_phase_real_cfr_matches_recursive_one_iteration_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::conservative_default(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 650,
            effective_stack: 9700,
            oop_range: RangeSpec::from_str("AsAh,KsKh").unwrap(),
            ip_range: RangeSpec::from_str("QsQh,JsJh").unwrap(),
            first_player: Player::Oop,
        })
        .unwrap();
        let mut recursive = RealCfrSolver::new(
            tree.clone(),
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let mut phased = RealCfrSolver::new(
            tree,
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let recursive = recursive
            .run(RealCfrConfig {
                iterations: 1,
                variant: RealCfrVariant::CfrPlus,
            })
            .unwrap();
        let phased = phased
            .run_three_phase(
                RealCfrConfig {
                    iterations: 1,
                    variant: RealCfrVariant::CfrPlus,
                },
                4,
                |_| {},
            )
            .unwrap();
        assert_eq!(recursive.terminal_evals, phased.terminal_evals);
        assert!((recursive.root_oop_value - phased.root_oop_value).abs() < 0.001);
        assert!((recursive.root_ip_value - phased.root_ip_value).abs() < 0.001);
    }

    #[test]
    #[ignore = "exact side-cache comparison traverses all turn/river runouts and is intentionally slow"]
    fn three_phase_terminal_side_cache_matches_uncached_one_iteration_on_small_ranges() {
        let board = Board::from_str("As7h2c").unwrap();
        let tree = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::conservative_default(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap()
        .build(Spot {
            board,
            pot: 650,
            effective_stack: 9700,
            oop_range: RangeSpec::from_str("AsAh,KsKh").unwrap(),
            ip_range: RangeSpec::from_str("QsQh,JsJh").unwrap(),
            first_player: Player::Oop,
        })
        .unwrap();
        let mut cached = RealCfrSolver::new(
            tree.clone(),
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let mut uncached = RealCfrSolver::new(
            tree,
            RangeSpec::from_str("AsAh,KsKh").unwrap(),
            RangeSpec::from_str("QsQh,JsJh").unwrap(),
        )
        .unwrap();
        let config = RealCfrConfig {
            iterations: 1,
            variant: RealCfrVariant::CfrPlus,
        };
        let cached = cached
            .run_three_phase_with_terminal_side_cache(config, 4, true, |_| {})
            .unwrap();
        let uncached = uncached
            .run_three_phase_with_terminal_side_cache(config, 4, false, |_| {})
            .unwrap();
        assert_eq!(cached.terminal_evals, uncached.terminal_evals);
        assert!((cached.root_oop_value - uncached.root_oop_value).abs() < 0.001);
        assert!((cached.root_ip_value - uncached.root_ip_value).abs() < 0.001);
    }

    #[test]
    fn terminal_phase_partitions_cover_terminal_suffix_and_balance_weights() {
        let board = Board::from_str("As7h2c").unwrap();
        let states = vec![
            PhaseState {
                node_id: 0,
                board: board.clone(),
                board_slot: 0,
                children: vec![1],
                chance_member_permutation_codes: Vec::new(),
                chance_concrete_events: 0,
                terminal_cache_indices: Vec::new(),
                terminal_cache_refs: Vec::new(),
            },
            PhaseState {
                node_id: 1,
                board: board.clone(),
                board_slot: 0,
                children: Vec::new(),
                chance_member_permutation_codes: Vec::new(),
                chance_concrete_events: 0,
                terminal_cache_indices: vec![0],
                terminal_cache_refs: terminal_cache_refs_for_test(1),
            },
            PhaseState {
                node_id: 2,
                board: board.clone(),
                board_slot: 0,
                children: Vec::new(),
                chance_member_permutation_codes: Vec::new(),
                chance_concrete_events: 0,
                terminal_cache_indices: vec![0, 1, 2],
                terminal_cache_refs: terminal_cache_refs_for_test(3),
            },
            PhaseState {
                node_id: 3,
                board,
                board_slot: 0,
                children: Vec::new(),
                chance_member_permutation_codes: Vec::new(),
                chance_concrete_events: 0,
                terminal_cache_indices: vec![0, 1],
                terminal_cache_refs: terminal_cache_refs_for_test(2),
            },
        ];

        let partitions = terminal_phase_partitions(&states, 1, 2);

        assert_eq!(partitions.first().map(|partition| partition.0), Some(1));
        assert_eq!(
            partitions.last().map(|partition| partition.1),
            Some(states.len())
        );
        for window in partitions.windows(2) {
            assert_eq!(window[0].1, window[1].0);
        }
        let weights = partitions
            .iter()
            .map(|(start, end)| {
                states[*start..*end]
                    .iter()
                    .map(terminal_phase_state_weight)
                    .sum::<usize>()
            })
            .collect::<Vec<_>>();
        assert_eq!(weights.iter().sum::<usize>(), 6);
        assert!(weights.iter().all(|weight| *weight <= 4), "{weights:?}");
    }

    fn terminal_cache_refs_for_test(len: usize) -> Vec<TerminalCacheRef> {
        (0..len)
            .map(|cache_index| TerminalCacheRef {
                cache_index,
                member_permutation_codes: vec![
                    crate::isomorphism::SuitPermutation::identity().code(),
                ],
            })
            .collect()
    }

    #[test]
    fn fast_opponent_weights_match_pairwise_fold_weights() {
        let board = Board::from_str("As7h2c").unwrap();
        let own = RangeSpec::from_str("AhAd,KsKh,QcQd,8s8h")
            .unwrap()
            .combos()
            .to_vec();
        let opponent = RangeSpec::from_str("AhAd,AcKd,KsKh,QhQs,8s8h,5c4c")
            .unwrap()
            .combos()
            .to_vec();
        let opponent_reach = vec![0.7, 0.0, 0.25, 1.0, 0.5, 0.125];
        let same = same_combo_indices(&own, &opponent);
        let slow = opponent_weights_for(&own, &opponent, &opponent_reach, &board);
        let fast = opponent_weights_for_fast(&own, &opponent, &opponent_reach, &same, &board);

        assert_eq!(slow.len(), fast.len());
        for (index, (slow, fast)) in slow.iter().zip(&fast).enumerate() {
            assert!(
                (*slow - *fast).abs() < 1e-6,
                "index={index} slow={slow} fast={fast}"
            );
        }
    }
}
