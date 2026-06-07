use crate::cards::{Board, Card};
use crate::range::{ComboWeight, RangeSpec};
use crate::terminal_cfv::{
    PreparedTerminalBoard, TerminalCfvScratch, terminal_cfv_prefix_blocker_board_targets_into,
    terminal_cfv_prefix_blocker_targets_into, terminal_cfv_sparse_board_targets_into,
    terminal_cfv_sparse_targets_into,
};
use crate::tree::{Player, PublicNodeKind, PublicTree, TerminalReason};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Debug, Clone, Copy, Default)]
struct TerminalWorkerProfile {
    worker_index: usize,
    tasks: usize,
    terminal_states: usize,
    fold_states: usize,
    sparse_tasks: usize,
    prefix_tasks: usize,
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
    terminal_cache_indices: Vec<usize>,
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
        let oop_same_ip_combo_indices = same_combo_indices(&oop_combos, &ip_combos);
        let ip_same_oop_combo_indices = same_combo_indices(&ip_combos, &oop_combos);
        let flop_board = tree.spot.board.clone();
        let turn_boards = turn_boards_from_flop(&flop_board)?;
        let river_boards = ordered_river_boards_from_flop(&flop_board)?;
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
            let oop_board_targets = prepared_board_targets(&oop_targets);
            let ip_board_targets = prepared_board_targets(&ip_targets);
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
            let combos = match player {
                Player::Oop => oop_combos.len(),
                Player::Ip => ip_combos.len(),
            };
            let board_count = board_count_for_len(node.state.board.cards().len())?;
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
            oop_same_ip_combo_indices,
            ip_same_oop_combo_indices,
        })
    }

    pub fn run(&mut self, config: RealCfrConfig) -> Result<RealCfrSummary, String> {
        self.run_with_progress(config, |_| {})
    }

    pub fn run_with_progress(
        &mut self,
        config: RealCfrConfig,
        mut progress: impl FnMut(RealCfrIterationSummary),
    ) -> Result<RealCfrSummary, String> {
        let mut root = Values::zero(self.oop_combos.len(), self.ip_combos.len());
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
            root = self.traverse(
                0,
                &board,
                &oop_reach,
                &ip_reach,
                average_weight,
                config.variant,
            )?;
            progress(RealCfrIterationSummary {
                iteration,
                terminal_evals: root.terminal_evals,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                root_oop_value: weighted_average(
                    &root.oop,
                    &self.oop_combos,
                    oop_weight,
                    ip_weight,
                ),
                root_ip_value: weighted_average(&root.ip, &self.ip_combos, ip_weight, oop_weight),
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
        Ok(RealCfrSummary {
            iterations: config.iterations,
            decision_nodes,
            action_slots,
            terminal_evals: root.terminal_evals,
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
        let oop_weight = self
            .oop_combos
            .iter()
            .map(|combo| combo.weight)
            .sum::<f32>();
        let ip_weight = self.ip_combos.iter().map(|combo| combo.weight).sum::<f32>();

        for iteration in 1..=config.iterations {
            let reach_started = std::time::Instant::now();
            self.forward_reaches_into(&states, &mut oop_reaches, &mut ip_reaches)?;
            let reach_ms = reach_started.elapsed().as_secs_f64() * 1000.0;

            let terminal_started = std::time::Instant::now();
            self.terminal_phase_into(
                &states,
                &oop_reaches,
                &ip_reaches,
                threads,
                &mut values,
                0.0,
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
        self.collect_phase_state_from(0, &self.flop_board, &mut states, &mut index_by_key)?;
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
    ) -> Result<usize, String> {
        let key = (node_id, ordered_board_key(board));
        if let Some(index) = index_by_key.get(&key) {
            return Ok(*index);
        }
        let state_index = states.len();
        index_by_key.insert(key, state_index);
        let board_slot = self.board_slot(board)?;
        let terminal_cache_indices = self.phase_state_terminal_cache_indices(node_id, board)?;
        states.push(PhaseState {
            node_id,
            board: board.clone(),
            board_slot,
            children: Vec::new(),
            terminal_cache_indices,
        });

        let node = &self.tree.nodes[node_id];
        let mut children = Vec::new();
        match &node.kind {
            PublicNodeKind::Terminal { .. } => {}
            PublicNodeKind::Chance(_) => {
                let Some(child) = node.children.first().copied() else {
                    states[state_index].children = children;
                    return Ok(state_index);
                };
                for card in board.remaining_deck() {
                    children.push(self.collect_phase_state_from(
                        child,
                        &board.push(card)?,
                        states,
                        index_by_key,
                    )?);
                }
            }
            PublicNodeKind::Decision { .. } => {
                for child in &node.children {
                    children.push(self.collect_phase_state_from(
                        *child,
                        board,
                        states,
                        index_by_key,
                    )?);
                }
            }
        }
        states[state_index].children = children;
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

    fn forward_reaches_into(
        &self,
        states: &[PhaseState],
        oop_reaches: &mut [Vec<f32>],
        ip_reaches: &mut [Vec<f32>],
    ) -> Result<(), String> {
        self.forward_reaches_for_mode_into(
            states,
            EvaluationMode::Profile,
            StrategySource::Current,
            oop_reaches,
            ip_reaches,
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
                    let chance_weight = 1.0 / state.children.len() as f32;
                    for child in &state.children {
                        add_scaled_reach(
                            &mut oop_reaches[*child],
                            &parent_oop_reach,
                            chance_weight,
                        );
                        add_scaled_reach(&mut ip_reaches[*child], &parent_ip_reach, chance_weight);
                    }
                }
                PublicNodeKind::Decision { player, actions } => {
                    let actions_len = actions.len();
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
    ) -> Result<(), String> {
        if oop_reaches.len() != states.len() || ip_reaches.len() != states.len() {
            return Err("reach scratch length does not match state count".to_string());
        }
        for reach in oop_reaches.iter_mut() {
            reach.fill(0.0);
        }
        for reach in ip_reaches.iter_mut() {
            reach.fill(0.0);
        }
        for (index, combo) in self.oop_combos.iter().enumerate() {
            oop_reaches[0][index] = combo.weight;
        }
        for (index, combo) in self.ip_combos.iter().enumerate() {
            ip_reaches[0][index] = combo.weight;
        }

        for (state_index, state) in states.iter().enumerate() {
            let node = &self.tree.nodes[state.node_id];
            match &node.kind {
                PublicNodeKind::Terminal { .. } => {}
                PublicNodeKind::Chance(_) => {
                    if state.children.is_empty() {
                        continue;
                    }
                    let chance_weight = 1.0 / state.children.len() as f32;
                    let (parent_oop, child_oop_reaches) =
                        split_reach_state_and_children(oop_reaches, state_index);
                    let (parent_ip, child_ip_reaches) =
                        split_reach_state_and_children(ip_reaches, state_index);
                    for child in &state.children {
                        add_scaled_reach(
                            child_reach_mut(child_oop_reaches, state_index, *child),
                            parent_oop,
                            chance_weight,
                        );
                        add_scaled_reach(
                            child_reach_mut(child_ip_reaches, state_index, *child),
                            parent_ip,
                            chance_weight,
                        );
                    }
                }
                PublicNodeKind::Decision { player, actions } => {
                    let actions_len = actions.len();
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
                    let (parent_oop, child_oop_reaches) =
                        split_reach_state_and_children(oop_reaches, state_index);
                    let (parent_ip, child_ip_reaches) =
                        split_reach_state_and_children(ip_reaches, state_index);
                    for action_index in 0..actions_len {
                        let child = state.children[action_index];
                        match player {
                            Player::Oop => {
                                add_reach(
                                    child_reach_mut(child_ip_reaches, state_index, child),
                                    parent_ip,
                                );
                                if mode == EvaluationMode::OopBestResponse {
                                    add_reach(
                                        child_reach_mut(child_oop_reaches, state_index, child),
                                        parent_oop,
                                    );
                                } else {
                                    add_strategy_reach(
                                        child_reach_mut(child_oop_reaches, state_index, child),
                                        parent_oop,
                                        &strategies,
                                        actions_len,
                                        action_index,
                                    );
                                }
                            }
                            Player::Ip => {
                                add_reach(
                                    child_reach_mut(child_oop_reaches, state_index, child),
                                    parent_oop,
                                );
                                if mode == EvaluationMode::IpBestResponse {
                                    add_reach(
                                        child_reach_mut(child_ip_reaches, state_index, child),
                                        parent_ip,
                                    );
                                } else {
                                    add_strategy_reach(
                                        child_reach_mut(child_ip_reaches, state_index, child),
                                        parent_ip,
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
        if values.len() != states.len() {
            return Err("terminal phase scratch length does not match state count".to_string());
        }
        let terminal_start = first_terminal_state_index(states);
        let terminal_len = states.len() - terminal_start;
        if terminal_len == 0 {
            return Ok(());
        }
        let threads = effective_worker_count(threads).min(terminal_len);
        let profile_terminal = std::env::var_os("POKEDR_REAL_CFR_TERMINAL_PROFILE").is_some();
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
        let sparse_nonzero_limit = terminal_sparse_nonzero_limit();
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
                    for cache_index in &state.terminal_cache_indices {
                        let cache = &self.terminal_cache[*cache_index];
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
                        let cfv_started = profile_terminal.then(Instant::now);
                        if oop_nonzero.len() <= sparse_nonzero_limit
                            && ip_nonzero.len() <= sparse_nonzero_limit
                        {
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
                            terminal_cfv_prefix_blocker_board_targets_into(
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
                        let accumulator_started = profile_terminal.then(Instant::now);
                        accumulator.add_board(cache, node.state.pot, &scratch);
                        if let Some(started) = accumulator_started {
                            profile.accumulator_ms += started.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                    accumulator.finish();
                    values_slot.copy_from(&accumulator.values);
                }
            }
        }
        profile.output_states = values_chunk.len();
        profile.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(profile)
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
                let decision_end = backup_run_decision_prefix_end(&self.tree, states, run);
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
                    backup_chance_state(&self.tree, states, values, state_index)?;
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

    fn traverse(
        &mut self,
        node_id: usize,
        board: &Board,
        oop_reach: &[f32],
        ip_reach: &[f32],
        average_weight: f32,
        variant: RealCfrVariant,
    ) -> Result<Values, String> {
        let node = self.tree.nodes[node_id].clone();
        match node.kind {
            PublicNodeKind::Terminal { reason } => self.terminal_values(
                board,
                node.state.pot,
                node.state.player,
                reason,
                oop_reach,
                ip_reach,
            ),
            PublicNodeKind::Chance(_) => {
                let Some(child) = node.children.first().copied() else {
                    let values = Values::zero(self.oop_combos.len(), self.ip_combos.len());
                    return Ok(values);
                };
                let mut values = Values::zero(self.oop_combos.len(), self.ip_combos.len());
                let next_cards = board.remaining_deck();
                let chance_weight = 1.0f32 / next_cards.len() as f32;
                for card in next_cards {
                    let next_board = board.push(card)?;
                    let child_values = self.traverse(
                        child,
                        &next_board,
                        oop_reach,
                        ip_reach,
                        average_weight,
                        variant,
                    )?;
                    values.add_scaled(&child_values, chance_weight);
                }
                Ok(values)
            }
            PublicNodeKind::Decision { player, actions } => {
                let actions_len = actions.len();
                let acting_combos = match player {
                    Player::Oop => self.oop_combos.len(),
                    Player::Ip => self.ip_combos.len(),
                };
                let board_slot = self.board_slot(board)?;
                let row_len = acting_combos * actions_len;
                let row_start = board_slot * row_len;
                let row_end = row_start + row_len;
                let strategies = {
                    let infoset = self.infosets[node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    debug_assert_eq!(infoset.player, player);
                    debug_assert_eq!(infoset.actions, actions_len);
                    debug_assert!(board_slot < infoset.board_count);
                    let slot_start = infoset.slots_start + row_start;
                    let slot_end = infoset.slots_start + row_end;
                    current_strategies(
                        &self.regrets[slot_start..slot_end],
                        acting_combos,
                        actions_len,
                    )
                };

                let mut action_values = Vec::with_capacity(actions_len);
                for action_index in 0..actions_len {
                    let mut next_oop = oop_reach.to_vec();
                    let mut next_ip = ip_reach.to_vec();
                    match player {
                        Player::Oop => apply_strategy_to_reach(
                            &mut next_oop,
                            &strategies,
                            actions_len,
                            action_index,
                        ),
                        Player::Ip => apply_strategy_to_reach(
                            &mut next_ip,
                            &strategies,
                            actions_len,
                            action_index,
                        ),
                    }
                    action_values.push(self.traverse(
                        node.children[action_index],
                        board,
                        &next_oop,
                        &next_ip,
                        average_weight,
                        variant,
                    )?);
                }

                let mut values = Values::zero(self.oop_combos.len(), self.ip_combos.len());
                match player {
                    Player::Oop => {
                        combine_acting_values(
                            &mut values.oop,
                            &action_values,
                            &strategies,
                            actions_len,
                            Player::Oop,
                        );
                        combine_nonacting_values(&mut values.ip, &action_values, Player::Ip);
                    }
                    Player::Ip => {
                        combine_acting_values(
                            &mut values.ip,
                            &action_values,
                            &strategies,
                            actions_len,
                            Player::Ip,
                        );
                        combine_nonacting_values(&mut values.oop, &action_values, Player::Oop);
                    }
                }
                values.terminal_evals =
                    action_values.iter().map(|value| value.terminal_evals).sum();

                let own_reach = match player {
                    Player::Oop => oop_reach,
                    Player::Ip => ip_reach,
                };
                let infoset = self.infosets[node_id]
                    .as_ref()
                    .expect("decision node must have infoset");
                let slots_start = infoset.slots_start;
                for combo in 0..acting_combos {
                    let node_value = match player {
                        Player::Oop => values.oop[combo],
                        Player::Ip => values.ip[combo],
                    };
                    for action_index in 0..actions_len {
                        let action_value = match player {
                            Player::Oop => action_values[action_index].oop[combo],
                            Player::Ip => action_values[action_index].ip[combo],
                        };
                        let local_slot = combo * actions_len + action_index;
                        let slot = slots_start + row_start + local_slot;
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
                Ok(values)
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

    fn terminal_values(
        &self,
        board: &Board,
        pot: u32,
        folding_player: Player,
        reason: TerminalReason,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) -> Result<Values, String> {
        match reason {
            TerminalReason::Fold => {
                Ok(self.fold_values(board, pot, folding_player, oop_reach, ip_reach))
            }
            TerminalReason::Showdown | TerminalReason::AllIn => {
                self.showdown_values(board, pot, oop_reach, ip_reach)
            }
        }
    }

    fn fold_values(
        &self,
        board: &Board,
        pot: u32,
        folding_player: Player,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) -> Values {
        let mut values = Values::zero(self.oop_combos.len(), self.ip_combos.len());
        self.fold_values_into(&mut values, board, pot, folding_player, oop_reach, ip_reach);
        values
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

    fn showdown_values(
        &self,
        board: &Board,
        pot: u32,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) -> Result<Values, String> {
        let terminal_boards = terminal_boards(board)?;
        let mut values = Values::zero(self.oop_combos.len(), self.ip_combos.len());
        let mut oop_counts = vec![0.0f32; self.oop_combos.len()];
        let mut ip_counts = vec![0.0f32; self.ip_combos.len()];
        let sparse_nonzero_limit = terminal_sparse_nonzero_limit();
        for final_board in &terminal_boards {
            let cache_index = self
                .terminal_cache_index_by_key
                .get(&unordered_board_key(final_board))
                .copied()
                .ok_or_else(|| "terminal board is outside the solver board cache".to_string())?;
            let cache = &self.terminal_cache[cache_index];
            let mut scratch = TerminalCfvScratch::new(&cache.prepared);
            let prepared_combos = cache.prepared.combos().len();
            let mut oop_live = vec![0.0f32; prepared_combos];
            let mut ip_live = vec![0.0f32; prepared_combos];
            let mut oop_nonzero = Vec::new();
            let mut ip_nonzero = Vec::new();
            reach_on_prepared_board_sparse_into(
                &cache.oop_combo_indices,
                oop_reach,
                &mut oop_live,
                &mut oop_nonzero,
            );
            reach_on_prepared_board_sparse_into(
                &cache.ip_combo_indices,
                ip_reach,
                &mut ip_live,
                &mut ip_nonzero,
            );
            if oop_nonzero.len() <= sparse_nonzero_limit && ip_nonzero.len() <= sparse_nonzero_limit
            {
                terminal_cfv_sparse_targets_into(
                    &cache.prepared,
                    &oop_live,
                    &ip_live,
                    &oop_nonzero,
                    &ip_nonzero,
                    &cache.oop_combo_indices,
                    &cache.ip_combo_indices,
                    &mut scratch,
                )?;
            } else {
                terminal_cfv_prefix_blocker_targets_into(
                    &cache.prepared,
                    &oop_live,
                    &ip_live,
                    &cache.oop_combo_indices,
                    &cache.ip_combo_indices,
                    &mut scratch,
                )?;
            }
            for index in 0..self.oop_combos.len() {
                if let Some(board_index) = cache.oop_combo_indices[index] {
                    values.oop[index] += scratch.hero_values()[board_index] * pot as f32;
                    oop_counts[index] += 1.0;
                }
            }
            for index in 0..self.ip_combos.len() {
                if let Some(board_index) = cache.ip_combo_indices[index] {
                    values.ip[index] += scratch.villain_values()[board_index] * pot as f32;
                    ip_counts[index] += 1.0;
                }
            }
        }
        for (value, count) in values.oop.iter_mut().zip(oop_counts) {
            if count > 0.0 {
                *value /= count;
            }
        }
        for (value, count) in values.ip.iter_mut().zip(ip_counts) {
            if count > 0.0 {
                *value /= count;
            }
        }
        values.terminal_evals = terminal_boards.len();
        Ok(values)
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

    fn add_board(&mut self, cache: &TerminalEvalCache, pot: u32, scratch: &TerminalCfvScratch) {
        let pot = pot as f32;
        for target in &cache.oop_targets {
            let index = target.range_index;
            self.values.oop[index] += scratch.hero_values()[target.board_index as usize] * pot;
            self.oop_counts[index] += 1.0;
        }
        for target in &cache.ip_targets {
            let index = target.range_index;
            self.values.ip[index] += scratch.villain_values()[target.board_index as usize] * pot;
            self.ip_counts[index] += 1.0;
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
    let mut strategies = vec![0.0; regrets.len()];
    for combo in 0..combos {
        let row = &regrets[combo * actions..(combo + 1) * actions];
        let positive_sum = row
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .sum::<f32>();
        if positive_sum > 0.0 {
            for action in 0..actions {
                strategies[combo * actions + action] = row[action].max(0.0) / positive_sum;
            }
        } else {
            let uniform = 1.0 / actions as f32;
            for action in 0..actions {
                strategies[combo * actions + action] = uniform;
            }
        }
    }
    strategies
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
    let mut strategies = vec![0.0; strategy_sum.len()];
    for combo in 0..combos {
        let row = &strategy_sum[combo * actions..(combo + 1) * actions];
        let total = row.iter().sum::<f32>();
        if total > 0.0 {
            for action in 0..actions {
                strategies[combo * actions + action] = row[action] / total;
            }
        } else {
            let uniform = 1.0 / actions as f32;
            for action in 0..actions {
                strategies[combo * actions + action] = uniform;
            }
        }
    }
    strategies
}

fn apply_strategy_to_reach(reach: &mut [f32], strategies: &[f32], actions: usize, action: usize) {
    for (combo, value) in reach.iter_mut().enumerate() {
        *value *= strategies[combo * actions + action];
    }
}

fn add_reach(out: &mut [f32], input: &[f32]) {
    for (out, input) in out.iter_mut().zip(input) {
        *out += *input;
    }
}

fn add_scaled_reach(out: &mut [f32], input: &[f32], scale: f32) {
    for (out, input) in out.iter_mut().zip(input) {
        *out += *input * scale;
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
    states: &[PhaseState],
    run: &BackupRun,
) -> usize {
    states[run.start..run.end]
        .iter()
        .position(|state| {
            !matches!(
                tree.nodes[state.node_id].kind,
                PublicNodeKind::Decision { .. }
            )
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
) -> Result<(), String> {
    let state = &states[state_index];
    match &tree.nodes[state.node_id].kind {
        PublicNodeKind::Chance(_) => {
            let (state_value, child_values) = split_state_and_children(values, state_index);
            state_value.reset();
            for child in &state.children {
                state_value.add_scaled(child_value(child_values, state_index, *child), 1.0);
            }
            Ok(())
        }
        PublicNodeKind::Terminal { .. } => Ok(()),
        PublicNodeKind::Decision { .. } => {
            Err("decision state reached chance backup path".to_string())
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
    let infoset = infosets[state.node_id]
        .as_ref()
        .expect("decision node must have infoset");
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

fn reach_on_prepared_board_sparse_into(
    combo_indices: &[Option<usize>],
    reach: &[f32],
    out: &mut [f32],
    nonzero: &mut Vec<u16>,
) {
    out.fill(0.0);
    nonzero.clear();
    for (index, reach) in combo_indices.iter().zip(reach) {
        if *reach == 0.0 {
            continue;
        }
        if let Some(index) = *index {
            if out[index] == 0.0 {
                nonzero.push(index as u16);
            }
            out[index] += *reach;
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
    state.terminal_cache_indices.len().max(1)
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
        "real_cfr_terminal_profile tasks={} accounted_tasks={} threads={} sparse_tasks={} prefix_tasks={} avg_oop_nonzero={:.2} avg_ip_nonzero={:.2} min_worker_ms={:.3} max_worker_ms={:.3}",
        total_tasks,
        tasks,
        threads,
        sparse_tasks,
        prefix_tasks,
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
            "real_cfr_terminal_worker worker={} terminal_states={} fold_states={} tasks={} sparse={} prefix={} avg_oop_nonzero={:.2} avg_ip_nonzero={:.2} max_oop_nonzero={} max_ip_nonzero={} output_states={} board_expand_ms={:.3} fold_ms={:.3} reach_map_ms={:.3} cfv_ms={:.3} accumulator_ms={:.3} elapsed_ms={:.3}",
            profile.worker_index,
            profile.terminal_states,
            profile.fold_states,
            profile.tasks,
            profile.sparse_tasks,
            profile.prefix_tasks,
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

fn board_count_for_len(cards: usize) -> Result<usize, String> {
    match cards {
        3 => Ok(1),
        4 => Ok(49),
        5 => Ok(49 * 48),
        other => Err(format!("invalid public board length {other}")),
    }
}

fn turn_boards_from_flop(flop: &Board) -> Result<Vec<Board>, String> {
    if flop.cards().len() != 3 {
        return Err("real CFR solver must start from a flop board".to_string());
    }
    let deck = flop.remaining_deck();
    let mut boards = Vec::with_capacity(deck.len());
    for card in deck {
        boards.push(flop.push(card)?);
    }
    Ok(boards)
}

fn ordered_river_boards_from_flop(flop: &Board) -> Result<Vec<Board>, String> {
    if flop.cards().len() != 3 {
        return Err("real CFR solver must start from a flop board".to_string());
    }
    let deck = flop.remaining_deck();
    let mut boards = Vec::with_capacity(deck.len() * (deck.len() - 1));
    for turn in 0..deck.len() {
        for river in 0..deck.len() {
            if turn == river {
                continue;
            }
            boards.push(flop.push(deck[turn])?.push(deck[river])?);
        }
    }
    Ok(boards)
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
    fn real_cfr_allocates_exact_board_indexed_state_on_small_ranges() {
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
        let action_slots = solver
            .infosets
            .iter()
            .filter_map(Option::as_ref)
            .map(|infoset| infoset.slots_len)
            .sum::<usize>();
        assert_eq!(action_slots, 5_981_008);
        assert_eq!(solver.turn_index_by_key.len(), 49);
        assert_eq!(solver.river_index_by_key.len(), 49 * 48);
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
    fn terminal_phase_partitions_cover_terminal_suffix_and_balance_weights() {
        let board = Board::from_str("As7h2c").unwrap();
        let states = vec![
            PhaseState {
                node_id: 0,
                board: board.clone(),
                board_slot: 0,
                children: vec![1],
                terminal_cache_indices: Vec::new(),
            },
            PhaseState {
                node_id: 1,
                board: board.clone(),
                board_slot: 0,
                children: Vec::new(),
                terminal_cache_indices: vec![0],
            },
            PhaseState {
                node_id: 2,
                board: board.clone(),
                board_slot: 0,
                children: Vec::new(),
                terminal_cache_indices: vec![0, 1, 2],
            },
            PhaseState {
                node_id: 3,
                board,
                board_slot: 0,
                children: Vec::new(),
                terminal_cache_indices: vec![0, 1],
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
