use crate::cards::Board;
use crate::range::{ComboWeight, RangeSpec};
use crate::terminal_cfv::{
    PreparedTerminalBoard, TerminalCfvScratch, terminal_cfv_prefix_blocker_into,
};
use crate::tree::{Player, PublicNodeKind, PublicTree, TerminalReason};
use std::collections::BTreeMap;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealCfrConfig {
    pub iterations: u32,
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

#[derive(Debug, Clone)]
pub struct RealCfrSolver {
    tree: PublicTree,
    oop_combos: Vec<ComboWeight>,
    ip_combos: Vec<ComboWeight>,
    infosets: Vec<Option<RealInfoset>>,
    flop_board: Board,
    turn_index_by_key: BTreeMap<u64, usize>,
    river_index_by_key: BTreeMap<u64, usize>,
    terminal_cache_index_by_key: BTreeMap<u64, usize>,
    terminal_cache: Vec<TerminalEvalCache>,
}

#[derive(Debug, Clone)]
struct RealInfoset {
    player: Player,
    board_count: usize,
    actions: usize,
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone)]
struct Values {
    oop: Vec<f32>,
    ip: Vec<f32>,
    terminal_evals: usize,
}

#[derive(Debug, Clone)]
struct TerminalEvalCache {
    prepared: PreparedTerminalBoard,
    oop_combo_indices: Vec<Option<usize>>,
    ip_combo_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct TerminalBoardTask {
    terminal_node: usize,
    board: Board,
    cache_index: usize,
}

#[derive(Debug, Clone)]
struct PhaseState {
    node_id: usize,
    board: Board,
    children: Vec<usize>,
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
            terminal_cache_index_by_key.insert(unordered_board_key(board), terminal_cache.len());
            terminal_cache.push(TerminalEvalCache {
                oop_combo_indices: prepared_combo_indices(&prepared, &oop_combos),
                ip_combo_indices: prepared_combo_indices(&prepared, &ip_combos),
                prepared,
            });
        }
        let mut infosets = vec![None; tree.nodes.len()];
        for node in &tree.nodes {
            let PublicNodeKind::Decision { player, actions } = &node.kind else {
                continue;
            };
            let combos = match player {
                Player::Oop => oop_combos.len(),
                Player::Ip => ip_combos.len(),
            };
            let board_count = board_count_for_len(node.state.board.cards().len())?;
            infosets[node.id] = Some(RealInfoset {
                player: *player,
                board_count,
                actions: actions.len(),
                regrets: vec![0.0; board_count * combos * actions.len()],
                strategy_sum: vec![0.0; board_count * combos * actions.len()],
            });
        }
        Ok(Self {
            tree,
            oop_combos,
            ip_combos,
            infosets,
            flop_board,
            turn_index_by_key,
            river_index_by_key,
            terminal_cache_index_by_key,
            terminal_cache,
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
            root = self.traverse(0, &board, &oop_reach, &ip_reach)?;
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
            .map(|infoset| infoset.regrets.len())
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
        let mut root = Values::zero(self.oop_combos.len(), self.ip_combos.len());
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
            let (oop_reaches, ip_reaches) = self.forward_reaches(&states)?;
            let reach_ms = reach_started.elapsed().as_secs_f64() * 1000.0;

            let terminal_started = std::time::Instant::now();
            let mut values = self.terminal_phase(&states, &oop_reaches, &ip_reaches, threads)?;
            let terminal_ms = terminal_started.elapsed().as_secs_f64() * 1000.0;

            let backup_started = std::time::Instant::now();
            last_terminal_evals =
                self.backup_phase(&states, &oop_reaches, &ip_reaches, &mut values)?;
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
            .map(|infoset| infoset.regrets.len())
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
        let started = std::time::Instant::now();
        let tasks = self.collect_terminal_board_tasks()?;
        let threads = if threads == 0 {
            thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        }
        .max(1)
        .min(tasks.len().max(1));
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
        let checksum = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for thread_index in 0..threads {
                let tasks = &tasks;
                let oop_reach = &oop_reach;
                let ip_reach = &ip_reach;
                handles.push(scope.spawn(move || -> Result<f64, String> {
                    let chunk = tasks.len().div_ceil(threads);
                    let start = thread_index * chunk;
                    let end = (start + chunk).min(tasks.len());
                    let mut checksum = 0.0f64;
                    let scratch_source = self
                        .terminal_cache
                        .first()
                        .ok_or_else(|| "terminal board cache is empty".to_string())?;
                    let combos = scratch_source.prepared.combos().len();
                    let mut scratch = TerminalCfvScratch::new(&scratch_source.prepared);
                    let mut oop_live = vec![0.0f32; combos];
                    let mut ip_live = vec![0.0f32; combos];
                    for task in &tasks[start..end] {
                        let cache = &self.terminal_cache[task.cache_index];
                        reach_on_prepared_board_into(
                            &cache.oop_combo_indices,
                            oop_reach,
                            &mut oop_live,
                        );
                        reach_on_prepared_board_into(
                            &cache.ip_combo_indices,
                            ip_reach,
                            &mut ip_live,
                        );
                        terminal_cfv_prefix_blocker_into(
                            &cache.prepared,
                            &oop_live,
                            &ip_live,
                            &mut scratch,
                        )?;
                        checksum += terminal_task_checksum(task, &cache.prepared, &scratch);
                    }
                    Ok(checksum)
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| "terminal board phase worker panicked".to_string())?
                })
                .try_fold(0.0f64, |total, checksum| {
                    checksum.map(|value| total + value)
                })
        })?;
        Ok(TerminalBoardPhaseSummary {
            terminal_evals: tasks.len(),
            threads,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            checksum,
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
        Ok(states)
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
        states.push(PhaseState {
            node_id,
            board: board.clone(),
            children: Vec::new(),
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

    fn forward_reaches(
        &self,
        states: &[PhaseState],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        self.forward_reaches_for_mode(states, EvaluationMode::Profile, StrategySource::Current)
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
                    let board_slot = self.board_slot(&state.board)?;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let infoset = self.infosets[state.node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    let strategies = match strategy_source {
                        StrategySource::Current => current_strategies(
                            &infoset.regrets[row_start..row_end],
                            acting_combos,
                            actions_len,
                        ),
                        StrategySource::Average => average_strategies(
                            &infoset.strategy_sum[row_start..row_end],
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
        let mut values =
            vec![Values::zero(self.oop_combos.len(), self.ip_combos.len()); states.len()];
        let terminals = states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| {
                matches!(
                    self.tree.nodes[state.node_id].kind,
                    PublicNodeKind::Terminal { .. }
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if terminals.is_empty() {
            return Ok(values);
        }
        let threads = if threads == 0 {
            thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        }
        .max(1)
        .min(terminals.len());
        let outputs = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for thread_index in 0..threads {
                let chunk = terminals.len().div_ceil(threads);
                let start = thread_index * chunk;
                let end = (start + chunk).min(terminals.len());
                let terminals = &terminals;
                handles.push(
                    scope.spawn(move || -> Result<Vec<(usize, Values)>, String> {
                        let mut out = Vec::with_capacity(end.saturating_sub(start));
                        for state_index in &terminals[start..end] {
                            let state = &states[*state_index];
                            let node = &self.tree.nodes[state.node_id];
                            let PublicNodeKind::Terminal { reason } = node.kind else {
                                continue;
                            };
                            out.push((
                                *state_index,
                                self.terminal_values(
                                    &state.board,
                                    node.state.pot,
                                    node.state.player,
                                    reason,
                                    &oop_reaches[*state_index],
                                    &ip_reaches[*state_index],
                                )?,
                            ));
                        }
                        Ok(out)
                    }),
                );
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| "terminal phase worker panicked".to_string())?
                })
                .try_fold(Vec::new(), |mut all, worker| {
                    worker.map(|mut values| {
                        all.append(&mut values);
                        all
                    })
                })
        })?;
        for (state_index, value) in outputs {
            values[state_index] = value;
        }
        Ok(values)
    }

    fn backup_phase(
        &mut self,
        states: &[PhaseState],
        oop_reaches: &[Vec<f32>],
        ip_reaches: &[Vec<f32>],
        values: &mut [Values],
    ) -> Result<usize, String> {
        for state_index in (0..states.len()).rev() {
            let state = &states[state_index];
            let node = self.tree.nodes[state.node_id].clone();
            match node.kind {
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
                    let board_slot = self.board_slot(&state.board)?;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let strategies = {
                        let infoset = self.infosets[state.node_id]
                            .as_ref()
                            .expect("decision node must have infoset");
                        current_strategies(
                            &infoset.regrets[row_start..row_end],
                            acting_combos,
                            actions_len,
                        )
                    };
                    let action_values = state
                        .children
                        .iter()
                        .map(|child| values[*child].clone())
                        .collect::<Vec<_>>();
                    let mut state_values =
                        Values::zero(self.oop_combos.len(), self.ip_combos.len());
                    match player {
                        Player::Oop => {
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
                        Player::Ip => {
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

                    let own_reach = match player {
                        Player::Oop => &oop_reaches[state_index],
                        Player::Ip => &ip_reaches[state_index],
                    };
                    let infoset = self.infosets[state.node_id]
                        .as_mut()
                        .expect("decision node must have infoset");
                    for combo in 0..acting_combos {
                        let node_value = match player {
                            Player::Oop => state_values.oop[combo],
                            Player::Ip => state_values.ip[combo],
                        };
                        for action_index in 0..actions_len {
                            let action_value = match player {
                                Player::Oop => action_values[action_index].oop[combo],
                                Player::Ip => action_values[action_index].ip[combo],
                            };
                            let local_slot = combo * actions_len + action_index;
                            let slot = row_start + local_slot;
                            infoset.regrets[slot] =
                                (infoset.regrets[slot] + action_value - node_value).max(0.0);
                            infoset.strategy_sum[slot] += own_reach[combo] * strategies[local_slot];
                        }
                    }
                    values[state_index] = state_values;
                }
            }
        }
        Ok(values[0].terminal_evals)
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
                    let board_slot = self.board_slot(&state.board)?;
                    let row_len = acting_combos * actions_len;
                    let row_start = board_slot * row_len;
                    let row_end = row_start + row_len;
                    let infoset = self.infosets[state.node_id]
                        .as_ref()
                        .expect("decision node must have infoset");
                    let strategies = average_strategies(
                        &infoset.strategy_sum[row_start..row_end],
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
                    let child_values = self.traverse(child, &next_board, oop_reach, ip_reach)?;
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
                    current_strategies(
                        &infoset.regrets[row_start..row_end],
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
                    .as_mut()
                    .expect("decision node must have infoset");
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
                        let slot = row_start + local_slot;
                        infoset.regrets[slot] =
                            (infoset.regrets[slot] + action_value - node_value).max(0.0);
                        infoset.strategy_sum[slot] += own_reach[combo] * strategies[local_slot];
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
        let pot = pot as f32;
        let oop_opp = opponent_weights_for(&self.oop_combos, &self.ip_combos, ip_reach, board);
        let ip_opp = opponent_weights_for(&self.ip_combos, &self.oop_combos, oop_reach, board);
        for (index, value) in values.oop.iter_mut().enumerate() {
            *value = if folding_player == Player::Oop {
                -pot
            } else {
                pot
            } * oop_opp[index];
        }
        for (index, value) in values.ip.iter_mut().enumerate() {
            *value = if folding_player == Player::Ip {
                -pot
            } else {
                pot
            } * ip_opp[index];
        }
        values
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
        for final_board in &terminal_boards {
            let cache_index = self
                .terminal_cache_index_by_key
                .get(&unordered_board_key(final_board))
                .copied()
                .ok_or_else(|| "terminal board is outside the solver board cache".to_string())?;
            let cache = &self.terminal_cache[cache_index];
            let mut scratch = TerminalCfvScratch::new(&cache.prepared);
            let prepared_combos = cache.prepared.combos().len();
            let oop_live =
                reach_on_prepared_board(&cache.oop_combo_indices, oop_reach, prepared_combos);
            let ip_live =
                reach_on_prepared_board(&cache.ip_combo_indices, ip_reach, prepared_combos);
            terminal_cfv_prefix_blocker_into(&cache.prepared, &oop_live, &ip_live, &mut scratch)?;
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

fn prepared_combo_indices(
    prepared: &PreparedTerminalBoard,
    combos: &[ComboWeight],
) -> Vec<Option<usize>> {
    combos
        .iter()
        .map(|combo| prepared.combo_index(combo.first, combo.second))
        .collect()
}

fn reach_on_prepared_board(
    combo_indices: &[Option<usize>],
    reach: &[f32],
    prepared_combos: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; prepared_combos];
    reach_on_prepared_board_into(combo_indices, reach, &mut out);
    out
}

fn reach_on_prepared_board_into(combo_indices: &[Option<usize>], reach: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for (index, reach) in combo_indices.iter().zip(reach) {
        if *reach == 0.0 {
            continue;
        }
        if let Some(index) = *index {
            out[index] += *reach;
        }
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
            .map(|infoset| infoset.regrets.len())
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
        let summary = solver.run(RealCfrConfig { iterations: 1 }).unwrap();
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
        let recursive = recursive.run(RealCfrConfig { iterations: 1 }).unwrap();
        let phased = phased
            .run_three_phase(RealCfrConfig { iterations: 1 }, 4, |_| {})
            .unwrap();
        assert_eq!(recursive.terminal_evals, phased.terminal_evals);
        assert!((recursive.root_oop_value - phased.root_oop_value).abs() < 0.001);
        assert!((recursive.root_ip_value - phased.root_ip_value).abs() < 0.001);
    }
}
