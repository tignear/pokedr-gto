pub use pokedr_core::{dense_cfr, postflop, postflop_dense, range};

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, VecDeque},
    hash::{Hash, Hasher},
    rc::Rc,
    time::Instant,
};

use pokedr_core::{
    cards::{Board, Card as PokedrCard, Rank as PokedrRank, Suit as PokedrSuit, evaluate},
    dense_cfr::gpu::{
        GpuCfrError, GpuDenseCfrBackend, GpuFinalBoard, GpuPrivateCombo, GpuPublicTreeNode,
        GpuRootTerminalValues, GpuShowdownTask,
    },
    dense_cfr::{CfrVariant, DenseCfrIteration, DenseCfrState},
    postflop::{
        ActionCandidate, ActionSetConfig, Player, PlayerAction, PublicNodeKind, PublicState,
        Street, SubgameTree, SubgameTreeConfig, TerminalKind,
    },
    postflop_dense::PostflopDenseLayout,
    range::{COMBO_COUNT, Combo, ComboIndexer},
};
use rs_poker::{
    arena::{
        Agent, Historian,
        action::Action,
        action::AgentAction,
        game_state::{GameState, Round},
        historian::{HistorianError, HistoryRecord, VecHistorian},
    },
    core::{Card as RsCard, Suit as RsSuit, Value as RsValue},
};

pub struct PokedrAgent {
    config: PokedrAgentConfig,
    history: SharedHistory,
    shared_plan: SharedPostflopPlan,
    postflop_plan: Option<PostflopPlan>,
}

#[derive(Debug, Clone)]
pub struct PokedrAgentConfig {
    pub cfr_iterations: usize,
    pub cfr_variant: CfrVariant,
    pub action_set: ActionSetConfig,
    pub max_raises_per_street: u8,
    pub max_showdown_runouts: usize,
}

impl Default for PokedrAgentConfig {
    fn default() -> Self {
        Self {
            cfr_iterations: 8,
            cfr_variant: CfrVariant::dcfr_schedule_default(),
            action_set: ActionSetConfig {
                max_aggressive_actions: 4,
                flop_bet_fractions: vec![0.5, 1.0, 1.5],
                turn_bet_fractions: vec![0.5, 1.0, 1.5],
                river_bet_fractions: vec![0.5, 1.0, 1.5],
                raise_fractions: vec![0.5, 1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMode {
    Gpu,
    CpuFallback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchSummary {
    pub hands: usize,
    pub hero_net: f32,
    pub villain_net: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandTrace {
    pub hand_index: usize,
    pub hero_cards: String,
    pub villain_cards: String,
    pub board: String,
    pub hero_net: f32,
    pub villain_net: f32,
    pub actions: Vec<String>,
    pub awards: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedFlopSolveSummary {
    pub board: String,
    pub iterations: usize,
    pub decisions: usize,
    pub chance: usize,
    pub terminals: usize,
    pub public_infosets: usize,
    pub private_infosets: usize,
    pub max_actions: usize,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedFlopCompactSmokeSummary {
    pub public_infosets: usize,
    pub public_actions: usize,
    pub combos: usize,
    pub compact_action_slots: usize,
    pub chunks: usize,
    pub largest_chunk_slots: usize,
    pub compact_context_action_slots: usize,
    pub uncovered_reach_tiles: usize,
    pub reach_tiles_requiring_split: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedFlopMetricRow {
    pub board: String,
    pub iterations: usize,
    pub elapsed_secs: f32,
    pub root_strategy_l1_delta: Option<f32>,
    pub root_action_probabilities: Vec<f32>,
    pub root_exploitability: Option<f32>,
    pub current_root_exploitability: Option<f32>,
    pub cpu_root_exploitability: Option<f32>,
    pub cpu_gpu_root_exploitability_delta: Option<f32>,
    pub hero_root_br_value: Option<f32>,
    pub villain_root_br_value: Option<f32>,
    pub hero_root_profile_value: Option<f32>,
    pub villain_root_profile_value: Option<f32>,
    pub hero_root_br_improvement: Option<f32>,
    pub villain_root_br_improvement: Option<f32>,
    pub cpu_hero_root_br_value: Option<f32>,
    pub cpu_villain_root_br_value: Option<f32>,
    pub cpu_hero_root_profile_value: Option<f32>,
    pub cpu_villain_root_profile_value: Option<f32>,
    pub cpu_hero_root_br_improvement: Option<f32>,
    pub cpu_villain_root_br_improvement: Option<f32>,
    pub current_hero_root_br_value: Option<f32>,
    pub current_villain_root_br_value: Option<f32>,
    pub current_hero_root_profile_value: Option<f32>,
    pub current_villain_root_profile_value: Option<f32>,
    pub current_hero_root_br_improvement: Option<f32>,
    pub current_villain_root_br_improvement: Option<f32>,
    pub root_br_gap: Option<f32>,
    pub local_br_gap: Option<f32>,
    pub recursive_root_br_gap: Option<f32>,
    pub recursive_local_br_gap: Option<f32>,
    pub local_gap_detail: Option<LocalGapDetail>,
    pub recursive_local_gap_detail: Option<LocalGapDetail>,
    pub positive_regret_mass: f32,
    pub negative_regret_mass: f32,
    pub negative_regret_count: usize,
    pub deep_negative_regret_count: usize,
    pub rbp_private_action_slots: usize,
    pub rbp_prunable_private_action_slots: usize,
    pub rbp_public_action_edges: usize,
    pub rbp_fully_prunable_public_action_edges: usize,
    pub rbp_public_edge_subtree_nodes: usize,
    pub rbp_fully_prunable_public_edge_subtree_nodes: usize,
    pub rbp_public_edge_terminal_nodes: usize,
    pub rbp_fully_prunable_public_edge_terminal_nodes: usize,
    pub rbp_terminal_combo_lanes: usize,
    pub rbp_active_terminal_combo_lanes: usize,
    pub rbp_showdown_combo_lanes: usize,
    pub rbp_active_showdown_combo_lanes: usize,
    pub illegal_strategy_mass: f32,
    pub current_strategy_norm_error: f32,
    pub average_strategy_norm_error: f32,
    pub finite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedFlopMetricOptions {
    pub br_metrics: bool,
    pub br_on_root_delta: Option<f32>,
    pub br_every_iterations: Option<usize>,
    pub current_br_metrics: bool,
    pub diagnostics: bool,
    pub local_gaps: bool,
}

impl FixedFlopMetricOptions {
    pub fn from_environment() -> Self {
        Self {
            br_metrics: std::env::var_os("POKEDR_METRIC_BR").is_some(),
            br_on_root_delta: std::env::var("POKEDR_METRIC_BR_ON_ROOT_DELTA")
                .ok()
                .and_then(|value| value.parse().ok()),
            br_every_iterations: std::env::var("POKEDR_METRIC_BR_EVERY")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|value| *value > 0),
            current_br_metrics: std::env::var_os("POKEDR_METRIC_CURRENT_BR").is_some(),
            diagnostics: std::env::var_os("POKEDR_METRIC_DIAGNOSTICS").is_some(),
            local_gaps: std::env::var_os("POKEDR_METRIC_LOCAL_GAPS").is_some(),
        }
    }
}

impl Default for FixedFlopMetricOptions {
    fn default() -> Self {
        Self::from_environment()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalGapDetail {
    pub gap: f32,
    pub weighted_gap: f32,
    pub reach_weight: f32,
    pub public_infoset: usize,
    pub node_index: usize,
    pub player: Player,
    pub combo_index: usize,
    pub combo: String,
    pub action_values: Vec<f32>,
    pub average_strategy: Vec<f32>,
    pub current_strategy: Vec<f32>,
    pub regrets: Vec<f32>,
    pub strategy_sum: Vec<f32>,
    pub actions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedFlopTreeDump {
    pub metrics: Option<FixedFlopTreeMetrics>,
    pub nodes: Vec<TreeNodeRecord>,
    pub actions: Vec<TreeActionRecord>,
    pub solver_nodes: Vec<SolverNodeRecord>,
    pub solver_combos: Vec<SolverComboRecord>,
    pub root_br_combos: Vec<RootBrComboRecord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedFlopTreeMetrics {
    pub iterations: usize,
    pub root_exploitability: Option<f32>,
    pub current_root_exploitability: Option<f32>,
    pub hero_root_br_value: Option<f32>,
    pub villain_root_br_value: Option<f32>,
    pub hero_root_profile_value: Option<f32>,
    pub villain_root_profile_value: Option<f32>,
    pub hero_root_br_improvement: Option<f32>,
    pub villain_root_br_improvement: Option<f32>,
    pub current_hero_root_br_value: Option<f32>,
    pub current_villain_root_br_value: Option<f32>,
    pub current_hero_root_profile_value: Option<f32>,
    pub current_villain_root_profile_value: Option<f32>,
    pub current_hero_root_br_improvement: Option<f32>,
    pub current_villain_root_br_improvement: Option<f32>,
    pub recursive_root_br_gap: Option<f32>,
    pub current_recursive_root_br_gap: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeNodeRecord {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub kind: String,
    pub infoset: Option<usize>,
    pub path: String,
    pub street: Option<String>,
    pub board: Option<String>,
    pub acting_player: Option<String>,
    pub pot: Option<u32>,
    pub to_call: Option<u32>,
    pub hero_invested: Option<u32>,
    pub villain_invested: Option<u32>,
    pub terminal_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeActionRecord {
    pub node_id: usize,
    pub action_index: usize,
    pub child_id: usize,
    pub action: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverNodeRecord {
    pub node_id: usize,
    pub infoset: usize,
    pub iterations: usize,
    pub acting_player: String,
    pub action_count: usize,
    pub legal_combo_count: usize,
    pub value_reach_sum: f32,
    pub avg_strategy_weight_sum: f32,
    pub current_strategy_weight_sum: f32,
    pub avg_strategy: Vec<f32>,
    pub current_strategy: Vec<f32>,
    pub avg_action_ev: Option<Vec<f32>>,
    pub current_action_ev: Option<Vec<f32>>,
    pub avg_policy_ev: Option<f32>,
    pub current_policy_ev: Option<f32>,
    pub avg_gap: Option<f32>,
    pub current_gap: Option<f32>,
    pub avg_recursive_action_ev: Option<Vec<f32>>,
    pub current_recursive_action_ev: Option<Vec<f32>>,
    pub avg_recursive_policy_ev: Option<f32>,
    pub current_recursive_policy_ev: Option<f32>,
    pub avg_recursive_gap: Option<f32>,
    pub current_recursive_gap: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverComboRecord {
    pub node_id: usize,
    pub combo_index: usize,
    pub combo: String,
    pub reach: f32,
    pub weighted_gap: f32,
    pub recursive_weighted_gap: f32,
    pub current_recursive_weighted_gap: f32,
    pub avg_strategy_weight: f32,
    pub current_strategy_weight: f32,
    pub avg_action_values: Option<Vec<f32>>,
    pub current_action_values: Option<Vec<f32>>,
    pub avg_recursive_action_values: Option<Vec<f32>>,
    pub current_recursive_action_values: Option<Vec<f32>>,
    pub avg_strategy: Vec<f32>,
    pub current_strategy: Vec<f32>,
    pub regrets: Vec<f32>,
    pub strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootBrComboRecord {
    pub profile: String,
    pub player: String,
    pub combo_index: usize,
    pub combo: String,
    pub root_weight: f32,
    pub opponent_nonblocking_weight: f32,
    pub root_value: f32,
    pub contribution: f32,
    pub contribution_bb100: f32,
}

impl Default for PokedrAgent {
    fn default() -> Self {
        Self::new(PokedrAgentConfig::default())
    }
}

impl Clone for PokedrAgent {
    fn clone(&self) -> Self {
        Self::new(self.config.clone())
    }
}

impl PokedrAgent {
    pub fn new(config: PokedrAgentConfig) -> Self {
        Self {
            config,
            history: SharedHistory::default(),
            shared_plan: SharedPostflopPlan::default(),
            postflop_plan: None,
        }
    }

    fn choose_action(&mut self, game_state: &GameState) -> AgentAction {
        match game_state.round {
            Round::Preflop => {
                self.postflop_plan = None;
                self.preflop_action(game_state)
            }
            Round::Flop | Round::Turn | Round::River => self.postflop_action(game_state),
            _ => AgentAction::Call,
        }
    }

    fn preflop_action(&self, game_state: &GameState) -> AgentAction {
        let to_call = amount_to_call(game_state);
        let class = game_state
            .hands
            .get(game_state.to_act_idx())
            .map(classify_preflop_hand)
            .unwrap_or(PreflopClass::Trash);
        let current_bet = game_state.current_round_bet();
        let big_blind = game_state.big_blind.max(1.0);
        let unopened_blind_spot = current_bet <= big_blind && to_call <= big_blind;
        if unopened_blind_spot {
            if class.three_bets() && can_raise(game_state) {
                return self.preflop_raise_to(game_state, big_blind * 3.0);
            }
            if class.calls_open() || to_call <= 0.0 {
                return AgentAction::Call;
            }
            return AgentAction::Fold;
        }
        if to_call <= 0.0 {
            if class.open_raises() {
                return self.preflop_raise_to(game_state, big_blind * 2.5);
            }
            return AgentAction::Call;
        }
        if current_bet <= big_blind * 2.6 {
            if class.three_bets() && can_raise(game_state) {
                return self.preflop_raise_to(game_state, big_blind * 7.5);
            }
            if class.calls_open() {
                return AgentAction::Call;
            }
            return AgentAction::Fold;
        }
        if current_bet <= big_blind * 8.0 {
            if class.four_bets() && can_raise(game_state) {
                return self.preflop_raise_to(game_state, big_blind * 18.0);
            }
            if class.calls_three_bet() {
                return AgentAction::Call;
            }
            return AgentAction::Fold;
        }
        if class.calls_large_preflop_bet() {
            AgentAction::Call
        } else {
            AgentAction::Fold
        }
    }

    fn postflop_action(&mut self, game_state: &GameState) -> AgentAction {
        let Some(public_state) = public_state_from_game(game_state) else {
            return AgentAction::Call;
        };
        let Some(hero_cards) = hero_cards_from_game(game_state) else {
            return AgentAction::Call;
        };
        let indexer = ComboIndexer::new();
        let hero_combo = hero_combo_index(&indexer, hero_cards);

        if self
            .postflop_plan
            .as_ref()
            .is_none_or(|plan| !plan.contains_state(&public_state))
        {
            self.postflop_plan = self.shared_plan.take_matching(&public_state).or_else(|| {
                Some(build_postflop_plan(
                    public_state.clone(),
                    game_state,
                    &self.config,
                    &self.history.records(),
                ))
            });
        }

        let plan = self
            .postflop_plan
            .as_ref()
            .expect("postflop plan should be available");
        let node_index = plan.node_for_state(&public_state).unwrap_or(0);
        let Some(public_infoset) = plan.layout.node_infoset(node_index) else {
            return AgentAction::Call;
        };
        let action_index = best_average_strategy_action(
            &plan.layout,
            &plan.state,
            public_infoset,
            Player::Hero,
            hero_combo,
        );
        let action = plan
            .layout
            .action(&plan.tree, public_infoset, action_index)
            .map(|candidate| candidate.action)
            .unwrap_or(PlayerAction::Check);
        to_rs_action(game_state, action)
    }

    fn preflop_raise_to(&self, game_state: &GameState, target: f32) -> AgentAction {
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let min_raise_to = game_state.current_round_bet() + game_state.current_round_min_raise();
        let amount = target.max(min_raise_to).min(player_bet + stack);
        if amount >= player_bet + stack {
            AgentAction::AllIn
        } else {
            AgentAction::Bet(amount)
        }
    }
}

impl Agent for PokedrAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        self.choose_action(game_state)
    }

    fn historian(&self) -> Option<Box<dyn Historian>> {
        Some(Box::new(PokedrAgentHistorian {
            history: self.history.clone(),
            shared_plan: self.shared_plan.clone(),
            config: self.config.clone(),
        }))
    }
}

pub fn run_heads_up_match(hands: usize, seed: u64) -> MatchSummary {
    run_heads_up_match_with_config(hands, seed, PokedrAgentConfig::default())
}

pub fn run_heads_up_match_with_config(
    hands: usize,
    seed: u64,
    config: PokedrAgentConfig,
) -> MatchSummary {
    use rand::{SeedableRng, rngs::StdRng};
    use rs_poker::arena::{HoldemSimulationBuilder, agent::RandomPotControlAgent};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut hero_net = 0.0;
    let mut villain_net = 0.0;
    for hand in 0..hands {
        let game_state = GameState::new_starting(vec![100.0, 100.0], 2.0, 1.0, 0.0, hand % 2);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(PokedrAgent::new(config.clone())),
                Box::new(RandomPotControlAgent::new(vec![0.65, 0.55, 0.45])),
            ])
            .build()
            .unwrap();
        sim.run(&mut rng);
        hero_net += sim.game_state.stacks[0] - sim.game_state.starting_stacks[0];
        villain_net += sim.game_state.stacks[1] - sim.game_state.starting_stacks[1];
    }
    MatchSummary {
        hands,
        hero_net,
        villain_net,
    }
}

pub fn run_traced_heads_up_match(hands: usize, seed: u64) -> Vec<HandTrace> {
    run_traced_heads_up_match_with_config(hands, seed, PokedrAgentConfig::default())
}

pub fn run_traced_heads_up_match_with_config(
    hands: usize,
    seed: u64,
    config: PokedrAgentConfig,
) -> Vec<HandTrace> {
    use rand::{SeedableRng, rngs::StdRng};
    use rs_poker::arena::{HoldemSimulationBuilder, agent::RandomPotControlAgent};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut traces = Vec::with_capacity(hands);
    for hand in 0..hands {
        let game_state = GameState::new_starting(vec![100.0, 100.0], 2.0, 1.0, 0.0, hand % 2);
        let historian = VecHistorian::new();
        let records = historian.get_storage();
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(PokedrAgent::new(config.clone())),
                Box::new(RandomPotControlAgent::new(vec![0.65, 0.55, 0.45])),
            ])
            .historians(vec![Box::new(historian)])
            .build()
            .unwrap();
        sim.run(&mut rng);
        let borrowed_records = records.borrow();
        let hole_cards = dealt_hole_cards(&borrowed_records, 2);
        traces.push(HandTrace {
            hand_index: hand,
            hero_cards: format_cards(&hole_cards[0]),
            villain_cards: format_cards(&hole_cards[1]),
            board: format_cards(&sim.game_state.board),
            hero_net: sim.game_state.stacks[0] - sim.game_state.starting_stacks[0],
            villain_net: sim.game_state.stacks[1] - sim.game_state.starting_stacks[1],
            actions: borrowed_records
                .iter()
                .filter_map(|record| format_trace_action(&record.action))
                .collect(),
            awards: borrowed_records
                .iter()
                .filter_map(|record| format_award(&record.action))
                .collect(),
        });
    }
    traces
}

pub fn gpu_backend_mode() -> BackendMode {
    match GpuDenseCfrBackend::new() {
        Ok(_) => BackendMode::Gpu,
        Err(GpuCfrError::NoAdapter)
        | Err(GpuCfrError::RequestDevice(_))
        | Err(GpuCfrError::MapFailed(_)) => BackendMode::CpuFallback,
    }
}

pub fn fixed_flop_compact_smoke(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
) -> Option<FixedFlopCompactSmokeSummary> {
    let public_state = PublicState {
        street: Street::Flop,
        board: Board::new(flop.to_vec()),
        pot: 4,
        hero_invested: 2,
        villain_invested: 2,
        effective_stack: 100,
        to_call: 0,
        min_aggressive_amount: 2,
        acting_player: Player::Hero,
        raises_this_street: 0,
        checks_this_street: 0,
    };
    let tree = SubgameTree::build(
        public_state,
        SubgameTreeConfig {
            action_set: config.action_set.clone(),
            max_raises_per_street: config.max_raises_per_street,
        },
    );
    let layout = PostflopDenseLayout::from_tree(&tree);
    let indexer = ComboIndexer::new();
    let root_dead = root_board(&tree).deck_mask();
    let (hero_weights, villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
    let backend = cfr_gpu_backend()?;
    let matrix_cache = RefCell::new(ShowdownMatrixCache::new(showdown_matrix_cache_capacity()));
    let linearized = linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)?;
    let combos = gpu_private_combos();
    let combo_legal: Vec<u32> = indexer
        .combos()
        .iter()
        .map(|combo| (!combo.collides_with(root_dead)) as u32)
        .collect();
    let max_chunk_bytes = (backend.max_storage_buffer_binding_size() as usize).min(128usize << 20);
    let compact_config = layout.compact_private_config(COMBO_COUNT, config.cfr_variant);
    let chunks = compact_config.chunk_by_action_bytes(max_chunk_bytes);
    let largest_chunk_slots = chunks
        .iter()
        .map(|chunk| chunk.action_slots)
        .max()
        .unwrap_or(0);
    let (compact_context_action_slots, uncovered_reach_tiles) = backend
        .compact_public_tree_context_smoke_with_chunks(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &linearized.showdown_boards,
            Some(&chunks),
        );
    Some(FixedFlopCompactSmokeSummary {
        public_infosets: compact_config.public_infosets(),
        public_actions: compact_config.public_actions(),
        combos: compact_config.combos,
        compact_action_slots: compact_config.total_action_slots(),
        chunks: chunks.len(),
        largest_chunk_slots,
        compact_context_action_slots,
        uncovered_reach_tiles,
        reach_tiles_requiring_split: uncovered_reach_tiles,
    })
}

pub fn fixed_flop_compact_context_smoke(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
) -> Option<usize> {
    fixed_flop_compact_smoke(flop, config).map(|summary| summary.compact_context_action_slots)
}

pub fn solve_fixed_flop_once(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
) -> FixedFlopSolveSummary {
    let started = Instant::now();
    let public_state = PublicState {
        street: Street::Flop,
        board: Board::new(flop.to_vec()),
        pot: 4,
        hero_invested: 2,
        villain_invested: 2,
        effective_stack: 100,
        to_call: 0,
        min_aggressive_amount: 2,
        acting_player: Player::Hero,
        raises_this_street: 0,
        checks_this_street: 0,
    };
    let tree = SubgameTree::build(
        public_state,
        SubgameTreeConfig {
            action_set: config.action_set.clone(),
            max_raises_per_street: config.max_raises_per_street,
        },
    );
    let layout = PostflopDenseLayout::from_tree(&tree);
    let indexer = ComboIndexer::new();
    let root_dead = root_board(&tree).deck_mask();
    let (hero_weights, villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
    let state = solve_public_tree_cfr(&tree, &layout, &config, &hero_weights, &villain_weights);
    FixedFlopSolveSummary {
        board: format_pokedr_cards(&flop),
        iterations: config.cfr_iterations.max(1),
        decisions: tree.decision_count(),
        chance: tree.chance_count(),
        terminals: tree.terminal_count(),
        public_infosets: layout.infoset_count(),
        private_infosets: state.infosets(),
        max_actions: layout.max_actions(),
        elapsed_secs: started.elapsed().as_secs_f32(),
    }
}

pub fn solve_fixed_flop_metrics(
    flop: [PokedrCard; 3],
    base_config: PokedrAgentConfig,
    iteration_counts: &[usize],
) -> Vec<FixedFlopMetricRow> {
    solve_fixed_flop_metrics_with_callback(flop, base_config, iteration_counts, |_| true)
}

pub fn solve_fixed_flop_metrics_with_callback<F>(
    flop: [PokedrCard; 3],
    base_config: PokedrAgentConfig,
    iteration_counts: &[usize],
    on_row: F,
) -> Vec<FixedFlopMetricRow>
where
    F: FnMut(&FixedFlopMetricRow) -> bool,
{
    solve_fixed_flop_metrics_with_options_and_callback(
        flop,
        base_config,
        iteration_counts,
        FixedFlopMetricOptions::default(),
        on_row,
    )
}

pub fn solve_fixed_flop_metrics_with_options_and_callback<F>(
    flop: [PokedrCard; 3],
    base_config: PokedrAgentConfig,
    iteration_counts: &[usize],
    metric_options: FixedFlopMetricOptions,
    mut on_row: F,
) -> Vec<FixedFlopMetricRow>
where
    F: FnMut(&FixedFlopMetricRow) -> bool,
{
    let public_state = PublicState {
        street: Street::Flop,
        board: Board::new(flop.to_vec()),
        pot: 4,
        hero_invested: 2,
        villain_invested: 2,
        effective_stack: 100,
        to_call: 0,
        min_aggressive_amount: 2,
        acting_player: Player::Hero,
        raises_this_street: 0,
        checks_this_street: 0,
    };
    let tree = SubgameTree::build(
        public_state,
        SubgameTreeConfig {
            action_set: base_config.action_set.clone(),
            max_raises_per_street: base_config.max_raises_per_street,
        },
    );
    let layout = PostflopDenseLayout::from_tree(&tree);
    let indexer = ComboIndexer::new();
    let root_dead = root_board(&tree).deck_mask();
    let (hero_weights, villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
    if let Some(rows) = try_solve_fixed_flop_metrics_gpu(
        &tree,
        &layout,
        &base_config,
        &hero_weights,
        &villain_weights,
        &indexer,
        root_dead,
        &flop,
        iteration_counts,
        metric_options,
        &mut on_row,
    ) {
        return rows;
    }
    let mut previous_root_strategy: Option<Vec<f32>> = None;
    let mut rows = Vec::with_capacity(iteration_counts.len());
    for iterations in iteration_counts.iter().copied() {
        let started = Instant::now();
        let mut config = base_config.clone();
        config.cfr_iterations = iterations.max(1);
        let state = solve_public_tree_cfr(&tree, &layout, &config, &hero_weights, &villain_weights);
        let elapsed_secs = started.elapsed().as_secs_f32();
        let root_strategy = root_average_strategy(&layout, &state, &indexer, root_dead);
        let root_strategy_l1_delta = previous_root_strategy
            .as_ref()
            .map(|previous| mean_l1_delta(previous, &root_strategy));
        let root_action_probabilities = weighted_root_action_probabilities(
            &layout,
            &root_strategy,
            &hero_weights,
            &indexer,
            root_dead,
        );
        let diagnostics =
            cfr_state_diagnostics(&state, Some((&tree, &layout, &indexer, root_dead)));
        rows.push(FixedFlopMetricRow {
            board: format_pokedr_cards(&flop),
            iterations: config.cfr_iterations,
            elapsed_secs,
            root_strategy_l1_delta,
            root_action_probabilities,
            root_exploitability: None,
            current_root_exploitability: None,
            cpu_root_exploitability: None,
            cpu_gpu_root_exploitability_delta: None,
            hero_root_br_value: None,
            villain_root_br_value: None,
            hero_root_profile_value: None,
            villain_root_profile_value: None,
            hero_root_br_improvement: None,
            villain_root_br_improvement: None,
            cpu_hero_root_br_value: None,
            cpu_villain_root_br_value: None,
            cpu_hero_root_profile_value: None,
            cpu_villain_root_profile_value: None,
            cpu_hero_root_br_improvement: None,
            cpu_villain_root_br_improvement: None,
            current_hero_root_br_value: None,
            current_villain_root_br_value: None,
            current_hero_root_profile_value: None,
            current_villain_root_profile_value: None,
            current_hero_root_br_improvement: None,
            current_villain_root_br_improvement: None,
            root_br_gap: None,
            local_br_gap: None,
            recursive_root_br_gap: None,
            recursive_local_br_gap: None,
            local_gap_detail: None,
            recursive_local_gap_detail: None,
            positive_regret_mass: diagnostics.positive_regret_mass,
            negative_regret_mass: diagnostics.negative_regret_mass,
            negative_regret_count: diagnostics.negative_regret_count,
            deep_negative_regret_count: diagnostics.deep_negative_regret_count,
            rbp_private_action_slots: diagnostics.rbp_private_action_slots,
            rbp_prunable_private_action_slots: diagnostics.rbp_prunable_private_action_slots,
            rbp_public_action_edges: diagnostics.rbp_public_action_edges,
            rbp_fully_prunable_public_action_edges: diagnostics
                .rbp_fully_prunable_public_action_edges,
            rbp_public_edge_subtree_nodes: diagnostics.rbp_public_edge_subtree_nodes,
            rbp_fully_prunable_public_edge_subtree_nodes: diagnostics
                .rbp_fully_prunable_public_edge_subtree_nodes,
            rbp_public_edge_terminal_nodes: diagnostics.rbp_public_edge_terminal_nodes,
            rbp_fully_prunable_public_edge_terminal_nodes: diagnostics
                .rbp_fully_prunable_public_edge_terminal_nodes,
            rbp_terminal_combo_lanes: diagnostics.rbp_terminal_combo_lanes,
            rbp_active_terminal_combo_lanes: diagnostics.rbp_active_terminal_combo_lanes,
            rbp_showdown_combo_lanes: diagnostics.rbp_showdown_combo_lanes,
            rbp_active_showdown_combo_lanes: diagnostics.rbp_active_showdown_combo_lanes,
            illegal_strategy_mass: diagnostics.illegal_strategy_mass,
            current_strategy_norm_error: diagnostics.current_strategy_norm_error,
            average_strategy_norm_error: diagnostics.average_strategy_norm_error,
            finite: diagnostics.finite,
        });
        previous_root_strategy = Some(root_strategy);
        if !on_row(rows.last().expect("metric row must exist")) {
            break;
        }
    }
    rows
}

pub fn dump_fixed_flop_tree(flop: [PokedrCard; 3], config: PokedrAgentConfig) -> Vec<String> {
    let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
    let solver_dump = DumpSolverContext::build(&tree, &layout, &config);
    tree.nodes()
        .iter()
        .enumerate()
        .map(|(index, node)| dump_tree_node_json(&tree, &layout, solver_dump.as_ref(), index, node))
        .collect()
}

pub fn dump_fixed_flop_tree_node(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
    node_index: usize,
) -> Option<String> {
    let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
    let solver_dump = DumpSolverContext::build(&tree, &layout, &config);
    let node = tree.nodes().get(node_index)?;
    Some(dump_tree_node_json(
        &tree,
        &layout,
        solver_dump.as_ref(),
        node_index,
        node,
    ))
}

#[derive(Debug, Clone)]
pub struct TerminalReachSharingReport {
    pub board: String,
    pub iterations: usize,
    pub use_current_strategy: bool,
    pub showdown_terminals: usize,
    pub river_showdown_terminals: usize,
    pub board_tables: usize,
    pub raw_unique_signatures: usize,
    pub normalized_unique_signatures: usize,
    pub support_unique_signatures: usize,
    pub rows: Vec<TerminalReachSharingRow>,
}

#[derive(Debug, Clone)]
pub struct TerminalReachSharingRow {
    pub board: String,
    pub board_cards: usize,
    pub terminals: usize,
    pub raw_unique_signatures: usize,
    pub normalized_unique_signatures: usize,
    pub support_unique_signatures: usize,
    pub hero_mass_min: f32,
    pub hero_mass_max: f32,
    pub villain_mass_min: f32,
    pub villain_mass_max: f32,
    pub hero_support_min: usize,
    pub hero_support_max: usize,
    pub villain_support_min: usize,
    pub villain_support_max: usize,
}

pub fn analyze_fixed_flop_terminal_reach_sharing(
    flop: [PokedrCard; 3],
    mut config: PokedrAgentConfig,
    use_current_strategy: bool,
    top_n: usize,
) -> TerminalReachSharingReport {
    config.cfr_iterations = config.cfr_iterations.max(1);
    let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
    let indexer = ComboIndexer::new();
    let root_dead = root_board(&tree).deck_mask();
    let (hero_weights, villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
    let state = solve_public_tree_cfr(&tree, &layout, &config, &hero_weights, &villain_weights);
    let combo_masks = indexer
        .combos()
        .iter()
        .map(|combo| combo.deck_mask())
        .collect::<Vec<_>>();
    let mut collector = TerminalReachSharingCollector::default();
    collect_terminal_reach_sharing(
        &tree,
        &layout,
        &state,
        &combo_masks,
        0,
        &hero_weights,
        &villain_weights,
        use_current_strategy,
        &mut collector,
    );
    collector.into_report(
        format_pokedr_cards(&flop),
        config.cfr_iterations,
        use_current_strategy,
        top_n,
    )
}

pub fn build_fixed_flop_tree_dump(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
) -> FixedFlopTreeDump {
    build_fixed_flop_tree_dump_with_combo_limit(flop, config, dump_solver_db_combo_limit())
}

pub fn build_fixed_flop_tree_dump_with_combo_limit(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
    combo_limit: usize,
) -> FixedFlopTreeDump {
    let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
    let solver_dump = Some(DumpSolverContext::build_for_mode(
        &tree,
        &layout,
        &config,
        DumpSolverMode::Summary,
        combo_limit,
    ));
    let mut nodes = Vec::with_capacity(tree.nodes().len());
    let mut actions = Vec::new();
    let mut solver_nodes = Vec::new();
    let mut solver_combos = Vec::new();

    for (node_id, node) in tree.nodes().iter().enumerate() {
        nodes.push(tree_node_record(&tree, &layout, node_id, node));
        if let PublicNodeKind::Decision {
            actions: candidates,
            ..
        } = &node.kind
        {
            for (action_index, candidate) in candidates.iter().enumerate() {
                actions.push(TreeActionRecord {
                    node_id,
                    action_index,
                    child_id: node.children[action_index],
                    action: format_player_action(candidate.action),
                    source: format!("{:?}", candidate.source),
                });
            }
        }
        if let Some(context) = solver_dump.as_ref() {
            if let Some((record, combo_records)) =
                solver_node_records(&tree, &layout, context, node_id)
            {
                solver_nodes.push(record);
                solver_combos.extend(combo_records);
            }
        }
    }

    FixedFlopTreeDump {
        metrics: solver_dump
            .as_ref()
            .map(|context| context.fixed_flop_tree_metrics()),
        nodes,
        actions,
        solver_nodes,
        solver_combos,
        root_br_combos: solver_dump
            .as_ref()
            .map(DumpSolverContext::root_br_combo_records)
            .unwrap_or_default(),
    }
}

fn fixed_flop_tree_and_layout(
    flop: [PokedrCard; 3],
    config: PokedrAgentConfig,
) -> (SubgameTree, PostflopDenseLayout) {
    let public_state = PublicState {
        street: Street::Flop,
        board: Board::new(flop.to_vec()),
        pot: 4,
        hero_invested: 2,
        villain_invested: 2,
        effective_stack: 100,
        to_call: 0,
        min_aggressive_amount: 2,
        acting_player: Player::Hero,
        raises_this_street: 0,
        checks_this_street: 0,
    };
    let tree = SubgameTree::build(
        public_state,
        SubgameTreeConfig {
            action_set: config.action_set,
            max_raises_per_street: config.max_raises_per_street,
        },
    );
    let layout = PostflopDenseLayout::from_tree(&tree);
    (tree, layout)
}

#[allow(clippy::too_many_arguments)]
fn try_solve_fixed_flop_metrics_gpu(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    base_config: &PokedrAgentConfig,
    hero_weights: &[f32],
    villain_weights: &[f32],
    indexer: &ComboIndexer,
    root_dead: u64,
    flop: &[PokedrCard; 3],
    iteration_counts: &[usize],
    metric_options: FixedFlopMetricOptions,
    on_row: &mut dyn FnMut(&FixedFlopMetricRow) -> bool,
) -> Option<Vec<FixedFlopMetricRow>> {
    let backend = cfr_gpu_backend()?;
    let matrix_cache = RefCell::new(ShowdownMatrixCache::new(showdown_matrix_cache_capacity()));
    let linearized = linearize_gpu_public_tree(tree, layout, &backend, base_config, &matrix_cache)?;
    let mut dense_config = layout.dense_config(base_config.cfr_variant);
    dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
    if !gpu_dense_state_fits_single_buffers(&backend, &dense_config) {
        if solver_progress_enabled() {
            eprintln!(
                "pokedr: skipping dense resident GPU metrics state; public_infosets={} private_infosets={} max_actions={} dense_action_slots={} compact_action_slots={}",
                layout.infoset_count(),
                dense_config.infosets,
                dense_config.actions,
                dense_config.infosets.saturating_mul(dense_config.actions),
                layout
                    .total_actions()
                    .saturating_mul(PRIVATE_INFOS_PER_PUBLIC)
            );
        }
        return None;
    }
    assert_gpu_dense_binding_feasible(&backend, &dense_config);
    let combos = gpu_private_combos();
    let combo_legal: Vec<u32> = indexer
        .combos()
        .iter()
        .map(|combo| (!combo.collides_with(root_dead)) as u32)
        .collect();
    let mut gpu_state =
        backend.zeroed_state_with_legal_actions(dense_config, private_legal_actions(layout));
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: gpu resident public tree checkpoints nodes={} showdown_boards={} iterations={:?}",
            linearized.nodes.len(),
            linearized.showdown_boards.len(),
            iteration_counts
        );
    }
    let mut checkpoints: Vec<_> = iteration_counts
        .iter()
        .copied()
        .map(|value| value.max(1))
        .collect();
    checkpoints.sort_unstable();
    checkpoints.dedup();
    let mut previous_root_strategy: Option<Vec<f32>> = None;
    let mut rows = Vec::with_capacity(checkpoints.len());
    let started = Instant::now();
    let mut completed_iterations = 0usize;
    let include_local_gaps = metric_options.local_gaps;
    let include_br_metrics = metric_options.br_metrics;
    let include_current_br = metric_options.current_br_metrics;
    let include_diagnostics = metric_options.diagnostics;
    for iterations in checkpoints {
        let delta = iterations.saturating_sub(completed_iterations);
        backend
            .public_tree_run_iterations_from(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                hero_weights,
                villain_weights,
                &linearized.showdown_boards,
                &mut gpu_state,
                completed_iterations + 1,
                delta,
            )
            .ok()?;
        backend.wait_idle().ok()?;
        completed_iterations = iterations;

        let should_compute_periodic_br = metric_options
            .br_every_iterations
            .is_some_and(|interval| iterations % interval == 0);
        let need_full_state_before_root = include_br_metrics
            || include_current_br
            || include_local_gaps
            || include_diagnostics
            || should_compute_periodic_br
            || metric_cpu_exploitability_enabled();
        let mut state = need_full_state_before_root
            .then(|| gpu_state.download(&backend))
            .transpose()
            .ok()?;
        let elapsed_secs = started.elapsed().as_secs_f32();
        let root_strategy = if let Some(state) = state.as_ref() {
            root_average_strategy(layout, state, indexer, root_dead)
        } else {
            let root_strategy_sum_len = COMBO_COUNT * layout.max_actions();
            let strategy_sum_prefix = gpu_state
                .download_strategy_sum_prefix(&backend, root_strategy_sum_len)
                .ok()?;
            root_average_strategy_from_sum_prefix(layout, &strategy_sum_prefix, indexer, root_dead)
        };
        let root_strategy_l1_delta = previous_root_strategy
            .as_ref()
            .map(|previous| mean_l1_delta(previous, &root_strategy));
        let should_compute_stable_br = metric_options
            .br_on_root_delta
            .zip(root_strategy_l1_delta)
            .is_some_and(|(threshold, delta)| delta <= threshold);
        let compute_br_metrics =
            include_br_metrics || should_compute_stable_br || should_compute_periodic_br;
        if compute_br_metrics && state.is_none() {
            state = Some(gpu_state.download(&backend).ok()?);
        }
        let root_action_probabilities = weighted_root_action_probabilities(
            layout,
            &root_strategy,
            hero_weights,
            indexer,
            root_dead,
        );
        let br_gap = include_local_gaps
            .then(|| {
                br_gap_metrics_gpu(
                    &backend,
                    tree,
                    &linearized,
                    &combos,
                    &combo_legal,
                    hero_weights,
                    villain_weights,
                    layout,
                    state.as_ref()?,
                )
            })
            .flatten();
        let recursive_br_gap = compute_br_metrics
            .then(|| {
                recursive_br_gap_metrics_gpu(
                    &backend,
                    tree,
                    &linearized,
                    &combos,
                    &combo_legal,
                    hero_weights,
                    villain_weights,
                    layout,
                    state.as_ref()?,
                    include_local_gaps,
                )
            })
            .flatten();
        let current_recursive_br_gap = include_current_br
            .then(|| {
                recursive_br_gap_metrics_gpu_with_profile(
                    &backend,
                    tree,
                    &linearized,
                    &combos,
                    &combo_legal,
                    hero_weights,
                    villain_weights,
                    layout,
                    state.as_ref()?,
                    false,
                )
            })
            .flatten();
        let cpu_exploitability = if metric_cpu_exploitability_enabled() {
            state.as_ref().map(|state| {
                cpu_infoset_root_exploitability(
                    tree,
                    layout,
                    state,
                    indexer,
                    hero_weights,
                    villain_weights,
                    base_config.max_showdown_runouts,
                )
            })
        } else {
            None
        };
        let root_exploitability = recursive_br_gap
            .as_ref()
            .map(|metrics| metrics.root_exploitability);
        let cpu_root_exploitability = cpu_exploitability
            .as_ref()
            .map(|metrics| metrics.exploitability);
        let cpu_gpu_root_exploitability_delta = root_exploitability
            .zip(cpu_root_exploitability)
            .map(|(gpu, cpu)| gpu - cpu);
        let diagnostics = if let (true, Some(state)) = (include_diagnostics, state.as_ref()) {
            cfr_state_diagnostics(state, Some((tree, layout, indexer, root_dead)))
        } else {
            CfrDiagnostics::skipped()
        };
        rows.push(FixedFlopMetricRow {
            board: format_pokedr_cards(flop),
            iterations,
            elapsed_secs,
            root_strategy_l1_delta,
            root_action_probabilities,
            root_exploitability,
            current_root_exploitability: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.root_exploitability),
            cpu_root_exploitability,
            cpu_gpu_root_exploitability_delta,
            hero_root_br_value: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_value),
            villain_root_br_value: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_value),
            hero_root_profile_value: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_profile_value),
            villain_root_profile_value: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_profile_value),
            hero_root_br_improvement: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_improvement),
            villain_root_br_improvement: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_improvement),
            cpu_hero_root_br_value: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.hero_br_value),
            cpu_villain_root_br_value: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.villain_br_value),
            cpu_hero_root_profile_value: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.hero_profile_value),
            cpu_villain_root_profile_value: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.villain_profile_value),
            cpu_hero_root_br_improvement: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.hero_improvement),
            cpu_villain_root_br_improvement: cpu_exploitability
                .as_ref()
                .map(|metrics| metrics.villain_improvement),
            current_hero_root_br_value: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_value),
            current_villain_root_br_value: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_value),
            current_hero_root_profile_value: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_profile_value),
            current_villain_root_profile_value: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_profile_value),
            current_hero_root_br_improvement: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_improvement),
            current_villain_root_br_improvement: current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_improvement),
            root_br_gap: br_gap.as_ref().map(|metrics| metrics.root_br_gap),
            local_br_gap: br_gap.as_ref().map(|metrics| metrics.local_br_gap),
            recursive_root_br_gap: recursive_br_gap.as_ref().map(|metrics| metrics.root_br_gap),
            recursive_local_br_gap: recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.local_br_gap),
            local_gap_detail: br_gap.and_then(|metrics| metrics.local_gap_detail),
            recursive_local_gap_detail: recursive_br_gap
                .and_then(|metrics| metrics.local_gap_detail),
            positive_regret_mass: diagnostics.positive_regret_mass,
            negative_regret_mass: diagnostics.negative_regret_mass,
            negative_regret_count: diagnostics.negative_regret_count,
            deep_negative_regret_count: diagnostics.deep_negative_regret_count,
            rbp_private_action_slots: diagnostics.rbp_private_action_slots,
            rbp_prunable_private_action_slots: diagnostics.rbp_prunable_private_action_slots,
            rbp_public_action_edges: diagnostics.rbp_public_action_edges,
            rbp_fully_prunable_public_action_edges: diagnostics
                .rbp_fully_prunable_public_action_edges,
            rbp_public_edge_subtree_nodes: diagnostics.rbp_public_edge_subtree_nodes,
            rbp_fully_prunable_public_edge_subtree_nodes: diagnostics
                .rbp_fully_prunable_public_edge_subtree_nodes,
            rbp_public_edge_terminal_nodes: diagnostics.rbp_public_edge_terminal_nodes,
            rbp_fully_prunable_public_edge_terminal_nodes: diagnostics
                .rbp_fully_prunable_public_edge_terminal_nodes,
            rbp_terminal_combo_lanes: diagnostics.rbp_terminal_combo_lanes,
            rbp_active_terminal_combo_lanes: diagnostics.rbp_active_terminal_combo_lanes,
            rbp_showdown_combo_lanes: diagnostics.rbp_showdown_combo_lanes,
            rbp_active_showdown_combo_lanes: diagnostics.rbp_active_showdown_combo_lanes,
            illegal_strategy_mass: diagnostics.illegal_strategy_mass,
            current_strategy_norm_error: diagnostics.current_strategy_norm_error,
            average_strategy_norm_error: diagnostics.average_strategy_norm_error,
            finite: diagnostics.finite,
        });
        previous_root_strategy = Some(root_strategy);
        backend.wait_idle().ok()?;
        if !on_row(rows.last().expect("metric row must exist")) {
            break;
        }
    }
    Some(rows)
}

struct CfrDiagnostics {
    positive_regret_mass: f32,
    negative_regret_mass: f32,
    negative_regret_count: usize,
    deep_negative_regret_count: usize,
    rbp_private_action_slots: usize,
    rbp_prunable_private_action_slots: usize,
    rbp_public_action_edges: usize,
    rbp_fully_prunable_public_action_edges: usize,
    rbp_public_edge_subtree_nodes: usize,
    rbp_fully_prunable_public_edge_subtree_nodes: usize,
    rbp_public_edge_terminal_nodes: usize,
    rbp_fully_prunable_public_edge_terminal_nodes: usize,
    rbp_terminal_combo_lanes: usize,
    rbp_active_terminal_combo_lanes: usize,
    rbp_showdown_combo_lanes: usize,
    rbp_active_showdown_combo_lanes: usize,
    illegal_strategy_mass: f32,
    current_strategy_norm_error: f32,
    average_strategy_norm_error: f32,
    finite: bool,
}

impl CfrDiagnostics {
    fn skipped() -> Self {
        Self {
            positive_regret_mass: f32::NAN,
            negative_regret_mass: f32::NAN,
            negative_regret_count: 0,
            deep_negative_regret_count: 0,
            rbp_private_action_slots: 0,
            rbp_prunable_private_action_slots: 0,
            rbp_public_action_edges: 0,
            rbp_fully_prunable_public_action_edges: 0,
            rbp_public_edge_subtree_nodes: 0,
            rbp_fully_prunable_public_edge_subtree_nodes: 0,
            rbp_public_edge_terminal_nodes: 0,
            rbp_fully_prunable_public_edge_terminal_nodes: 0,
            rbp_terminal_combo_lanes: 0,
            rbp_active_terminal_combo_lanes: 0,
            rbp_showdown_combo_lanes: 0,
            rbp_active_showdown_combo_lanes: 0,
            illegal_strategy_mass: f32::NAN,
            current_strategy_norm_error: 0.0,
            average_strategy_norm_error: 0.0,
            finite: true,
        }
    }
}

fn cfr_state_diagnostics(
    state: &DenseCfrState,
    pruning_context: Option<(&SubgameTree, &PostflopDenseLayout, &ComboIndexer, u64)>,
) -> CfrDiagnostics {
    let positive_regret_mass = state
        .regrets()
        .iter()
        .copied()
        .map(|value| value.max(0.0))
        .sum();
    let negative_regret_mass = state
        .regrets()
        .iter()
        .copied()
        .map(|value| (-value).max(0.0))
        .sum();
    let negative_regret_count = state.regrets().iter().filter(|value| **value < 0.0).count();
    let deep_negative_regret_count = state
        .regrets()
        .iter()
        .filter(|value| **value <= -1.0)
        .count();
    let illegal_strategy_mass = state
        .strategy_sum()
        .iter()
        .zip(state.legal_actions())
        .filter(|(_, legal)| !**legal)
        .map(|(value, _)| value.abs())
        .sum();
    let finite = state.regrets().iter().all(|value| value.is_finite())
        && state.strategy_sum().iter().all(|value| value.is_finite());
    let mut current = vec![0.0; state.actions()];
    let mut average = vec![0.0; state.actions()];
    let mut current_strategy_norm_error = 0.0f32;
    let mut average_strategy_norm_error = 0.0f32;
    for infoset in 0..state.infosets() {
        state.strategy_for(infoset, &mut current);
        state.average_strategy_for(infoset, &mut average);
        current_strategy_norm_error =
            current_strategy_norm_error.max((current.iter().sum::<f32>() - 1.0).abs());
        average_strategy_norm_error =
            average_strategy_norm_error.max((average.iter().sum::<f32>() - 1.0).abs());
    }
    let pruning = pruning_context
        .map(|(tree, layout, indexer, root_dead)| {
            regret_based_pruning_estimate(state, tree, layout, indexer, root_dead)
        })
        .unwrap_or_default();
    CfrDiagnostics {
        positive_regret_mass,
        negative_regret_mass,
        negative_regret_count,
        deep_negative_regret_count,
        rbp_private_action_slots: pruning.private_action_slots,
        rbp_prunable_private_action_slots: pruning.prunable_private_action_slots,
        rbp_public_action_edges: pruning.public_action_edges,
        rbp_fully_prunable_public_action_edges: pruning.fully_prunable_public_action_edges,
        rbp_public_edge_subtree_nodes: pruning.public_edge_subtree_nodes,
        rbp_fully_prunable_public_edge_subtree_nodes: pruning
            .fully_prunable_public_edge_subtree_nodes,
        rbp_public_edge_terminal_nodes: pruning.public_edge_terminal_nodes,
        rbp_fully_prunable_public_edge_terminal_nodes: pruning
            .fully_prunable_public_edge_terminal_nodes,
        rbp_terminal_combo_lanes: pruning.terminal_combo_lanes,
        rbp_active_terminal_combo_lanes: pruning.active_terminal_combo_lanes,
        rbp_showdown_combo_lanes: pruning.showdown_combo_lanes,
        rbp_active_showdown_combo_lanes: pruning.active_showdown_combo_lanes,
        illegal_strategy_mass,
        current_strategy_norm_error,
        average_strategy_norm_error,
        finite,
    }
}

#[derive(Default)]
struct TerminalReachSharingCollector {
    showdown_terminals: usize,
    river_showdown_terminals: usize,
    raw_signatures: HashMap<ReachSignature, usize>,
    normalized_signatures: HashMap<ReachSignature, usize>,
    support_signatures: HashMap<ReachSignature, usize>,
    boards: BTreeMap<u64, TerminalReachBoardStats>,
}

#[derive(Default)]
struct TerminalReachBoardStats {
    board: String,
    board_cards: usize,
    terminals: usize,
    raw_signatures: HashMap<ReachSignature, usize>,
    normalized_signatures: HashMap<ReachSignature, usize>,
    support_signatures: HashMap<ReachSignature, usize>,
    hero_mass_min: f32,
    hero_mass_max: f32,
    villain_mass_min: f32,
    villain_mass_max: f32,
    hero_support_min: usize,
    hero_support_max: usize,
    villain_support_min: usize,
    villain_support_max: usize,
}

type ReachSignature = (u64, u64);

impl TerminalReachSharingCollector {
    fn record(&mut self, board: &Board, hero_reach: &[f32], villain_reach: &[f32]) {
        self.showdown_terminals += 1;
        if board.cards().len() == 5 {
            self.river_showdown_terminals += 1;
        }

        let hero_summary = reach_vector_summary(hero_reach);
        let villain_summary = reach_vector_summary(villain_reach);
        let raw_signature = (
            reach_vector_hash(hero_reach, ReachHashMode::Raw),
            reach_vector_hash(villain_reach, ReachHashMode::Raw),
        );
        let normalized_signature = (
            reach_vector_hash(hero_reach, ReachHashMode::Normalized),
            reach_vector_hash(villain_reach, ReachHashMode::Normalized),
        );
        let support_signature = (
            reach_vector_hash(hero_reach, ReachHashMode::Support),
            reach_vector_hash(villain_reach, ReachHashMode::Support),
        );
        *self.raw_signatures.entry(raw_signature).or_insert(0) += 1;
        *self
            .normalized_signatures
            .entry(normalized_signature)
            .or_insert(0) += 1;
        *self
            .support_signatures
            .entry(support_signature)
            .or_insert(0) += 1;

        let entry =
            self.boards
                .entry(board.deck_mask())
                .or_insert_with(|| TerminalReachBoardStats {
                    board: format_pokedr_cards(board.cards()),
                    board_cards: board.cards().len(),
                    hero_mass_min: f32::INFINITY,
                    villain_mass_min: f32::INFINITY,
                    hero_support_min: usize::MAX,
                    villain_support_min: usize::MAX,
                    ..TerminalReachBoardStats::default()
                });
        entry.terminals += 1;
        *entry.raw_signatures.entry(raw_signature).or_insert(0) += 1;
        *entry
            .normalized_signatures
            .entry(normalized_signature)
            .or_insert(0) += 1;
        *entry
            .support_signatures
            .entry(support_signature)
            .or_insert(0) += 1;
        entry.hero_mass_min = entry.hero_mass_min.min(hero_summary.mass);
        entry.hero_mass_max = entry.hero_mass_max.max(hero_summary.mass);
        entry.villain_mass_min = entry.villain_mass_min.min(villain_summary.mass);
        entry.villain_mass_max = entry.villain_mass_max.max(villain_summary.mass);
        entry.hero_support_min = entry.hero_support_min.min(hero_summary.support);
        entry.hero_support_max = entry.hero_support_max.max(hero_summary.support);
        entry.villain_support_min = entry.villain_support_min.min(villain_summary.support);
        entry.villain_support_max = entry.villain_support_max.max(villain_summary.support);
    }

    fn into_report(
        self,
        board: String,
        iterations: usize,
        use_current_strategy: bool,
        top_n: usize,
    ) -> TerminalReachSharingReport {
        let board_tables = self.boards.len();
        let mut rows = self
            .boards
            .into_values()
            .map(|stats| TerminalReachSharingRow {
                board: stats.board,
                board_cards: stats.board_cards,
                terminals: stats.terminals,
                raw_unique_signatures: stats.raw_signatures.len(),
                normalized_unique_signatures: stats.normalized_signatures.len(),
                support_unique_signatures: stats.support_signatures.len(),
                hero_mass_min: stats.hero_mass_min,
                hero_mass_max: stats.hero_mass_max,
                villain_mass_min: stats.villain_mass_min,
                villain_mass_max: stats.villain_mass_max,
                hero_support_min: stats.hero_support_min,
                hero_support_max: stats.hero_support_max,
                villain_support_min: stats.villain_support_min,
                villain_support_max: stats.villain_support_max,
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| std::cmp::Reverse(row.terminals));
        rows.truncate(top_n);
        TerminalReachSharingReport {
            board,
            iterations,
            use_current_strategy,
            showdown_terminals: self.showdown_terminals,
            river_showdown_terminals: self.river_showdown_terminals,
            board_tables,
            raw_unique_signatures: self.raw_signatures.len(),
            normalized_unique_signatures: self.normalized_signatures.len(),
            support_unique_signatures: self.support_signatures.len(),
            rows,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReachVectorSummary {
    mass: f32,
    support: usize,
}

#[derive(Debug, Clone, Copy)]
enum ReachHashMode {
    Raw,
    Normalized,
    Support,
}

fn collect_terminal_reach_sharing(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    combo_masks: &[u64],
    node_index: usize,
    hero_reach: &[f32],
    villain_reach: &[f32],
    use_current_strategy: bool,
    collector: &mut TerminalReachSharingCollector,
) {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision {
            state: public_state,
            actions,
        } => {
            let Some(public_infoset) = layout.node_infoset(node_index) else {
                return;
            };
            let mut strategy = vec![0.0; state.actions()];
            for (action_index, _) in actions.iter().enumerate() {
                let Some(child) = layout.child_for_action(public_infoset, action_index) else {
                    continue;
                };
                let mut next_hero = hero_reach.to_vec();
                let mut next_villain = villain_reach.to_vec();
                let actor_reach = match public_state.acting_player {
                    Player::Hero => &mut next_hero,
                    Player::Villain => &mut next_villain,
                };
                for (combo_index, reach) in actor_reach.iter_mut().enumerate() {
                    if *reach <= 0.0 {
                        continue;
                    }
                    let private_infoset =
                        private_infoset(public_infoset, public_state.acting_player, combo_index);
                    if use_current_strategy {
                        state.strategy_for(private_infoset, &mut strategy);
                    } else {
                        state.average_strategy_for(private_infoset, &mut strategy);
                    }
                    *reach *= strategy.get(action_index).copied().unwrap_or(0.0);
                }
                collect_terminal_reach_sharing(
                    tree,
                    layout,
                    state,
                    combo_masks,
                    child,
                    &next_hero,
                    &next_villain,
                    use_current_strategy,
                    collector,
                );
            }
        }
        PublicNodeKind::Chance { cards, .. } => {
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                let card_mask = card.deck_mask();
                let mut next_hero = hero_reach.to_vec();
                let mut next_villain = villain_reach.to_vec();
                for (combo_index, combo_mask) in combo_masks.iter().copied().enumerate() {
                    if combo_mask & card_mask != 0 {
                        next_hero[combo_index] = 0.0;
                        next_villain[combo_index] = 0.0;
                    }
                }
                collect_terminal_reach_sharing(
                    tree,
                    layout,
                    state,
                    combo_masks,
                    *child,
                    &next_hero,
                    &next_villain,
                    use_current_strategy,
                    collector,
                );
            }
        }
        PublicNodeKind::Terminal { kind, board, .. } => {
            if matches!(kind, TerminalKind::Showdown) {
                collector.record(board, hero_reach, villain_reach);
            }
        }
    }
}

fn reach_vector_summary(values: &[f32]) -> ReachVectorSummary {
    let mut mass = 0.0f32;
    let mut support = 0usize;
    for &value in values {
        if value > 1.0e-9 {
            mass += value;
            support += 1;
        }
    }
    ReachVectorSummary { mass, support }
}

fn reach_vector_hash(values: &[f32], mode: ReachHashMode) -> u64 {
    let summary = reach_vector_summary(values);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match mode {
        ReachHashMode::Raw => {
            0x7261_775f_7265_6163u64.hash(&mut hasher);
            for &value in values {
                quantize_reach(value).hash(&mut hasher);
            }
        }
        ReachHashMode::Normalized => {
            0x6e6f_726d_7265_6163u64.hash(&mut hasher);
            let mass = summary.mass.max(1.0e-12);
            for &value in values {
                quantize_reach(value / mass).hash(&mut hasher);
            }
        }
        ReachHashMode::Support => {
            0x7375_7070_7265_6163u64.hash(&mut hasher);
            for &value in values {
                (value > 1.0e-9).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn quantize_reach(value: f32) -> i64 {
    (value.clamp(-1.0e9, 1.0e9) as f64 * 1_000_000.0).round() as i64
}

#[derive(Debug, Clone, Copy, Default)]
struct RegretBasedPruningEstimate {
    private_action_slots: usize,
    prunable_private_action_slots: usize,
    public_action_edges: usize,
    fully_prunable_public_action_edges: usize,
    public_edge_subtree_nodes: usize,
    fully_prunable_public_edge_subtree_nodes: usize,
    public_edge_terminal_nodes: usize,
    fully_prunable_public_edge_terminal_nodes: usize,
    terminal_combo_lanes: usize,
    active_terminal_combo_lanes: usize,
    showdown_combo_lanes: usize,
    active_showdown_combo_lanes: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct SubtreeSize {
    nodes: usize,
    terminals: usize,
}

fn regret_based_pruning_estimate(
    state: &DenseCfrState,
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    indexer: &ComboIndexer,
    root_dead: u64,
) -> RegretBasedPruningEstimate {
    let threshold = regret_based_pruning_threshold();
    let legal_combos = legal_private_combos(indexer, root_dead).collect::<Vec<_>>();
    let combo_masks = indexer
        .combos()
        .iter()
        .map(|combo| combo.deck_mask())
        .collect::<Vec<_>>();
    let subtree_sizes = subtree_sizes(tree);
    let mut estimate = RegretBasedPruningEstimate::default();

    for (node_index, node) in tree.nodes().iter().enumerate() {
        let PublicNodeKind::Decision {
            state: public_state,
            actions,
        } = &node.kind
        else {
            continue;
        };
        let Some(public_infoset) = layout.node_infoset(node_index) else {
            continue;
        };
        for (action_index, _) in actions.iter().enumerate() {
            let Some(child) = layout.child_for_action(public_infoset, action_index) else {
                continue;
            };
            estimate.public_action_edges += 1;
            estimate.public_edge_subtree_nodes += subtree_sizes[child].nodes;
            estimate.public_edge_terminal_nodes += subtree_sizes[child].terminals;

            let mut legal_slots = 0usize;
            let mut prunable_slots = 0usize;
            for &combo_index in &legal_combos {
                let private_infoset =
                    private_infoset(public_infoset, public_state.acting_player, combo_index);
                let offset = private_infoset * state.actions() + action_index;
                if offset >= state.legal_actions().len() || !state.legal_actions()[offset] {
                    continue;
                }
                legal_slots += 1;
                if state.regrets()[offset] <= threshold {
                    prunable_slots += 1;
                }
            }

            estimate.private_action_slots += legal_slots;
            estimate.prunable_private_action_slots += prunable_slots;
            if legal_slots > 0 && legal_slots == prunable_slots {
                estimate.fully_prunable_public_action_edges += 1;
                estimate.fully_prunable_public_edge_subtree_nodes += subtree_sizes[child].nodes;
                estimate.fully_prunable_public_edge_terminal_nodes +=
                    subtree_sizes[child].terminals;
            }
        }
    }

    estimate_active_terminal_lanes(
        &mut estimate,
        state,
        tree,
        layout,
        &legal_combos,
        &combo_masks,
        threshold,
    );
    estimate
}

fn estimate_active_terminal_lanes(
    estimate: &mut RegretBasedPruningEstimate,
    state: &DenseCfrState,
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    legal_combos: &[usize],
    combo_masks: &[u64],
    threshold: f32,
) {
    let combo_count = combo_masks.len();
    let mut root_active = vec![0u8; combo_count].into_boxed_slice();
    for &combo_index in legal_combos {
        root_active[combo_index] = 1;
    }
    let mut hero_active: Vec<Option<Box<[u8]>>> = vec![None; tree.nodes().len()];
    let mut villain_active: Vec<Option<Box<[u8]>>> = vec![None; tree.nodes().len()];
    hero_active[0] = Some(root_active.clone());
    villain_active[0] = Some(root_active);

    for node_index in 0..tree.nodes().len() {
        let Some(hero_here) = hero_active[node_index].take() else {
            continue;
        };
        let Some(villain_here) = villain_active[node_index].take() else {
            continue;
        };
        match &tree.nodes()[node_index].kind {
            PublicNodeKind::Decision {
                state: public_state,
                actions,
            } => {
                let Some(public_infoset) = layout.node_infoset(node_index) else {
                    continue;
                };
                for (action_index, _) in actions.iter().enumerate() {
                    let Some(child) = layout.child_for_action(public_infoset, action_index) else {
                        continue;
                    };
                    let mut next_hero = hero_here.clone();
                    let mut next_villain = villain_here.clone();
                    let actor_active = match public_state.acting_player {
                        Player::Hero => &mut next_hero,
                        Player::Villain => &mut next_villain,
                    };
                    for &combo_index in legal_combos {
                        if actor_active[combo_index] == 0 {
                            continue;
                        }
                        let private_infoset = private_infoset(
                            public_infoset,
                            public_state.acting_player,
                            combo_index,
                        );
                        let offset = private_infoset * state.actions() + action_index;
                        if offset >= state.legal_actions().len()
                            || !state.legal_actions()[offset]
                            || state.regrets()[offset] <= threshold
                        {
                            actor_active[combo_index] = 0;
                        }
                    }
                    merge_active(&mut hero_active[child], &next_hero);
                    merge_active(&mut villain_active[child], &next_villain);
                }
            }
            PublicNodeKind::Chance { cards, .. } => {
                for (card, &child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                    let card_mask = card.deck_mask();
                    let mut next_hero = hero_here.clone();
                    let mut next_villain = villain_here.clone();
                    for &combo_index in legal_combos {
                        if combo_masks[combo_index] & card_mask != 0 {
                            next_hero[combo_index] = 0;
                            next_villain[combo_index] = 0;
                        }
                    }
                    merge_active(&mut hero_active[child], &next_hero);
                    merge_active(&mut villain_active[child], &next_villain);
                }
            }
            PublicNodeKind::Terminal { kind, .. } => {
                let active = legal_combos
                    .iter()
                    .filter(|&&combo_index| {
                        hero_here[combo_index] != 0 || villain_here[combo_index] != 0
                    })
                    .count();
                estimate.terminal_combo_lanes += legal_combos.len();
                estimate.active_terminal_combo_lanes += active;
                if matches!(kind, TerminalKind::Showdown) {
                    estimate.showdown_combo_lanes += legal_combos.len();
                    estimate.active_showdown_combo_lanes += active;
                }
            }
        }
    }
}

fn merge_active(target: &mut Option<Box<[u8]>>, source: &[u8]) {
    if let Some(target) = target {
        for (left, right) in target.iter_mut().zip(source) {
            *left |= *right;
        }
    } else {
        *target = Some(source.to_vec().into_boxed_slice());
    }
}

fn regret_based_pruning_threshold() -> f32 {
    std::env::var("POKEDR_RBP_PRUNE_THRESHOLD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(-1.0)
}

fn subtree_sizes(tree: &SubgameTree) -> Vec<SubtreeSize> {
    let mut sizes = vec![SubtreeSize::default(); tree.nodes().len()];
    for node_index in (0..tree.nodes().len()).rev() {
        let mut size = SubtreeSize {
            nodes: 1,
            terminals: usize::from(matches!(
                tree.nodes()[node_index].kind,
                PublicNodeKind::Terminal { .. }
            )),
        };
        for &child in &tree.nodes()[node_index].children {
            size.nodes += sizes[child].nodes;
            size.terminals += sizes[child].terminals;
        }
        sizes[node_index] = size;
    }
    sizes
}

struct BrGapMetrics {
    root_br_gap: f32,
    local_br_gap: f32,
    root_exploitability: f32,
    hero_root_br_value: f32,
    villain_root_br_value: f32,
    hero_root_profile_value: f32,
    villain_root_profile_value: f32,
    hero_root_br_improvement: f32,
    villain_root_br_improvement: f32,
    local_gap_detail: Option<LocalGapDetail>,
}

fn br_gap_metrics_gpu(
    backend: &GpuDenseCfrBackend,
    tree: &SubgameTree,
    linearized: &GpuLinearizedPublicTree,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
) -> Option<BrGapMetrics> {
    let profile = state.average_strategy_profile_state();
    let values = backend
        .public_tree_iteration_values(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            &linearized.showdown_boards,
            &profile,
        )
        .ok()?;
    backend.wait_idle().ok()?;
    Some(br_gap_metrics_from_values(
        tree, layout, state, &profile, &values,
    ))
}

fn recursive_br_gap_metrics_gpu(
    backend: &GpuDenseCfrBackend,
    tree: &SubgameTree,
    linearized: &GpuLinearizedPublicTree,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    include_local_gaps: bool,
) -> Option<BrGapMetrics> {
    let profile = state.average_strategy_profile_state();
    recursive_br_gap_metrics_gpu_for_profile(
        backend,
        tree,
        linearized,
        combos,
        combo_legal,
        hero_weights,
        villain_weights,
        layout,
        state,
        &profile,
        include_local_gaps,
    )
}

fn recursive_br_gap_metrics_gpu_with_profile(
    backend: &GpuDenseCfrBackend,
    tree: &SubgameTree,
    linearized: &GpuLinearizedPublicTree,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    include_local_gaps: bool,
) -> Option<BrGapMetrics> {
    recursive_br_gap_metrics_gpu_for_profile(
        backend,
        tree,
        linearized,
        combos,
        combo_legal,
        hero_weights,
        villain_weights,
        layout,
        state,
        state,
        include_local_gaps,
    )
}

#[allow(clippy::too_many_arguments)]
fn recursive_br_gap_metrics_gpu_for_profile(
    backend: &GpuDenseCfrBackend,
    tree: &SubgameTree,
    linearized: &GpuLinearizedPublicTree,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    profile: &DenseCfrState,
    include_local_gaps: bool,
) -> Option<BrGapMetrics> {
    let profile_values = backend
        .public_tree_iteration_values(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            &linearized.showdown_boards,
            profile,
        )
        .ok()?;
    backend.wait_idle().ok()?;
    let hero_values = backend
        .public_tree_best_response_values(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            &linearized.showdown_boards,
            profile,
            0,
        )
        .ok()?;
    backend.wait_idle().ok()?;
    let villain_values = backend
        .public_tree_best_response_values(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            combos,
            combo_legal,
            hero_weights,
            villain_weights,
            &linearized.showdown_boards,
            profile,
            1,
        )
        .ok()?;
    backend.wait_idle().ok()?;
    Some(recursive_br_gap_metrics_from_values(
        tree,
        layout,
        state,
        profile,
        combos,
        combo_legal,
        hero_weights,
        villain_weights,
        &hero_values,
        &villain_values,
        &profile_values,
        include_local_gaps,
    ))
}

fn br_gap_metrics_from_values(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    profile: &DenseCfrState,
    values: &GpuRootTerminalValues,
) -> BrGapMetrics {
    let mut strategy = vec![0.0; layout.max_actions()];
    let mut current_strategy = vec![0.0; layout.max_actions()];
    let mut root_gap_sum = 0.0;
    let mut root_weight_sum = 0.0;
    let mut local_gap_sum = 0.0;
    let mut local_weight_sum = 0.0;
    let mut local_gap_detail = None;

    for public_infoset in 0..layout.infoset_count() {
        let action_count = layout.action_count(public_infoset);
        let player = infoset_acting_player(tree, layout, public_infoset);
        {
            for combo_index in 0..COMBO_COUNT {
                let infoset = private_infoset(public_infoset, player, combo_index);
                let offset = infoset * layout.max_actions();
                let action_values = &values.action_values[offset..offset + action_count];
                let best = action_values
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if !best.is_finite() {
                    continue;
                }
                profile.strategy_for(infoset, &mut strategy);
                let policy_value = action_values
                    .iter()
                    .zip(&strategy)
                    .map(|(value, probability)| value * probability)
                    .sum::<f32>();
                let gap = (best - policy_value).max(0.0);
                let reach_weight = values.reach_weights.get(infoset).copied().unwrap_or(0.0);
                if reach_weight > 0.0 {
                    if public_infoset == 0 && player == Player::Hero {
                        root_gap_sum += gap * reach_weight;
                        root_weight_sum += reach_weight;
                    }
                    let weighted_gap = gap * reach_weight;
                    local_gap_sum += weighted_gap;
                    local_weight_sum += reach_weight;
                    update_local_gap_detail(
                        &mut local_gap_detail,
                        tree,
                        layout,
                        public_infoset,
                        player,
                        combo_index,
                        gap,
                        weighted_gap,
                        reach_weight,
                        action_values,
                        &strategy[..action_count],
                        state,
                        &mut current_strategy,
                    );
                }
            }
        }
    }

    BrGapMetrics {
        root_br_gap: if root_weight_sum > 0.0 {
            root_gap_sum / root_weight_sum
        } else {
            0.0
        },
        local_br_gap: if local_weight_sum > 0.0 {
            local_gap_sum / local_weight_sum
        } else {
            0.0
        },
        root_exploitability: 0.0,
        hero_root_br_value: 0.0,
        villain_root_br_value: 0.0,
        hero_root_profile_value: 0.0,
        villain_root_profile_value: 0.0,
        hero_root_br_improvement: 0.0,
        villain_root_br_improvement: 0.0,
        local_gap_detail,
    }
}

fn recursive_br_gap_metrics_from_values(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    _profile: &DenseCfrState,
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    hero_values: &GpuRootTerminalValues,
    villain_values: &GpuRootTerminalValues,
    profile_values: &GpuRootTerminalValues,
    include_local_gaps: bool,
) -> BrGapMetrics {
    let mut strategy = vec![0.0; layout.max_actions()];
    let mut current_strategy = vec![0.0; layout.max_actions()];
    let mut root_gap_sum = 0.0;
    let mut root_weight_sum = 0.0;
    let mut local_gap_sum = 0.0;
    let mut local_weight_sum = 0.0;
    let mut local_gap_detail = None;

    let public_infoset_count = if include_local_gaps {
        layout.infoset_count()
    } else {
        1
    };
    for public_infoset in 0..public_infoset_count {
        let action_count = layout.action_count(public_infoset);
        let player = infoset_acting_player(tree, layout, public_infoset);
        {
            if !include_local_gaps && !(public_infoset == 0 && player == Player::Hero) {
                continue;
            }
            let values = match player {
                Player::Hero => hero_values,
                Player::Villain => villain_values,
            };
            for combo_index in 0..COMBO_COUNT {
                let infoset = private_infoset(public_infoset, player, combo_index);
                let offset = infoset * layout.max_actions();
                let action_values = &values.action_values[offset..offset + action_count];
                let best = action_values
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if !best.is_finite() {
                    continue;
                }
                state.average_strategy_for(infoset, &mut strategy);
                let policy_value = action_values
                    .iter()
                    .zip(&strategy)
                    .map(|(value, probability)| value * probability)
                    .sum::<f32>();
                let gap = (best - policy_value).max(0.0);
                let reach_weight = values.reach_weights.get(infoset).copied().unwrap_or(0.0);
                if reach_weight > 0.0 {
                    if public_infoset == 0 && player == Player::Hero {
                        root_gap_sum += gap * reach_weight;
                        root_weight_sum += reach_weight;
                    }
                    if include_local_gaps {
                        let weighted_gap = gap * reach_weight;
                        local_gap_sum += weighted_gap;
                        local_weight_sum += reach_weight;
                        update_local_gap_detail(
                            &mut local_gap_detail,
                            tree,
                            layout,
                            public_infoset,
                            player,
                            combo_index,
                            gap,
                            weighted_gap,
                            reach_weight,
                            action_values,
                            &strategy[..action_count],
                            state,
                            &mut current_strategy,
                        );
                    }
                }
            }
        }
    }

    let root_exploitability = root_exploitability_from_recursive_values(
        combos,
        combo_legal,
        hero_weights,
        villain_weights,
        hero_values,
        villain_values,
        profile_values,
    );

    BrGapMetrics {
        root_br_gap: if root_weight_sum > 0.0 {
            root_gap_sum / root_weight_sum
        } else {
            0.0
        },
        local_br_gap: if local_weight_sum > 0.0 {
            local_gap_sum / local_weight_sum
        } else {
            0.0
        },
        root_exploitability: root_exploitability.exploitability,
        hero_root_br_value: root_exploitability.hero_br_value,
        villain_root_br_value: root_exploitability.villain_br_value,
        hero_root_profile_value: root_exploitability.hero_profile_value,
        villain_root_profile_value: root_exploitability.villain_profile_value,
        hero_root_br_improvement: root_exploitability.hero_improvement,
        villain_root_br_improvement: root_exploitability.villain_improvement,
        local_gap_detail,
    }
}

#[derive(Debug, Clone, Copy)]
struct RootExploitability {
    exploitability: f32,
    hero_br_value: f32,
    villain_br_value: f32,
    hero_profile_value: f32,
    villain_profile_value: f32,
    hero_improvement: f32,
    villain_improvement: f32,
}

fn root_exploitability_from_recursive_values(
    combos: &[GpuPrivateCombo],
    combo_legal: &[u32],
    hero_weights: &[f32],
    villain_weights: &[f32],
    hero_values: &GpuRootTerminalValues,
    villain_values: &GpuRootTerminalValues,
    profile_values: &GpuRootTerminalValues,
) -> RootExploitability {
    let mut hero_br_sum = 0.0;
    let mut villain_br_sum = 0.0;
    let mut hero_profile_sum = 0.0;
    let mut villain_profile_sum = 0.0;
    let mut joint_weight_sum = 0.0;

    for (combo_index, combo) in combos.iter().enumerate() {
        if combo_legal.get(combo_index).copied().unwrap_or(0) == 0 {
            continue;
        }

        let mut villain_nonblocking_weight = 0.0;
        for (opponent_index, opponent) in combos.iter().enumerate() {
            if opponent_index == combo_index
                || combo_legal.get(opponent_index).copied().unwrap_or(0) == 0
                || combo.cards[0] == opponent.cards[0]
                || combo.cards[0] == opponent.cards[1]
                || combo.cards[1] == opponent.cards[0]
                || combo.cards[1] == opponent.cards[1]
            {
                continue;
            }
            villain_nonblocking_weight +=
                villain_weights.get(opponent_index).copied().unwrap_or(0.0);
        }

        let hero_weight = hero_weights.get(combo_index).copied().unwrap_or(0.0);
        if hero_weight > 0.0 {
            joint_weight_sum += hero_weight * villain_nonblocking_weight;
            let hero_value = hero_values
                .root_hero_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if hero_value.is_finite() {
                hero_br_sum += hero_value * hero_weight;
            }
            let hero_profile_value = profile_values
                .root_hero_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if hero_profile_value.is_finite() {
                hero_profile_sum += hero_profile_value * hero_weight;
            }
        }

        let villain_weight = villain_weights.get(combo_index).copied().unwrap_or(0.0);
        if villain_weight > 0.0 {
            let villain_value = villain_values
                .root_villain_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if villain_value.is_finite() {
                villain_br_sum += villain_value * villain_weight;
            }
            let villain_profile_value = profile_values
                .root_villain_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if villain_profile_value.is_finite() {
                villain_profile_sum += villain_profile_value * villain_weight;
            }
        }
    }

    if joint_weight_sum <= 0.0 {
        return RootExploitability {
            exploitability: 0.0,
            hero_br_value: 0.0,
            villain_br_value: 0.0,
            hero_profile_value: 0.0,
            villain_profile_value: 0.0,
            hero_improvement: 0.0,
            villain_improvement: 0.0,
        };
    }

    let hero_br_value = hero_br_sum / joint_weight_sum;
    let villain_br_value = villain_br_sum / joint_weight_sum;
    let hero_profile_value = hero_profile_sum / joint_weight_sum;
    let villain_profile_value = villain_profile_sum / joint_weight_sum;
    let hero_improvement = (hero_br_value - hero_profile_value).max(0.0);
    let villain_improvement = (villain_br_value - villain_profile_value).max(0.0);
    RootExploitability {
        exploitability: hero_improvement + villain_improvement,
        hero_br_value,
        villain_br_value,
        hero_profile_value,
        villain_profile_value,
        hero_improvement,
        villain_improvement,
    }
}

fn metric_cpu_exploitability_enabled() -> bool {
    std::env::var_os("POKEDR_METRIC_CPU_EXPLOITABILITY").is_some()
}

#[derive(Clone, Copy)]
struct ActiveCombo {
    index: usize,
    cards: [PokedrCard; 2],
    mask: u64,
    weight: f32,
}

fn cpu_infoset_root_exploitability(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    indexer: &ComboIndexer,
    hero_weights: &[f32],
    villain_weights: &[f32],
    max_showdown_runouts: usize,
) -> RootExploitability {
    let root_dead = root_board(tree).deck_mask();
    let heroes = active_weighted_combos(indexer, root_dead, hero_weights);
    let villains = active_weighted_combos(indexer, root_dead, villain_weights);
    let pair_count = heroes.len() * villains.len();
    let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));

    let mut hero_reach = vec![0.0; pair_count];
    let mut villain_reach = vec![0.0; pair_count];
    let mut joint_weight_sum = 0.0;
    for (h, hero) in heroes.iter().enumerate() {
        for (v, villain) in villains.iter().enumerate() {
            if hero.mask & villain.mask != 0 {
                continue;
            }
            let pair = h * villains.len() + v;
            hero_reach[pair] = villain.weight;
            villain_reach[pair] = hero.weight;
            joint_weight_sum += hero.weight * villain.weight;
        }
    }

    if joint_weight_sum <= 0.0 {
        return RootExploitability {
            exploitability: 0.0,
            hero_br_value: 0.0,
            villain_br_value: 0.0,
            hero_profile_value: 0.0,
            villain_profile_value: 0.0,
            hero_improvement: 0.0,
            villain_improvement: 0.0,
        };
    }

    let hero_values = cpu_infoset_br_hero_payoff_matrix(
        tree,
        layout,
        0,
        None,
        None,
        &heroes,
        &villains,
        Player::Hero,
        state,
        &matrix_cache,
        max_showdown_runouts,
        &hero_reach,
    );
    let villain_values = cpu_infoset_br_hero_payoff_matrix(
        tree,
        layout,
        0,
        None,
        None,
        &heroes,
        &villains,
        Player::Villain,
        state,
        &matrix_cache,
        max_showdown_runouts,
        &villain_reach,
    );
    let profile_values = cpu_infoset_profile_hero_payoff_matrix(
        tree,
        layout,
        0,
        None,
        None,
        &heroes,
        &villains,
        state,
        &matrix_cache,
        max_showdown_runouts,
    );

    let mut hero_br_sum = 0.0;
    let mut villain_br_sum = 0.0;
    let mut profile_hero_sum = 0.0;
    for (h, hero) in heroes.iter().enumerate() {
        for (v, villain) in villains.iter().enumerate() {
            if hero.mask & villain.mask != 0 {
                continue;
            }
            let pair = h * villains.len() + v;
            let joint = hero.weight * villain.weight;
            hero_br_sum += joint * hero_values[pair];
            villain_br_sum += joint * -villain_values[pair];
            profile_hero_sum += joint * profile_values[pair];
        }
    }

    let hero_br_value = hero_br_sum / joint_weight_sum;
    let villain_br_value = villain_br_sum / joint_weight_sum;
    let hero_profile_value = profile_hero_sum / joint_weight_sum;
    let villain_profile_value = -hero_profile_value;
    let hero_improvement = (hero_br_value - hero_profile_value).max(0.0);
    let villain_improvement = (villain_br_value - villain_profile_value).max(0.0);
    RootExploitability {
        exploitability: hero_improvement + villain_improvement,
        hero_br_value,
        villain_br_value,
        hero_profile_value,
        villain_profile_value,
        hero_improvement,
        villain_improvement,
    }
}

fn active_weighted_combos(
    indexer: &ComboIndexer,
    root_dead: u64,
    weights: &[f32],
) -> Vec<ActiveCombo> {
    legal_private_combos(indexer, root_dead)
        .filter_map(|combo_index| {
            let weight = weights.get(combo_index).copied().unwrap_or(0.0);
            if weight <= 0.0 {
                return None;
            }
            let cards = combo_cards(indexer.combo(combo_index));
            Some(ActiveCombo {
                index: combo_index,
                cards,
                mask: hero_mask(cards),
                weight,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_infoset_br_hero_payoff_matrix(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    parent_state: Option<&PublicState>,
    parent_action: Option<PlayerAction>,
    heroes: &[ActiveCombo],
    villains: &[ActiveCombo],
    br_player: Player,
    state: &DenseCfrState,
    matrix_cache: &RefCell<ShowdownMatrixCache>,
    max_showdown_runouts: usize,
    reach: &[f32],
) -> Vec<f32> {
    let pair_count = heroes.len() * villains.len();
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision {
            state: public_state,
            actions,
        } => {
            let public_infoset = layout
                .node_infoset(node_index)
                .expect("decision nodes have infosets");
            let action_count = actions.len();
            let mut child_values = Vec::with_capacity(action_count);
            let mut strategy = vec![0.0; layout.max_actions()];
            for action_index in 0..action_count {
                let child = layout
                    .child_for_action(public_infoset, action_index)
                    .expect("legal action must have a child");
                let mut child_reach = reach.to_vec();
                if public_state.acting_player != br_player {
                    for (h, hero) in heroes.iter().enumerate() {
                        for (v, villain) in villains.iter().enumerate() {
                            let pair = h * villains.len() + v;
                            if child_reach[pair] <= 0.0 {
                                continue;
                            }
                            let acting_combo = match public_state.acting_player {
                                Player::Hero => hero.index,
                                Player::Villain => villain.index,
                            };
                            let infoset = private_infoset(
                                public_infoset,
                                public_state.acting_player,
                                acting_combo,
                            );
                            state.average_strategy_for(infoset, &mut strategy);
                            child_reach[pair] *= strategy[action_index];
                        }
                    }
                }
                child_values.push(cpu_infoset_br_hero_payoff_matrix(
                    tree,
                    layout,
                    child,
                    Some(public_state),
                    Some(actions[action_index].action),
                    heroes,
                    villains,
                    br_player,
                    state,
                    matrix_cache,
                    max_showdown_runouts,
                    &child_reach,
                ));
            }

            if public_state.acting_player == br_player {
                cpu_infoset_choose_br_action(
                    &child_values,
                    reach,
                    heroes.len(),
                    villains.len(),
                    br_player,
                )
            } else {
                let mut result = vec![0.0; pair_count];
                for (action_index, child) in child_values.iter().enumerate() {
                    for (h, hero) in heroes.iter().enumerate() {
                        for (v, villain) in villains.iter().enumerate() {
                            let pair = h * villains.len() + v;
                            let acting_combo = match public_state.acting_player {
                                Player::Hero => hero.index,
                                Player::Villain => villain.index,
                            };
                            let infoset = private_infoset(
                                public_infoset,
                                public_state.acting_player,
                                acting_combo,
                            );
                            state.average_strategy_for(infoset, &mut strategy);
                            result[pair] += strategy[action_index] * child[pair];
                        }
                    }
                }
                result
            }
        }
        PublicNodeKind::Chance { cards, .. } => {
            let mut result = vec![0.0; pair_count];
            let mut valid_counts = vec![0usize; pair_count];
            let pair_dead = pair_dead_masks(heroes, villains);
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                let card_mask = card.deck_mask();
                let mut child_reach = vec![0.0; pair_count];
                for pair in 0..pair_count {
                    if card_mask & pair_dead[pair] == 0 {
                        valid_counts[pair] += 1;
                    }
                }
                for pair in 0..pair_count {
                    if card_mask & pair_dead[pair] == 0 {
                        let count = cards
                            .iter()
                            .filter(|candidate| candidate.deck_mask() & pair_dead[pair] == 0)
                            .count();
                        if count > 0 {
                            child_reach[pair] = reach[pair] / count as f32;
                        }
                    }
                }
                let child_values = cpu_infoset_br_hero_payoff_matrix(
                    tree,
                    layout,
                    *child,
                    None,
                    None,
                    heroes,
                    villains,
                    br_player,
                    state,
                    matrix_cache,
                    max_showdown_runouts,
                    &child_reach,
                );
                for pair in 0..pair_count {
                    if card_mask & pair_dead[pair] == 0 {
                        result[pair] += child_values[pair];
                    }
                }
            }
            for pair in 0..pair_count {
                if valid_counts[pair] > 0 {
                    result[pair] /= valid_counts[pair] as f32;
                }
            }
            result
        }
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            ..
        } => {
            let mut result = vec![0.0; pair_count];
            for (h, hero) in heroes.iter().enumerate() {
                for (v, villain) in villains.iter().enumerate() {
                    let pair = h * villains.len() + v;
                    if hero.mask & villain.mask != 0 {
                        continue;
                    }
                    result[pair] = match kind {
                        TerminalKind::Fold => fold_utility(
                            parent_state.expect("fold terminal must have a parent decision"),
                            parent_action.expect("fold terminal must have a parent action"),
                            *pot,
                            *hero_invested,
                        ),
                        TerminalKind::Showdown => {
                            let mut ctx = PostflopEvaluationContext {
                                hero_cards: hero.cards,
                                villain_cards: villain.cards,
                                hero_combo: hero.index,
                                villain_combo: villain.index,
                                gpu_backend: None,
                                matrix_cache,
                                max_showdown_runouts,
                                equity_cache: HashMap::new(),
                            };
                            showdown_utility(*pot, *hero_invested, board, &mut ctx)
                        }
                    };
                }
            }
            result
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_infoset_profile_hero_payoff_matrix(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    parent_state: Option<&PublicState>,
    parent_action: Option<PlayerAction>,
    heroes: &[ActiveCombo],
    villains: &[ActiveCombo],
    state: &DenseCfrState,
    matrix_cache: &RefCell<ShowdownMatrixCache>,
    max_showdown_runouts: usize,
) -> Vec<f32> {
    let pair_count = heroes.len() * villains.len();
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision {
            state: public_state,
            actions,
        } => {
            let public_infoset = layout
                .node_infoset(node_index)
                .expect("decision nodes have infosets");
            let action_count = actions.len();
            let mut child_values = Vec::with_capacity(action_count);
            for action_index in 0..action_count {
                let child = layout
                    .child_for_action(public_infoset, action_index)
                    .expect("legal action must have a child");
                child_values.push(cpu_infoset_profile_hero_payoff_matrix(
                    tree,
                    layout,
                    child,
                    Some(public_state),
                    Some(actions[action_index].action),
                    heroes,
                    villains,
                    state,
                    matrix_cache,
                    max_showdown_runouts,
                ));
            }

            let mut result = vec![0.0; pair_count];
            let mut strategy = vec![0.0; layout.max_actions()];
            for (action_index, child) in child_values.iter().enumerate() {
                for (h, hero) in heroes.iter().enumerate() {
                    for (v, villain) in villains.iter().enumerate() {
                        let pair = h * villains.len() + v;
                        if hero.mask & villain.mask != 0 {
                            continue;
                        }
                        let acting_combo = match public_state.acting_player {
                            Player::Hero => hero.index,
                            Player::Villain => villain.index,
                        };
                        let infoset = private_infoset(
                            public_infoset,
                            public_state.acting_player,
                            acting_combo,
                        );
                        state.average_strategy_for(infoset, &mut strategy);
                        result[pair] += strategy[action_index] * child[pair];
                    }
                }
            }
            result
        }
        PublicNodeKind::Chance { cards, .. } => {
            let mut result = vec![0.0; pair_count];
            let mut valid_counts = vec![0usize; pair_count];
            let pair_dead = pair_dead_masks(heroes, villains);
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                let card_mask = card.deck_mask();
                let child_values = cpu_infoset_profile_hero_payoff_matrix(
                    tree,
                    layout,
                    *child,
                    None,
                    None,
                    heroes,
                    villains,
                    state,
                    matrix_cache,
                    max_showdown_runouts,
                );
                for pair in 0..pair_count {
                    if card_mask & pair_dead[pair] == 0 {
                        valid_counts[pair] += 1;
                        result[pair] += child_values[pair];
                    }
                }
            }
            for pair in 0..pair_count {
                if valid_counts[pair] > 0 {
                    result[pair] /= valid_counts[pair] as f32;
                }
            }
            result
        }
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            ..
        } => {
            let mut result = vec![0.0; pair_count];
            for (h, hero) in heroes.iter().enumerate() {
                for (v, villain) in villains.iter().enumerate() {
                    let pair = h * villains.len() + v;
                    if hero.mask & villain.mask != 0 {
                        continue;
                    }
                    result[pair] = match kind {
                        TerminalKind::Fold => fold_utility(
                            parent_state.expect("fold terminal must have a parent decision"),
                            parent_action.expect("fold terminal must have a parent action"),
                            *pot,
                            *hero_invested,
                        ),
                        TerminalKind::Showdown => {
                            let mut ctx = PostflopEvaluationContext {
                                hero_cards: hero.cards,
                                villain_cards: villain.cards,
                                hero_combo: hero.index,
                                villain_combo: villain.index,
                                gpu_backend: None,
                                matrix_cache,
                                max_showdown_runouts,
                                equity_cache: HashMap::new(),
                            };
                            showdown_utility(*pot, *hero_invested, board, &mut ctx)
                        }
                    };
                }
            }
            result
        }
    }
}

fn cpu_infoset_choose_br_action(
    child_values: &[Vec<f32>],
    reach: &[f32],
    hero_count: usize,
    villain_count: usize,
    br_player: Player,
) -> Vec<f32> {
    let mut result = vec![0.0; hero_count * villain_count];
    match br_player {
        Player::Hero => {
            for h in 0..hero_count {
                let mut best_action = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                for (action_index, values) in child_values.iter().enumerate() {
                    let mut value = 0.0;
                    let mut weight = 0.0;
                    for v in 0..villain_count {
                        let pair = h * villain_count + v;
                        value += reach[pair] * values[pair];
                        weight += reach[pair];
                    }
                    if weight > 0.0 && value > best_value {
                        best_value = value;
                        best_action = action_index;
                    }
                }
                for v in 0..villain_count {
                    let pair = h * villain_count + v;
                    result[pair] = child_values[best_action][pair];
                }
            }
        }
        Player::Villain => {
            for v in 0..villain_count {
                let mut best_action = 0usize;
                let mut best_value = f32::INFINITY;
                for (action_index, values) in child_values.iter().enumerate() {
                    let mut value = 0.0;
                    let mut weight = 0.0;
                    for h in 0..hero_count {
                        let pair = h * villain_count + v;
                        value += reach[pair] * values[pair];
                        weight += reach[pair];
                    }
                    if weight > 0.0 && value < best_value {
                        best_value = value;
                        best_action = action_index;
                    }
                }
                for h in 0..hero_count {
                    let pair = h * villain_count + v;
                    result[pair] = child_values[best_action][pair];
                }
            }
        }
    }
    result
}

fn pair_dead_masks(heroes: &[ActiveCombo], villains: &[ActiveCombo]) -> Vec<u64> {
    let mut masks = Vec::with_capacity(heroes.len() * villains.len());
    for hero in heroes {
        for villain in villains {
            masks.push(hero.mask | villain.mask);
        }
    }
    masks
}

fn infoset_acting_player(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    public_infoset: usize,
) -> Player {
    let node_index = layout.infoset_node(public_infoset);
    let PublicNodeKind::Decision { state, .. } = &tree.nodes()[node_index].kind else {
        unreachable!("infoset nodes are decisions");
    };
    state.acting_player
}

#[allow(clippy::too_many_arguments)]
fn update_local_gap_detail(
    current: &mut Option<LocalGapDetail>,
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    public_infoset: usize,
    player: Player,
    combo_index: usize,
    gap: f32,
    weighted_gap: f32,
    reach_weight: f32,
    action_values: &[f32],
    average_strategy: &[f32],
    state: &DenseCfrState,
    current_strategy_buffer: &mut [f32],
) {
    if current
        .as_ref()
        .is_some_and(|detail| detail.weighted_gap >= weighted_gap)
    {
        return;
    }
    let indexer = ComboIndexer::new();
    let combo = indexer.combo(combo_index);
    let node_index = layout.infoset_node(public_infoset);
    let infoset = private_infoset(public_infoset, player, combo_index);
    let offset = infoset * layout.max_actions();
    state.strategy_for(infoset, current_strategy_buffer);
    let actions = (0..layout.action_count(public_infoset))
        .filter_map(|action| layout.action(tree, public_infoset, action))
        .map(format_action_candidate)
        .collect();
    *current = Some(LocalGapDetail {
        gap,
        weighted_gap,
        reach_weight,
        public_infoset,
        node_index,
        player,
        combo_index,
        combo: format_pokedr_cards(&[combo.first, combo.second]),
        action_values: action_values.to_vec(),
        average_strategy: average_strategy.to_vec(),
        current_strategy: current_strategy_buffer[..action_values.len()].to_vec(),
        regrets: state.regrets()[offset..offset + action_values.len()].to_vec(),
        strategy_sum: state.strategy_sum()[offset..offset + action_values.len()].to_vec(),
        actions,
    });
}

fn root_average_strategy(
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    indexer: &ComboIndexer,
    root_dead: u64,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(COMBO_COUNT * layout.max_actions());
    let mut strategy = vec![0.0; layout.max_actions()];
    for (combo_index, combo) in indexer.combos().iter().enumerate() {
        if combo.collides_with(root_dead) {
            result.extend(std::iter::repeat_n(0.0, layout.max_actions()));
            continue;
        }
        state.average_strategy_for(private_infoset(0, Player::Hero, combo_index), &mut strategy);
        result.extend_from_slice(&strategy);
    }
    result
}

fn root_average_strategy_from_sum_prefix(
    layout: &PostflopDenseLayout,
    strategy_sum_prefix: &[f32],
    indexer: &ComboIndexer,
    root_dead: u64,
) -> Vec<f32> {
    let max_actions = layout.max_actions();
    let action_count = layout.action_count(0);
    assert!(strategy_sum_prefix.len() >= COMBO_COUNT * max_actions);
    let mut result = Vec::with_capacity(COMBO_COUNT * max_actions);
    for (combo_index, combo) in indexer.combos().iter().enumerate() {
        let offset = combo_index * max_actions;
        let sums = &strategy_sum_prefix[offset..offset + max_actions];
        if combo.collides_with(root_dead) {
            result.extend(std::iter::repeat_n(0.0, max_actions));
            continue;
        }
        let normalizer: f32 = sums[..action_count].iter().sum();
        if normalizer > f32::EPSILON {
            result.extend(sums.iter().enumerate().map(|(action, value)| {
                if action < action_count {
                    *value / normalizer
                } else {
                    0.0
                }
            }));
        } else {
            let uniform = 1.0 / action_count as f32;
            result.extend((0..max_actions).map(
                |action| {
                    if action < action_count { uniform } else { 0.0 }
                },
            ));
        }
    }
    result
}

fn weighted_root_action_probabilities(
    layout: &PostflopDenseLayout,
    root_strategy: &[f32],
    hero_weights: &[f32],
    indexer: &ComboIndexer,
    root_dead: u64,
) -> Vec<f32> {
    let mut probabilities = vec![0.0; layout.max_actions()];
    let mut weight_sum = 0.0f32;
    for ((strategy, combo), weight) in root_strategy
        .chunks(layout.max_actions())
        .zip(indexer.combos())
        .zip(hero_weights)
    {
        if *weight <= 0.0 || combo.collides_with(root_dead) {
            continue;
        }
        let mass: f32 = strategy.iter().sum();
        if mass <= 0.0 {
            continue;
        }
        for (target, value) in probabilities.iter_mut().zip(strategy) {
            *target += *value * *weight;
        }
        weight_sum += *weight;
    }
    if weight_sum > 0.0 {
        for probability in &mut probabilities {
            *probability /= weight_sum;
        }
    }
    probabilities
}

fn mean_l1_delta(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    if left.is_empty() {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / left.len() as f32
}

struct PostflopEvaluationContext<'a> {
    hero_cards: [PokedrCard; 2],
    villain_cards: [PokedrCard; 2],
    hero_combo: usize,
    villain_combo: usize,
    gpu_backend: Option<&'a GpuDenseCfrBackend>,
    matrix_cache: &'a RefCell<ShowdownMatrixCache>,
    max_showdown_runouts: usize,
    equity_cache: HashMap<u64, f32>,
}

struct ShowdownMatrixCache {
    entries: HashMap<u64, Vec<f32>>,
    order: VecDeque<u64>,
    capacity: usize,
}

impl ShowdownMatrixCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn contains_key(&self, key: &u64) -> bool {
        self.entries.contains_key(key)
    }

    fn get(&self, key: &u64) -> Option<&Vec<f32>> {
        self.entries.get(key)
    }

    fn insert(&mut self, key: u64, matrix: Vec<f32>) {
        if !self.entries.contains_key(&key) {
            self.order.push_back(key);
        }
        self.entries.insert(key, matrix);
        while self.entries.len() > self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if oldest != key {
                self.entries.remove(&oldest);
            }
        }
    }
}

struct PostflopPlan {
    tree: SubgameTree,
    layout: PostflopDenseLayout,
    state: DenseCfrState,
    node_by_state: HashMap<PublicStateKey, usize>,
}

#[derive(Clone, Default)]
struct SharedPostflopPlan(Rc<RefCell<Option<PostflopPlan>>>);

impl SharedPostflopPlan {
    fn set(&self, plan: PostflopPlan) {
        *self.0.borrow_mut() = Some(plan);
    }

    fn take_matching(&self, state: &PublicState) -> Option<PostflopPlan> {
        if self
            .0
            .borrow()
            .as_ref()
            .is_some_and(|plan| plan.contains_state(state))
        {
            self.0.borrow_mut().take()
        } else {
            None
        }
    }
}

impl PostflopPlan {
    fn new(tree: SubgameTree, layout: PostflopDenseLayout, state: DenseCfrState) -> Self {
        let mut node_by_state = HashMap::new();
        for (node_index, node) in tree.nodes().iter().enumerate() {
            if let PublicNodeKind::Decision { state, .. } = &node.kind {
                node_by_state.insert(PublicStateKey::from(state), node_index);
            }
        }
        Self {
            tree,
            layout,
            state,
            node_by_state,
        }
    }

    fn contains_state(&self, state: &PublicState) -> bool {
        self.node_by_state
            .contains_key(&PublicStateKey::from(state))
    }

    fn node_for_state(&self, state: &PublicState) -> Option<usize> {
        self.node_by_state
            .get(&PublicStateKey::from(state))
            .copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PublicStateKey {
    street: u8,
    board_mask: u64,
    pot: u32,
    to_call: u32,
    acting_player: u8,
    raises_this_street: u8,
}

impl From<&PublicState> for PublicStateKey {
    fn from(state: &PublicState) -> Self {
        Self {
            street: match state.street {
                Street::Flop => 0,
                Street::Turn => 1,
                Street::River => 2,
            },
            board_mask: state.board.deck_mask(),
            pot: state.pot,
            to_call: state.to_call,
            acting_player: match state.acting_player {
                Player::Hero => 0,
                Player::Villain => 1,
            },
            raises_this_street: state.raises_this_street,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SharedHistory(Rc<RefCell<Vec<HistoryRecord>>>);

impl SharedHistory {
    fn records(&self) -> Vec<HistoryRecord> {
        self.0.borrow().clone()
    }
}

#[derive(Clone)]
struct PokedrAgentHistorian {
    history: SharedHistory,
    shared_plan: SharedPostflopPlan,
    config: PokedrAgentConfig,
}

impl Historian for PokedrAgentHistorian {
    fn record_action(
        &mut self,
        _id: u128,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        let build_on_flop_start = matches!(&action, Action::RoundAdvance(Round::Flop));
        self.history.0.borrow_mut().push(HistoryRecord {
            before_game_state: None,
            action,
            after_game_state: game_state.clone(),
        });
        if build_on_flop_start && game_state.board.len() == 3 {
            if let Some(public_state) = public_state_from_game(game_state) {
                if self
                    .shared_plan
                    .0
                    .borrow()
                    .as_ref()
                    .is_none_or(|plan| !plan.contains_state(&public_state))
                {
                    let records = self.history.records();
                    self.shared_plan.set(build_postflop_plan(
                        public_state,
                        game_state,
                        &self.config,
                        &records,
                    ));
                }
            }
        }
        Ok(())
    }
}

fn build_postflop_plan(
    public_state: PublicState,
    game_state: &GameState,
    config: &PokedrAgentConfig,
    records: &[HistoryRecord],
) -> PostflopPlan {
    let started = Instant::now();
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: build postflop plan street={:?} board_cards={} pot={} to_call={} iterations={}",
            public_state.street,
            public_state.board.cards().len(),
            public_state.pot,
            public_state.to_call,
            config.cfr_iterations
        );
    }
    let tree = SubgameTree::build(
        public_state,
        SubgameTreeConfig {
            action_set: config.action_set.clone(),
            max_raises_per_street: config.max_raises_per_street,
        },
    );
    let layout = PostflopDenseLayout::from_tree(&tree);
    let indexer = ComboIndexer::new();
    let hero_weights = observed_hero_range_weights(game_state, &indexer);
    let villain_weights = observed_villain_range_weights(records, game_state, &indexer);
    let state = solve_public_tree_cfr(&tree, &layout, config, &hero_weights, &villain_weights);
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: plan ready decisions={} chance={} terminals={} elapsed={:.2}s",
            tree.decision_count(),
            tree.chance_count(),
            tree.terminal_count(),
            started.elapsed().as_secs_f32()
        );
    }
    PostflopPlan::new(tree, layout, state)
}

fn solve_public_tree_cfr(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    config: &PokedrAgentConfig,
    hero_weights: &[f32],
    villain_weights: &[f32],
) -> DenseCfrState {
    let mut dense_config = layout.dense_config(config.cfr_variant);
    dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
    let indexer = ComboIndexer::new();
    let gpu_backend = cfr_gpu_backend();
    let matrix_cache = RefCell::new(ShowdownMatrixCache::new(showdown_matrix_cache_capacity()));
    if let Some(backend) = gpu_backend.as_deref()
        && let Some(gpu_state) = try_solve_gpu_public_tree_resident(
            tree,
            layout,
            &indexer,
            backend,
            &dense_config,
            config,
            hero_weights,
            villain_weights,
            &matrix_cache,
        )
    {
        return gpu_state;
    }

    assert_dense_private_state_feasible(layout);
    let mut state =
        DenseCfrState::new_with_legal_actions(dense_config.clone(), private_legal_actions(layout));
    let mut batch = DenseCfrIteration::new(&dense_config);
    for iteration in 1..=config.cfr_iterations.max(1) {
        let iteration_started = Instant::now();
        if solver_progress_enabled() && should_report_iteration(iteration, config.cfr_iterations) {
            eprintln!(
                "pokedr: cfr iteration {}/{} start",
                iteration,
                config.cfr_iterations.max(1)
            );
        }
        fill_public_tree_iteration(
            tree,
            layout,
            &indexer,
            gpu_backend.as_deref(),
            &state,
            config,
            hero_weights,
            villain_weights,
            &matrix_cache,
            &mut batch,
        );
        if !config.cfr_variant.is_dcfr_plus() {
            let average_weight = iteration as f32;
            for weight in &mut batch.strategy_weights {
                *weight *= average_weight;
            }
        }
        batch.validate(&dense_config);
        if let Some(backend) = gpu_backend.as_deref() {
            backend
                .update_all_infosets(
                    &mut state,
                    &batch.action_values,
                    &batch.reach_weights,
                    &batch.strategy_weights,
                    iteration,
                )
                .expect("GPU CFR regret update failed");
            continue;
        }
        assert!(
            cfg!(test),
            "GPU CFR backend is required for postflop solving"
        );
        state.update_all_infosets(
            &batch.action_values,
            &batch.reach_weights,
            &batch.strategy_weights,
            iteration,
        );
        if solver_progress_enabled() && should_report_iteration(iteration, config.cfr_iterations) {
            eprintln!(
                "pokedr: cfr iteration {}/{} elapsed={:.2}s",
                iteration,
                config.cfr_iterations.max(1),
                iteration_started.elapsed().as_secs_f32()
            );
        }
    }
    state
}

fn assert_dense_private_state_feasible(layout: &PostflopDenseLayout) {
    let dense_action_slots = layout
        .infoset_count()
        .saturating_mul(PRIVATE_INFOS_PER_PUBLIC)
        .saturating_mul(layout.max_actions());
    let dense_action_bytes = dense_action_slots.saturating_mul(std::mem::size_of::<f32>());
    const MAX_CPU_FALLBACK_ACTION_BYTES: usize = 1usize << 30;
    assert!(
        dense_action_bytes <= MAX_CPU_FALLBACK_ACTION_BYTES,
        "postflop solver cannot use dense private fallback for this tree: public_infosets={} max_actions={} dense_action_slots={} dense_f32_bytes={} compact_action_slots={}. The solver needs chunked compact GPU state for this tree.",
        layout.infoset_count(),
        layout.max_actions(),
        dense_action_slots,
        dense_action_bytes,
        layout
            .total_actions()
            .saturating_mul(PRIVATE_INFOS_PER_PUBLIC)
    );
}

fn solver_progress_enabled() -> bool {
    !cfg!(test) && std::env::var_os("POKEDR_SOLVER_PROGRESS_OFF").is_none()
}

fn should_report_iteration(iteration: usize, total: usize) -> bool {
    let total = total.max(1);
    iteration == 1 || iteration == total || iteration % (total / 4).max(1) == 0
}

fn cfr_gpu_backend() -> Option<Rc<GpuDenseCfrBackend>> {
    if cfg!(test) || std::env::var_os("POKEDR_DISABLE_GPU_CFR").is_some() {
        return None;
    }
    thread_local! {
        static SHARED_GPU_BACKEND: RefCell<Option<Rc<GpuDenseCfrBackend>>> = const { RefCell::new(None) };
    }
    SHARED_GPU_BACKEND.with(|backend| {
        if backend.borrow().is_none() {
            *backend.borrow_mut() = GpuDenseCfrBackend::new().ok().map(Rc::new);
        }
        backend.borrow().clone()
    })
}

fn showdown_matrix_cache_capacity() -> usize {
    std::env::var("POKEDR_SHOWDOWN_MATRIX_CACHE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2)
}

#[allow(clippy::too_many_arguments)]
fn try_solve_gpu_public_tree_resident(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    indexer: &ComboIndexer,
    backend: &GpuDenseCfrBackend,
    dense_config: &pokedr_core::dense_cfr::DenseCfrConfig,
    config: &PokedrAgentConfig,
    hero_weights: &[f32],
    villain_weights: &[f32],
    matrix_cache: &RefCell<ShowdownMatrixCache>,
) -> Option<DenseCfrState> {
    let linearized = linearize_gpu_public_tree(tree, layout, backend, config, matrix_cache)?;
    if !gpu_dense_state_fits_single_buffers(backend, dense_config) {
        if solver_progress_enabled() {
            eprintln!(
                "pokedr: skipping dense resident GPU state; public_infosets={} private_infosets={} max_actions={} dense_action_slots={} compact_action_slots={}",
                layout.infoset_count(),
                dense_config.infosets,
                dense_config.actions,
                dense_config.infosets.saturating_mul(dense_config.actions),
                layout
                    .total_actions()
                    .saturating_mul(PRIVATE_INFOS_PER_PUBLIC)
            );
        }
        return None;
    }
    assert_gpu_dense_binding_feasible(backend, dense_config);
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: gpu resident public tree nodes={} showdown_boards={} iterations={}",
            linearized.nodes.len(),
            linearized.showdown_boards.len(),
            config.cfr_iterations.max(1)
        );
    }
    let combos = gpu_private_combos();
    let root_dead = root_board(tree).deck_mask();
    let combo_legal: Vec<u32> = indexer
        .combos()
        .iter()
        .map(|combo| (!combo.collides_with(root_dead)) as u32)
        .collect();
    let mut gpu_state = backend
        .zeroed_state_with_legal_actions(dense_config.clone(), private_legal_actions(layout));
    let cfr_started = Instant::now();
    backend
        .public_tree_run_iterations(
            &linearized.nodes,
            &linearized.children,
            &linearized.child_cards,
            &combos,
            &combo_legal,
            hero_weights,
            villain_weights,
            &linearized.showdown_boards,
            &mut gpu_state,
            config.cfr_iterations.max(1),
        )
        .ok()?;
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: gpu resident cfr queued {} iterations elapsed={:.2}s",
            config.cfr_iterations.max(1),
            cfr_started.elapsed().as_secs_f32()
        );
    }
    gpu_state.download(backend).ok()
}

fn gpu_dense_state_fits_single_buffers(
    backend: &GpuDenseCfrBackend,
    config: &pokedr_core::dense_cfr::DenseCfrConfig,
) -> bool {
    let max_buffer_bytes = backend.max_buffer_size() as usize;
    let max_binding_bytes = backend.max_storage_buffer_binding_size() as usize;
    let action_slots = config.infosets.saturating_mul(config.actions);
    let f32_action_bytes = action_slots.saturating_mul(std::mem::size_of::<f32>());
    let u32_action_bytes = action_slots.saturating_mul(std::mem::size_of::<u32>());
    let infoset_bytes = config.infosets.saturating_mul(std::mem::size_of::<f32>());
    f32_action_bytes <= max_buffer_bytes
        && f32_action_bytes <= max_binding_bytes
        && u32_action_bytes <= max_buffer_bytes
        && u32_action_bytes <= max_binding_bytes
        && infoset_bytes <= max_buffer_bytes
        && infoset_bytes <= max_binding_bytes
}

fn fill_public_tree_iteration(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    indexer: &ComboIndexer,
    gpu_backend: Option<&GpuDenseCfrBackend>,
    cfr_state: &DenseCfrState,
    config: &PokedrAgentConfig,
    hero_weights: &[f32],
    villain_weights: &[f32],
    matrix_cache: &RefCell<ShowdownMatrixCache>,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(0.0);
    batch.strategy_weights.fill(0.0);
    let mut value_weights = vec![0.0; batch.action_values.len()];
    let root_dead = root_board(tree).deck_mask();
    if try_fill_gpu_public_tree_iteration(
        tree,
        layout,
        indexer,
        gpu_backend,
        cfr_state,
        config,
        hero_weights,
        villain_weights,
        matrix_cache,
        batch,
        root_dead,
    ) {
        return;
    }
    assert!(
        cfg!(test),
        "GPU public-tree CFR path is required for postflop solving"
    );
    let legal_combos: Vec<_> = legal_private_combos(indexer, root_dead)
        .map(|combo_index| {
            let cards = combo_cards(indexer.combo(combo_index));
            (combo_index, cards, hero_mask(cards))
        })
        .collect();
    for (hero_offset, (hero_combo, hero_cards, hero_dead)) in legal_combos.iter().enumerate() {
        let hero_weight = hero_weights[*hero_combo];
        if hero_weight <= 0.0 {
            continue;
        }
        if solver_progress_enabled() && hero_offset % 64 == 0 {
            eprintln!(
                "pokedr: cfr combo block {}/{}",
                hero_offset + 1,
                legal_combos.len()
            );
        }
        for (villain_combo, villain_cards, villain_dead) in &legal_combos {
            let villain_weight = villain_weights[*villain_combo];
            if villain_weight <= 0.0 {
                continue;
            }
            if hero_dead & villain_dead != 0 {
                continue;
            }
            let mut ctx = PostflopEvaluationContext {
                hero_cards: *hero_cards,
                villain_cards: *villain_cards,
                hero_combo: *hero_combo,
                villain_combo: *villain_combo,
                gpu_backend,
                matrix_cache,
                max_showdown_runouts: config.max_showdown_runouts.max(1),
                equity_cache: HashMap::new(),
            };
            traverse_cfr_node(
                tree,
                layout,
                0,
                None,
                None,
                *hero_combo,
                *villain_combo,
                hero_weight,
                villain_weight,
                1.0,
                cfr_state,
                &mut ctx,
                batch,
                &mut value_weights,
            );
        }
    }

    for (value, weight) in batch.action_values.iter_mut().zip(value_weights) {
        if weight > 0.0 {
            *value /= weight;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_fill_gpu_public_tree_iteration(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    indexer: &ComboIndexer,
    gpu_backend: Option<&GpuDenseCfrBackend>,
    cfr_state: &DenseCfrState,
    config: &PokedrAgentConfig,
    hero_weights: &[f32],
    villain_weights: &[f32],
    matrix_cache: &RefCell<ShowdownMatrixCache>,
    batch: &mut DenseCfrIteration,
    root_dead: u64,
) -> bool {
    let Some(backend) = gpu_backend else {
        return false;
    };
    let Some(linearized) = linearize_gpu_public_tree(tree, layout, backend, config, matrix_cache)
    else {
        return false;
    };
    if solver_progress_enabled() {
        eprintln!(
            "pokedr: gpu public tree nodes={} showdown_boards={}",
            linearized.nodes.len(),
            linearized.showdown_boards.len()
        );
    }
    let combos = gpu_private_combos();
    let combo_legal: Vec<u32> = indexer
        .combos()
        .iter()
        .map(|combo| (!combo.collides_with(root_dead)) as u32)
        .collect();
    let Ok(values) = backend.public_tree_iteration_values(
        &linearized.nodes,
        &linearized.children,
        &linearized.child_cards,
        &combos,
        &combo_legal,
        hero_weights,
        villain_weights,
        &linearized.showdown_boards,
        cfr_state,
    ) else {
        return false;
    };
    if values.action_values.len() != batch.action_values.len()
        || values.reach_weights.len() != batch.reach_weights.len()
        || values.strategy_weights.len() != batch.strategy_weights.len()
    {
        return false;
    }
    batch.action_values.copy_from_slice(&values.action_values);
    batch.reach_weights.copy_from_slice(&values.reach_weights);
    batch
        .strategy_weights
        .copy_from_slice(&values.strategy_weights);
    true
}

struct GpuLinearizedPublicTree {
    nodes: Vec<GpuPublicTreeNode>,
    children: Vec<u32>,
    child_cards: Vec<u32>,
    showdown_boards: Vec<GpuFinalBoard>,
}

fn linearize_gpu_public_tree(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    _backend: &GpuDenseCfrBackend,
    _config: &PokedrAgentConfig,
    _matrix_cache: &RefCell<ShowdownMatrixCache>,
) -> Option<GpuLinearizedPublicTree> {
    let mut nodes = Vec::with_capacity(tree.nodes().len());
    let mut children = Vec::new();
    let mut child_cards = Vec::new();
    let mut showdown_boards = Vec::new();
    let mut showdown_offsets = HashMap::new();
    let public_chance_reach = public_chance_reaches(tree);

    for (node_index, node) in tree.nodes().iter().enumerate() {
        let first_child = children.len() as u32;
        let child_count = node.children.len() as u32;
        children.extend(node.children.iter().map(|child| *child as u32));
        match &node.kind {
            PublicNodeKind::Decision { state, .. } => {
                child_cards.extend(std::iter::repeat_n(52, node.children.len()));
                nodes.push(GpuPublicTreeNode {
                    kind: 0,
                    acting_player: player_code(state.acting_player),
                    public_infoset: layout.node_infoset(node_index)? as u32,
                    first_child,
                    child_count,
                    terminal_kind: 0,
                    showdown_offset: 0,
                    _pad0: 0,
                    pot: 0.0,
                    hero_invested: 0.0,
                    _pad1: public_chance_reach[node_index],
                    _pad2: 0.0,
                });
            }
            PublicNodeKind::Chance { cards, .. } => {
                if cards.len() != node.children.len() {
                    return None;
                }
                child_cards.extend(cards.iter().map(|card| card.index() as u32));
                nodes.push(GpuPublicTreeNode {
                    kind: 1,
                    acting_player: 0,
                    public_infoset: 0,
                    first_child,
                    child_count,
                    terminal_kind: 0,
                    showdown_offset: 0,
                    _pad0: 0,
                    pot: 0.0,
                    hero_invested: 0.0,
                    _pad1: public_chance_reach[node_index],
                    _pad2: 0.0,
                });
            }
            PublicNodeKind::Terminal {
                kind,
                board,
                pot,
                hero_invested,
                ..
            } => {
                child_cards.extend(std::iter::repeat_n(52, node.children.len()));
                let (terminal_kind, showdown_offset, showdown_count) = match kind {
                    TerminalKind::Fold => (fold_terminal_code(tree, node_index)?, 0, 0),
                    TerminalKind::Showdown => {
                        let key = board.deck_mask();
                        let (offset, count) = if let Some(offset_count) = showdown_offsets.get(&key)
                        {
                            *offset_count
                        } else {
                            let offset = showdown_boards.len() as u32;
                            let final_boards = gpu_full_final_boards(board);
                            let count = final_boards.len() as u32;
                            showdown_boards.extend(final_boards);
                            showdown_offsets.insert(key, (offset, count));
                            (offset, count)
                        };
                        (2, offset, count)
                    }
                };
                nodes.push(GpuPublicTreeNode {
                    kind: 2,
                    acting_player: 0,
                    public_infoset: 0,
                    first_child,
                    child_count,
                    terminal_kind,
                    showdown_offset,
                    _pad0: showdown_count,
                    pot: *pot as f32,
                    hero_invested: *hero_invested as f32,
                    _pad1: public_chance_reach[node_index],
                    _pad2: full_runout_pair_denominator(board.cards().len()) as f32,
                });
            }
        }
    }
    Some(GpuLinearizedPublicTree {
        nodes,
        children,
        child_cards,
        showdown_boards,
    })
}

fn public_chance_reaches(tree: &SubgameTree) -> Vec<f32> {
    let mut reaches = vec![0.0; tree.nodes().len()];
    reaches[0] = 1.0;
    for node_index in 0..tree.nodes().len() {
        let reach = reaches[node_index];
        if reach == 0.0 {
            continue;
        }
        match &tree.nodes()[node_index].kind {
            PublicNodeKind::Decision { .. } => {
                for child in &tree.nodes()[node_index].children {
                    reaches[*child] += reach;
                }
            }
            PublicNodeKind::Chance { cards, .. } => {
                let denominator = cards.len().saturating_sub(4).max(1) as f32;
                for child in &tree.nodes()[node_index].children {
                    reaches[*child] += reach / denominator;
                }
            }
            PublicNodeKind::Terminal { .. } => {}
        }
    }
    reaches
}

fn player_code(player: Player) -> u32 {
    match player {
        Player::Hero => 0,
        Player::Villain => 1,
    }
}

fn fold_terminal_code(tree: &SubgameTree, node_index: usize) -> Option<u32> {
    let parent = tree.nodes()[node_index].parent?;
    let child_position = tree.nodes()[parent]
        .children
        .iter()
        .position(|child| *child == node_index)?;
    let PublicNodeKind::Decision { state, actions } = &tree.nodes()[parent].kind else {
        return None;
    };
    if actions.get(child_position)?.action != PlayerAction::Fold {
        return None;
    }
    Some(match state.acting_player {
        Player::Hero => 0,
        Player::Villain => 1,
    })
}

#[allow(clippy::too_many_arguments)]
fn traverse_cfr_node(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    parent_state: Option<&PublicState>,
    parent_action: Option<PlayerAction>,
    hero_combo: usize,
    villain_combo: usize,
    hero_reach: f32,
    villain_reach: f32,
    public_reach: f32,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
    batch: &mut DenseCfrIteration,
    value_weights: &mut [f32],
) -> f32 {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision { state, actions } => {
            let Some(public_infoset) = layout.node_infoset(node_index) else {
                return 0.0;
            };
            let acting_combo = match state.acting_player {
                Player::Hero => hero_combo,
                Player::Villain => villain_combo,
            };
            let private_infoset =
                private_infoset(public_infoset, state.acting_player, acting_combo);
            let offset = private_infoset * layout.max_actions();
            let mut strategy = vec![0.0; layout.max_actions()];
            cfr_state.strategy_for(private_infoset, &mut strategy);

            let mut action_values = vec![0.0; actions.len()];
            for (action_index, action) in actions.iter().enumerate() {
                let child = layout
                    .child_for_action(public_infoset, action_index)
                    .expect("legal action must have a child");
                let probability = strategy[action_index];
                let (next_hero_reach, next_villain_reach) = match state.acting_player {
                    Player::Hero => (hero_reach * probability, villain_reach),
                    Player::Villain => (hero_reach, villain_reach * probability),
                };
                action_values[action_index] = traverse_cfr_node(
                    tree,
                    layout,
                    child,
                    Some(state),
                    Some(action.action),
                    hero_combo,
                    villain_combo,
                    next_hero_reach,
                    next_villain_reach,
                    public_reach,
                    cfr_state,
                    ctx,
                    batch,
                    value_weights,
                );
            }

            let opponent_reach = match state.acting_player {
                Player::Hero => public_reach * villain_reach,
                Player::Villain => public_reach * hero_reach,
            };
            let own_reach = match state.acting_player {
                Player::Hero => public_reach * hero_reach,
                Player::Villain => public_reach * villain_reach,
            };
            if opponent_reach > 0.0 || own_reach > 0.0 {
                for (action_index, hero_value) in action_values.iter().copied().enumerate() {
                    let player_value = match state.acting_player {
                        Player::Hero => hero_value,
                        Player::Villain => -hero_value,
                    };
                    batch.action_values[offset + action_index] += opponent_reach * player_value;
                    value_weights[offset + action_index] += opponent_reach;
                }
                batch.reach_weights[private_infoset] += opponent_reach;
                batch.strategy_weights[private_infoset] += own_reach;
            }

            action_values
                .iter()
                .zip(strategy)
                .map(|(value, probability)| value * probability)
                .sum()
        }
        PublicNodeKind::Chance { cards, .. } => {
            let valid_count = cards
                .iter()
                .filter(|card| card.deck_mask() & private_dead_mask(ctx) == 0)
                .count();
            if valid_count == 0 {
                return 0.0;
            }
            let child_public_reach = public_reach / valid_count as f32;
            let mut sum = 0.0;
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                if card.deck_mask() & private_dead_mask(ctx) != 0 {
                    continue;
                }
                sum += traverse_cfr_node(
                    tree,
                    layout,
                    *child,
                    None,
                    None,
                    hero_combo,
                    villain_combo,
                    hero_reach,
                    villain_reach,
                    child_public_reach,
                    cfr_state,
                    ctx,
                    batch,
                    value_weights,
                );
            }
            sum / valid_count as f32
        }
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            ..
        } => match kind {
            TerminalKind::Fold => fold_utility(
                parent_state.expect("fold terminal must have a parent decision"),
                parent_action.expect("fold terminal must have a parent action"),
                *pot,
                *hero_invested,
            ),
            TerminalKind::Showdown => showdown_utility(*pot, *hero_invested, board, ctx),
        },
    }
}

fn best_average_strategy_action(
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    public_infoset: usize,
    player: Player,
    hero_combo: usize,
) -> usize {
    let mut strategy = vec![0.0; layout.max_actions()];
    state.average_strategy_for(
        private_infoset(public_infoset, player, hero_combo),
        &mut strategy,
    );
    let mut best = 0;
    let mut best_probability = f32::NEG_INFINITY;
    for (action, probability) in strategy
        .iter()
        .copied()
        .enumerate()
        .take(layout.action_count(public_infoset))
    {
        if probability > best_probability {
            best = action;
            best_probability = probability;
        }
    }
    best
}

fn observed_villain_range_weights(
    _records: &[HistoryRecord],
    game_state: &GameState,
    indexer: &ComboIndexer,
) -> Vec<f32> {
    let hero_mask = hero_cards_from_game(game_state)
        .map(hero_mask)
        .unwrap_or_default();
    simple_preflop_range_weights(
        indexer,
        PreflopClass::Speculative,
        hero_mask | board_mask_from_game(game_state),
    )
}

fn observed_hero_range_weights(game_state: &GameState, indexer: &ComboIndexer) -> Vec<f32> {
    let board_mask = board_mask_from_game(game_state);
    if let Some(cards) = hero_cards_from_game(game_state)
        && let Some(index) = indexer.index(cards[0], cards[1])
    {
        let mut weights = vec![0.0; COMBO_COUNT];
        if hero_mask(cards) & board_mask == 0 {
            weights[index] = 1.0;
        }
        return weights;
    }
    simple_preflop_range_weights(indexer, PreflopClass::Playable, board_mask)
}

fn fixed_flop_root_weights(indexer: &ComboIndexer, root_dead: u64) -> (Vec<f32>, Vec<f32>) {
    let mut hero_weights = simple_preflop_range_weights(indexer, PreflopClass::Playable, root_dead);
    let mut villain_weights =
        simple_preflop_range_weights(indexer, PreflopClass::Speculative, root_dead);
    apply_env_range_limit("POKEDR_HERO_RANGE_LIMIT", &mut hero_weights);
    apply_env_range_limit("POKEDR_VILLAIN_RANGE_LIMIT", &mut villain_weights);
    (hero_weights, villain_weights)
}

fn apply_env_range_limit(name: &str, weights: &mut [f32]) {
    let Some(limit) = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return;
    };
    let mut kept = 0usize;
    for weight in weights {
        if *weight <= 0.0 {
            continue;
        }
        kept += 1;
        if kept > limit {
            *weight = 0.0;
        }
    }
}

fn simple_preflop_range_weights(
    indexer: &ComboIndexer,
    minimum_class: PreflopClass,
    dead_mask: u64,
) -> Vec<f32> {
    indexer
        .combos()
        .iter()
        .map(|combo| {
            if combo.collides_with(dead_mask) {
                0.0
            } else if classify_pokedr_combo(*combo) >= minimum_class {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn classify_pokedr_combo(combo: Combo) -> PreflopClass {
    classify_preflop_values(
        pokedr_rank_value(combo.first.rank()),
        pokedr_rank_value(combo.second.rank()),
        combo.first.suit() == combo.second.suit(),
    )
}

fn pokedr_rank_value(rank: PokedrRank) -> u8 {
    rank as u8 + 2
}

fn private_infoset(public_infoset: usize, _player: Player, private_combo: usize) -> usize {
    public_infoset * PRIVATE_INFOS_PER_PUBLIC + private_combo
}

fn hero_combo_index(indexer: &ComboIndexer, hero_cards: [PokedrCard; 2]) -> usize {
    indexer
        .index(hero_cards[0], hero_cards[1])
        .expect("hero cards must form a valid private combo")
}

fn private_legal_actions(layout: &PostflopDenseLayout) -> Vec<bool> {
    let mut legal = Vec::with_capacity(
        layout.infoset_count() * PRIVATE_INFOS_PER_PUBLIC * layout.max_actions(),
    );
    for infoset in 0..layout.infoset_count() {
        let offset = infoset * layout.max_actions();
        for _ in 0..PRIVATE_INFOS_PER_PUBLIC {
            legal.extend_from_slice(&layout.legal_actions()[offset..offset + layout.max_actions()]);
        }
    }
    legal
}

fn assert_gpu_dense_binding_feasible(
    backend: &GpuDenseCfrBackend,
    config: &pokedr_core::dense_cfr::DenseCfrConfig,
) {
    let max_binding_bytes = backend.max_storage_buffer_binding_size() as usize;
    let infoset_bytes = config.infosets.saturating_mul(std::mem::size_of::<f32>());
    let action_slots = config.infosets.saturating_mul(config.actions);
    let action_bytes = action_slots.saturating_mul(std::mem::size_of::<f32>());
    if infoset_bytes <= max_binding_bytes && action_bytes <= max_binding_bytes {
        return;
    }

    if solver_progress_enabled() {
        let public_infosets = config.infosets / PRIVATE_INFOS_PER_PUBLIC.max(1);
        eprintln!(
            "pokedr: gpu dense CFR state requires tiled bindings public_infosets={} private_infosets={} actions={} infoset_bytes={} action_bytes={} max_binding_bytes={}",
            public_infosets,
            config.infosets,
            config.actions,
            infoset_bytes,
            action_bytes,
            max_binding_bytes
        );
    }
}

fn fold_utility(
    parent_state: &PublicState,
    parent_action: PlayerAction,
    pot: u32,
    hero_invested: u32,
) -> f32 {
    if parent_action != PlayerAction::Fold {
        return 0.0;
    }
    if parent_state.acting_player == Player::Hero {
        -(hero_invested as f32)
    } else {
        pot as f32 - hero_invested as f32
    }
}

fn showdown_utility(
    pot: u32,
    hero_invested: u32,
    board: &Board,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    let equity = showdown_equity(board, ctx);
    equity * pot as f32 - hero_invested as f32
}

fn showdown_equity(board: &Board, ctx: &mut PostflopEvaluationContext) -> f32 {
    if let Some(equity) = gpu_showdown_matrix_equity(board, ctx) {
        return equity;
    }

    let key = board.deck_mask() ^ private_dead_mask(ctx).rotate_left(17);
    if let Some(equity) = ctx.equity_cache.get(&key) {
        return *equity;
    }

    let dead = board.deck_mask() | private_dead_mask(ctx);
    let runouts = completion_runouts(board, dead, ctx.max_showdown_runouts);
    if let Some(equity) = gpu_showdown_equity(board, &runouts, ctx) {
        ctx.equity_cache.insert(key, equity);
        return equity;
    }
    let mut equity_sum = 0.0;
    let mut matchup_count = 0.0;

    for runout in &runouts {
        let mut final_board = board.cards().to_vec();
        final_board.extend(runout.iter().copied());
        let board_mask = final_board
            .iter()
            .fold(0u64, |mask, card| mask | card.deck_mask());
        let hero_strength = evaluate_seven(ctx.hero_cards, &final_board);
        if ctx
            .villain_cards
            .iter()
            .any(|card| card.deck_mask() & board_mask != 0)
        {
            continue;
        }
        equity_sum += heads_up_equity(
            hero_strength,
            Combo::new(ctx.villain_cards[0], ctx.villain_cards[1]),
            &final_board,
        );
        matchup_count += 1.0;
    }

    let equity = if matchup_count > 0.0 {
        equity_sum / matchup_count
    } else {
        0.5
    };
    ctx.equity_cache.insert(key, equity);
    equity
}

fn gpu_showdown_matrix_equity(
    board: &Board,
    ctx: &mut PostflopEvaluationContext<'_>,
) -> Option<f32> {
    let backend = ctx.gpu_backend?;
    let key = board.deck_mask();
    if !ctx.matrix_cache.borrow().contains_key(&key) {
        let combos = gpu_private_combos();
        let final_boards = gpu_final_boards(board, ctx.max_showdown_runouts);
        let matrix = backend.showdown_matrix(&combos, &final_boards).ok()?;
        ctx.matrix_cache.borrow_mut().insert(key, matrix);
    }
    let cache = ctx.matrix_cache.borrow();
    let matrix = cache.get(&key)?;
    matrix
        .get(ctx.hero_combo * COMBO_COUNT + ctx.villain_combo)
        .copied()
}

fn gpu_showdown_equity(
    board: &Board,
    runouts: &[Vec<PokedrCard>],
    ctx: &PostflopEvaluationContext<'_>,
) -> Option<f32> {
    let backend = ctx.gpu_backend?;
    if runouts.len() < 8 {
        return None;
    }
    let mut tasks = Vec::with_capacity(runouts.len());
    for runout in runouts {
        let mut final_board = board.cards().to_vec();
        final_board.extend(runout.iter().copied());
        if final_board.len() != 5 {
            return None;
        }
        if ctx.villain_cards.iter().any(|card| {
            card.deck_mask()
                & final_board
                    .iter()
                    .fold(0u64, |mask, card| mask | card.deck_mask())
                != 0
        }) {
            continue;
        }
        tasks.push(GpuShowdownTask {
            cards: [
                ctx.hero_cards[0].index() as u32,
                ctx.hero_cards[1].index() as u32,
                ctx.villain_cards[0].index() as u32,
                ctx.villain_cards[1].index() as u32,
                final_board[0].index() as u32,
                final_board[1].index() as u32,
                final_board[2].index() as u32,
                final_board[3].index() as u32,
                final_board[4].index() as u32,
            ],
        });
    }
    if tasks.is_empty() {
        return Some(0.5);
    }
    let equities = backend.showdown_equities(&tasks).ok()?;
    Some(equities.iter().sum::<f32>() / equities.len() as f32)
}

fn completion_runouts(board: &Board, dead: u64, limit: usize) -> Vec<Vec<PokedrCard>> {
    let missing = 5usize.saturating_sub(board.cards().len());
    if missing == 0 {
        return vec![Vec::new()];
    }
    let deck: Vec<_> = (0..PokedrCard::COUNT as u8)
        .map(PokedrCard::from_index)
        .filter(|card| card.deck_mask() & dead == 0)
        .collect();
    let mut runouts = Vec::new();
    if missing == 1 {
        for card in deck {
            runouts.push(vec![card]);
            if runouts.len() >= limit {
                break;
            }
        }
    } else {
        for first in 0..deck.len() {
            for second in first + 1..deck.len() {
                runouts.push(vec![deck[first], deck[second]]);
                if runouts.len() >= limit {
                    return runouts;
                }
            }
        }
    }
    runouts
}

fn gpu_private_combos() -> Vec<GpuPrivateCombo> {
    ComboIndexer::new()
        .combos()
        .iter()
        .map(|combo| GpuPrivateCombo {
            cards: [combo.first.index() as u32, combo.second.index() as u32],
        })
        .collect()
}

fn gpu_final_boards(board: &Board, limit: usize) -> Vec<GpuFinalBoard> {
    let runouts = completion_runouts(board, board.deck_mask(), limit.max(1));
    gpu_final_boards_from_runouts(board, runouts)
}

fn gpu_full_final_boards(board: &Board) -> Vec<GpuFinalBoard> {
    let runouts = completion_runouts(board, board.deck_mask(), usize::MAX);
    gpu_final_boards_from_runouts(board, runouts)
}

fn gpu_final_boards_from_runouts(
    board: &Board,
    runouts: Vec<Vec<PokedrCard>>,
) -> Vec<GpuFinalBoard> {
    runouts
        .into_iter()
        .filter_map(|runout| {
            let mut final_board = board.cards().to_vec();
            final_board.extend(runout);
            (final_board.len() == 5).then(|| GpuFinalBoard {
                cards: [
                    final_board[0].index() as u32,
                    final_board[1].index() as u32,
                    final_board[2].index() as u32,
                    final_board[3].index() as u32,
                    final_board[4].index() as u32,
                ],
            })
        })
        .collect()
}

fn full_runout_pair_denominator(public_cards: usize) -> usize {
    let missing = 5usize.saturating_sub(public_cards);
    let remaining_after_public_and_pair = PokedrCard::COUNT - public_cards - 4;
    match missing {
        0 => 1,
        1 => remaining_after_public_and_pair,
        2 => remaining_after_public_and_pair * (remaining_after_public_and_pair - 1) / 2,
        _ => 1,
    }
}

fn heads_up_equity(
    hero_strength: pokedr_core::cards::HandStrength,
    villain: Combo,
    board: &[PokedrCard],
) -> f32 {
    let villain_strength = evaluate_seven([villain.first, villain.second], board);
    match hero_strength.cmp(&villain_strength) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    }
}

fn evaluate_seven(
    private_cards: [PokedrCard; 2],
    board: &[PokedrCard],
) -> pokedr_core::cards::HandStrength {
    let mut cards = Vec::with_capacity(7);
    cards.extend(private_cards);
    cards.extend_from_slice(board);
    evaluate(&cards)
}

fn hero_mask(hero_cards: [PokedrCard; 2]) -> u64 {
    hero_cards[0].deck_mask() | hero_cards[1].deck_mask()
}

fn private_dead_mask(ctx: &PostflopEvaluationContext) -> u64 {
    hero_mask(ctx.hero_cards) | hero_mask(ctx.villain_cards)
}

fn combo_cards(combo: Combo) -> [PokedrCard; 2] {
    [combo.first, combo.second]
}

fn legal_private_combos(
    indexer: &ComboIndexer,
    dead_mask: u64,
) -> impl Iterator<Item = usize> + '_ {
    indexer
        .combos()
        .iter()
        .enumerate()
        .filter_map(move |(index, combo)| (!combo.collides_with(dead_mask)).then_some(index))
}

fn root_board(tree: &SubgameTree) -> &Board {
    let PublicNodeKind::Decision { state, .. } = &tree.nodes()[0].kind else {
        panic!("root should be a decision");
    };
    &state.board
}

fn dealt_hole_cards(records: &[HistoryRecord], players: usize) -> Vec<Vec<RsCard>> {
    let mut cards = vec![Vec::new(); players];
    for record in records {
        let Action::DealStartingHand(payload) = &record.action else {
            continue;
        };
        if payload.idx < cards.len() {
            cards[payload.idx].push(payload.card);
        }
    }
    cards
}

fn format_cards(cards: &[RsCard]) -> String {
    cards
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_pokedr_cards(cards: &[PokedrCard]) -> String {
    cards
        .iter()
        .map(|card| {
            let rank = match card.rank() {
                PokedrRank::Two => "2",
                PokedrRank::Three => "3",
                PokedrRank::Four => "4",
                PokedrRank::Five => "5",
                PokedrRank::Six => "6",
                PokedrRank::Seven => "7",
                PokedrRank::Eight => "8",
                PokedrRank::Nine => "9",
                PokedrRank::Ten => "T",
                PokedrRank::Jack => "J",
                PokedrRank::Queen => "Q",
                PokedrRank::King => "K",
                PokedrRank::Ace => "A",
            };
            let suit = match card.suit() {
                PokedrSuit::Clubs => "c",
                PokedrSuit::Diamonds => "d",
                PokedrSuit::Hearts => "h",
                PokedrSuit::Spades => "s",
            };
            format!("{rank}{suit}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_action_candidate(candidate: &ActionCandidate) -> String {
    format!(
        "{}:{:?}",
        format_player_action(candidate.action),
        candidate.source
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DumpSolverMode {
    Summary,
    Full,
}

struct DumpSolverContext {
    state: DenseCfrState,
    average_values: Option<GpuRootTerminalValues>,
    current_values: Option<GpuRootTerminalValues>,
    average_recursive_hero_values: Option<GpuRootTerminalValues>,
    average_recursive_villain_values: Option<GpuRootTerminalValues>,
    current_recursive_hero_values: Option<GpuRootTerminalValues>,
    current_recursive_villain_values: Option<GpuRootTerminalValues>,
    recursive_br_gap: Option<BrGapMetrics>,
    current_recursive_br_gap: Option<BrGapMetrics>,
    indexer: ComboIndexer,
    root_dead: u64,
    iterations: usize,
    mode: DumpSolverMode,
    combo_limit: usize,
}

impl DumpSolverContext {
    fn build(
        tree: &SubgameTree,
        layout: &PostflopDenseLayout,
        config: &PokedrAgentConfig,
    ) -> Option<Self> {
        let mode = dump_solver_mode()?;
        Some(Self::build_for_mode(
            tree,
            layout,
            config,
            mode,
            dump_solver_combo_limit(mode),
        ))
    }

    fn build_for_mode(
        tree: &SubgameTree,
        layout: &PostflopDenseLayout,
        config: &PokedrAgentConfig,
        mode: DumpSolverMode,
        combo_limit: usize,
    ) -> Self {
        let indexer = ComboIndexer::new();
        let root_dead = root_board(tree).deck_mask();
        let (hero_weights, villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        let state = solve_public_tree_cfr(tree, layout, config, &hero_weights, &villain_weights);
        let (
            average_values,
            current_values,
            average_recursive_hero_values,
            average_recursive_villain_values,
            current_recursive_hero_values,
            current_recursive_villain_values,
            recursive_br_gap,
            current_recursive_br_gap,
        ) = cfr_gpu_backend()
            .and_then(|backend| {
                let matrix_cache =
                    RefCell::new(ShowdownMatrixCache::new(showdown_matrix_cache_capacity()));
                let linearized =
                    linearize_gpu_public_tree(tree, layout, &backend, config, &matrix_cache)?;
                let combos = gpu_private_combos();
                let combo_legal = indexer
                    .combos()
                    .iter()
                    .map(|combo| (!combo.collides_with(root_dead)) as u32)
                    .collect::<Vec<_>>();
                let profile = state.average_strategy_profile_state();
                let average_values = backend
                    .public_tree_iteration_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &profile,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let current_values = backend
                    .public_tree_iteration_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &state,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let average_recursive_hero_values = backend
                    .public_tree_best_response_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &profile,
                        0,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let average_recursive_villain_values = backend
                    .public_tree_best_response_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &profile,
                        1,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let current_recursive_hero_values = backend
                    .public_tree_best_response_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &state,
                        0,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let current_recursive_villain_values = backend
                    .public_tree_best_response_values(
                        &linearized.nodes,
                        &linearized.children,
                        &linearized.child_cards,
                        &combos,
                        &combo_legal,
                        &hero_weights,
                        &villain_weights,
                        &linearized.showdown_boards,
                        &state,
                        1,
                    )
                    .ok()?;
                backend.wait_idle().ok()?;
                let recursive_br_gap = recursive_br_gap_metrics_from_values(
                    tree,
                    layout,
                    &state,
                    &profile,
                    &combos,
                    &combo_legal,
                    &hero_weights,
                    &villain_weights,
                    &average_recursive_hero_values,
                    &average_recursive_villain_values,
                    &average_values,
                    false,
                );
                let current_recursive_br_gap = recursive_br_gap_metrics_from_values(
                    tree,
                    layout,
                    &state,
                    &state,
                    &combos,
                    &combo_legal,
                    &hero_weights,
                    &villain_weights,
                    &current_recursive_hero_values,
                    &current_recursive_villain_values,
                    &current_values,
                    false,
                );
                Some((
                    Some(average_values),
                    Some(current_values),
                    Some(average_recursive_hero_values),
                    Some(average_recursive_villain_values),
                    Some(current_recursive_hero_values),
                    Some(current_recursive_villain_values),
                    Some(recursive_br_gap),
                    Some(current_recursive_br_gap),
                ))
            })
            .unwrap_or((None, None, None, None, None, None, None, None));
        Self {
            state,
            average_values,
            current_values,
            average_recursive_hero_values,
            average_recursive_villain_values,
            current_recursive_hero_values,
            current_recursive_villain_values,
            recursive_br_gap,
            current_recursive_br_gap,
            indexer,
            root_dead,
            iterations: config.cfr_iterations.max(1),
            mode,
            combo_limit,
        }
    }

    fn fixed_flop_tree_metrics(&self) -> FixedFlopTreeMetrics {
        FixedFlopTreeMetrics {
            iterations: self.iterations,
            root_exploitability: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.root_exploitability),
            current_root_exploitability: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.root_exploitability),
            hero_root_br_value: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_value),
            villain_root_br_value: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_value),
            hero_root_profile_value: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_profile_value),
            villain_root_profile_value: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_profile_value),
            hero_root_br_improvement: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_improvement),
            villain_root_br_improvement: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_improvement),
            current_hero_root_br_value: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_value),
            current_villain_root_br_value: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_value),
            current_hero_root_profile_value: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_profile_value),
            current_villain_root_profile_value: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_profile_value),
            current_hero_root_br_improvement: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.hero_root_br_improvement),
            current_villain_root_br_improvement: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.villain_root_br_improvement),
            recursive_root_br_gap: self
                .recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.root_br_gap),
            current_recursive_root_br_gap: self
                .current_recursive_br_gap
                .as_ref()
                .map(|metrics| metrics.root_br_gap),
        }
    }

    fn root_br_combo_records(&self) -> Vec<RootBrComboRecord> {
        let (hero_weights, villain_weights) =
            fixed_flop_root_weights(&self.indexer, self.root_dead);
        let joint_weight_sum = root_joint_weight_sum(
            self.indexer.combos(),
            self.root_dead,
            &hero_weights,
            &villain_weights,
        );
        if joint_weight_sum <= 0.0 {
            return Vec::new();
        }

        let mut records = Vec::new();
        if let (Some(hero_values), Some(villain_values)) = (
            self.average_recursive_hero_values.as_ref(),
            self.average_recursive_villain_values.as_ref(),
        ) {
            push_root_br_combo_records(
                &mut records,
                "average",
                self.indexer.combos(),
                self.root_dead,
                &hero_weights,
                &villain_weights,
                hero_values,
                villain_values,
                joint_weight_sum,
            );
        }
        if let (Some(hero_values), Some(villain_values)) = (
            self.current_recursive_hero_values.as_ref(),
            self.current_recursive_villain_values.as_ref(),
        ) {
            push_root_br_combo_records(
                &mut records,
                "current",
                self.indexer.combos(),
                self.root_dead,
                &hero_weights,
                &villain_weights,
                hero_values,
                villain_values,
                joint_weight_sum,
            );
        }
        records
    }
}

fn root_joint_weight_sum(
    combos: &[Combo],
    root_dead: u64,
    hero_weights: &[f32],
    villain_weights: &[f32],
) -> f32 {
    let mut joint_weight_sum = 0.0;
    for (combo_index, combo) in combos.iter().enumerate() {
        if combo.collides_with(root_dead) {
            continue;
        }
        let hero_weight = hero_weights.get(combo_index).copied().unwrap_or(0.0);
        if hero_weight <= 0.0 {
            continue;
        }
        joint_weight_sum += hero_weight
            * opponent_nonblocking_weight(combos, root_dead, villain_weights, combo_index);
    }
    joint_weight_sum
}

#[allow(clippy::too_many_arguments)]
fn push_root_br_combo_records(
    records: &mut Vec<RootBrComboRecord>,
    profile: &str,
    combos: &[Combo],
    root_dead: u64,
    hero_weights: &[f32],
    villain_weights: &[f32],
    hero_values: &GpuRootTerminalValues,
    villain_values: &GpuRootTerminalValues,
    joint_weight_sum: f32,
) {
    for (combo_index, combo) in combos.iter().enumerate() {
        if combo.collides_with(root_dead) {
            continue;
        }
        let hero_weight = hero_weights.get(combo_index).copied().unwrap_or(0.0);
        if hero_weight > 0.0 {
            let root_value = hero_values
                .root_hero_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if root_value.is_finite() {
                let contribution = root_value * hero_weight / joint_weight_sum;
                records.push(RootBrComboRecord {
                    profile: profile.to_string(),
                    player: "Hero".to_string(),
                    combo_index,
                    combo: format_pokedr_cards(&[combo.first, combo.second]),
                    root_weight: hero_weight,
                    opponent_nonblocking_weight: opponent_nonblocking_weight(
                        combos,
                        root_dead,
                        villain_weights,
                        combo_index,
                    ),
                    root_value,
                    contribution,
                    contribution_bb100: contribution * 100.0,
                });
            }
        }

        let villain_weight = villain_weights.get(combo_index).copied().unwrap_or(0.0);
        if villain_weight > 0.0 {
            let root_value = villain_values
                .root_villain_values
                .get(combo_index)
                .copied()
                .unwrap_or(0.0);
            if root_value.is_finite() {
                let contribution = root_value * villain_weight / joint_weight_sum;
                records.push(RootBrComboRecord {
                    profile: profile.to_string(),
                    player: "Villain".to_string(),
                    combo_index,
                    combo: format_pokedr_cards(&[combo.first, combo.second]),
                    root_weight: villain_weight,
                    opponent_nonblocking_weight: opponent_nonblocking_weight(
                        combos,
                        root_dead,
                        hero_weights,
                        combo_index,
                    ),
                    root_value,
                    contribution,
                    contribution_bb100: contribution * 100.0,
                });
            }
        }
    }
}

fn opponent_nonblocking_weight(
    combos: &[Combo],
    root_dead: u64,
    weights: &[f32],
    combo_index: usize,
) -> f32 {
    let combo = &combos[combo_index];
    let mut total = 0.0;
    for (opponent_index, opponent) in combos.iter().enumerate() {
        if opponent_index == combo_index
            || opponent.collides_with(root_dead)
            || combo.first == opponent.first
            || combo.first == opponent.second
            || combo.second == opponent.first
            || combo.second == opponent.second
        {
            continue;
        }
        total += weights.get(opponent_index).copied().unwrap_or(0.0);
    }
    total
}

fn dump_solver_mode() -> Option<DumpSolverMode> {
    let value = std::env::var("POKEDR_DUMP_SOLVER_STATE").ok()?;
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "summary" => Some(DumpSolverMode::Summary),
        "full" | "all" | "combos" => Some(DumpSolverMode::Full),
        "0" | "false" | "no" | "off" => None,
        _ => Some(DumpSolverMode::Summary),
    }
}

fn dump_solver_combo_limit(mode: DumpSolverMode) -> usize {
    std::env::var("POKEDR_DUMP_COMBO_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(match mode {
            DumpSolverMode::Summary => 8,
            DumpSolverMode::Full => usize::MAX,
        })
}

fn dump_solver_db_combo_limit() -> usize {
    std::env::var("POKEDR_DUMP_COMBO_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32)
}

fn dump_tree_node_json(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    solver_dump: Option<&DumpSolverContext>,
    index: usize,
    node: &pokedr_core::postflop::PublicNode,
) -> String {
    let parent = node
        .parent
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let children = json_usize_array(&node.children);
    let path = dump_tree_path_json(tree, index);
    let infoset = layout
        .node_infoset(index)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let kind = match &node.kind {
        PublicNodeKind::Decision { state, actions } => format!(
            r#""decision","state":{},"actions":[{}]"#,
            dump_public_state_json(state),
            actions
                .iter()
                .enumerate()
                .map(|(action_index, action)| {
                    let child = tree.nodes()[index].children[action_index];
                    format!(
                        r#"{{"index":{},"child":{},"action":"{}","source":"{:?}"}}"#,
                        action_index,
                        child,
                        json_escape(&format_player_action(action.action)),
                        action.source
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        ),
        PublicNodeKind::Chance {
            street,
            board,
            cards,
        } => format!(
            r#""chance","street":"{:?}","board":"{}","cards":[{}]"#,
            street,
            json_escape(&format_pokedr_cards(board.cards())),
            cards
                .iter()
                .map(|card| format!(r#""{}""#, json_escape(&format_pokedr_cards(&[*card]))))
                .collect::<Vec<_>>()
                .join(",")
        ),
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            villain_invested,
        } => format!(
            r#""terminal","terminal_kind":"{:?}","board":"{}","pot":{},"hero_invested":{},"villain_invested":{}"#,
            kind,
            json_escape(&format_pokedr_cards(board.cards())),
            pot,
            hero_invested,
            villain_invested
        ),
    };
    let solver = solver_dump
        .and_then(|context| dump_solver_node_json(tree, layout, context, index))
        .map(|json| format!(r#","solver":{json}"#))
        .unwrap_or_default();
    format!(
        r#"{{"index":{index},"parent":{parent},"children":{children},"path":[{path}],"infoset":{infoset},"kind":{kind}{solver}}}"#
    )
}

fn dump_solver_node_json(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    context: &DumpSolverContext,
    node_index: usize,
) -> Option<String> {
    let public_infoset = layout.node_infoset(node_index)?;
    let PublicNodeKind::Decision { state, actions } = &tree.nodes()[node_index].kind else {
        return None;
    };
    let action_count = actions.len();
    let mut current_strategy = vec![0.0; layout.max_actions()];
    let mut average_strategy = vec![0.0; layout.max_actions()];
    let mut average_sum = vec![0.0; action_count];
    let mut current_sum = vec![0.0; action_count];
    let mut avg_action_value_sum = vec![0.0; action_count];
    let mut current_action_value_sum = vec![0.0; action_count];
    let mut avg_policy_value_sum = 0.0;
    let mut current_policy_value_sum = 0.0;
    let mut action_value_weight_sum = 0.0;
    let mut avg_strategy_weight_sum = 0.0;
    let mut current_strategy_weight_sum = 0.0;
    let mut legal_combo_count = 0usize;
    let mut gap_rows = Vec::new();

    for (combo_index, combo) in context.indexer.combos().iter().enumerate() {
        if combo.collides_with(context.root_dead) {
            continue;
        }
        legal_combo_count += 1;
        let infoset = private_infoset(public_infoset, state.acting_player, combo_index);
        let offset = infoset * layout.max_actions();
        context.state.strategy_for(infoset, &mut current_strategy);
        context
            .state
            .average_strategy_for(infoset, &mut average_strategy);
        let row = dump_solver_combo_row(
            context,
            state.acting_player,
            combo_index,
            offset,
            action_count,
            &average_strategy,
            &current_strategy,
        );
        for action in 0..action_count {
            average_sum[action] += average_strategy[action] * row.avg_strategy_weight;
            current_sum[action] += current_strategy[action] * row.current_strategy_weight;
        }
        avg_strategy_weight_sum += row.avg_strategy_weight;
        current_strategy_weight_sum += row.current_strategy_weight;
        if row.reach_weight > 0.0 && row.avg_strategy_weight > 0.0 {
            let value_weight = row.reach_weight;
            if let Some(values) = &row.avg_action_values {
                for action in 0..action_count {
                    avg_action_value_sum[action] += values[action] * value_weight;
                    avg_policy_value_sum +=
                        values[action] * average_strategy[action] * value_weight;
                }
            }
            if let Some(values) = &row.current_action_values {
                for action in 0..action_count {
                    current_action_value_sum[action] += values[action] * value_weight;
                    current_policy_value_sum +=
                        values[action] * current_strategy[action] * value_weight;
                }
            }
            action_value_weight_sum += value_weight;
        }
        gap_rows.push(row);
    }

    if avg_strategy_weight_sum > 0.0 {
        for action in 0..action_count {
            average_sum[action] /= avg_strategy_weight_sum;
        }
    } else if legal_combo_count > 0 {
        for action in 0..action_count {
            average_sum[action] /= legal_combo_count as f32;
        }
    }
    if current_strategy_weight_sum > 0.0 {
        for action in 0..action_count {
            current_sum[action] /= current_strategy_weight_sum;
        }
    } else if legal_combo_count > 0 {
        for action in 0..action_count {
            current_sum[action] /= legal_combo_count as f32;
        }
    }
    if action_value_weight_sum > 0.0 {
        for action in 0..action_count {
            avg_action_value_sum[action] /= action_value_weight_sum;
            current_action_value_sum[action] /= action_value_weight_sum;
        }
        avg_policy_value_sum /= action_value_weight_sum;
        current_policy_value_sum /= action_value_weight_sum;
    }

    gap_rows.sort_by(|left, right| {
        right
            .weighted_gap
            .partial_cmp(&left.weighted_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_gap = gap_rows.first().map(DumpSolverComboRow::to_json);
    let include_combos = matches!(context.mode, DumpSolverMode::Full) || context.combo_limit > 0;
    let combos = if include_combos {
        let rows = gap_rows
            .iter()
            .take(context.combo_limit)
            .map(DumpSolverComboRow::to_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(r#","combos":[{rows}]"#)
    } else {
        String::new()
    };
    Some(format!(
        r#"{{"mode":"{:?}","iterations":{},"public_infoset":{},"acting_player":"{:?}","action_count":{},"legal_combo_count":{},"actions":[{}],"avg_strategy":{},"current_strategy":{},"avg_action_ev":{},"current_action_ev":{},"avg_policy_ev":{},"current_policy_ev":{},"max_gap":{}{}}}"#,
        context.mode,
        context.iterations,
        public_infoset,
        state.acting_player,
        action_count,
        legal_combo_count,
        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                format!(
                    r#"{{"index":{},"action":"{}","source":"{:?}"}}"#,
                    action_index,
                    json_escape(&format_player_action(action.action)),
                    action.source
                )
            })
            .collect::<Vec<_>>()
            .join(","),
        json_f32_array(&average_sum),
        json_f32_array(&current_sum),
        json_f32_array_or_null(&avg_action_value_sum, action_value_weight_sum > 0.0),
        json_f32_array_or_null(&current_action_value_sum, action_value_weight_sum > 0.0),
        json_f32_or_null(avg_policy_value_sum, action_value_weight_sum > 0.0),
        json_f32_or_null(current_policy_value_sum, action_value_weight_sum > 0.0),
        max_gap.unwrap_or_else(|| "null".to_string()),
        combos
    ))
}

struct DumpSolverComboRow {
    combo_index: usize,
    combo: String,
    reach_weight: f32,
    gap: f32,
    weighted_gap: f32,
    recursive_weighted_gap: f32,
    current_recursive_weighted_gap: f32,
    avg_strategy_weight: f32,
    current_strategy_weight: f32,
    best_action: Option<usize>,
    policy_value: Option<f32>,
    avg_action_values: Option<Vec<f32>>,
    current_action_values: Option<Vec<f32>>,
    avg_recursive_action_values: Option<Vec<f32>>,
    current_recursive_action_values: Option<Vec<f32>>,
    average_strategy: Vec<f32>,
    current_strategy: Vec<f32>,
    regrets: Vec<f32>,
    strategy_sum: Vec<f32>,
}

impl DumpSolverComboRow {
    fn to_json(&self) -> String {
        format!(
            r#"{{"combo_index":{},"combo":"{}","reach":{},"gap":{},"weighted_gap":{},"avg_strategy_weight":{},"current_strategy_weight":{},"best_action":{},"policy_value":{},"avg_action_values":{},"current_action_values":{},"avg_strategy":{},"current_strategy":{},"regrets":{},"strategy_sum":{}}}"#,
            self.combo_index,
            json_escape(&self.combo),
            json_f32(self.reach_weight),
            json_f32(self.gap),
            json_f32(self.weighted_gap),
            json_f32(self.avg_strategy_weight),
            json_f32(self.current_strategy_weight),
            self.best_action
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_string()),
            self.policy_value
                .map(json_f32)
                .unwrap_or_else(|| "null".to_string()),
            self.avg_action_values
                .as_ref()
                .map(|values| json_f32_array(values))
                .unwrap_or_else(|| "null".to_string()),
            self.current_action_values
                .as_ref()
                .map(|values| json_f32_array(values))
                .unwrap_or_else(|| "null".to_string()),
            json_f32_array(&self.average_strategy),
            json_f32_array(&self.current_strategy),
            json_f32_array(&self.regrets),
            json_f32_array(&self.strategy_sum)
        )
    }
}

fn dump_solver_combo_row(
    context: &DumpSolverContext,
    player: Player,
    combo_index: usize,
    offset: usize,
    action_count: usize,
    average_strategy: &[f32],
    current_strategy: &[f32],
) -> DumpSolverComboRow {
    let combo = context.indexer.combo(combo_index);
    let regrets = context.state.regrets()[offset..offset + action_count].to_vec();
    let strategy_sum = context.state.strategy_sum()[offset..offset + action_count].to_vec();
    let avg_action_values = context
        .average_values
        .as_ref()
        .map(|values| values.action_values[offset..offset + action_count].to_vec());
    let current_action_values = context
        .current_values
        .as_ref()
        .map(|values| values.action_values[offset..offset + action_count].to_vec());
    let avg_recursive_values = match context
        .average_recursive_hero_values
        .as_ref()
        .zip(context.average_recursive_villain_values.as_ref())
    {
        Some((hero_values, villain_values)) => Some(match player {
            Player::Hero => hero_values.action_values[offset..offset + action_count].to_vec(),
            Player::Villain => villain_values.action_values[offset..offset + action_count].to_vec(),
        }),
        None => None,
    };
    let current_recursive_values = match context
        .current_recursive_hero_values
        .as_ref()
        .zip(context.current_recursive_villain_values.as_ref())
    {
        Some((hero_values, villain_values)) => Some(match player {
            Player::Hero => hero_values.action_values[offset..offset + action_count].to_vec(),
            Player::Villain => villain_values.action_values[offset..offset + action_count].to_vec(),
        }),
        None => None,
    };
    let reach_weight = context
        .average_values
        .as_ref()
        .and_then(|values| values.reach_weights.get(offset / context.state.actions()))
        .copied()
        .unwrap_or(0.0);
    let avg_strategy_weight = context
        .average_values
        .as_ref()
        .and_then(|values| {
            values
                .strategy_weights
                .get(offset / context.state.actions())
        })
        .copied()
        .unwrap_or(0.0);
    let current_strategy_weight = context
        .current_values
        .as_ref()
        .and_then(|values| {
            values
                .strategy_weights
                .get(offset / context.state.actions())
        })
        .copied()
        .unwrap_or(0.0);
    let current_reach_weight = context
        .current_values
        .as_ref()
        .and_then(|values| values.reach_weights.get(offset / context.state.actions()))
        .copied()
        .unwrap_or(0.0);
    let (best_action, gap, policy_value) = avg_action_values
        .as_ref()
        .map(|values| {
            let (best_action, best_value) = values
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or((0, 0.0));
            let policy_value = values
                .iter()
                .zip(average_strategy)
                .take(action_count)
                .map(|(value, probability)| value * probability)
                .sum::<f32>();
            (
                Some(best_action),
                (best_value - policy_value).max(0.0),
                Some(policy_value),
            )
        })
        .unwrap_or((None, 0.0, None));
    let recursive_weighted_gap = avg_recursive_values
        .as_ref()
        .map(|values| combo_gap(values, average_strategy, action_count).max(0.0) * reach_weight)
        .unwrap_or(0.0);
    let current_recursive_weighted_gap = current_recursive_values
        .as_ref()
        .map(|values| {
            combo_gap(values, current_strategy, action_count).max(0.0) * current_reach_weight
        })
        .unwrap_or(0.0);
    DumpSolverComboRow {
        combo_index,
        combo: format_pokedr_cards(&[combo.first, combo.second]),
        reach_weight,
        gap,
        weighted_gap: gap * reach_weight,
        recursive_weighted_gap,
        current_recursive_weighted_gap,
        avg_strategy_weight,
        current_strategy_weight,
        best_action,
        policy_value,
        avg_action_values,
        current_action_values,
        avg_recursive_action_values: avg_recursive_values,
        current_recursive_action_values: current_recursive_values,
        average_strategy: average_strategy[..action_count].to_vec(),
        current_strategy: current_strategy[..action_count].to_vec(),
        regrets,
        strategy_sum,
    }
}

fn combo_gap(values: &[f32], strategy: &[f32], action_count: usize) -> f32 {
    let best = values
        .iter()
        .take(action_count)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !best.is_finite() {
        return 0.0;
    }
    let policy = values
        .iter()
        .zip(strategy)
        .take(action_count)
        .map(|(value, probability)| value * probability)
        .sum::<f32>();
    best - policy
}

fn tree_node_record(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_id: usize,
    node: &pokedr_core::postflop::PublicNode,
) -> TreeNodeRecord {
    let path = format!("[{}]", dump_tree_path_json(tree, node_id));
    match &node.kind {
        PublicNodeKind::Decision { state, .. } => TreeNodeRecord {
            node_id,
            parent_id: node.parent,
            kind: "decision".to_string(),
            infoset: layout.node_infoset(node_id),
            path,
            street: Some(format!("{:?}", state.street)),
            board: Some(format_pokedr_cards(state.board.cards())),
            acting_player: Some(format!("{:?}", state.acting_player)),
            pot: Some(state.pot),
            to_call: Some(state.to_call),
            hero_invested: Some(state.hero_invested),
            villain_invested: Some(state.villain_invested),
            terminal_kind: None,
        },
        PublicNodeKind::Chance { street, board, .. } => TreeNodeRecord {
            node_id,
            parent_id: node.parent,
            kind: "chance".to_string(),
            infoset: None,
            path,
            street: Some(format!("{:?}", street)),
            board: Some(format_pokedr_cards(board.cards())),
            acting_player: None,
            pot: None,
            to_call: None,
            hero_invested: None,
            villain_invested: None,
            terminal_kind: None,
        },
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            villain_invested,
        } => TreeNodeRecord {
            node_id,
            parent_id: node.parent,
            kind: "terminal".to_string(),
            infoset: None,
            path,
            street: None,
            board: Some(format_pokedr_cards(board.cards())),
            acting_player: None,
            pot: Some(*pot),
            to_call: None,
            hero_invested: Some(*hero_invested),
            villain_invested: Some(*villain_invested),
            terminal_kind: Some(format!("{:?}", kind)),
        },
    }
}

fn solver_node_records(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    context: &DumpSolverContext,
    node_id: usize,
) -> Option<(SolverNodeRecord, Vec<SolverComboRecord>)> {
    let public_infoset = layout.node_infoset(node_id)?;
    let PublicNodeKind::Decision { state, actions } = &tree.nodes()[node_id].kind else {
        return None;
    };
    let action_count = actions.len();
    let mut current_strategy = vec![0.0; layout.max_actions()];
    let mut average_strategy = vec![0.0; layout.max_actions()];
    let mut average_sum = vec![0.0; action_count];
    let mut current_sum = vec![0.0; action_count];
    let mut avg_action_value_sum = vec![0.0; action_count];
    let mut current_action_value_sum = vec![0.0; action_count];
    let mut avg_policy_value_sum = 0.0;
    let mut current_policy_value_sum = 0.0;
    let mut avg_recursive_action_value_sum = vec![0.0; action_count];
    let mut current_recursive_action_value_sum = vec![0.0; action_count];
    let mut avg_recursive_policy_value_sum = 0.0;
    let mut current_recursive_policy_value_sum = 0.0;
    let mut avg_recursive_value_weight_sum = 0.0;
    let mut current_recursive_value_weight_sum = 0.0;
    let mut avg_recursive_combo_gap_weighted_sum = 0.0;
    let mut current_recursive_combo_gap_weighted_sum = 0.0;
    let mut avg_action_value_weight_sum = 0.0;
    let mut current_action_value_weight_sum = 0.0;
    let mut avg_combo_gap_weighted_sum = 0.0;
    let mut current_combo_gap_weighted_sum = 0.0;
    let mut avg_strategy_weight_sum = 0.0;
    let mut current_strategy_weight_sum = 0.0;
    let mut legal_combo_count = 0usize;
    let mut combo_rows = Vec::new();

    for (combo_index, combo) in context.indexer.combos().iter().enumerate() {
        if combo.collides_with(context.root_dead) {
            continue;
        }
        legal_combo_count += 1;
        let infoset = private_infoset(public_infoset, state.acting_player, combo_index);
        let offset = infoset * layout.max_actions();
        context.state.strategy_for(infoset, &mut current_strategy);
        context
            .state
            .average_strategy_for(infoset, &mut average_strategy);
        let row = dump_solver_combo_row(
            context,
            state.acting_player,
            combo_index,
            offset,
            action_count,
            &average_strategy,
            &current_strategy,
        );
        for action in 0..action_count {
            average_sum[action] += average_strategy[action] * row.avg_strategy_weight;
            current_sum[action] += current_strategy[action] * row.current_strategy_weight;
        }
        avg_strategy_weight_sum += row.avg_strategy_weight;
        current_strategy_weight_sum += row.current_strategy_weight;
        if row.reach_weight > 0.0 {
            let value_weight = row.reach_weight;
            if let Some(values) = &row.avg_action_values {
                for action in 0..action_count {
                    avg_action_value_sum[action] += values[action] * value_weight;
                    avg_policy_value_sum +=
                        values[action] * average_strategy[action] * value_weight;
                }
            }
            avg_combo_gap_weighted_sum += row.weighted_gap;
            avg_action_value_weight_sum += value_weight;
        }
        let current_value_weight = context
            .current_values
            .as_ref()
            .and_then(|values| values.reach_weights.get(offset / context.state.actions()))
            .copied()
            .unwrap_or(0.0);
        if current_value_weight > 0.0 && row.current_strategy_weight > 0.0 {
            if let Some(values) = &row.current_action_values {
                for action in 0..action_count {
                    current_action_value_sum[action] += values[action] * current_value_weight;
                    current_policy_value_sum +=
                        values[action] * current_strategy[action] * current_value_weight;
                }
                let best_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let policy_value = values
                    .iter()
                    .zip(&current_strategy)
                    .take(action_count)
                    .map(|(value, probability)| value * probability)
                    .sum::<f32>();
                current_combo_gap_weighted_sum +=
                    (best_value - policy_value).max(0.0) * current_value_weight;
            }
            current_action_value_weight_sum += current_value_weight;
        }
        let avg_recursive_values = match state.acting_player {
            Player::Hero => context.average_recursive_hero_values.as_ref(),
            Player::Villain => context.average_recursive_villain_values.as_ref(),
        };
        if let Some(values) = avg_recursive_values {
            let value_weight = values
                .reach_weights
                .get(offset / context.state.actions())
                .copied()
                .unwrap_or(0.0);
            if value_weight > 0.0 {
                for action in 0..action_count {
                    let value = values.action_values[offset + action];
                    avg_recursive_action_value_sum[action] += value * value_weight;
                    avg_recursive_policy_value_sum +=
                        value * average_strategy[action] * value_weight;
                }
                avg_recursive_combo_gap_weighted_sum += row.recursive_weighted_gap;
                avg_recursive_value_weight_sum += value_weight;
            }
        }
        let current_recursive_values = match state.acting_player {
            Player::Hero => context.current_recursive_hero_values.as_ref(),
            Player::Villain => context.current_recursive_villain_values.as_ref(),
        };
        if let Some(values) = current_recursive_values {
            let value_weight = values
                .reach_weights
                .get(offset / context.state.actions())
                .copied()
                .unwrap_or(0.0);
            if value_weight > 0.0 {
                for action in 0..action_count {
                    let value = values.action_values[offset + action];
                    current_recursive_action_value_sum[action] += value * value_weight;
                    current_recursive_policy_value_sum +=
                        value * current_strategy[action] * value_weight;
                }
                if let Some(values) = &row.current_recursive_action_values {
                    current_recursive_combo_gap_weighted_sum +=
                        combo_gap(values, &current_strategy, action_count).max(0.0) * value_weight;
                }
                current_recursive_value_weight_sum += value_weight;
            }
        }
        combo_rows.push(row);
    }

    if avg_strategy_weight_sum > 0.0 {
        for action in 0..action_count {
            average_sum[action] /= avg_strategy_weight_sum;
        }
    } else if legal_combo_count > 0 {
        for action in 0..action_count {
            average_sum[action] /= legal_combo_count as f32;
        }
    }
    if current_strategy_weight_sum > 0.0 {
        for action in 0..action_count {
            current_sum[action] /= current_strategy_weight_sum;
        }
    } else if legal_combo_count > 0 {
        for action in 0..action_count {
            current_sum[action] /= legal_combo_count as f32;
        }
    }
    let has_avg_ev = avg_action_value_weight_sum > 0.0;
    if has_avg_ev {
        for action in 0..action_count {
            avg_action_value_sum[action] /= avg_action_value_weight_sum;
        }
        avg_policy_value_sum /= avg_action_value_weight_sum;
    }
    let has_current_ev = current_action_value_weight_sum > 0.0;
    if has_current_ev {
        for action in 0..action_count {
            current_action_value_sum[action] /= current_action_value_weight_sum;
        }
        current_policy_value_sum /= current_action_value_weight_sum;
    }
    let has_avg_recursive_ev = avg_recursive_value_weight_sum > 0.0;
    if has_avg_recursive_ev {
        for action in 0..action_count {
            avg_recursive_action_value_sum[action] /= avg_recursive_value_weight_sum;
        }
        avg_recursive_policy_value_sum /= avg_recursive_value_weight_sum;
    }
    let has_current_recursive_ev = current_recursive_value_weight_sum > 0.0;
    if has_current_recursive_ev {
        for action in 0..action_count {
            current_recursive_action_value_sum[action] /= current_recursive_value_weight_sum;
        }
        current_recursive_policy_value_sum /= current_recursive_value_weight_sum;
    }

    let avg_gap = has_avg_ev.then(|| avg_combo_gap_weighted_sum / avg_action_value_weight_sum);
    let current_gap =
        has_current_ev.then(|| current_combo_gap_weighted_sum / current_action_value_weight_sum);
    let avg_recursive_gap = has_avg_recursive_ev
        .then(|| avg_recursive_combo_gap_weighted_sum / avg_recursive_value_weight_sum);
    let current_recursive_gap = has_current_recursive_ev
        .then(|| current_recursive_combo_gap_weighted_sum / current_recursive_value_weight_sum);
    combo_rows.sort_by(|left, right| {
        right
            .weighted_gap
            .partial_cmp(&left.weighted_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let combo_records = combo_rows
        .iter()
        .filter(|row| row.avg_strategy_weight > 0.0 || row.current_strategy_weight > 0.0)
        .take(context.combo_limit)
        .map(|row| SolverComboRecord {
            node_id,
            combo_index: row.combo_index,
            combo: row.combo.clone(),
            reach: row.reach_weight,
            weighted_gap: row.weighted_gap,
            recursive_weighted_gap: row.recursive_weighted_gap,
            current_recursive_weighted_gap: row.current_recursive_weighted_gap,
            avg_strategy_weight: row.avg_strategy_weight,
            current_strategy_weight: row.current_strategy_weight,
            avg_action_values: row.avg_action_values.clone(),
            current_action_values: row.current_action_values.clone(),
            avg_recursive_action_values: row.avg_recursive_action_values.clone(),
            current_recursive_action_values: row.current_recursive_action_values.clone(),
            avg_strategy: row.average_strategy.clone(),
            current_strategy: row.current_strategy.clone(),
            regrets: row.regrets.clone(),
            strategy_sum: row.strategy_sum.clone(),
        })
        .collect();

    Some((
        SolverNodeRecord {
            node_id,
            infoset: public_infoset,
            iterations: context.iterations,
            acting_player: format!("{:?}", state.acting_player),
            action_count,
            legal_combo_count,
            value_reach_sum: avg_action_value_weight_sum,
            avg_strategy_weight_sum,
            current_strategy_weight_sum,
            avg_strategy: average_sum,
            current_strategy: current_sum,
            avg_action_ev: has_avg_ev.then_some(avg_action_value_sum),
            current_action_ev: has_current_ev.then_some(current_action_value_sum),
            avg_policy_ev: has_avg_ev.then_some(avg_policy_value_sum),
            current_policy_ev: has_current_ev.then_some(current_policy_value_sum),
            avg_gap: avg_gap.map(|value| value.max(0.0)),
            current_gap: current_gap.map(|value| value.max(0.0)),
            avg_recursive_action_ev: has_avg_recursive_ev.then_some(avg_recursive_action_value_sum),
            current_recursive_action_ev: has_current_recursive_ev
                .then_some(current_recursive_action_value_sum),
            avg_recursive_policy_ev: has_avg_recursive_ev.then_some(avg_recursive_policy_value_sum),
            current_recursive_policy_ev: has_current_recursive_ev
                .then_some(current_recursive_policy_value_sum),
            avg_recursive_gap: avg_recursive_gap.map(|value| value.max(0.0)),
            current_recursive_gap: current_recursive_gap.map(|value| value.max(0.0)),
        },
        combo_records,
    ))
}

fn dump_tree_path_json(tree: &SubgameTree, index: usize) -> String {
    let mut edges = Vec::new();
    let mut child_index = index;
    while let Some(parent_index) = tree.nodes()[child_index].parent {
        edges.push(dump_tree_edge_json(tree, parent_index, child_index));
        child_index = parent_index;
    }
    edges.reverse();
    edges.join(",")
}

fn dump_tree_edge_json(tree: &SubgameTree, parent_index: usize, child_index: usize) -> String {
    let action_index = tree.nodes()[parent_index]
        .children
        .iter()
        .position(|candidate| *candidate == child_index);
    match (&tree.nodes()[parent_index].kind, action_index) {
        (PublicNodeKind::Decision { actions, .. }, Some(action_index)) => {
            let action = &actions[action_index];
            format!(
                r#"{{"from":{},"to":{},"kind":"action","index":{},"action":"{}","source":"{:?}"}}"#,
                parent_index,
                child_index,
                action_index,
                json_escape(&format_player_action(action.action)),
                action.source
            )
        }
        (PublicNodeKind::Chance { cards, .. }, Some(action_index)) => format!(
            r#"{{"from":{},"to":{},"kind":"card","index":{},"card":"{}"}}"#,
            parent_index,
            child_index,
            action_index,
            json_escape(&format_pokedr_cards(&[cards[action_index]]))
        ),
        _ => format!(
            r#"{{"from":{},"to":{},"kind":"unknown"}}"#,
            parent_index, child_index
        ),
    }
}

fn dump_public_state_json(state: &PublicState) -> String {
    format!(
        r#"{{"street":"{:?}","board":"{}","pot":{},"hero_invested":{},"villain_invested":{},"effective_stack":{},"to_call":{},"min_aggressive_amount":{},"acting_player":"{:?}","raises_this_street":{},"checks_this_street":{}}}"#,
        state.street,
        json_escape(&format_pokedr_cards(state.board.cards())),
        state.pot,
        state.hero_invested,
        state.villain_invested,
        state.effective_stack,
        state.to_call,
        state.min_aggressive_amount,
        state.acting_player,
        state.raises_this_street,
        state.checks_this_street
    )
}

fn format_player_action(action: PlayerAction) -> String {
    match action {
        PlayerAction::Fold => "fold".to_string(),
        PlayerAction::Check => "check".to_string(),
        PlayerAction::Call { amount } => format!("call:{amount}"),
        PlayerAction::Bet { amount } => format!("bet:{amount}"),
        PlayerAction::Raise { amount } => format!("raise:{amount}"),
        PlayerAction::AllIn { amount } => format!("allin:{amount}"),
    }
}

fn json_usize_array(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_f32(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".to_string()
    }
}

fn json_f32_or_null(value: f32, present: bool) -> String {
    if present {
        json_f32(value)
    } else {
        "null".to_string()
    }
}

fn json_f32_array(values: &[f32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_f32(*value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_f32_array_or_null(values: &[f32], present: bool) -> String {
    if present {
        json_f32_array(values)
    } else {
        "null".to_string()
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn format_trace_action(action: &Action) -> Option<String> {
    match action {
        Action::ForcedBet(payload) => Some(format!(
            "p{} forced {:?} {:.2} stack {:.2}",
            payload.idx, payload.forced_bet_type, payload.bet, payload.player_stack
        )),
        Action::PlayedAction(payload) => Some(format!(
            "{:?} p{} {:?} pot {:.2}->{:.2} bet {:.2}->{:.2} stack {:.2}",
            payload.round,
            payload.idx,
            payload.action,
            payload.starting_pot,
            payload.final_pot,
            payload.starting_bet,
            payload.final_bet,
            payload.player_stack
        )),
        Action::FailedAction(payload) => Some(format!(
            "{:?} p{} failed {:?} -> {:?}",
            payload.result.round, payload.result.idx, payload.action, payload.result.action
        )),
        Action::DealCommunity(card) => Some(format!("deal {card}")),
        Action::RoundAdvance(round) => Some(format!("round {round}")),
        _ => None,
    }
}

fn format_award(action: &Action) -> Option<String> {
    let Action::Award(payload) = action else {
        return None;
    };
    Some(format!(
        "p{} award {:.2} total_pot {:.2} rank {:?} hand {:?}",
        payload.idx, payload.award_amount, payload.total_pot, payload.rank, payload.hand
    ))
}

fn public_state_from_game(game_state: &GameState) -> Option<PublicState> {
    let street = match game_state.round {
        Round::Flop => Street::Flop,
        Round::Turn => Street::Turn,
        Round::River => Street::River,
        _ => return None,
    };
    let to_call = amount_to_call(game_state).max(0.0).round() as u32;
    let stack = game_state.current_player_stack().max(1.0).round() as u32;
    let hero_invested = (game_state.starting_stacks[0] - game_state.stacks[0])
        .max(0.0)
        .round() as u32;
    let villain_invested = (game_state.starting_stacks[1] - game_state.stacks[1])
        .max(0.0)
        .round() as u32;
    Some(PublicState {
        street,
        board: Board::new(
            game_state
                .board
                .iter()
                .copied()
                .map(to_pokedr_card)
                .collect(),
        ),
        pot: game_state.total_pot.max(1.0).round() as u32,
        hero_invested,
        villain_invested,
        effective_stack: stack.max(to_call),
        to_call,
        min_aggressive_amount: game_state.current_round_min_raise().max(1.0).round() as u32,
        acting_player: if game_state.to_act_idx() == 0 {
            Player::Hero
        } else {
            Player::Villain
        },
        raises_this_street: game_state.round_data.total_raise_count,
        checks_this_street: 0,
    })
}

fn to_rs_action(game_state: &GameState, action: PlayerAction) -> AgentAction {
    match action {
        PlayerAction::Fold => AgentAction::Fold,
        PlayerAction::Check | PlayerAction::Call { .. } => AgentAction::Call,
        PlayerAction::Bet { amount } | PlayerAction::Raise { amount } => {
            let player_bet = game_state.current_round_current_player_bet();
            let stack = game_state.current_player_stack();
            let target = amount as f32;
            if target >= player_bet + stack {
                AgentAction::AllIn
            } else {
                AgentAction::Bet(target.max(game_state.current_round_bet()))
            }
        }
        PlayerAction::AllIn { .. } => AgentAction::AllIn,
    }
}

fn amount_to_call(game_state: &GameState) -> f32 {
    (game_state.current_round_bet() - game_state.current_round_current_player_bet()).max(0.0)
}

fn board_mask_from_game(game_state: &GameState) -> u64 {
    game_state
        .board
        .iter()
        .copied()
        .map(to_pokedr_card)
        .fold(0u64, |mask, card| mask | card.deck_mask())
}

fn can_raise(game_state: &GameState) -> bool {
    game_state.current_player_stack()
        > amount_to_call(game_state) + game_state.current_round_min_raise()
}

fn hero_cards_from_game(game_state: &GameState) -> Option<[PokedrCard; 2]> {
    let hand = game_state.hands.get(game_state.to_act_idx())?;
    let cards: Vec<_> = hand.iter().collect();
    if cards.len() != 2 {
        return None;
    }
    Some([to_pokedr_card(cards[0]), to_pokedr_card(cards[1])])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PreflopClass {
    Trash = 0,
    Speculative = 1,
    Playable = 2,
    Strong = 3,
    Premium = 4,
}

const PRIVATE_INFOS_PER_PUBLIC: usize = COMBO_COUNT;

impl PreflopClass {
    fn open_raises(self) -> bool {
        self >= Self::Playable
    }

    fn calls_open(self) -> bool {
        self >= Self::Speculative
    }

    fn three_bets(self) -> bool {
        self >= Self::Strong
    }

    fn calls_three_bet(self) -> bool {
        self >= Self::Playable
    }

    fn four_bets(self) -> bool {
        self >= Self::Premium
    }

    fn calls_large_preflop_bet(self) -> bool {
        self >= Self::Strong
    }
}

fn classify_preflop_hand(hand: &rs_poker::core::Hand) -> PreflopClass {
    let cards: Vec<_> = hand.iter().collect();
    if cards.len() < 2 {
        return PreflopClass::Trash;
    }
    let first = cards[0];
    let second = cards[1];
    let first_value = rs_value_index(first.value);
    let second_value = rs_value_index(second.value);
    classify_preflop_values(first_value, second_value, first.suit == second.suit)
}

fn classify_preflop_values(first_value: u8, second_value: u8, suited: bool) -> PreflopClass {
    let high = first_value.max(second_value);
    let low = first_value.min(second_value);
    let is_pair = first_value == second_value;
    let gap = high.abs_diff(low);

    if is_pair {
        return match high {
            12..=14 => PreflopClass::Premium,
            10..=11 => PreflopClass::Strong,
            7..=9 => PreflopClass::Playable,
            2..=6 => PreflopClass::Speculative,
            _ => PreflopClass::Trash,
        };
    }
    if high == 14 && low >= 13 {
        return PreflopClass::Strong;
    }
    if high == 14 && suited {
        return if low >= 8 {
            PreflopClass::Playable
        } else {
            PreflopClass::Speculative
        };
    }
    if high == 14 && low >= 10 {
        return if suited {
            PreflopClass::Strong
        } else {
            PreflopClass::Playable
        };
    }
    if high == 14 {
        return PreflopClass::Speculative;
    }
    if high == 13 && suited {
        return if low >= 8 {
            PreflopClass::Playable
        } else {
            PreflopClass::Speculative
        };
    }
    if high == 13 && low >= 10 {
        return PreflopClass::Playable;
    }
    if high >= 13 && low >= 11 {
        return if suited {
            PreflopClass::Strong
        } else {
            PreflopClass::Playable
        };
    }
    if suited && high >= 11 && low >= 9 {
        return PreflopClass::Playable;
    }
    if high >= 12 && low >= 10 {
        return PreflopClass::Speculative;
    }
    if suited && gap <= 1 && high >= 8 {
        return PreflopClass::Speculative;
    }
    if gap <= 1 && high >= 11 {
        return PreflopClass::Speculative;
    }
    PreflopClass::Trash
}

fn to_pokedr_card(card: RsCard) -> PokedrCard {
    PokedrCard::new(to_pokedr_rank(card.value), to_pokedr_suit(card.suit))
}

fn to_pokedr_suit(suit: RsSuit) -> PokedrSuit {
    match suit {
        RsSuit::Club => PokedrSuit::Clubs,
        RsSuit::Diamond => PokedrSuit::Diamonds,
        RsSuit::Heart => PokedrSuit::Hearts,
        RsSuit::Spade => PokedrSuit::Spades,
    }
}

fn to_pokedr_rank(value: RsValue) -> PokedrRank {
    match value {
        RsValue::Two => PokedrRank::Two,
        RsValue::Three => PokedrRank::Three,
        RsValue::Four => PokedrRank::Four,
        RsValue::Five => PokedrRank::Five,
        RsValue::Six => PokedrRank::Six,
        RsValue::Seven => PokedrRank::Seven,
        RsValue::Eight => PokedrRank::Eight,
        RsValue::Nine => PokedrRank::Nine,
        RsValue::Ten => PokedrRank::Ten,
        RsValue::Jack => PokedrRank::Jack,
        RsValue::Queen => PokedrRank::Queen,
        RsValue::King => PokedrRank::King,
        RsValue::Ace => PokedrRank::Ace,
    }
}

fn rs_value_index(value: RsValue) -> u8 {
    match value {
        RsValue::Two => 2,
        RsValue::Three => 3,
        RsValue::Four => 4,
        RsValue::Five => 5,
        RsValue::Six => 6,
        RsValue::Seven => 7,
        RsValue::Eight => 8,
        RsValue::Nine => 9,
        RsValue::Ten => 10,
        RsValue::Jack => 11,
        RsValue::Queen => 12,
        RsValue::King => 13,
        RsValue::Ace => 14,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_up_match_runs_to_completion() {
        let summary = run_heads_up_match_with_config(
            2,
            7,
            PokedrAgentConfig {
                cfr_iterations: 1,
                max_showdown_runouts: 1,
                ..PokedrAgentConfig::default()
            },
        );
        assert_eq!(summary.hands, 2);
        assert!((summary.hero_net + summary.villain_net).abs() < 1e-3);
    }

    #[test]
    fn postflop_showdown_equity_prefers_strong_hand() {
        let board = Board::new(vec![
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
            PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
            PokedrCard::new(PokedrRank::Three, PokedrSuit::Spades),
        ]);
        let indexer = ComboIndexer::new();
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let mut strong = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
            ],
            villain_cards: [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
            hero_combo: hero_combo_index(
                &indexer,
                [
                    PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
                ],
            ),
            villain_combo: hero_combo_index(
                &indexer,
                [
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
                ],
            ),
            gpu_backend: None,
            matrix_cache: &matrix_cache,
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let mut weak = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
            ],
            villain_cards: [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
            hero_combo: hero_combo_index(
                &indexer,
                [
                    PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
                ],
            ),
            villain_combo: hero_combo_index(
                &indexer,
                [
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
                ],
            ),
            gpu_backend: None,
            matrix_cache: &matrix_cache,
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };

        assert!(showdown_equity(&board, &mut strong) > showdown_equity(&board, &mut weak));
    }

    #[test]
    fn board_locked_showdown_is_exact_tie_for_both_players() {
        let board = Board::new(vec![
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::King, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Queen, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Jack, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Ten, PokedrSuit::Spades),
        ]);
        let indexer = ComboIndexer::new();
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let hero_cards = [
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
            PokedrCard::new(PokedrRank::Three, PokedrSuit::Diamonds),
        ];
        let villain_cards = [
            PokedrCard::new(PokedrRank::Four, PokedrSuit::Clubs),
            PokedrCard::new(PokedrRank::Five, PokedrSuit::Diamonds),
        ];
        let mut hero_view = PostflopEvaluationContext {
            hero_cards,
            villain_cards,
            hero_combo: hero_combo_index(&indexer, hero_cards),
            villain_combo: hero_combo_index(&indexer, villain_cards),
            gpu_backend: None,
            matrix_cache: &matrix_cache,
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let mut villain_view = PostflopEvaluationContext {
            hero_cards: villain_cards,
            villain_cards: hero_cards,
            hero_combo: hero_combo_index(&indexer, villain_cards),
            villain_combo: hero_combo_index(&indexer, hero_cards),
            gpu_backend: None,
            matrix_cache: &matrix_cache,
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };

        assert!((showdown_equity(&board, &mut hero_view) - 0.5).abs() < 1e-6);
        assert!((showdown_equity(&board, &mut villain_view) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fold_terminal_payoff_matches_pot_accounting() {
        let hero_facing_bet = PublicState {
            street: Street::River,
            board: Board::new(vec![]),
            pot: 100,
            hero_invested: 40,
            villain_invested: 60,
            effective_stack: 100,
            to_call: 20,
            min_aggressive_amount: 60,
            acting_player: Player::Hero,
            raises_this_street: 0,
            checks_this_street: 0,
        };
        let villain_facing_bet = PublicState {
            acting_player: Player::Villain,
            hero_invested: 60,
            villain_invested: 40,
            ..hero_facing_bet.clone()
        };

        assert_eq!(
            fold_utility(&hero_facing_bet, PlayerAction::Fold, 100, 40),
            -40.0
        );
        assert_eq!(
            fold_utility(&villain_facing_bet, PlayerAction::Fold, 100, 60),
            40.0
        );
    }

    #[test]
    fn river_fold_call_values_match_hand_calculation() {
        let tree = SubgameTree::build(
            PublicState {
                street: Street::River,
                board: Board::new(vec![
                    PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
                    PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
                    PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
                    PokedrCard::new(PokedrRank::Three, PokedrSuit::Spades),
                ]),
                pot: 100,
                hero_invested: 40,
                villain_invested: 60,
                effective_stack: 100,
                to_call: 20,
                min_aggressive_amount: 60,
                acting_player: Player::Hero,
                raises_this_street: 0,
                checks_this_street: 0,
            },
            SubgameTreeConfig {
                action_set: ActionSetConfig {
                    max_aggressive_actions: 0,
                    ..ActionSetConfig::default()
                },
                max_raises_per_street: 0,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let mut dense_config = layout.dense_config(CfrVariant::CfrPlus);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let cfr_state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let indexer = ComboIndexer::new();
        let hero_cards = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
        ];
        let villain_cards = [
            PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
            PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
        ];
        let hero_combo = hero_combo_index(&indexer, hero_cards);
        let villain_combo = hero_combo_index(&indexer, villain_cards);
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let mut ctx = PostflopEvaluationContext {
            hero_cards,
            villain_cards,
            hero_combo,
            villain_combo,
            gpu_backend: None,
            matrix_cache: &matrix_cache,
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let mut batch = DenseCfrIteration::new(&dense_config);
        let mut value_weights = vec![0.0; batch.action_values.len()];
        traverse_cfr_node(
            &tree,
            &layout,
            0,
            None,
            None,
            hero_combo,
            villain_combo,
            1.0,
            1.0,
            1.0,
            &cfr_state,
            &mut ctx,
            &mut batch,
            &mut value_weights,
        );

        let offset = private_infoset(0, Player::Hero, hero_combo) * layout.max_actions();
        assert!((batch.action_values[offset] - -40.0).abs() < 1e-6);
        assert!((batch.action_values[offset + 1] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn public_tree_cfr_produces_normalized_root_strategy() {
        let tree = SubgameTree::build(
            PublicState {
                street: Street::Flop,
                board: Board::new(vec![
                    PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
                    PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
                    PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
                ]),
                pot: 10,
                hero_invested: 5,
                villain_invested: 5,
                effective_stack: 30,
                to_call: 0,
                min_aggressive_amount: 5,
                acting_player: Player::Hero,
                raises_this_street: 0,
                checks_this_street: 0,
            },
            SubgameTreeConfig {
                action_set: ActionSetConfig {
                    max_aggressive_actions: 1,
                    flop_bet_fractions: vec![0.5],
                    turn_bet_fractions: vec![0.5],
                    river_bet_fractions: vec![0.5],
                    raise_fractions: vec![1.0],
                    ..ActionSetConfig::default()
                },
                max_raises_per_street: 1,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let hero_weights = vec![1.0; COMBO_COUNT];
        let villain_weights = vec![1.0; COMBO_COUNT];
        let state = solve_public_tree_cfr(
            &tree,
            &layout,
            &PokedrAgentConfig {
                cfr_iterations: 1,
                max_showdown_runouts: 1,
                ..PokedrAgentConfig::default()
            },
            &hero_weights,
            &villain_weights,
        );
        let hero_combo = hero_combo_index(
            &ComboIndexer::new(),
            [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
            ],
        );
        let mut strategy = vec![0.0; layout.max_actions()];
        state.average_strategy_for(private_infoset(0, Player::Hero, hero_combo), &mut strategy);
        let legal_sum: f32 = strategy.iter().take(layout.action_count(0)).sum();

        assert!((legal_sum - 1.0).abs() < 1e-5);
        assert!(strategy.iter().all(|value| value.is_finite()));
        assert_eq!(
            state.infosets(),
            layout.infoset_count() * PRIVATE_INFOS_PER_PUBLIC
        );
    }

    #[test]
    fn cpu_depth_three_chance_game_converges_on_sparse_ranges() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 256,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let mut hero_weights = vec![0.0; COMBO_COUNT];
        let mut villain_weights = vec![0.0; COMBO_COUNT];
        for combo in [
            [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
            ],
            [
                PokedrCard::new(PokedrRank::King, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
            ],
        ] {
            hero_weights[hero_combo_index(&indexer, combo)] = 1.0;
        }
        for combo in [
            [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
            [
                PokedrCard::new(PokedrRank::Jack, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Jack, PokedrSuit::Diamonds),
            ],
        ] {
            villain_weights[hero_combo_index(&indexer, combo)] = 1.0;
        }
        let state = solve_public_tree_cfr(&tree, &layout, &config, &hero_weights, &villain_weights);
        let cpu_exploitability = cpu_infoset_root_exploitability(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );

        assert!(
            cpu_exploitability.exploitability * 100.0 < 1.0,
            "sparse depth3 CPU infoset exploitability stayed at {:.3} bb/100",
            cpu_exploitability.exploitability * 100.0
        );
    }

    #[test]
    fn cpu_infoset_br_is_bounded_by_pairwise_oracle_on_sparse_ranges() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 64,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let (hero_weights, villain_weights) = sparse_test_ranges(&indexer);
        let state = solve_public_tree_cfr(&tree, &layout, &config, &hero_weights, &villain_weights);

        let infoset = cpu_infoset_root_exploitability(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );
        let oracle_gap = cpu_pairwise_oracle_root_gap(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );

        assert!(infoset.exploitability.is_finite());
        assert!(infoset.hero_br_value.is_finite());
        assert!(infoset.villain_br_value.is_finite());
        assert!(
            infoset.exploitability <= oracle_gap + 1e-4,
            "infoset exploitability {:.6} exceeded pairwise oracle gap {:.6}",
            infoset.exploitability,
            oracle_gap
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_public_tree_iteration_matches_cpu_on_sparse_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 32,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let (hero_weights, villain_weights) = sparse_test_ranges(&indexer);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let mut cpu_batch = DenseCfrIteration::new(&dense_config);
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        fill_public_tree_iteration(
            &tree,
            &layout,
            &indexer,
            None,
            &state,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
            &mut cpu_batch,
        );

        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let root_dead = root_board(&tree).deck_mask();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &state,
            )
            .expect("GPU public tree values should run");

        let mut max_action_delta = 0.0f32;
        let mut max_action_site = None;
        let mut max_reach_delta = 0.0f32;
        let mut max_reach_site = None;
        for public_infoset in 0..layout.infoset_count() {
            let action_count = layout.action_count(public_infoset);
            let player = infoset_acting_player(&tree, &layout, public_infoset);
            for combo_index in 0..COMBO_COUNT {
                let infoset = private_infoset(public_infoset, player, combo_index);
                if cpu_batch.strategy_weights[infoset] <= 1e-6
                    && gpu_values.strategy_weights[infoset] <= 1e-6
                {
                    continue;
                }
                let reach_delta =
                    (cpu_batch.reach_weights[infoset] - gpu_values.reach_weights[infoset]).abs();
                if reach_delta > max_reach_delta {
                    max_reach_delta = reach_delta;
                    max_reach_site = Some((
                        public_infoset,
                        combo_index,
                        cpu_batch.reach_weights[infoset],
                        gpu_values.reach_weights[infoset],
                    ));
                }
                let offset = infoset * layout.max_actions();
                for action in 0..action_count {
                    let action_delta = (cpu_batch.action_values[offset + action]
                        - gpu_values.action_values[offset + action])
                        .abs();
                    if action_delta > max_action_delta {
                        max_action_delta = action_delta;
                        max_action_site = Some((
                            public_infoset,
                            combo_index,
                            action,
                            cpu_batch.action_values[offset + action],
                            gpu_values.action_values[offset + action],
                        ));
                    }
                }
            }
        }

        assert!(
            max_reach_delta < 1e-3 && max_action_delta < 1e-3,
            "GPU/CPU iteration mismatch: max_reach_delta={max_reach_delta} site={max_reach_site:?}, max_action_delta={max_action_delta} site={max_action_site:?}"
        );
        backend.wait_idle().ok();
        std::mem::forget(backend);
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_public_tree_iteration_matches_cpu_on_blocker_range_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 1,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let mut hero_weights = vec![0.0; COMBO_COUNT];
        let mut villain_weights = vec![0.0; COMBO_COUNT];
        for combo_index in legal_private_combos(&indexer, root_dead).take(64) {
            hero_weights[combo_index] = 1.0;
            villain_weights[combo_index] = 1.0;
        }
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let mut cpu_batch = DenseCfrIteration::new(&dense_config);
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        fill_public_tree_iteration(
            &tree,
            &layout,
            &indexer,
            None,
            &state,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
            &mut cpu_batch,
        );

        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &state,
            )
            .expect("GPU public tree values should run");

        let mut max_action_delta = 0.0f32;
        let mut max_action_site = None;
        let mut max_reach_delta = 0.0f32;
        let mut max_reach_site = None;
        for public_infoset in 0..layout.infoset_count() {
            let action_count = layout.action_count(public_infoset);
            let player = infoset_acting_player(&tree, &layout, public_infoset);
            for combo_index in 0..COMBO_COUNT {
                let own_weight = match player {
                    Player::Hero => hero_weights[combo_index],
                    Player::Villain => villain_weights[combo_index],
                };
                if own_weight <= 0.0 {
                    continue;
                }
                let infoset = private_infoset(public_infoset, player, combo_index);
                if cpu_batch.reach_weights[infoset] <= 1e-6
                    && gpu_values.reach_weights[infoset] <= 1e-6
                {
                    continue;
                }
                let reach_delta =
                    (cpu_batch.reach_weights[infoset] - gpu_values.reach_weights[infoset]).abs();
                if reach_delta > max_reach_delta {
                    max_reach_delta = reach_delta;
                    max_reach_site = Some((
                        public_infoset,
                        combo_index,
                        cpu_batch.reach_weights[infoset],
                        gpu_values.reach_weights[infoset],
                    ));
                }
                let offset = infoset * layout.max_actions();
                for action in 0..action_count {
                    let action_delta = (cpu_batch.action_values[offset + action]
                        - gpu_values.action_values[offset + action])
                        .abs();
                    if action_delta > max_action_delta {
                        max_action_delta = action_delta;
                        max_action_site = Some((
                            public_infoset,
                            combo_index,
                            action,
                            cpu_batch.action_values[offset + action],
                            gpu_values.action_values[offset + action],
                        ));
                    }
                }
            }
        }

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            max_reach_delta < 1e-3 && max_action_delta < 1e-3,
            "GPU/CPU full-range iteration mismatch: max_reach_delta={max_reach_delta} site={max_reach_site:?}, max_action_delta={max_action_delta} site={max_action_site:?}"
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_public_tree_iteration_matches_cpu_on_limited_fixed_flop_ranges_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 8,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");

        let mut cpu_batch = DenseCfrIteration::new(&dense_config);
        fill_public_tree_iteration(
            &tree,
            &layout,
            &indexer,
            None,
            &state,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
            &mut cpu_batch,
        );

        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &state,
            )
            .expect("GPU public tree values should run");

        let mut max_action_delta = 0.0f32;
        let mut max_action_site = None;
        let mut max_reach_delta = 0.0f32;
        let mut max_reach_site = None;
        for public_infoset in 0..layout.infoset_count() {
            let action_count = layout.action_count(public_infoset);
            let player = infoset_acting_player(&tree, &layout, public_infoset);
            for combo_index in 0..COMBO_COUNT {
                let own_weight = match player {
                    Player::Hero => hero_weights[combo_index],
                    Player::Villain => villain_weights[combo_index],
                };
                if own_weight <= 0.0 {
                    continue;
                }
                let infoset = private_infoset(public_infoset, player, combo_index);
                if cpu_batch.reach_weights[infoset] <= 1e-6
                    && gpu_values.reach_weights[infoset] <= 1e-6
                {
                    continue;
                }
                let reach_delta =
                    (cpu_batch.reach_weights[infoset] - gpu_values.reach_weights[infoset]).abs();
                if reach_delta > max_reach_delta {
                    max_reach_delta = reach_delta;
                    max_reach_site = Some((
                        public_infoset,
                        combo_index,
                        cpu_batch.reach_weights[infoset],
                        gpu_values.reach_weights[infoset],
                    ));
                }
                let offset = infoset * layout.max_actions();
                for action in 0..action_count {
                    let action_delta = (cpu_batch.action_values[offset + action]
                        - gpu_values.action_values[offset + action])
                        .abs();
                    if action_delta > max_action_delta {
                        max_action_delta = action_delta;
                        max_action_site = Some((
                            public_infoset,
                            combo_index,
                            action,
                            cpu_batch.action_values[offset + action],
                            gpu_values.action_values[offset + action],
                        ));
                    }
                }
            }
        }

        backend.wait_idle().ok();
        std::mem::forget(backend);
        let max_reach_detail = max_reach_site.map(|(public_infoset, combo_index, _, _)| {
            let node_index = layout.infoset_node(public_infoset);
            let board_mask = match &tree.nodes()[node_index].kind {
                PublicNodeKind::Decision { state, .. } => state.board.deck_mask(),
                _ => 0,
            };
            let combo = indexer.combo(combo_index);
            (
                node_index,
                combo_cards(combo),
                combo.collides_with(board_mask),
                board_mask,
            )
        });
        assert!(
            max_reach_delta < 1e-3 && max_action_delta < 1e-3,
            "GPU/CPU limited fixed-flop-range iteration mismatch: max_reach_delta={max_reach_delta} site={max_reach_site:?} detail={max_reach_detail:?}, max_action_delta={max_action_delta} site={max_action_site:?}"
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_public_tree_initial_iteration_matches_cpu_on_limited_fixed_flop_ranges_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 1,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let mut cpu_batch = DenseCfrIteration::new(&dense_config);
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        fill_public_tree_iteration(
            &tree,
            &layout,
            &indexer,
            None,
            &state,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
            &mut cpu_batch,
        );

        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &state,
            )
            .expect("GPU public tree values should run");

        let mut max_action_delta = 0.0f32;
        let mut max_action_site = None;
        for public_infoset in 0..layout.infoset_count() {
            let action_count = layout.action_count(public_infoset);
            let player = infoset_acting_player(&tree, &layout, public_infoset);
            for combo_index in 0..COMBO_COUNT {
                let own_weight = match player {
                    Player::Hero => hero_weights[combo_index],
                    Player::Villain => villain_weights[combo_index],
                };
                if own_weight <= 0.0 {
                    continue;
                }
                let infoset = private_infoset(public_infoset, player, combo_index);
                let offset = infoset * layout.max_actions();
                for action in 0..action_count {
                    let action_delta = (cpu_batch.action_values[offset + action]
                        - gpu_values.action_values[offset + action])
                        .abs();
                    if action_delta > max_action_delta {
                        max_action_delta = action_delta;
                        max_action_site = Some((
                            public_infoset,
                            combo_index,
                            action,
                            cpu_batch.action_values[offset + action],
                            gpu_values.action_values[offset + action],
                        ));
                    }
                }
            }
        }

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            max_action_delta < 1e-3,
            "GPU/CPU initial limited fixed-flop iteration mismatch: max_action_delta={max_action_delta} site={max_action_site:?}"
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_public_tree_first_update_matches_cpu_on_limited_fixed_flop_ranges_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 1,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let mut cpu_state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let mut cpu_batch = DenseCfrIteration::new(&dense_config);
        fill_public_tree_iteration(
            &tree,
            &layout,
            &indexer,
            None,
            &cpu_state,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
            &mut cpu_batch,
        );
        cpu_state.update_all_infosets(
            &cpu_batch.action_values,
            &cpu_batch.reach_weights,
            &cpu_batch.strategy_weights,
            1,
        );

        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_initial_values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &cpu_state,
            )
            .expect("GPU public tree values should run");
        let profile_zero_sum = root_exploitability_from_recursive_values(
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &gpu_initial_values,
            &gpu_initial_values,
            &gpu_initial_values,
        );
        assert!(
            (profile_zero_sum.hero_profile_value + profile_zero_sum.villain_profile_value).abs()
                < 1e-3,
            "GPU profile root values are not zero-sum: hero={} villain={} sum={}",
            profile_zero_sum.hero_profile_value,
            profile_zero_sum.villain_profile_value,
            profile_zero_sum.hero_profile_value + profile_zero_sum.villain_profile_value,
        );
        let gpu_state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");

        let mut max_regret_delta = 0.0f32;
        let mut max_regret_index = 0usize;
        for (index, (cpu, gpu)) in cpu_state
            .regrets()
            .iter()
            .zip(gpu_state.regrets())
            .enumerate()
        {
            let delta = (cpu - gpu).abs();
            if delta > max_regret_delta {
                max_regret_delta = delta;
                max_regret_index = index;
            }
        }
        let mut max_strategy_sum_delta = 0.0f32;
        let mut max_strategy_sum_index = 0usize;
        for (index, (cpu, gpu)) in cpu_state
            .strategy_sum()
            .iter()
            .zip(gpu_state.strategy_sum())
            .enumerate()
        {
            let delta = (cpu - gpu).abs();
            if delta > max_strategy_sum_delta {
                max_strategy_sum_delta = delta;
                max_strategy_sum_index = index;
            }
        }

        backend.wait_idle().ok();
        std::mem::forget(backend);
        let regret_infoset = max_regret_index / layout.max_actions();
        let regret_action = max_regret_index % layout.max_actions();
        let strategy_infoset = max_strategy_sum_index / layout.max_actions();
        let strategy_action = max_strategy_sum_index % layout.max_actions();
        let strategy_combo = strategy_infoset % PRIVATE_INFOS_PER_PUBLIC;
        let strategy_cards = combo_cards(indexer.combo(strategy_combo));
        let strategy_mask = hero_mask(strategy_cards);
        let compatible_villain_mass: f32 = indexer
            .combos()
            .iter()
            .enumerate()
            .filter(|(combo_index, combo)| {
                villain_weights[*combo_index] > 0.0
                    && !combo.collides_with(root_dead)
                    && hero_mask(combo_cards(**combo)) & strategy_mask == 0
            })
            .map(|(combo_index, _)| villain_weights[combo_index])
            .sum();
        let villain_total_mass: f32 = indexer
            .combos()
            .iter()
            .enumerate()
            .filter(|(_, combo)| !combo.collides_with(root_dead))
            .map(|(combo_index, _)| villain_weights[combo_index])
            .sum();
        let villain_first_card_mass: f32 = indexer
            .combos()
            .iter()
            .enumerate()
            .filter(|(_, combo)| {
                !combo.collides_with(root_dead)
                    && (combo.first == strategy_cards[0] || combo.second == strategy_cards[0])
            })
            .map(|(combo_index, _)| villain_weights[combo_index])
            .sum();
        let villain_second_card_mass: f32 = indexer
            .combos()
            .iter()
            .enumerate()
            .filter(|(_, combo)| {
                !combo.collides_with(root_dead)
                    && (combo.first == strategy_cards[1] || combo.second == strategy_cards[1])
            })
            .map(|(combo_index, _)| villain_weights[combo_index])
            .sum();
        let villain_same_combo_mass = villain_weights[strategy_combo];
        let compatible_villain_mass_by_aggregate =
            villain_total_mass - villain_first_card_mass - villain_second_card_mass
                + villain_same_combo_mass;
        assert!(
            max_regret_delta < 1e-3 && max_strategy_sum_delta < 1e-3,
            "GPU/CPU first update mismatch: profile_hero={} profile_villain={} profile_sum={} max_regret_delta={max_regret_delta} index={max_regret_index} infoset={regret_infoset} action={regret_action} legal={} reach={} strategy_weight={} gpu_initial_strategy_weight={} action_value={} cpu_regret={} gpu_regret={}, max_strategy_sum_delta={max_strategy_sum_delta} index={max_strategy_sum_index} infoset={strategy_infoset} combo={strategy_combo} cards={:?} action={strategy_action} legal={} reach={} strategy_weight={} gpu_initial_strategy_weight={} compatible_villain_mass={} compatible_villain_mass_by_aggregate={} total={} card0={} card1={} same={} action_value={} cpu_strategy_sum={} gpu_strategy_sum={}",
            profile_zero_sum.hero_profile_value,
            profile_zero_sum.villain_profile_value,
            profile_zero_sum.hero_profile_value + profile_zero_sum.villain_profile_value,
            private_legal_actions(&layout)[max_regret_index],
            cpu_batch.reach_weights[regret_infoset],
            cpu_batch.strategy_weights[regret_infoset],
            gpu_initial_values.strategy_weights[regret_infoset],
            cpu_batch.action_values[max_regret_index],
            cpu_state.regrets()[max_regret_index],
            gpu_state.regrets()[max_regret_index],
            strategy_cards,
            private_legal_actions(&layout)[max_strategy_sum_index],
            cpu_batch.reach_weights[strategy_infoset],
            cpu_batch.strategy_weights[strategy_infoset],
            gpu_initial_values.strategy_weights[strategy_infoset],
            compatible_villain_mass,
            compatible_villain_mass_by_aggregate,
            villain_total_mass,
            villain_first_card_mass,
            villain_second_card_mass,
            villain_same_combo_mass,
            cpu_batch.action_values[max_strategy_sum_index],
            cpu_state.strategy_sum()[max_strategy_sum_index],
            gpu_state.strategy_sum()[max_strategy_sum_index],
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_depth_three_chance_game_converges_on_sparse_ranges() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 256,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 0,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let (hero_weights, villain_weights) = sparse_test_ranges(&indexer);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let root_dead = root_board(&tree).deck_mask();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let profile = state.average_strategy_profile_state();
        let gpu_metrics = recursive_br_gap_metrics_gpu_for_profile(
            &backend,
            &tree,
            &linearized,
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &layout,
            &state,
            &profile,
            false,
        )
        .expect("GPU recursive BR metric should run");
        let cpu_exploitability = cpu_infoset_root_exploitability(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            cpu_exploitability.exploitability * 100.0 < 1.0,
            "GPU sparse depth3 CPU infoset exploitability stayed at {:.3} bb/100",
            cpu_exploitability.exploitability * 100.0
        );
        assert!(
            (gpu_metrics.root_exploitability - cpu_exploitability.exploitability).abs() < 1e-3,
            "GPU metric exploitability {:.6} differs from CPU infoset exploitability {:.6}",
            gpu_metrics.root_exploitability,
            cpu_exploitability.exploitability
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_depth_three_chance_game_converges_on_limited_fixed_flop_ranges() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 256,
            cfr_variant: CfrVariant::dcfr_plus_default(),
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let profile = state.average_strategy_profile_state();
        let gpu_metrics = recursive_br_gap_metrics_gpu_for_profile(
            &backend,
            &tree,
            &linearized,
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &layout,
            &state,
            &profile,
            false,
        )
        .expect("GPU recursive BR metric should run");
        assert!(
            (gpu_metrics.hero_root_profile_value + gpu_metrics.villain_root_profile_value).abs()
                < 1e-3,
            "GPU resident average profile values are not zero-sum: hero={} villain={} sum={}",
            gpu_metrics.hero_root_profile_value,
            gpu_metrics.villain_root_profile_value,
            gpu_metrics.hero_root_profile_value + gpu_metrics.villain_root_profile_value,
        );

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            gpu_metrics.root_exploitability * 100.0 < 10.0,
            "GPU limited fixed-flop depth3 exploitability stayed at {:.3} bb/100",
            gpu_metrics.root_exploitability * 100.0
        );
    }

    #[test]
    fn fixed_flop_tree_terminals_preserve_pot_investment_sum_at_depth_five() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
            ..PokedrAgentConfig::default()
        };
        let (tree, _) = fixed_flop_tree_and_layout(flop, config);
        for (node_index, node) in tree.nodes().iter().enumerate() {
            let PublicNodeKind::Terminal {
                pot,
                hero_invested,
                villain_invested,
                ..
            } = &node.kind
            else {
                continue;
            };
            assert_eq!(
                *pot,
                hero_invested + villain_invested,
                "terminal node {node_index} violates pot investment invariant"
            );
        }
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_depth_five_average_profile_values_remain_zero_sum() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 8,
            cfr_variant: CfrVariant::dcfr_plus_default(),
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");
        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let profile = state.average_strategy_profile_state();
        let values = backend
            .public_tree_iteration_values(
                &linearized.nodes,
                &linearized.children,
                &linearized.child_cards,
                &combos,
                &combo_legal,
                &hero_weights,
                &villain_weights,
                &linearized.showdown_boards,
                &profile,
            )
            .expect("GPU profile values should run");
        let metrics = root_exploitability_from_recursive_values(
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &values,
            &values,
            &values,
        );

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            (metrics.hero_profile_value + metrics.villain_profile_value).abs() < 1e-3,
            "GPU depth5 average profile values are not zero-sum: hero={} villain={} sum={}",
            metrics.hero_profile_value,
            metrics.villain_profile_value,
            metrics.hero_profile_value + metrics.villain_profile_value,
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_recursive_br_matches_cpu_on_blocker_range_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 1,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 0,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let mut hero_weights = vec![0.0; COMBO_COUNT];
        let mut villain_weights = vec![0.0; COMBO_COUNT];
        for combo_index in legal_private_combos(&indexer, root_dead).take(64) {
            hero_weights[combo_index] = 1.0;
            villain_weights[combo_index] = 1.0;
        }

        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");
        let cpu_exploitability = cpu_infoset_root_exploitability(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );

        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_metrics = recursive_br_gap_metrics_gpu(
            &backend,
            &tree,
            &linearized,
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &layout,
            &state,
            false,
        )
        .expect("GPU recursive BR metrics should run");

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            (gpu_metrics.root_exploitability - cpu_exploitability.exploitability).abs() < 1e-3,
            "GPU recursive exploitability mismatch: gpu={} cpu={}",
            gpu_metrics.root_exploitability,
            cpu_exploitability.exploitability
        );
    }

    #[test]
    #[ignore = "requires a working local GPU backend"]
    fn gpu_recursive_br_matches_cpu_on_limited_fixed_flop_ranges_depth_three() {
        let flop = [
            PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
            PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
            PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
        ];
        let config = PokedrAgentConfig {
            cfr_iterations: 8,
            cfr_variant: CfrVariant::CfrPlus,
            action_set: ActionSetConfig {
                max_aggressive_actions: 1,
                flop_bet_fractions: vec![1.0],
                turn_bet_fractions: vec![1.0],
                river_bet_fractions: vec![1.0],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_showdown_runouts: usize::MAX,
        };
        let (tree, layout) = fixed_flop_tree_and_layout(flop, config.clone());
        let indexer = ComboIndexer::new();
        let root_dead = root_board(&tree).deck_mask();
        let (mut hero_weights, mut villain_weights) = fixed_flop_root_weights(&indexer, root_dead);
        limit_nonzero_weights(&mut hero_weights, 32);
        limit_nonzero_weights(&mut villain_weights, 32);

        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let backend = GpuDenseCfrBackend::new().expect("GPU backend should initialize");
        let mut dense_config = layout.dense_config(config.cfr_variant);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let state = try_solve_gpu_public_tree_resident(
            &tree,
            &layout,
            &indexer,
            &backend,
            &dense_config,
            &config,
            &hero_weights,
            &villain_weights,
            &matrix_cache,
        )
        .expect("GPU resident solve should run");
        let cpu_exploitability = cpu_infoset_root_exploitability(
            &tree,
            &layout,
            &state,
            &indexer,
            &hero_weights,
            &villain_weights,
            config.max_showdown_runouts,
        );

        let linearized =
            linearize_gpu_public_tree(&tree, &layout, &backend, &config, &matrix_cache)
                .expect("tree should linearize");
        let combos = gpu_private_combos();
        let combo_legal = indexer
            .combos()
            .iter()
            .map(|combo| (!combo.collides_with(root_dead)) as u32)
            .collect::<Vec<_>>();
        let gpu_metrics = recursive_br_gap_metrics_gpu(
            &backend,
            &tree,
            &linearized,
            &combos,
            &combo_legal,
            &hero_weights,
            &villain_weights,
            &layout,
            &state,
            false,
        )
        .expect("GPU recursive BR metrics should run");

        backend.wait_idle().ok();
        std::mem::forget(backend);
        assert!(
            (gpu_metrics.root_exploitability - cpu_exploitability.exploitability).abs() < 1e-3,
            "GPU limited fixed-flop-range recursive exploitability mismatch: gpu={} cpu={}",
            gpu_metrics.root_exploitability,
            cpu_exploitability.exploitability
        );
    }

    fn limit_nonzero_weights(weights: &mut [f32], max_nonzero: usize) {
        let mut kept = 0usize;
        for weight in weights {
            if *weight <= 0.0 {
                continue;
            }
            kept += 1;
            if kept > max_nonzero {
                *weight = 0.0;
            }
        }
    }

    fn sparse_test_ranges(indexer: &ComboIndexer) -> (Vec<f32>, Vec<f32>) {
        let mut hero_weights = vec![0.0; COMBO_COUNT];
        let mut villain_weights = vec![0.0; COMBO_COUNT];
        for combo in [
            [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
            ],
            [
                PokedrCard::new(PokedrRank::King, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
            ],
        ] {
            hero_weights[hero_combo_index(indexer, combo)] = 1.0;
        }
        for combo in [
            [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
            [
                PokedrCard::new(PokedrRank::Jack, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Jack, PokedrSuit::Diamonds),
            ],
        ] {
            villain_weights[hero_combo_index(indexer, combo)] = 1.0;
        }
        (hero_weights, villain_weights)
    }

    fn infoset_br_hero_payoff(action_values: &[Vec<f32>], opponent_weights: &[f32]) -> f32 {
        assert!(!action_values.is_empty());
        assert!(
            action_values
                .iter()
                .all(|values| values.len() == opponent_weights.len())
        );
        action_values
            .iter()
            .map(|values| {
                values
                    .iter()
                    .zip(opponent_weights)
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>()
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn clairvoyant_pairwise_br_hero_payoff(
        action_values: &[Vec<f32>],
        opponent_weights: &[f32],
    ) -> f32 {
        assert!(!action_values.is_empty());
        let opponent_count = opponent_weights.len();
        assert!(
            action_values
                .iter()
                .all(|values| values.len() == opponent_count)
        );
        (0..opponent_count)
            .map(|opponent| {
                let best = action_values
                    .iter()
                    .map(|values| values[opponent])
                    .fold(f32::NEG_INFINITY, f32::max);
                best * opponent_weights[opponent]
            })
            .sum()
    }

    fn infoset_br_villain_payoff(action_values: &[Vec<f32>], opponent_weights: &[f32]) -> f32 {
        assert!(!action_values.is_empty());
        assert!(
            action_values
                .iter()
                .all(|values| values.len() == opponent_weights.len())
        );
        action_values
            .iter()
            .map(|values| {
                values
                    .iter()
                    .zip(opponent_weights)
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>()
            })
            .fold(f32::INFINITY, f32::min)
    }

    fn clairvoyant_pairwise_villain_payoff(
        action_values: &[Vec<f32>],
        opponent_weights: &[f32],
    ) -> f32 {
        assert!(!action_values.is_empty());
        let opponent_count = opponent_weights.len();
        assert!(
            action_values
                .iter()
                .all(|values| values.len() == opponent_count)
        );
        (0..opponent_count)
            .map(|opponent| {
                let best = action_values
                    .iter()
                    .map(|values| values[opponent])
                    .fold(f32::INFINITY, f32::min);
                best * opponent_weights[opponent]
            })
            .sum()
    }

    #[test]
    fn infoset_br_chooses_one_action_before_observing_opponent_combo() {
        let opponent_weights = [0.5, 0.5];
        let action_values = vec![vec![10.0, -10.0], vec![0.0, 0.0]];

        let infoset_br = infoset_br_hero_payoff(&action_values, &opponent_weights);
        let clairvoyant_br = clairvoyant_pairwise_br_hero_payoff(&action_values, &opponent_weights);

        assert!((infoset_br - 0.0).abs() < 1e-6);
        assert!((clairvoyant_br - 5.0).abs() < 1e-6);
        assert!(clairvoyant_br > infoset_br);
    }

    #[test]
    fn villain_infoset_br_chooses_one_action_before_observing_hero_combo() {
        let opponent_weights = [0.5, 0.5];
        let action_values = vec![vec![10.0, -10.0], vec![0.0, 0.0]];

        let infoset_br = infoset_br_villain_payoff(&action_values, &opponent_weights);
        let clairvoyant_br = clairvoyant_pairwise_villain_payoff(&action_values, &opponent_weights);

        assert!((infoset_br - 0.0).abs() < 1e-6);
        assert!((clairvoyant_br - -5.0).abs() < 1e-6);
        assert!(clairvoyant_br < infoset_br);
    }

    #[test]
    fn infoset_br_weights_reachable_opponent_combos_before_choosing_action() {
        let opponent_weights = [0.9, 0.1];
        let action_values = vec![vec![10.0, -10.0], vec![0.0, 0.0]];

        let infoset_br = infoset_br_hero_payoff(&action_values, &opponent_weights);
        let clairvoyant_br = clairvoyant_pairwise_br_hero_payoff(&action_values, &opponent_weights);

        assert!((infoset_br - 8.0).abs() < 1e-6);
        assert!((clairvoyant_br - 9.0).abs() < 1e-6);
        assert!(clairvoyant_br > infoset_br);
    }

    fn cpu_pairwise_oracle_root_gap(
        tree: &SubgameTree,
        layout: &PostflopDenseLayout,
        state: &DenseCfrState,
        indexer: &ComboIndexer,
        hero_weights: &[f32],
        villain_weights: &[f32],
        max_showdown_runouts: usize,
    ) -> f32 {
        let legal_combos = legal_private_combos(indexer, root_board(tree).deck_mask())
            .map(|combo_index| {
                let cards = combo_cards(indexer.combo(combo_index));
                (combo_index, cards, hero_mask(cards))
            })
            .collect::<Vec<_>>();
        let matrix_cache = RefCell::new(ShowdownMatrixCache::new(1));
        let mut hero_br_sum = 0.0;
        let mut villain_br_sum = 0.0;
        let mut joint_weight_sum = 0.0;

        for (hero_combo, hero_cards, hero_dead) in &legal_combos {
            let hero_weight = hero_weights[*hero_combo];
            if hero_weight > 0.0 {
                for (villain_combo, villain_cards, villain_dead) in &legal_combos {
                    let villain_weight = villain_weights[*villain_combo];
                    if villain_weight <= 0.0 || hero_dead & villain_dead != 0 {
                        continue;
                    }
                    let joint_weight = hero_weight * villain_weight;
                    joint_weight_sum += joint_weight;
                    let mut ctx = PostflopEvaluationContext {
                        hero_cards: *hero_cards,
                        villain_cards: *villain_cards,
                        hero_combo: *hero_combo,
                        villain_combo: *villain_combo,
                        gpu_backend: None,
                        matrix_cache: &matrix_cache,
                        max_showdown_runouts,
                        equity_cache: HashMap::new(),
                    };
                    hero_br_sum += joint_weight
                        * cpu_pairwise_oracle_br_hero_payoff(
                            tree,
                            layout,
                            0,
                            None,
                            None,
                            *hero_combo,
                            *villain_combo,
                            Player::Hero,
                            state,
                            &mut ctx,
                        );
                }
            }

            let villain_weight = villain_weights[*hero_combo];
            if villain_weight <= 0.0 {
                continue;
            }
            let mut weighted_value = 0.0;
            for (opponent_combo, opponent_cards, opponent_dead) in &legal_combos {
                let hero_opponent_weight = hero_weights[*opponent_combo];
                if hero_opponent_weight <= 0.0 || hero_dead & opponent_dead != 0 {
                    continue;
                }
                let joint_weight = villain_weight * hero_opponent_weight;
                let mut ctx = PostflopEvaluationContext {
                    hero_cards: *opponent_cards,
                    villain_cards: *hero_cards,
                    hero_combo: *opponent_combo,
                    villain_combo: *hero_combo,
                    gpu_backend: None,
                    matrix_cache: &matrix_cache,
                    max_showdown_runouts,
                    equity_cache: HashMap::new(),
                };
                weighted_value += joint_weight
                    * -cpu_pairwise_oracle_br_hero_payoff(
                        tree,
                        layout,
                        0,
                        None,
                        None,
                        *opponent_combo,
                        *hero_combo,
                        Player::Villain,
                        state,
                        &mut ctx,
                    );
            }
            villain_br_sum += weighted_value;
        }

        if joint_weight_sum == 0.0 {
            return 0.0;
        }
        (hero_br_sum / joint_weight_sum + villain_br_sum / joint_weight_sum).max(0.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_pairwise_oracle_br_hero_payoff(
        tree: &SubgameTree,
        layout: &PostflopDenseLayout,
        node_index: usize,
        parent_state: Option<&PublicState>,
        parent_action: Option<PlayerAction>,
        hero_combo: usize,
        villain_combo: usize,
        br_player: Player,
        state: &DenseCfrState,
        ctx: &mut PostflopEvaluationContext,
    ) -> f32 {
        match &tree.nodes()[node_index].kind {
            PublicNodeKind::Decision {
                state: public_state,
                actions,
            } => {
                let public_infoset = layout
                    .node_infoset(node_index)
                    .expect("decision nodes have infosets");
                let acting_combo = match public_state.acting_player {
                    Player::Hero => hero_combo,
                    Player::Villain => villain_combo,
                };
                let private_infoset =
                    private_infoset(public_infoset, public_state.acting_player, acting_combo);
                let mut strategy = vec![0.0; layout.max_actions()];
                state.average_strategy_for(private_infoset, &mut strategy);
                let mut values = Vec::with_capacity(actions.len());
                for (action_index, action) in actions.iter().enumerate() {
                    let child = layout
                        .child_for_action(public_infoset, action_index)
                        .expect("legal action must have a child");
                    values.push(cpu_pairwise_oracle_br_hero_payoff(
                        tree,
                        layout,
                        child,
                        Some(public_state),
                        Some(action.action),
                        hero_combo,
                        villain_combo,
                        br_player,
                        state,
                        ctx,
                    ));
                }
                if public_state.acting_player == br_player {
                    if br_player == Player::Hero {
                        values.into_iter().fold(f32::NEG_INFINITY, f32::max)
                    } else {
                        values.into_iter().fold(f32::INFINITY, f32::min)
                    }
                } else {
                    values
                        .iter()
                        .zip(strategy)
                        .map(|(value, probability)| value * probability)
                        .sum()
                }
            }
            PublicNodeKind::Chance { cards, .. } => {
                let valid_count = cards
                    .iter()
                    .filter(|card| card.deck_mask() & private_dead_mask(ctx) == 0)
                    .count();
                if valid_count == 0 {
                    return 0.0;
                }
                let mut sum = 0.0;
                for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                    if card.deck_mask() & private_dead_mask(ctx) != 0 {
                        continue;
                    }
                    sum += cpu_pairwise_oracle_br_hero_payoff(
                        tree,
                        layout,
                        *child,
                        None,
                        None,
                        hero_combo,
                        villain_combo,
                        br_player,
                        state,
                        ctx,
                    );
                }
                sum / valid_count as f32
            }
            PublicNodeKind::Terminal {
                kind,
                board,
                pot,
                hero_invested,
                ..
            } => match kind {
                TerminalKind::Fold => fold_utility(
                    parent_state.expect("fold terminal must have a parent decision"),
                    parent_action.expect("fold terminal must have a parent action"),
                    *pot,
                    *hero_invested,
                ),
                TerminalKind::Showdown => showdown_utility(*pot, *hero_invested, board, ctx),
            },
        }
    }

    #[test]
    fn villain_decision_uses_private_strategy() {
        let tree = SubgameTree::build(
            PublicState {
                street: Street::River,
                board: Board::new(vec![
                    PokedrCard::new(PokedrRank::Ace, PokedrSuit::Spades),
                    PokedrCard::new(PokedrRank::Seven, PokedrSuit::Hearts),
                    PokedrCard::new(PokedrRank::Two, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
                    PokedrCard::new(PokedrRank::Three, PokedrSuit::Spades),
                ]),
                pot: 100,
                hero_invested: 50,
                villain_invested: 50,
                effective_stack: 100,
                to_call: 20,
                min_aggressive_amount: 60,
                acting_player: Player::Villain,
                raises_this_street: 0,
                checks_this_street: 0,
            },
            SubgameTreeConfig {
                action_set: ActionSetConfig {
                    max_aggressive_actions: 1,
                    raise_fractions: vec![1.0],
                    ..ActionSetConfig::default()
                },
                max_raises_per_street: 0,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let mut dense_config = layout.dense_config(CfrVariant::CfrPlus);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let cfr_state = DenseCfrState::new_with_legal_actions(
            dense_config.clone(),
            private_legal_actions(&layout),
        );
        let hero_combo = hero_combo_index(
            &ComboIndexer::new(),
            [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
            ],
        );
        let villain_combo = hero_combo_index(
            &ComboIndexer::new(),
            [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
        );
        let mut ctx = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
            ],
            villain_cards: [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
            hero_combo,
            villain_combo,
            gpu_backend: None,
            matrix_cache: &RefCell::new(ShowdownMatrixCache::new(1)),
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let mut batch = DenseCfrIteration::new(&dense_config);
        let mut value_weights = vec![0.0; batch.action_values.len()];
        let value = traverse_cfr_node(
            &tree,
            &layout,
            0,
            None,
            None,
            hero_combo,
            villain_combo,
            1.0,
            1.0,
            1.0,
            &cfr_state,
            &mut ctx,
            &mut batch,
            &mut value_weights,
        );

        assert!(value.is_finite());
    }

    #[test]
    fn preflop_classifier_keeps_medium_pairs_out_of_four_bet_range() {
        let nines = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::Nine, RsSuit::Club),
            RsCard::new(RsValue::Nine, RsSuit::Diamond),
        ]);
        let queens = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::Queen, RsSuit::Club),
            RsCard::new(RsValue::Queen, RsSuit::Diamond),
        ]);
        let ace_king = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::Ace, RsSuit::Club),
            RsCard::new(RsValue::King, RsSuit::Diamond),
        ]);
        let ace_eight = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::Ace, RsSuit::Heart),
            RsCard::new(RsValue::Eight, RsSuit::Club),
        ]);
        let king_ten = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::King, RsSuit::Spade),
            RsCard::new(RsValue::Ten, RsSuit::Club),
        ]);
        let king_two_suited = rs_poker::core::Hand::new_with_cards(vec![
            RsCard::new(RsValue::King, RsSuit::Spade),
            RsCard::new(RsValue::Two, RsSuit::Spade),
        ]);

        assert_eq!(classify_preflop_hand(&nines), PreflopClass::Playable);
        assert!(!classify_preflop_hand(&nines).four_bets());
        assert!(classify_preflop_hand(&nines).calls_three_bet());
        assert_eq!(classify_preflop_hand(&queens), PreflopClass::Premium);
        assert_eq!(classify_preflop_hand(&ace_king), PreflopClass::Strong);
        assert!(!classify_preflop_hand(&ace_king).four_bets());
        assert_eq!(classify_preflop_hand(&ace_eight), PreflopClass::Speculative);
        assert_eq!(classify_preflop_hand(&king_ten), PreflopClass::Playable);
        assert_eq!(
            classify_preflop_hand(&king_two_suited),
            PreflopClass::Speculative
        );
    }
}
