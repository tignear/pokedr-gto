pub use pokedr_core::{dense_cfr, postflop, postflop_dense, range};

use std::collections::HashMap;

use pokedr_core::{
    cards::{Board, Card as PokedrCard, Rank as PokedrRank, Suit as PokedrSuit, evaluate},
    dense_cfr::gpu::{GpuCfrError, GpuDenseCfrBackend},
    dense_cfr::{CfrVariant, DenseCfrIteration, DenseCfrState},
    postflop::{
        ActionSetConfig, Player, PlayerAction, PublicNodeKind, PublicState, Street, SubgameTree,
        SubgameTreeConfig, TerminalKind,
    },
    postflop_dense::PostflopDenseLayout,
    range::{COMBO_COUNT, Combo, ComboIndexer},
};
use rs_poker::{
    arena::{
        Agent,
        action::Action,
        action::AgentAction,
        game_state::{GameState, Round},
        historian::{HistoryRecord, VecHistorian},
    },
    core::{Card as RsCard, Suit as RsSuit, Value as RsValue},
};

#[derive(Debug, Clone)]
pub struct PokedrAgent {
    config: PokedrAgentConfig,
}

#[derive(Debug, Clone)]
pub struct PokedrAgentConfig {
    pub cfr_iterations: usize,
    pub action_set: ActionSetConfig,
    pub max_raises_per_street: u8,
    pub max_depth: usize,
    pub max_showdown_runouts: usize,
}

impl Default for PokedrAgentConfig {
    fn default() -> Self {
        Self {
            cfr_iterations: 8,
            action_set: ActionSetConfig {
                max_aggressive_actions: 2,
                flop_bet_fractions: vec![0.5],
                turn_bet_fractions: vec![0.5],
                river_bet_fractions: vec![0.5],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
            max_depth: 5,
            max_showdown_runouts: 128,
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

impl Default for PokedrAgent {
    fn default() -> Self {
        Self::new(PokedrAgentConfig::default())
    }
}

impl PokedrAgent {
    pub fn new(config: PokedrAgentConfig) -> Self {
        Self { config }
    }

    fn choose_action(&self, game_state: &GameState) -> AgentAction {
        match game_state.round {
            Round::Preflop => self.preflop_action(game_state),
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

    fn postflop_action(&self, game_state: &GameState) -> AgentAction {
        let Some(public_state) = public_state_from_game(game_state) else {
            return AgentAction::Call;
        };
        let tree = SubgameTree::build(
            public_state,
            SubgameTreeConfig {
                action_set: self.config.action_set.clone(),
                max_raises_per_street: self.config.max_raises_per_street,
                max_depth: self.config.max_depth,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let Some(hero_cards) = hero_cards_from_game(game_state) else {
            return AgentAction::Call;
        };
        let indexer = ComboIndexer::new();
        let hero_combo = hero_combo_index(&indexer, hero_cards);
        let state = solve_public_tree_cfr(&tree, &layout, &self.config);
        let action_index = best_average_strategy_action(&layout, &state, Player::Hero, hero_combo);
        let action = layout
            .action(&tree, 0, action_index)
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
}

pub fn run_heads_up_match(hands: usize, seed: u64) -> MatchSummary {
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
                Box::new(PokedrAgent::default()),
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
                Box::new(PokedrAgent::default()),
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

struct PostflopEvaluationContext {
    hero_cards: [PokedrCard; 2],
    villain_cards: [PokedrCard; 2],
    max_showdown_runouts: usize,
    equity_cache: HashMap<u64, f32>,
}

fn solve_public_tree_cfr(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    config: &PokedrAgentConfig,
) -> DenseCfrState {
    let mut dense_config = layout.dense_config(CfrVariant::CfrPlus);
    dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
    let mut state =
        DenseCfrState::new_with_legal_actions(dense_config.clone(), private_legal_actions(layout));
    let mut batch = DenseCfrIteration::new(&dense_config);
    let indexer = ComboIndexer::new();
    let gpu_backend = cfr_gpu_backend();

    for iteration in 1..=config.cfr_iterations.max(1) {
        fill_public_tree_iteration(tree, layout, &indexer, &state, config, &mut batch);
        batch.validate(&dense_config);
        if let Some(backend) = &gpu_backend {
            if backend
                .update_all_infosets(
                    &mut state,
                    &batch.action_values,
                    &batch.reach_weights,
                    &batch.strategy_weights,
                    iteration,
                )
                .is_ok()
            {
                continue;
            }
        }
        state.update_all_infosets(
            &batch.action_values,
            &batch.reach_weights,
            &batch.strategy_weights,
            iteration,
        );
    }
    state
}

fn cfr_gpu_backend() -> Option<GpuDenseCfrBackend> {
    if cfg!(test) || std::env::var_os("POKEDR_DISABLE_GPU_CFR").is_some() {
        None
    } else {
        GpuDenseCfrBackend::new().ok()
    }
}

fn fill_public_tree_iteration(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    indexer: &ComboIndexer,
    cfr_state: &DenseCfrState,
    config: &PokedrAgentConfig,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(0.0);
    batch.strategy_weights.fill(0.0);
    let mut value_counts = vec![0.0; batch.action_values.len()];
    let root_dead = root_board(tree).deck_mask();

    for infoset in 0..layout.infoset_count() {
        let node_index = layout.infoset_node(infoset);
        let PublicNodeKind::Decision {
            state: public_state,
            actions,
        } = &tree.nodes()[node_index].kind
        else {
            unreachable!("infoset nodes are decisions");
        };
        for hero_combo in legal_private_combos(indexer, root_dead) {
            let hero_cards = combo_cards(indexer.combo(hero_combo));
            for villain_combo in legal_private_combos(indexer, root_dead | hero_mask(hero_cards)) {
                let villain_cards = combo_cards(indexer.combo(villain_combo));
                let mut ctx = PostflopEvaluationContext {
                    hero_cards,
                    villain_cards,
                    max_showdown_runouts: config.max_showdown_runouts.max(1),
                    equity_cache: HashMap::new(),
                };
                let acting_combo = match public_state.acting_player {
                    Player::Hero => hero_combo,
                    Player::Villain => villain_combo,
                };
                let private_infoset =
                    private_infoset(infoset, public_state.acting_player, acting_combo);
                let offset = private_infoset * layout.max_actions();
                for action_index in 0..layout.action_count(infoset) {
                    let child = layout
                        .child_for_action(infoset, action_index)
                        .expect("legal action must have a child");
                    let hero_value = evaluate_action_child(
                        tree,
                        layout,
                        child,
                        public_state,
                        actions[action_index].action,
                        hero_combo,
                        villain_combo,
                        cfr_state,
                        &mut ctx,
                    );
                    let player_value = match public_state.acting_player {
                        Player::Hero => hero_value,
                        Player::Villain => -hero_value,
                    };
                    batch.action_values[offset + action_index] += player_value;
                    value_counts[offset + action_index] += 1.0;
                }
                batch.reach_weights[private_infoset] = 1.0;
                batch.strategy_weights[private_infoset] = 1.0;
            }
        }
    }

    for (value, count) in batch.action_values.iter_mut().zip(value_counts) {
        if count > 0.0 {
            *value /= count;
        }
    }
}

fn evaluate_action_child(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    parent_state: &PublicState,
    parent_action: PlayerAction,
    hero_combo: usize,
    villain_combo: usize,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision { state, .. } => evaluate_decision(
            tree,
            layout,
            node_index,
            state,
            hero_combo,
            villain_combo,
            cfr_state,
            ctx,
        ),
        PublicNodeKind::Chance { cards, .. } => {
            let mut sum = 0.0;
            let mut count = 0;
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                if card.deck_mask() & private_dead_mask(ctx) != 0 {
                    continue;
                }
                sum += evaluate_node(
                    tree,
                    layout,
                    *child,
                    hero_combo,
                    villain_combo,
                    cfr_state,
                    ctx,
                );
                count += 1;
            }
            if count == 0 { 0.0 } else { sum / count as f32 }
        }
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            ..
        } => match kind {
            TerminalKind::Fold => fold_utility(parent_state, parent_action, *pot, *hero_invested),
            TerminalKind::Showdown => showdown_utility(*pot, *hero_invested, board, ctx),
        },
    }
}

fn evaluate_node(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    hero_combo: usize,
    villain_combo: usize,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision { state, .. } => evaluate_decision(
            tree,
            layout,
            node_index,
            state,
            hero_combo,
            villain_combo,
            cfr_state,
            ctx,
        ),
        PublicNodeKind::Chance { cards, .. } => {
            let mut sum = 0.0;
            let mut count = 0;
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                if card.deck_mask() & private_dead_mask(ctx) != 0 {
                    continue;
                }
                sum += evaluate_node(
                    tree,
                    layout,
                    *child,
                    hero_combo,
                    villain_combo,
                    cfr_state,
                    ctx,
                );
                count += 1;
            }
            if count == 0 { 0.0 } else { sum / count as f32 }
        }
        PublicNodeKind::Terminal {
            kind,
            board,
            pot,
            hero_invested,
            ..
        } => match kind {
            TerminalKind::Fold => 0.0,
            TerminalKind::Showdown => showdown_utility(*pot, *hero_invested, board, ctx),
        },
    }
}

fn evaluate_decision(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    state: &PublicState,
    hero_combo: usize,
    villain_combo: usize,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    let PublicNodeKind::Decision { actions, .. } = &tree.nodes()[node_index].kind else {
        unreachable!("decision node expected");
    };
    let mut values = Vec::with_capacity(actions.len());
    for (action, child) in actions.iter().zip(&tree.nodes()[node_index].children) {
        values.push(evaluate_action_child(
            tree,
            layout,
            *child,
            state,
            action.action,
            hero_combo,
            villain_combo,
            cfr_state,
            ctx,
        ));
    }
    let Some(infoset) = layout.node_infoset(node_index) else {
        return values.iter().sum::<f32>() / values.len().max(1) as f32;
    };
    let mut strategy = vec![0.0; layout.max_actions()];
    let acting_combo = match state.acting_player {
        Player::Hero => hero_combo,
        Player::Villain => villain_combo,
    };
    cfr_state.strategy_for(
        private_infoset(infoset, state.acting_player, acting_combo),
        &mut strategy,
    );
    values
        .iter()
        .zip(strategy)
        .map(|(value, probability)| value * probability)
        .sum()
}

fn best_average_strategy_action(
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    player: Player,
    hero_combo: usize,
) -> usize {
    let mut strategy = vec![0.0; layout.max_actions()];
    state.average_strategy_for(private_infoset(0, player, hero_combo), &mut strategy);
    let mut best = 0;
    let mut best_probability = f32::NEG_INFINITY;
    for (action, probability) in strategy
        .iter()
        .copied()
        .enumerate()
        .take(layout.action_count(0))
    {
        if probability > best_probability {
            best = action;
            best_probability = probability;
        }
    }
    best
}

fn private_infoset(public_infoset: usize, player: Player, private_combo: usize) -> usize {
    let player_offset = match player {
        Player::Hero => 0,
        Player::Villain => PRIVATE_HANDS,
    };
    public_infoset * PRIVATE_INFOS_PER_PUBLIC + player_offset + private_combo
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
    let key = board.deck_mask() ^ private_dead_mask(ctx).rotate_left(17);
    if let Some(equity) = ctx.equity_cache.get(&key) {
        return *equity;
    }

    let dead = board.deck_mask() | private_dead_mask(ctx);
    let runouts = completion_runouts(board, dead, ctx.max_showdown_runouts);
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

const PRIVATE_HANDS: usize = COMBO_COUNT;
const PRIVATE_INFOS_PER_PUBLIC: usize = PRIVATE_HANDS * 2;

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
        self >= Self::Strong
    }

    fn four_bets(self) -> bool {
        self >= Self::Premium
    }

    fn calls_large_preflop_bet(self) -> bool {
        self >= Self::Premium
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
    if high == 14 && low >= 10 {
        return if suited {
            PreflopClass::Strong
        } else {
            PreflopClass::Playable
        };
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
        let summary = run_heads_up_match(4, 7);
        assert_eq!(summary.hands, 4);
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
        let mut strong = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Diamonds),
            ],
            villain_cards: [
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
            ],
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
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };

        assert!(showdown_equity(&board, &mut strong) > showdown_equity(&board, &mut weak));
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
                max_depth: 1,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let state = solve_public_tree_cfr(
            &tree,
            &layout,
            &PokedrAgentConfig {
                cfr_iterations: 1,
                max_showdown_runouts: 1,
                ..PokedrAgentConfig::default()
            },
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
                max_depth: 1,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let mut dense_config = layout.dense_config(CfrVariant::CfrPlus);
        dense_config.infosets *= PRIVATE_INFOS_PER_PUBLIC;
        let cfr_state =
            DenseCfrState::new_with_legal_actions(dense_config, private_legal_actions(&layout));
        let hero_combo = hero_combo_index(
            &ComboIndexer::new(),
            [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
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
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let value = evaluate_decision(
            &tree,
            &layout,
            0,
            &root_public_state(&tree),
            hero_combo,
            hero_combo_index(
                &ComboIndexer::new(),
                [
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Clubs),
                    PokedrCard::new(PokedrRank::Queen, PokedrSuit::Diamonds),
                ],
            ),
            &cfr_state,
            &mut ctx,
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

        assert_eq!(classify_preflop_hand(&nines), PreflopClass::Playable);
        assert!(!classify_preflop_hand(&nines).four_bets());
        assert_eq!(classify_preflop_hand(&queens), PreflopClass::Premium);
        assert_eq!(classify_preflop_hand(&ace_king), PreflopClass::Strong);
        assert!(!classify_preflop_hand(&ace_king).four_bets());
    }

    fn root_public_state(tree: &SubgameTree) -> PublicState {
        let PublicNodeKind::Decision { state, .. } = &tree.nodes()[0].kind else {
            panic!("root should be a decision");
        };
        state.clone()
    }
}
