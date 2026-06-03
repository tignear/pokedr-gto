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
    range::{Combo, ComboIndexer},
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
        let hand_score = game_state
            .hands
            .get(game_state.to_act_idx())
            .map(preflop_hand_score)
            .unwrap_or(0);
        if to_call <= 0.0 {
            if hand_score >= 22 {
                return self.bet_fraction(game_state, 0.75);
            }
            return AgentAction::Call;
        }
        if hand_score < 12 && to_call > game_state.big_blind * 2.0 {
            AgentAction::Fold
        } else if hand_score >= 24 && can_raise(game_state) {
            self.bet_fraction(game_state, 0.75)
        } else {
            AgentAction::Call
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
        let state = solve_public_tree_cfr(&tree, &layout, hero_cards, &self.config);
        let action_index = best_average_strategy_action(&layout, &state);
        let action = layout
            .action(&tree, 0, action_index)
            .map(|candidate| candidate.action)
            .unwrap_or(PlayerAction::Check);
        to_rs_action(game_state, action)
    }

    fn bet_fraction(&self, game_state: &GameState, fraction: f32) -> AgentAction {
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let target = game_state.current_round_bet()
            + (game_state.total_pot.max(game_state.big_blind) * fraction).round();
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
    indexer: ComboIndexer,
    max_showdown_runouts: usize,
    equity_cache: HashMap<u64, f32>,
}

fn solve_public_tree_cfr(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    hero_cards: [PokedrCard; 2],
    config: &PokedrAgentConfig,
) -> DenseCfrState {
    let dense_config = layout.dense_config(CfrVariant::CfrPlus);
    let mut state = DenseCfrState::new_with_legal_actions(
        dense_config.clone(),
        layout.legal_actions().to_vec(),
    );
    let mut batch = DenseCfrIteration::new(&dense_config);
    let mut ctx = PostflopEvaluationContext {
        hero_cards,
        indexer: ComboIndexer::new(),
        max_showdown_runouts: config.max_showdown_runouts.max(1),
        equity_cache: HashMap::new(),
    };

    for iteration in 1..=config.cfr_iterations.max(1) {
        fill_public_tree_iteration(tree, layout, &state, &mut ctx, &mut batch);
        batch.validate(&dense_config);
        state.update_all_infosets(
            &batch.action_values,
            &batch.reach_weights,
            &batch.strategy_weights,
            iteration,
        );
    }
    state
}

fn fill_public_tree_iteration(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(0.0);
    batch.strategy_weights.fill(0.0);

    for infoset in 0..layout.infoset_count() {
        let node_index = layout.infoset_node(infoset);
        let PublicNodeKind::Decision {
            state: public_state,
            actions,
        } = &tree.nodes()[node_index].kind
        else {
            unreachable!("infoset nodes are decisions");
        };
        let offset = infoset * layout.max_actions();
        for action_index in 0..layout.action_count(infoset) {
            let child = layout
                .child_for_action(infoset, action_index)
                .expect("legal action must have a child");
            batch.action_values[offset + action_index] = evaluate_action_child(
                tree,
                layout,
                child,
                public_state,
                actions[action_index].action,
                cfr_state,
                ctx,
            );
        }
        if public_state.acting_player == Player::Hero {
            batch.reach_weights[infoset] = 1.0;
            batch.strategy_weights[infoset] = 1.0;
        }
    }
}

fn evaluate_action_child(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    parent_state: &PublicState,
    parent_action: PlayerAction,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision { state, .. } => {
            evaluate_decision(tree, layout, node_index, state, cfr_state, ctx)
        }
        PublicNodeKind::Chance { cards, .. } => {
            let mut sum = 0.0;
            let mut count = 0;
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                if card.deck_mask() & hero_mask(ctx.hero_cards) != 0 {
                    continue;
                }
                sum += evaluate_node(tree, layout, *child, cfr_state, ctx);
                count += 1;
            }
            if count == 0 { 0.0 } else { sum / count as f32 }
        }
        PublicNodeKind::Terminal { kind, board, pot } => match kind {
            TerminalKind::Fold => fold_utility(parent_state, parent_action, *pot),
            TerminalKind::Showdown => showdown_utility(*pot, board, ctx),
        },
    }
}

fn evaluate_node(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    cfr_state: &DenseCfrState,
    ctx: &mut PostflopEvaluationContext,
) -> f32 {
    match &tree.nodes()[node_index].kind {
        PublicNodeKind::Decision { state, .. } => {
            evaluate_decision(tree, layout, node_index, state, cfr_state, ctx)
        }
        PublicNodeKind::Chance { cards, .. } => {
            let mut sum = 0.0;
            let mut count = 0;
            for (card, child) in cards.iter().zip(&tree.nodes()[node_index].children) {
                if card.deck_mask() & hero_mask(ctx.hero_cards) != 0 {
                    continue;
                }
                sum += evaluate_node(tree, layout, *child, cfr_state, ctx);
                count += 1;
            }
            if count == 0 { 0.0 } else { sum / count as f32 }
        }
        PublicNodeKind::Terminal { kind, board, pot } => match kind {
            TerminalKind::Fold => 0.0,
            TerminalKind::Showdown => showdown_utility(*pot, board, ctx),
        },
    }
}

fn evaluate_decision(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    node_index: usize,
    state: &PublicState,
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
            cfr_state,
            ctx,
        ));
    }
    let Some(infoset) = layout.node_infoset(node_index) else {
        return values.iter().sum::<f32>() / values.len().max(1) as f32;
    };
    if state.acting_player == Player::Villain {
        return values.into_iter().fold(f32::INFINITY, f32::min);
    }
    let mut strategy = vec![0.0; layout.max_actions()];
    cfr_state.strategy_for(infoset, &mut strategy);
    values
        .iter()
        .zip(strategy)
        .map(|(value, probability)| value * probability)
        .sum()
}

fn best_average_strategy_action(layout: &PostflopDenseLayout, state: &DenseCfrState) -> usize {
    let mut strategy = vec![0.0; layout.max_actions()];
    state.average_strategy_for(0, &mut strategy);
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

fn fold_utility(parent_state: &PublicState, parent_action: PlayerAction, pot: u32) -> f32 {
    if parent_action != PlayerAction::Fold {
        return 0.0;
    }
    let value = pot as f32 * 0.5;
    if parent_state.acting_player == Player::Hero {
        -value
    } else {
        value
    }
}

fn showdown_utility(pot: u32, board: &Board, ctx: &mut PostflopEvaluationContext) -> f32 {
    let equity = showdown_equity(board, ctx);
    (equity * 2.0 - 1.0) * pot as f32 * 0.5
}

fn showdown_equity(board: &Board, ctx: &mut PostflopEvaluationContext) -> f32 {
    let key = board.deck_mask();
    if let Some(equity) = ctx.equity_cache.get(&key) {
        return *equity;
    }

    let dead = board.deck_mask() | hero_mask(ctx.hero_cards);
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
        for combo in ctx.indexer.combos() {
            if combo.collides_with(board_mask | hero_mask(ctx.hero_cards)) {
                continue;
            }
            equity_sum += heads_up_equity(hero_strength, *combo, &final_board);
            matchup_count += 1.0;
        }
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

fn preflop_hand_score(hand: &rs_poker::core::Hand) -> u8 {
    let cards: Vec<_> = hand.iter().collect();
    if cards.len() < 2 {
        return 0;
    }
    let first = cards[0];
    let second = cards[1];
    let first_value = rs_value_index(first.value);
    let second_value = rs_value_index(second.value);
    let high = first_value.max(second_value);
    let low = first_value.min(second_value);
    let pair_bonus = if first_value == second_value { 12 } else { 0 };
    let suited_bonus = if first.suit == second.suit { 3 } else { 0 };
    let connector_bonus = if high.abs_diff(low) <= 1 { 2 } else { 0 };
    high + low / 2 + pair_bonus + suited_bonus + connector_bonus
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
            indexer: ComboIndexer::new(),
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let mut weak = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
            ],
            indexer: ComboIndexer::new(),
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
                max_depth: 3,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let state = solve_public_tree_cfr(
            &tree,
            &layout,
            [
                PokedrCard::new(PokedrRank::Ace, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::King, PokedrSuit::Diamonds),
            ],
            &PokedrAgentConfig {
                cfr_iterations: 4,
                max_showdown_runouts: 8,
                ..PokedrAgentConfig::default()
            },
        );
        let mut strategy = vec![0.0; layout.max_actions()];
        state.average_strategy_for(0, &mut strategy);
        let legal_sum: f32 = strategy.iter().take(layout.action_count(0)).sum();

        assert!((legal_sum - 1.0).abs() < 1e-5);
        assert!(strategy.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn villain_decision_minimizes_hero_value() {
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
        let cfr_state = DenseCfrState::new_with_legal_actions(
            layout.dense_config(CfrVariant::CfrPlus),
            layout.legal_actions().to_vec(),
        );
        let mut ctx = PostflopEvaluationContext {
            hero_cards: [
                PokedrCard::new(PokedrRank::Nine, PokedrSuit::Clubs),
                PokedrCard::new(PokedrRank::Ten, PokedrSuit::Diamonds),
            ],
            indexer: ComboIndexer::new(),
            max_showdown_runouts: 1,
            equity_cache: HashMap::new(),
        };
        let value = evaluate_decision(
            &tree,
            &layout,
            0,
            &root_public_state(&tree),
            &cfr_state,
            &mut ctx,
        );

        assert!(value <= 0.0);
    }

    fn root_public_state(tree: &SubgameTree) -> PublicState {
        let PublicNodeKind::Decision { state, .. } = &tree.nodes()[0].kind else {
            panic!("root should be a decision");
        };
        state.clone()
    }
}
