pub use pokedr_core::{dense_cfr, postflop, postflop_dense, range};

use pokedr_core::{
    cards::{Board, Card as PokedrCard, Rank as PokedrRank, Suit as PokedrSuit},
    dense_cfr::gpu::{GpuCfrError, GpuDenseCfrBackend},
    dense_cfr::{CfrVariant, DenseCfrIteration},
    postflop::{
        ActionSetConfig, Player, PlayerAction, PublicState, Street, SubgameTree, SubgameTreeConfig,
    },
    postflop_dense::PostflopDenseLayout,
};
use rs_poker::{
    arena::{
        Agent,
        action::AgentAction,
        game_state::{GameState, Round},
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
        let values = solve_fixture_values(&layout, self.config.cfr_iterations);
        let action_index = best_legal_action(&layout, &values);
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

pub fn gpu_backend_mode() -> BackendMode {
    match GpuDenseCfrBackend::new() {
        Ok(_) => BackendMode::Gpu,
        Err(GpuCfrError::NoAdapter)
        | Err(GpuCfrError::RequestDevice(_))
        | Err(GpuCfrError::MapFailed(_)) => BackendMode::CpuFallback,
    }
}

fn solve_fixture_values(layout: &PostflopDenseLayout, iterations: usize) -> Vec<f32> {
    let mut values = vec![0.0; layout.infoset_count() * layout.max_actions()];
    let config = layout.dense_config(CfrVariant::CfrPlus);
    let mut batch = DenseCfrIteration::new(&config);
    for iteration in 1..=iterations.max(1) {
        fill_postflop_fixture_iteration(iteration, layout, &mut batch);
        values.copy_from_slice(&batch.action_values);
    }
    values
}

fn fill_postflop_fixture_iteration(
    iteration: usize,
    layout: &PostflopDenseLayout,
    batch: &mut DenseCfrIteration,
) {
    batch.action_values.fill(0.0);
    batch.reach_weights.fill(1.0);
    batch.strategy_weights.fill(1.0);
    for infoset in 0..layout.infoset_count() {
        let offset = infoset * layout.max_actions();
        for action in 0..layout.action_count(infoset) {
            batch.action_values[offset + action] =
                ((infoset as f32 * 0.17) + (action as f32 * 0.31) + iteration as f32).sin();
        }
    }
}

fn best_legal_action(layout: &PostflopDenseLayout, values: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for action in 0..layout.action_count(0) {
        let value = values[action];
        if value > best_value {
            best = action;
            best_value = value;
        }
    }
    best
}

fn public_state_from_game(game_state: &GameState) -> Option<PublicState> {
    let street = match game_state.round {
        Round::Flop => Street::Flop,
        Round::Turn => Street::Turn,
        Round::River => Street::River,
        _ => return None,
    };
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
        effective_stack: game_state.current_player_stack().max(1.0).round() as u32,
        to_call: amount_to_call(game_state).max(0.0).round() as u32,
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
}
