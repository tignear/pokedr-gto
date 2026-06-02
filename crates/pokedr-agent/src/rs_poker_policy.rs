use std::collections::HashSet;

use pokedr_core::cards::Card as PokedrCard;
use pokedr_core::hand_class::HandClass;
use pokedr_core::hand_class::all_hand_classes;
use pokedr_core::subgame::{
    ActionAbstraction, ActionKind, BetSize, ChancePolicy, Player, PotState, RangeState,
    SubgameSolveRequest, SubgameSpec,
};
use rs_poker::{
    arena::{
        action::AgentAction,
        agent::Agent,
        game_state::{GameState, Round},
    },
    core::{Hand, Rank, Rankable},
    holdem::MonteCarloGame,
};

#[derive(Debug, Clone)]
pub struct PreflopRanges {
    open: HashSet<HandClass>,
    continue_vs_raise: HashSet<HandClass>,
    value_raise: HashSet<HandClass>,
}

impl PreflopRanges {
    pub fn open_class_count(&self) -> usize {
        self.open.len()
    }

    pub fn opens(&self, class: HandClass) -> bool {
        self.open.contains(&class)
    }

    pub fn continues_vs_raise(&self, class: HandClass) -> bool {
        self.continue_vs_raise.contains(&class)
    }

    pub fn value_raises(&self, class: HandClass) -> bool {
        self.value_raise.contains(&class)
    }
}

impl Default for PreflopRanges {
    fn default() -> Self {
        // Public 6-max references put BTN/SB opening ranges around the low-to-mid
        // 40% band. This is an implementable baseline, not a solved range.
        Self {
            open: hand_classes(&[
                "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
                "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "ATs", "AJs", "AQs", "AKs",
                "K2s", "K3s", "K4s", "K5s", "K6s", "K7s", "K8s", "K9s", "KTs", "KJs", "KQs", "Q5s",
                "Q6s", "Q7s", "Q8s", "Q9s", "QTs", "QJs", "J7s", "J8s", "J9s", "JTs", "T7s", "T8s",
                "T9s", "97s", "98s", "87s", "76s", "65s", "54s", "A2o", "A3o", "A4o", "A5o", "A6o",
                "A7o", "A8o", "A9o", "ATo", "AJo", "AQo", "AKo", "K8o", "K9o", "KTo", "KJo", "KQo",
                "Q9o", "QTo", "QJo", "J9o", "JTo", "T9o",
            ]),
            continue_vs_raise: hand_classes(&[
                "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA", "A8s", "A9s", "ATs",
                "AJs", "AQs", "AKs", "KTs", "KJs", "KQs", "QJs", "JTs", "ATo", "AJo", "AQo", "AKo",
                "KQo", "A5s", "A4s",
            ]),
            value_raise: hand_classes(&[
                "TT", "JJ", "QQ", "KK", "AA", "AQs", "AKs", "AKo", "A5s", "A4s",
            ]),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EquityPolicyAgent {
    preflop: PreflopRanges,
}

impl EquityPolicyAgent {
    pub fn new(preflop: PreflopRanges) -> Self {
        Self { preflop }
    }
}

#[derive(Debug, Clone)]
pub struct CfrPolicyAgent {
    fallback: EquityPolicyAgent,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
}

impl Default for CfrPolicyAgent {
    fn default() -> Self {
        Self {
            fallback: EquityPolicyAgent::default(),
            iterations: 1_000,
            range_classes: 48,
            runouts: 8,
        }
    }
}

impl CfrPolicyAgent {
    pub fn new(iterations: usize, range_classes: usize, runouts: usize) -> Self {
        Self {
            fallback: EquityPolicyAgent::default(),
            iterations,
            range_classes,
            runouts,
        }
    }
}

impl Agent for CfrPolicyAgent {
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        if game_state.round == Round::Preflop || game_state.board.len() < 3 {
            return self.fallback.act(id, game_state);
        }

        let idx = game_state.round_data.to_act_idx;
        cfr_postflop_action(
            game_state,
            idx,
            self.iterations,
            self.range_classes,
            self.runouts,
        )
        .unwrap_or_else(|| self.fallback.act(id, game_state))
    }
}

impl Agent for EquityPolicyAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let idx = game_state.round_data.to_act_idx;
        let to_call = amount_to_call(game_state, idx);
        let stack = game_state.stacks[idx];
        let equity = estimate_equity(game_state, idx).unwrap_or(0.0);

        if game_state.round == Round::Preflop {
            return self.preflop_action(game_state, idx, to_call, stack, equity);
        }

        postflop_action(game_state, idx, to_call, stack, equity)
    }
}

impl EquityPolicyAgent {
    fn preflop_action(
        &self,
        game_state: &GameState,
        idx: usize,
        to_call: f32,
        stack: f32,
        equity: f32,
    ) -> AgentAction {
        let Some(class) = hand_class(game_state, idx) else {
            return AgentAction::Call;
        };

        if to_call <= 0.0 {
            if self.preflop.value_raise.contains(&class) {
                AgentAction::Bet((game_state.big_blind * 3.0).min(stack))
            } else if self.preflop.open.contains(&class) || equity >= 0.53 {
                AgentAction::Call
            } else {
                AgentAction::Fold
            }
        } else if self.preflop.value_raise.contains(&class) && to_call <= game_state.big_blind * 4.0
        {
            AgentAction::Bet((game_state.round_data.bet + game_state.big_blind * 3.0).min(stack))
        } else if self.preflop.continue_vs_raise.contains(&class)
            || equity * (game_state.total_pot + to_call) >= to_call
        {
            AgentAction::Call
        } else {
            AgentAction::Fold
        }
    }
}

fn postflop_action(
    game_state: &GameState,
    idx: usize,
    to_call: f32,
    stack: f32,
    equity: f32,
) -> AgentAction {
    let strength = made_hand_strength(game_state, idx);
    if to_call <= 0.0 {
        if strength >= 2 || equity >= 0.62 {
            let size = (game_state.total_pot * 0.75).max(game_state.big_blind * 2.0);
            AgentAction::Bet((game_state.round_data.bet + size).min(stack))
        } else {
            AgentAction::Call
        }
    } else if strength >= 2 || equity * (game_state.total_pot + to_call) >= to_call {
        AgentAction::Call
    } else {
        AgentAction::Fold
    }
}

fn cfr_postflop_action(
    game_state: &GameState,
    idx: usize,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
) -> Option<AgentAction> {
    let hero_class = hand_class(game_state, idx)?;
    let board = pokedr_board(game_state)?;
    let to_call = amount_to_call(game_state, idx);
    let stack = game_state.stacks[idx];
    let (oop_idx, ip_idx) = heads_up_postflop_positions(game_state)?;
    let root_player = if idx == oop_idx {
        Player::Oop
    } else if idx == ip_idx {
        Player::Ip
    } else {
        return None;
    };
    let hero_mask = pokedr_hand_mask(game_state, idx)?;

    let oop_range = mc_range_prior(
        game_state,
        oop_idx,
        range_classes,
        (idx == oop_idx).then_some(hero_class),
        if idx == oop_idx { 0 } else { hero_mask },
    );
    let ip_range = mc_range_prior(
        game_state,
        ip_idx,
        range_classes,
        (idx == ip_idx).then_some(hero_class),
        if idx == ip_idx { 0 } else { hero_mask },
    );
    if oop_range.is_empty() || ip_range.is_empty() {
        return None;
    }

    let pot = PotState {
        pot: game_state.total_pot as f64,
        stacks: [
            game_state.stacks[oop_idx] as f64,
            game_state.stacks[ip_idx] as f64,
        ],
        committed: [
            game_state.round_data.player_bet[oop_idx] as f64,
            game_state.round_data.player_bet[ip_idx] as f64,
        ],
        current_bet: game_state.round_data.bet as f64,
        min_raise: game_state.round_data.min_raise as f64,
    };
    let spec = SubgameSpec::postflop(
        board.clone(),
        pot,
        RangeState::new(oop_range, ip_range),
        cfr_action_abstraction(game_state),
        ChancePolicy::Sample(runouts.max(1)),
    )
    .ok()?
    .with_root_player(root_player);
    let result = spec
        .solve_cfr_with_request(SubgameSolveRequest {
            iterations,
            focused_oop_combo_mask: (root_player == Player::Oop).then_some(hero_mask),
            focused_ip_combo_mask: (root_player == Player::Ip).then_some(hero_mask),
            focused_sampling_rate: 0.5,
        })
        .ok()?;
    let root = result.strategies.iter().find(|strategy| {
        strategy.node == result.root
            && strategy.player == root_player
            && strategy.combo.mask == hero_mask
    })?;
    let best = root
        .actions
        .iter()
        .max_by(|left, right| left.frequency.total_cmp(&right.frequency))?;

    Some(match best.action {
        ActionKind::Fold => AgentAction::Fold,
        ActionKind::Check | ActionKind::Call => {
            if to_call <= 0.0 {
                AgentAction::Call
            } else {
                AgentAction::Bet(game_state.round_data.bet)
            }
        }
        ActionKind::Bet(amount) | ActionKind::Raise(amount) | ActionKind::AllIn(amount) => {
            let capped = amount
                .max(game_state.round_data.bet as f64)
                .min((game_state.round_data.player_bet[idx] + stack) as f64);
            AgentAction::Bet(capped as f32)
        }
    })
}

fn cfr_action_abstraction(game_state: &GameState) -> ActionAbstraction {
    let pot = game_state.total_pot.max(game_state.big_blind);
    ActionAbstraction {
        bet_sizes: vec![
            BetSize::Chips((pot * 0.5) as f64),
            BetSize::Chips((pot * 0.75) as f64),
        ],
        raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
        reraise_sizes: vec![BetSize::CurrentBetMultiple(2.2)],
        allow_all_in: true,
        max_raises: 2,
    }
}

fn amount_to_call(game_state: &GameState, idx: usize) -> f32 {
    (game_state.round_data.bet - game_state.round_data.player_bet[idx]).max(0.0)
}

fn estimate_equity(game_state: &GameState, idx: usize) -> Option<f32> {
    let mut default_hand = Hand::new();
    default_hand.extend(game_state.board.iter().copied());
    let hands = game_state
        .hands
        .iter()
        .enumerate()
        .map(
            |(hand_idx, hand)| {
                if hand_idx == idx { *hand } else { default_hand }
            },
        )
        .collect();
    let mut monte = MonteCarloGame::new(hands).ok()?;
    monte.estimate_equity(400).get(idx).copied()
}

fn mc_range_prior(
    game_state: &GameState,
    idx: usize,
    max_classes: usize,
    required: Option<HandClass>,
    extra_dead_mask: u64,
) -> Vec<HandClass> {
    let board_mask = pokedr_board(game_state)
        .map(|board| board.iter().fold(0_u64, |mask, card| mask | card.mask()))
        .unwrap_or(0);
    let dead_mask = board_mask | extra_dead_mask;
    let opponent_idx = heads_up_opponent(game_state, idx).unwrap_or(idx);
    let mut scored: Vec<_> = all_hand_classes()
        .into_iter()
        .filter_map(|class| {
            let combo = class
                .combos()
                .into_iter()
                .find(|cards| cards.iter().all(|card| card.mask() & dead_mask == 0))?;
            let equity = estimate_class_equity(game_state, idx, opponent_idx, combo)?;
            Some((class, equity))
        })
        .collect();

    scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    let mut classes = stratified_classes(&scored, max_classes.max(1));
    if let Some(required) = required {
        if !classes.contains(&required) {
            classes.push(required);
        }
    }
    classes
}

fn stratified_classes(scored: &[(HandClass, f32)], max_classes: usize) -> Vec<HandClass> {
    if scored.is_empty() {
        return Vec::new();
    }
    if scored.len() <= max_classes {
        return scored.iter().map(|(class, _)| *class).collect();
    }

    let mut classes = Vec::with_capacity(max_classes);
    let mut seen = HashSet::new();
    for slot in 0..max_classes {
        let index = if max_classes == 1 {
            0
        } else {
            slot * (scored.len() - 1) / (max_classes - 1)
        };
        let class = scored[index].0;
        if seen.insert(class) {
            classes.push(class);
        }
    }
    classes
}

fn estimate_class_equity(
    game_state: &GameState,
    idx: usize,
    opponent_idx: usize,
    combo: [PokedrCard; 2],
) -> Option<f32> {
    let mut hero_hand = Hand::new();
    hero_hand.insert(rs_card(combo[0]));
    hero_hand.insert(rs_card(combo[1]));
    hero_hand.extend(game_state.board.iter().copied());

    let mut opponent_default = Hand::new();
    opponent_default.extend(game_state.board.iter().copied());

    let mut hands = game_state.hands.clone();
    hands[idx] = hero_hand;
    hands[opponent_idx] = opponent_default;
    let mut monte = MonteCarloGame::new(hands).ok()?;
    monte.estimate_equity(32).get(idx).copied()
}

fn made_hand_strength(game_state: &GameState, idx: usize) -> u8 {
    if game_state.board.len() + game_state.hands[idx].count() < 5 {
        return 0;
    }
    rank_category(combined_hand_rank(game_state, idx))
}

fn heads_up_opponent(game_state: &GameState, idx: usize) -> Option<usize> {
    (0..game_state.num_players).find(|&other| {
        other != idx && (game_state.player_active.get(other) || game_state.player_all_in.get(other))
    })
}

fn heads_up_postflop_positions(game_state: &GameState) -> Option<(usize, usize)> {
    if game_state.num_players != 2 {
        return None;
    }
    let ip_idx = game_state.dealer_idx;
    let oop_idx = (0..game_state.num_players).find(|&idx| idx != ip_idx)?;
    if game_state.player_active.get(ip_idx) || game_state.player_all_in.get(ip_idx) {
        Some((oop_idx, ip_idx))
    } else {
        None
    }
}

fn pokedr_hand_mask(game_state: &GameState, idx: usize) -> Option<u64> {
    let mut mask = 0_u64;
    for card in game_state.hands[idx].iter() {
        mask |= pokedr_card(card)?.mask();
    }
    Some(mask)
}

fn pokedr_board(game_state: &GameState) -> Option<Vec<PokedrCard>> {
    game_state
        .board
        .iter()
        .map(|card| pokedr_card(*card))
        .collect()
}

fn pokedr_card(card: rs_poker::core::Card) -> Option<PokedrCard> {
    Some(PokedrCard::new(
        u8::from(card.value) + 2,
        match card.suit {
            rs_poker::core::Suit::Club => 0,
            rs_poker::core::Suit::Diamond => 1,
            rs_poker::core::Suit::Heart => 2,
            rs_poker::core::Suit::Spade => 3,
        },
    ))
}

fn rs_card(card: PokedrCard) -> rs_poker::core::Card {
    rs_poker::core::Card::new(
        rs_poker::core::Value::from(card.rank() - 2),
        match card.suit() {
            0 => rs_poker::core::Suit::Club,
            1 => rs_poker::core::Suit::Diamond,
            2 => rs_poker::core::Suit::Heart,
            _ => rs_poker::core::Suit::Spade,
        },
    )
}

fn combined_hand_rank(game_state: &GameState, idx: usize) -> Rank {
    let mut hand = Hand::new();
    hand.extend(game_state.board.iter().copied());
    hand.extend(game_state.hands[idx].iter());
    hand.rank()
}

fn rank_category(rank: Rank) -> u8 {
    match rank {
        Rank::HighCard(_) => 0,
        Rank::OnePair(_) => 1,
        Rank::TwoPair(_) => 2,
        Rank::ThreeOfAKind(_) => 3,
        Rank::Straight(_) => 4,
        Rank::Flush(_) => 5,
        Rank::FullHouse(_) => 6,
        Rank::FourOfAKind(_) => 7,
        Rank::StraightFlush(_) => 8,
    }
}

fn hand_class(game_state: &GameState, idx: usize) -> Option<HandClass> {
    let mut cards = game_state.hands[idx].iter();
    let first = cards.next()?;
    let second = cards.next()?;
    let first_rank = u8::from(first.value) + 2;
    let second_rank = u8::from(second.value) + 2;
    let high = first_rank.max(second_rank);
    let low = first_rank.min(second_rank);
    Some(HandClass::new(
        high,
        low,
        first.suit == second.suit && high != low,
    ))
}

fn hand_classes(labels: &[&str]) -> HashSet<HandClass> {
    labels
        .iter()
        .map(|label| hand_class_from_label(label))
        .collect()
}

fn hand_class_from_label(label: &str) -> HandClass {
    let bytes = label.as_bytes();
    let high = rank_from_byte(bytes[0]);
    let low = rank_from_byte(bytes[1]);
    let suited = bytes.get(2).copied() == Some(b's');
    HandClass::new(high, low, suited && high != low)
}

fn rank_from_byte(rank: u8) -> u8 {
    match rank {
        b'A' => 14,
        b'K' => 13,
        b'Q' => 12,
        b'J' => 11,
        b'T' => 10,
        b'9' => 9,
        b'8' => 8,
        b'7' => 7,
        b'6' => 6,
        b'5' => 5,
        b'4' => 4,
        b'3' => 3,
        b'2' => 2,
        _ => panic!("invalid hand class rank"),
    }
}
