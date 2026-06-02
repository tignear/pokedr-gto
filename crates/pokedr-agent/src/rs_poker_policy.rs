use std::collections::HashSet;

use pokedr_core::hand_class::HandClass;
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
                "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "ATs", "AJs",
                "AQs", "AKs", "K2s", "K3s", "K4s", "K5s", "K6s", "K7s", "K8s", "K9s",
                "KTs", "KJs", "KQs", "Q5s", "Q6s", "Q7s", "Q8s", "Q9s", "QTs", "QJs",
                "J7s", "J8s", "J9s", "JTs", "T7s", "T8s", "T9s", "97s", "98s", "87s",
                "76s", "65s", "54s", "A2o", "A3o", "A4o", "A5o", "A6o", "A7o", "A8o",
                "A9o", "ATo", "AJo", "AQo", "AKo", "K8o", "K9o", "KTo", "KJo", "KQo",
                "Q9o", "QTo", "QJo", "J9o", "JTo", "T9o",
            ]),
            continue_vs_raise: hand_classes(&[
                "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA", "A8s", "A9s",
                "ATs", "AJs", "AQs", "AKs", "KTs", "KJs", "KQs", "QJs", "JTs", "ATo",
                "AJo", "AQo", "AKo", "KQo", "A5s", "A4s",
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
        .map(|(hand_idx, hand)| {
            if hand_idx == idx {
                *hand
            } else {
                default_hand
            }
        })
        .collect();
    let mut monte = MonteCarloGame::new(hands).ok()?;
    monte.estimate_equity(400).get(idx).copied()
}

fn made_hand_strength(game_state: &GameState, idx: usize) -> u8 {
    if game_state.board.len() + game_state.hands[idx].count() < 5 {
        return 0;
    }
    rank_category(combined_hand_rank(game_state, idx))
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
    Some(HandClass::new(high, low, first.suit == second.suit && high != low))
}

fn hand_classes(labels: &[&str]) -> HashSet<HandClass> {
    labels.iter().map(|label| hand_class_from_label(label)).collect()
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
