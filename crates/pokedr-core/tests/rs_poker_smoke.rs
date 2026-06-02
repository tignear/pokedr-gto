use std::cmp::Ordering;

use pokedr_core::{cards::Card as PokedrCard, hand_eval::evaluate_seven};
use rs_poker::{
    arena::{
        AgentGenerator, CloneAgentGenerator, CloneGameStateGenerator,
        action::AgentAction,
        agent::{Agent, RandomPotControlAgent},
        competition::{HoldemCompetition, StandardSimulationIterator},
        game_state::{GameState, Round},
    },
    core::{Hand, Rank, Rankable},
    holdem::MonteCarloGame,
};

#[test]
fn hand_ordering_matches_rs_poker_on_sampled_showdowns() {
    let cases = [
        ("AsKsQsJsTs9d8c", "AhAdAcKhKd2s3c"),
        ("9s9h9d9c2s3d4h", "AsAhAcKsKd2d3h"),
        ("As9s7s5s2sKdQc", "KhQdJsTc9h2c3d"),
        ("5s4h3d2cAsKdQh", "AhAdKsKc2d3h4s"),
        ("AhAdKsKc2d3h4s", "AhAdAcKsQd3h4s"),
    ];

    for (left, right) in cases {
        let pokedr_order = pokedr_value(left).cmp(&pokedr_value(right));
        let rs_order = rs_poker_value(left).cmp(&rs_poker_value(right));
        assert_eq!(pokedr_order, rs_order, "{left} vs {right}");
    }
}

#[test]
fn rs_poker_ranked_river_policy_smoke_does_not_get_crushed() {
    let deck = deck_labels();
    let mut hero_ev = 0.0;
    let hands = 800;

    for hand_index in 0..hands {
        let offset = (hand_index * 11) % deck.len();
        let sample: Vec<_> = (0..9).map(|step| deck[(offset + step) % deck.len()]).collect();
        let hero = format!("{}{}", sample[0], sample[1]);
        let villain = format!("{}{}", sample[2], sample[3]);
        let board = format!("{}{}{}{}{}", sample[4], sample[5], sample[6], sample[7], sample[8]);

        hero_ev += river_battle_ev(&hero, &villain, &board, hand_index);
    }

    let bb_per_hand = hero_ev / hands as f64 / 10.0;
    assert!(
        bb_per_hand > -0.50,
        "rs_poker-ranked river smoke policy is losing too much: {bb_per_hand:.3} bb/hand"
    );
}

#[test]
fn rs_poker_arena_policy_smoke_does_not_get_crushed_by_pot_control() {
    let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
        Box::new(CloneAgentGenerator::new(EquityPolicyAgent)),
        Box::new(CloneAgentGenerator::new(RandomPotControlAgent::new(vec![
            0.35, 0.25,
        ]))),
    ];
    let game_state = GameState::new_starting(vec![1_000.0, 1_000.0], 10.0, 5.0, 0.0, 0);
    let simulation_gen = StandardSimulationIterator::new(
        agent_gens,
        vec![],
        CloneGameStateGenerator::new(game_state),
    );
    let mut competition = HoldemCompetition::new(simulation_gen);

    competition.run(400).expect("rs_poker arena should run");
    let hero_bb_per_hand = competition.total_change[0] as f64 / competition.num_rounds as f64;
    eprintln!("rs_poker arena smoke hero_bb_per_hand={hero_bb_per_hand:.3}");

    assert!(
        hero_bb_per_hand > -0.75,
        "rs_poker arena smoke policy is losing too much: {hero_bb_per_hand:.3} bb/hand"
    );
}

#[derive(Clone)]
struct EquityPolicyAgent;

impl Agent for EquityPolicyAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let idx = game_state.round_data.to_act_idx;
        let to_call = (game_state.round_data.bet - game_state.round_data.player_bet[idx]).max(0.0);
        let stack = game_state.stacks[idx];
        let equity = estimate_equity(game_state, idx).unwrap_or(0.0);

        if game_state.round == Round::Preflop {
            return preflop_action(game_state, idx, to_call, stack, equity);
        }

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
}

fn preflop_action(
    game_state: &GameState,
    idx: usize,
    to_call: f32,
    stack: f32,
    equity: f32,
) -> AgentAction {
    let mut cards = game_state.hands[idx].iter();
    let Some(first) = cards.next() else {
        return AgentAction::Call;
    };
    let Some(second) = cards.next() else {
        return AgentAction::Call;
    };
    let high = u8::from(first.value).max(u8::from(second.value));
    let low = u8::from(first.value).min(u8::from(second.value));
    let pair = high == low;
    let suited = first.suit == second.suit;
    let playable = pair
        || high >= 10 && low >= 8
        || suited && high >= 9 && low >= 6
        || high >= 12 && low >= 5
        || equity >= 0.53;
    let premium = pair && high >= 8 || high >= 11 && low >= 9 || suited && high >= 10 && low >= 8;

    if to_call <= 0.0 && premium {
        AgentAction::Bet((game_state.big_blind * 3.0).min(stack))
    } else if to_call <= game_state.big_blind || playable {
        AgentAction::Call
    } else {
        AgentAction::Fold
    }
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

fn river_battle_ev(hero: &str, villain: &str, board: &str, hand_index: usize) -> f64 {
    let pot = 10.0;
    let bet = 7.5;
    let hero_rank = rs_poker_value(&format!("{hero}{board}"));
    let villain_rank = rs_poker_value(&format!("{villain}{board}"));
    let hero_category = rank_category(hero_rank);
    let villain_category = rank_category(villain_rank);
    let villain_bets = rank_category(villain_rank) >= 1 || hand_index % 11 == 0;

    if villain_bets {
        if hero_category >= 1 {
            showdown_ev(hero_rank.cmp(&villain_rank), pot * 0.5 + bet)
        } else {
            -pot * 0.5
        }
    } else if hero_category >= 1 {
        if villain_category >= 1 {
            showdown_ev(hero_rank.cmp(&villain_rank), pot * 0.5 + bet)
        } else {
            pot * 0.5
        }
    } else {
        showdown_ev(hero_rank.cmp(&villain_rank), pot * 0.5)
    }
}

fn showdown_ev(ordering: Ordering, win_amount: f64) -> f64 {
    match ordering {
        Ordering::Greater => win_amount,
        Ordering::Equal => 0.0,
        Ordering::Less => -win_amount,
    }
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

fn deck_labels() -> Vec<&'static str> {
    let ranks = [
        "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2",
    ];
    let suits = ["s", "h", "d", "c"];
    ranks
        .iter()
        .flat_map(|rank| {
            suits
                .iter()
                .map(move |suit| match (*rank, *suit) {
                    ("A", "s") => "As",
                    ("A", "h") => "Ah",
                    ("A", "d") => "Ad",
                    ("A", "c") => "Ac",
                    ("K", "s") => "Ks",
                    ("K", "h") => "Kh",
                    ("K", "d") => "Kd",
                    ("K", "c") => "Kc",
                    ("Q", "s") => "Qs",
                    ("Q", "h") => "Qh",
                    ("Q", "d") => "Qd",
                    ("Q", "c") => "Qc",
                    ("J", "s") => "Js",
                    ("J", "h") => "Jh",
                    ("J", "d") => "Jd",
                    ("J", "c") => "Jc",
                    ("T", "s") => "Ts",
                    ("T", "h") => "Th",
                    ("T", "d") => "Td",
                    ("T", "c") => "Tc",
                    ("9", "s") => "9s",
                    ("9", "h") => "9h",
                    ("9", "d") => "9d",
                    ("9", "c") => "9c",
                    ("8", "s") => "8s",
                    ("8", "h") => "8h",
                    ("8", "d") => "8d",
                    ("8", "c") => "8c",
                    ("7", "s") => "7s",
                    ("7", "h") => "7h",
                    ("7", "d") => "7d",
                    ("7", "c") => "7c",
                    ("6", "s") => "6s",
                    ("6", "h") => "6h",
                    ("6", "d") => "6d",
                    ("6", "c") => "6c",
                    ("5", "s") => "5s",
                    ("5", "h") => "5h",
                    ("5", "d") => "5d",
                    ("5", "c") => "5c",
                    ("4", "s") => "4s",
                    ("4", "h") => "4h",
                    ("4", "d") => "4d",
                    ("4", "c") => "4c",
                    ("3", "s") => "3s",
                    ("3", "h") => "3h",
                    ("3", "d") => "3d",
                    ("3", "c") => "3c",
                    ("2", "s") => "2s",
                    ("2", "h") => "2h",
                    ("2", "d") => "2d",
                    ("2", "c") => "2c",
                    _ => unreachable!(),
                })
        })
        .collect()
}

fn pokedr_value(cards: &str) -> u32 {
    let cards: Vec<_> = parse_pokedr_cards(cards).collect();
    evaluate_seven([
        cards[0], cards[1], cards[2], cards[3], cards[4], cards[5], cards[6],
    ])
}

fn rs_poker_value(cards: &str) -> Rank {
    Hand::new_from_str(cards).expect("valid rs_poker hand").rank()
}

fn parse_pokedr_cards(cards: &str) -> impl Iterator<Item = PokedrCard> + '_ {
    cards
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| PokedrCard::new(parse_rank(chunk[0]), parse_suit(chunk[1])))
}

fn parse_rank(rank: u8) -> u8 {
    match rank {
        b'A' | b'a' => 14,
        b'K' | b'k' => 13,
        b'Q' | b'q' => 12,
        b'J' | b'j' => 11,
        b'T' | b't' => 10,
        b'9' => 9,
        b'8' => 8,
        b'7' => 7,
        b'6' => 6,
        b'5' => 5,
        b'4' => 4,
        b'3' => 3,
        b'2' => 2,
        _ => panic!("invalid rank"),
    }
}

fn parse_suit(suit: u8) -> u8 {
    match suit {
        b'c' | b'C' => 0,
        b'd' | b'D' => 1,
        b'h' | b'H' => 2,
        b's' | b'S' => 3,
        _ => panic!("invalid suit"),
    }
}
