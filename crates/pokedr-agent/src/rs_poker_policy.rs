use std::collections::{HashMap, HashSet};

use pokedr_core::{
    cards::Card as PokedrCard,
    hand_class::HandClass,
    postflop::{
        PostflopCfrConfig, PostflopCfrResult, PostflopNode, PostflopRole, parse_range,
        postflop_combos, solve_postflop_check_bet,
    },
};
use rs_poker::{
    arena::{
        action::AgentAction,
        agent::Agent,
        game_state::{GameState, Round},
    },
    core::{Card, Hand, Rank, Rankable, Suit, Value},
    holdem::MonteCarloGame,
};

#[derive(Debug, Clone)]
pub struct BaselinePreflopRanges {
    open: HashSet<HandClass>,
    continue_vs_raise: HashSet<HandClass>,
    value_raise: HashSet<HandClass>,
}

impl BaselinePreflopRanges {
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

impl Default for BaselinePreflopRanges {
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
    preflop: BaselinePreflopRanges,
    postflop_cfr: PostflopCfrPolicy,
    current_hand_id: Option<u128>,
    cfr_action_cache: HashMap<String, AgentAction>,
    cfr_result_cache: HashMap<String, PostflopCfrResult>,
}

impl EquityPolicyAgent {
    pub fn new(preflop: BaselinePreflopRanges) -> Self {
        Self {
            preflop,
            postflop_cfr: PostflopCfrPolicy::default(),
            current_hand_id: None,
            cfr_action_cache: HashMap::new(),
            cfr_result_cache: HashMap::new(),
        }
    }

    pub fn with_postflop_cfr(mut self, postflop_cfr: PostflopCfrPolicy) -> Self {
        self.postflop_cfr = postflop_cfr;
        self
    }
}

impl Agent for EquityPolicyAgent {
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        if self.current_hand_id != Some(id) {
            self.current_hand_id = Some(id);
            self.cfr_action_cache.clear();
            self.cfr_result_cache.clear();
        }

        let idx = game_state.round_data.to_act_idx;
        let to_call = amount_to_call(game_state, idx);
        let stack = game_state.stacks[idx];
        let equity = estimate_equity(game_state, idx).unwrap_or(0.0);

        if game_state.round == Round::Preflop {
            return self.preflop_action(game_state, idx, to_call, stack, equity);
        }

        self.postflop_action(game_state, idx, to_call, stack, equity)
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

        let pot_odds_ok = equity * (game_state.total_pot + to_call) >= to_call;
        let clear_open = equity >= 0.53 || self.preflop.opens(class);
        let clear_continue =
            equity >= 0.49 || pot_odds_ok || self.preflop.continues_vs_raise(class);
        let clear_raise = equity >= 0.63 || self.preflop.value_raises(class);

        if to_call <= 0.0 {
            if clear_raise {
                AgentAction::Bet((game_state.big_blind * 3.0).min(stack))
            } else if clear_open {
                AgentAction::Call
            } else {
                AgentAction::Fold
            }
        } else if clear_raise && to_call <= game_state.big_blind * 4.0 {
            AgentAction::Bet((game_state.round_data.bet + game_state.big_blind * 3.0).min(stack))
        } else if clear_continue {
            AgentAction::Call
        } else {
            AgentAction::Fold
        }
    }

    fn postflop_action(
        &mut self,
        game_state: &GameState,
        idx: usize,
        to_call: f32,
        stack: f32,
        equity: f32,
    ) -> AgentAction {
        if let Some(action) =
            self.postflop_cfr_action(game_state, idx, to_call, stack, equity)
        {
            return action;
        }
        monte_carlo_postflop_action(game_state, idx, to_call, stack, equity)
    }

    fn postflop_cfr_action(
        &mut self,
        game_state: &GameState,
        idx: usize,
        to_call: f32,
        stack: f32,
        equity: f32,
    ) -> Option<AgentAction> {
        if !self.postflop_cfr.enabled || !(3..=5).contains(&game_state.board.len()) {
            return None;
        }

        let board = convert_board(&game_state.board)?;
        let hero_mask = hand_mask(&game_state.hands[idx])?;
        let node = if to_call > 0.0 {
            PostflopNode::FacingBet
        } else {
            PostflopNode::AfterCheck
        };
        let role = if to_call > 0.0 {
            PostflopRole::Oop
        } else {
            PostflopRole::Ip
        };
        let action_cache_key = format!(
            "{}:{}:{}:{}:{:.1}:{:.1}:{:.1}",
            board_label(&board),
            postflop_role_label(role),
            postflop_node_label(node),
            hero_mask,
            game_state.total_pot,
            game_state.round_data.bet,
            to_call
        );
        if let Some(action) = self.cfr_action_cache.get(&action_cache_key) {
            return Some(action.clone());
        }

        let strategy = self.find_or_solve_strategy(game_state, &board, role, node, hero_mask)?;
        let action = choose_cfr_action(
            &strategy,
            game_state.round_data.bet,
            game_state.big_blind,
            stack,
            equity,
        );
        self.cfr_action_cache
            .insert(action_cache_key, action.clone());
        Some(action)
    }

    fn find_or_solve_strategy(
        &mut self,
        game_state: &GameState,
        board: &[PokedrCard],
        role: PostflopRole,
        node: PostflopNode,
        hero_mask: u64,
    ) -> Option<pokedr_core::postflop::PostflopComboStrategy> {
        if board.len() > 3 {
            let flop_board = board.get(..3)?.to_vec();
            let flop_key = board_label(&flop_board);
            if !self.cfr_result_cache.contains_key(&flop_key) {
                let result = self.solve_postflop_cfr_root(game_state, &flop_board)?;
                self.cfr_result_cache.insert(flop_key.clone(), result);
            }
            if let Some(strategy) =
                lookup_strategy(self.cfr_result_cache.get(&flop_key)?, board, role, node, hero_mask)
            {
                return Some(strategy);
            }
        }

        let root_key = board_label(board);
        if !self.cfr_result_cache.contains_key(&root_key) {
            let result = self.solve_postflop_cfr_root(game_state, board)?;
            self.cfr_result_cache.insert(root_key.clone(), result);
        }
        lookup_strategy(
            self.cfr_result_cache.get(&root_key)?,
            board,
            role,
            node,
            hero_mask,
        )
    }

    fn solve_postflop_cfr_root(
        &self,
        game_state: &GameState,
        board: &[PokedrCard],
    ) -> Option<PostflopCfrResult> {
        let range_classes = parse_range(&self.postflop_cfr.range).ok()?;
        let oop_range = postflop_combos(&range_classes, board);
        let ip_range = postflop_combos(&range_classes, board);
        if oop_range.is_empty() || ip_range.is_empty() {
            return None;
        }
        let bet = game_state
            .round_data
            .bet
            .max((game_state.total_pot * self.postflop_cfr.bet_fraction).max(game_state.big_blind))
            as f64;
        let raise = bet * self.postflop_cfr.raise_multiplier;
        let result = solve_postflop_check_bet(PostflopCfrConfig {
            board: board.to_vec(),
            oop_range,
            ip_range,
            pot: game_state.total_pot as f64,
            bet,
            raise,
            reraise: raise * self.postflop_cfr.reraise_multiplier,
            iterations: self.postflop_cfr.iterations,
            max_runouts: self.postflop_cfr.max_runouts,
        });
        Some(result)
    }
}

fn lookup_strategy(
    result: &PostflopCfrResult,
    board: &[PokedrCard],
    role: PostflopRole,
    node: PostflopNode,
    hero_mask: u64,
) -> Option<pokedr_core::postflop::PostflopComboStrategy> {
    result
        .strategies
        .iter()
        .find(|strategy| {
            strategy.board == board
                && strategy.role == role
                && strategy.node == node
                && strategy.combo.mask == hero_mask
        })
        .cloned()
}

#[derive(Debug, Clone)]
pub struct PostflopCfrPolicy {
    pub enabled: bool,
    pub range: String,
    pub iterations: usize,
    pub max_runouts: usize,
    pub bet_fraction: f32,
    pub raise_multiplier: f64,
    pub reraise_multiplier: f64,
}

impl Default for PostflopCfrPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            range: "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,87s,76s,65s,54s,A2o+,K8o+,Q9o+,J9o+,T9o".to_string(),
            iterations: 300,
            max_runouts: 4,
            bet_fraction: 0.75,
            raise_multiplier: 3.0,
            reraise_multiplier: 2.5,
        }
    }
}

fn monte_carlo_postflop_action(
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

fn choose_cfr_action(
    strategy: &pokedr_core::postflop::PostflopComboStrategy,
    current_bet: f32,
    big_blind: f32,
    stack: f32,
    equity: f32,
) -> AgentAction {
    let best = strategy
        .actions
        .iter()
        .max_by(|left, right| left.frequency.total_cmp(&right.frequency))
        .map(|action| action.action)
        .unwrap_or("call");

    match best {
        "fold" => AgentAction::Fold,
        "bet" => AgentAction::Bet(stack.min(big_blind * 3.0)),
        "raise" | "reraise" if equity >= 0.58 => {
            AgentAction::Bet(stack.min(current_bet + big_blind * 3.0))
        }
        "raise" | "reraise" => AgentAction::Call,
        "check" | "call" => AgentAction::Call,
        _ => AgentAction::Call,
    }
}

fn amount_to_call(game_state: &GameState, idx: usize) -> f32 {
    (game_state.round_data.bet - game_state.round_data.player_bet[idx]).max(0.0)
}

fn convert_board(board: &[Card]) -> Option<Vec<PokedrCard>> {
    board.iter().copied().map(convert_card).collect()
}

fn hand_mask(hand: &Hand) -> Option<u64> {
    let mut cards = hand.iter().map(convert_card);
    let first = cards.next()??;
    let second = cards.next()??;
    Some(first.mask() | second.mask())
}

fn convert_card(card: Card) -> Option<PokedrCard> {
    Some(PokedrCard::new(convert_rank(card.value), convert_suit(card.suit)))
}

fn convert_rank(rank: Value) -> u8 {
    u8::from(rank) + 2
}

fn convert_suit(suit: Suit) -> u8 {
    match suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    }
}

fn board_label(board: &[PokedrCard]) -> String {
    board
        .iter()
        .map(|card| format!("{}{}", rank_label(card.rank()), suit_label(card.suit())))
        .collect()
}

fn rank_label(rank: u8) -> char {
    match rank {
        14 => 'A',
        13 => 'K',
        12 => 'Q',
        11 => 'J',
        10 => 'T',
        9 => '9',
        8 => '8',
        7 => '7',
        6 => '6',
        5 => '5',
        4 => '4',
        3 => '3',
        2 => '2',
        _ => '?',
    }
}

fn suit_label(suit: u8) -> char {
    match suit {
        0 => 'c',
        1 => 'd',
        2 => 'h',
        3 => 's',
        _ => '?',
    }
}

fn postflop_role_label(role: PostflopRole) -> &'static str {
    match role {
        PostflopRole::Oop => "oop",
        PostflopRole::Ip => "ip",
    }
}

fn postflop_node_label(node: PostflopNode) -> &'static str {
    match node {
        PostflopNode::AfterCheck => "after-check",
        PostflopNode::FacingBet => "facing-bet",
        PostflopNode::FacingRaise => "facing-raise",
        PostflopNode::FacingReraise => "facing-reraise",
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
