use std::collections::{HashMap, HashSet};

use crate::cards::{Card, deck};
use crate::hand_class::{HandClass, all_hand_classes};
use crate::hand_eval::evaluate_seven;
use crate::river::Combo;

#[derive(Debug, Clone)]
pub struct PostflopCombo {
    pub combo: Combo,
    pub class: HandClass,
    pub mask: u64,
}

#[derive(Debug, Clone)]
pub struct PostflopEquityReport {
    pub combo: PostflopCombo,
    pub equity: f64,
    pub win: f64,
    pub tie: f64,
    pub total: f64,
}

#[derive(Debug, Clone)]
pub struct PostflopCfrConfig {
    pub board: Vec<Card>,
    pub oop_range: Vec<PostflopCombo>,
    pub ip_range: Vec<PostflopCombo>,
    pub pot: f64,
    pub bet: f64,
    pub iterations: usize,
    pub max_runouts: usize,
}

#[derive(Debug, Clone)]
pub struct PostflopCfrResult {
    pub iterations: usize,
    pub expected_value_oop: f64,
    pub pot: f64,
    pub bet: f64,
    pub board_cards: usize,
    pub oop_combo_count: usize,
    pub ip_combo_count: usize,
    pub strategies: Vec<PostflopComboStrategy>,
}

#[derive(Debug, Clone)]
pub struct PostflopComboStrategy {
    pub board: Vec<Card>,
    pub combo: PostflopCombo,
    pub role: PostflopRole,
    pub node: PostflopNode,
    pub equity: f64,
    pub actions: Vec<PostflopActionFrequency>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostflopRole {
    Oop,
    Ip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostflopNode {
    FacingBet,
    AfterCheck,
}

#[derive(Debug, Clone)]
pub struct PostflopActionFrequency {
    pub action: &'static str,
    pub frequency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeParseError {
    InvalidToken,
}

impl PostflopCombo {
    pub fn label(&self) -> String {
        self.combo.label()
    }
}

pub fn parse_range(spec: &str) -> Result<Vec<HandClass>, RangeParseError> {
    let trimmed = spec.trim();
    if trimmed == "*" || trimmed.eq_ignore_ascii_case("all") {
        return Ok(all_hand_classes());
    }

    let mut classes = Vec::new();
    let mut seen = HashSet::new();
    for token in trimmed
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        for class in parse_range_token(token)? {
            if seen.insert(class) {
                classes.push(class);
            }
        }
    }

    Ok(classes)
}

pub fn postflop_combos(classes: &[HandClass], board: &[Card]) -> Vec<PostflopCombo> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let mut combos = Vec::new();
    let mut seen = HashSet::new();

    for &class in classes {
        for [first, second] in class.combos() {
            let Some(combo) = Combo::new(first, second) else {
                continue;
            };
            let mask = combo.mask();
            if mask & board_mask != 0 || !seen.insert(mask) {
                continue;
            }
            combos.push(PostflopCombo { combo, class, mask });
        }
    }

    combos
}

pub fn postflop_equity_reports(
    board: &[Card],
    hero_range: &[PostflopCombo],
    villain_range: &[PostflopCombo],
    max_runouts: usize,
) -> Vec<PostflopEquityReport> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let mut reports: Vec<_> = hero_range
        .iter()
        .map(|hero| {
            let mut win = 0.0;
            let mut tie = 0.0;
            let mut total = 0.0;

            for villain in villain_range {
                if hero.mask & villain.mask != 0 {
                    continue;
                }

                let used_mask = board_mask | hero.mask | villain.mask;
                for completed_board in sampled_runouts(board, used_mask, max_runouts) {
                    let hero_value = evaluate_seven([
                        hero.combo.first,
                        hero.combo.second,
                        completed_board[0],
                        completed_board[1],
                        completed_board[2],
                        completed_board[3],
                        completed_board[4],
                    ]);
                    let villain_value = evaluate_seven([
                        villain.combo.first,
                        villain.combo.second,
                        completed_board[0],
                        completed_board[1],
                        completed_board[2],
                        completed_board[3],
                        completed_board[4],
                    ]);
                    match hero_value.cmp(&villain_value) {
                        std::cmp::Ordering::Greater => win += 1.0,
                        std::cmp::Ordering::Equal => tie += 1.0,
                        std::cmp::Ordering::Less => {}
                    }
                    total += 1.0;
                }
            }

            PostflopEquityReport {
                combo: hero.clone(),
                equity: if total == 0.0 {
                    0.0
                } else {
                    (win + tie * 0.5) / total
                },
                win,
                tie,
                total,
            }
        })
        .collect();

    reports.sort_by(|left, right| {
        right
            .equity
            .total_cmp(&left.equity)
            .then_with(|| left.combo.label().cmp(&right.combo.label()))
    });
    reports
}

pub fn solve_postflop_check_bet(config: PostflopCfrConfig) -> PostflopCfrResult {
    let deals = postflop_deals(&config.oop_range, &config.ip_range);
    if deals.is_empty() {
        return PostflopCfrResult {
            iterations: config.iterations,
            expected_value_oop: 0.0,
            pot: config.pot,
            bet: config.bet,
            board_cards: config.board.len(),
            oop_combo_count: config.oop_range.len(),
            ip_combo_count: config.ip_range.len(),
            strategies: Vec::new(),
        };
    }
    let mut solver = PostflopCfrTrainer {
        oop_range: &config.oop_range,
        ip_range: &config.ip_range,
        bet: config.bet,
        max_runouts: config.max_runouts,
        nodes: HashMap::new(),
    };
    let mut utility_sum = 0.0;

    for iteration in 0..config.iterations {
        let (oop_index, ip_index) = deals[sampled_index(iteration, deals.len())];
        utility_sum += solver.cfr(
            PostflopHistory::IpDecision,
            &config.board,
            config.pot,
            oop_index,
            ip_index,
            [1.0, 1.0],
        );
    }

    let strategies = solver.combo_strategies();
    PostflopCfrResult {
        iterations: config.iterations,
        expected_value_oop: if config.iterations == 0 {
            0.0
        } else {
            utility_sum / config.iterations as f64
        },
        pot: config.pot,
        bet: config.bet,
        board_cards: config.board.len(),
        oop_combo_count: config.oop_range.len(),
        ip_combo_count: config.ip_range.len(),
        strategies,
    }
}

#[derive(Debug, Clone, Copy)]
enum PostflopHistory {
    IpDecision,
    OopFacingBet,
}

struct PostflopCfrTrainer<'a> {
    oop_range: &'a [PostflopCombo],
    ip_range: &'a [PostflopCombo],
    bet: f64,
    max_runouts: usize,
    nodes: HashMap<String, PostflopCfrNode>,
}

#[derive(Debug, Clone)]
struct PostflopCfrNode {
    board: Vec<Card>,
    role: PostflopRole,
    node: PostflopNode,
    combo: PostflopCombo,
    equity: f64,
    action_labels: [&'static str; 2],
    regret_sum: [f64; 2],
    strategy_sum: [f64; 2],
}

impl PostflopCfrTrainer<'_> {
    fn cfr(
        &mut self,
        history: PostflopHistory,
        board: &[Card],
        pot: f64,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        let player = match history {
            PostflopHistory::IpDecision => 1,
            PostflopHistory::OopFacingBet => 0,
        };
        let key = self.infoset_key(history, board, oop_index, ip_index);
        let strategy = self.strategy_for(&key, history, board, oop_index, ip_index);
        let mut action_values = [0.0; 2];
        let mut node_value = 0.0;

        for action in 0..2 {
            let mut next_reach = reach;
            next_reach[player] *= strategy[action];
            action_values[action] = match history {
                PostflopHistory::IpDecision => {
                    if action == 0 {
                        self.advance_or_showdown(board, pot, oop_index, ip_index, next_reach)
                    } else {
                        self.cfr(
                            PostflopHistory::OopFacingBet,
                            board,
                            pot,
                            oop_index,
                            ip_index,
                            next_reach,
                        )
                    }
                }
                PostflopHistory::OopFacingBet => {
                    if action == 0 {
                        -pot * 0.5
                    } else {
                        self.advance_or_showdown(
                            board,
                            pot + self.bet * 2.0,
                            oop_index,
                            ip_index,
                            next_reach,
                        )
                    }
                }
            };
            node_value += strategy[action] * action_values[action];
        }

        let node = self
            .nodes
            .get_mut(&key)
            .expect("postflop CFR node should exist after strategy lookup");
        let opponent_reach = reach[1 - player];
        for action in 0..2 {
            let regret = if player == 0 {
                action_values[action] - node_value
            } else {
                node_value - action_values[action]
            };
            node.regret_sum[action] += opponent_reach * regret;
            node.strategy_sum[action] += reach[player] * strategy[action];
        }

        node_value
    }

    fn advance_or_showdown(
        &mut self,
        board: &[Card],
        pot: f64,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        if board.len() == 5 {
            return self.showdown_utility(board, oop_index, ip_index, pot * 0.5);
        }

        let cards = sampled_next_cards(
            board,
            self.oop_range[oop_index].mask | self.ip_range[ip_index].mask,
            self.max_runouts,
        );
        if cards.is_empty() {
            return 0.0;
        }

        let probability = 1.0 / cards.len() as f64;
        cards
            .into_iter()
            .map(|card| {
                let mut next_board = board.to_vec();
                next_board.push(card);
                probability
                    * self.cfr(
                        PostflopHistory::IpDecision,
                        &next_board,
                        pot,
                        oop_index,
                        ip_index,
                        reach,
                    )
            })
            .sum()
    }

    fn showdown_utility(
        &self,
        board: &[Card],
        oop_index: usize,
        ip_index: usize,
        win_amount: f64,
    ) -> f64 {
        let oop = &self.oop_range[oop_index];
        let ip = &self.ip_range[ip_index];
        debug_assert_eq!(board.len(), 5);
        let oop_value = evaluate_seven([
            oop.combo.first,
            oop.combo.second,
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
        ]);
        let ip_value = evaluate_seven([
            ip.combo.first,
            ip.combo.second,
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
        ]);
        match oop_value.cmp(&ip_value) {
            std::cmp::Ordering::Greater => win_amount,
            std::cmp::Ordering::Less => -win_amount,
            std::cmp::Ordering::Equal => 0.0,
        }
    }

    fn infoset_key(
        &self,
        history: PostflopHistory,
        board: &[Card],
        oop_index: usize,
        ip_index: usize,
    ) -> String {
        let board = board_label(board);
        match history {
            PostflopHistory::IpDecision => {
                format!("IP:{board}:{}:after-check", self.ip_range[ip_index].label())
            }
            PostflopHistory::OopFacingBet => {
                format!(
                    "OOP:{board}:{}:facing-bet",
                    self.oop_range[oop_index].label()
                )
            }
        }
    }

    fn strategy_for(
        &mut self,
        key: &str,
        history: PostflopHistory,
        board: &[Card],
        oop_index: usize,
        ip_index: usize,
    ) -> [f64; 2] {
        if let Some(node) = self.nodes.get(key) {
            return node.strategy();
        }

        let oop_equity = combo_equity(
            board,
            &self.oop_range[oop_index],
            &self.ip_range[ip_index],
            self.max_runouts,
        );
        let node = match history {
            PostflopHistory::IpDecision => PostflopCfrNode {
                board: board.to_vec(),
                role: PostflopRole::Ip,
                node: PostflopNode::AfterCheck,
                combo: self.ip_range[ip_index].clone(),
                equity: 1.0 - oop_equity,
                action_labels: ["check", "bet"],
                regret_sum: [0.0, 0.0],
                strategy_sum: [0.0, 0.0],
            },
            PostflopHistory::OopFacingBet => PostflopCfrNode {
                board: board.to_vec(),
                role: PostflopRole::Oop,
                node: PostflopNode::FacingBet,
                combo: self.oop_range[oop_index].clone(),
                equity: oop_equity,
                action_labels: ["fold", "call"],
                regret_sum: [0.0, 0.0],
                strategy_sum: [0.0, 0.0],
            },
        };
        let strategy = node.strategy();
        self.nodes.insert(key.to_string(), node);
        strategy
    }

    fn combo_strategies(&self) -> Vec<PostflopComboStrategy> {
        let mut strategies: Vec<_> = self
            .nodes
            .values()
            .map(|node| PostflopComboStrategy {
                board: node.board.clone(),
                combo: node.combo.clone(),
                role: node.role,
                node: node.node,
                equity: node.equity,
                actions: node.average_strategy(),
            })
            .collect();
        strategies.sort_by(|left, right| {
            postflop_role_key(left.role)
                .cmp(&postflop_role_key(right.role))
                .then_with(|| postflop_node_key(left.node).cmp(&postflop_node_key(right.node)))
                .then_with(|| right.equity.total_cmp(&left.equity))
                .then_with(|| left.combo.label().cmp(&right.combo.label()))
        });
        strategies
    }
}

impl PostflopCfrNode {
    fn strategy(&self) -> [f64; 2] {
        let positive = [self.regret_sum[0].max(0.0), self.regret_sum[1].max(0.0)];
        let normalizer = positive[0] + positive[1];

        if normalizer > 0.0 {
            [positive[0] / normalizer, positive[1] / normalizer]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self) -> Vec<PostflopActionFrequency> {
        let normalizer = self.strategy_sum[0] + self.strategy_sum[1];
        let probabilities = if normalizer > 0.0 {
            [
                self.strategy_sum[0] / normalizer,
                self.strategy_sum[1] / normalizer,
            ]
        } else {
            [0.5, 0.5]
        };

        self.action_labels
            .iter()
            .zip(probabilities)
            .map(|(&action, frequency)| PostflopActionFrequency { action, frequency })
            .collect()
    }
}

fn combo_equity(
    board: &[Card],
    hero: &PostflopCombo,
    villain: &PostflopCombo,
    max_runouts: usize,
) -> f64 {
    if hero.mask & villain.mask != 0 {
        return 0.0;
    }

    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let used_mask = board_mask | hero.mask | villain.mask;
    let mut win = 0.0;
    let mut tie = 0.0;
    let mut total = 0.0;
    for completed_board in sampled_runouts(board, used_mask, max_runouts) {
        let hero_value = evaluate_seven([
            hero.combo.first,
            hero.combo.second,
            completed_board[0],
            completed_board[1],
            completed_board[2],
            completed_board[3],
            completed_board[4],
        ]);
        let villain_value = evaluate_seven([
            villain.combo.first,
            villain.combo.second,
            completed_board[0],
            completed_board[1],
            completed_board[2],
            completed_board[3],
            completed_board[4],
        ]);
        match hero_value.cmp(&villain_value) {
            std::cmp::Ordering::Greater => win += 1.0,
            std::cmp::Ordering::Equal => tie += 1.0,
            std::cmp::Ordering::Less => {}
        }
        total += 1.0;
    }

    if total == 0.0 {
        0.0
    } else {
        (win + tie * 0.5) / total
    }
}

fn postflop_deals(oop_range: &[PostflopCombo], ip_range: &[PostflopCombo]) -> Vec<(usize, usize)> {
    let mut deals = Vec::new();
    for oop_index in 0..oop_range.len() {
        for ip_index in 0..ip_range.len() {
            if oop_range[oop_index].mask & ip_range[ip_index].mask == 0 {
                deals.push((oop_index, ip_index));
            }
        }
    }
    deals
}

fn postflop_role_key(role: PostflopRole) -> u8 {
    match role {
        PostflopRole::Ip => 0,
        PostflopRole::Oop => 1,
    }
}

fn postflop_node_key(node: PostflopNode) -> u8 {
    match node {
        PostflopNode::AfterCheck => 0,
        PostflopNode::FacingBet => 1,
    }
}

fn sampled_next_cards(board: &[Card], hand_mask: u64, max_cards: usize) -> Vec<Card> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let available: Vec<_> = deck()
        .into_iter()
        .filter(|card| (board_mask | hand_mask) & card.mask() == 0)
        .collect();

    if max_cards == 0 || available.len() <= max_cards {
        return available;
    }

    let mut sampled = Vec::with_capacity(max_cards);
    let mut seen = HashSet::new();
    let mut iteration = 0;
    while sampled.len() < max_cards {
        let index = sampled_index(iteration, available.len());
        if seen.insert(index) {
            sampled.push(available[index]);
        }
        iteration += 1;
    }
    sampled
}

fn board_label(board: &[Card]) -> String {
    board.iter().map(|&card| card_label(card)).collect()
}

fn card_label(card: Card) -> String {
    let rank = match card.rank() {
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
    };
    let suit = match card.suit() {
        0 => 'c',
        1 => 'd',
        2 => 'h',
        3 => 's',
        _ => '?',
    };
    format!("{rank}{suit}")
}

fn parse_range_token(token: &str) -> Result<Vec<HandClass>, RangeParseError> {
    let plus = token.ends_with('+');
    let token = token.trim_end_matches('+');
    let chars: Vec<_> = token.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return Err(RangeParseError::InvalidToken);
    }

    let high = parse_rank(chars[0]).ok_or(RangeParseError::InvalidToken)?;
    let low = parse_rank(chars[1]).ok_or(RangeParseError::InvalidToken)?;
    let suitedness = chars.get(2).copied();
    if high < low {
        return Err(RangeParseError::InvalidToken);
    }

    if high == low {
        if suitedness.is_some() {
            return Err(RangeParseError::InvalidToken);
        }
        if plus {
            return Ok((low..=14)
                .rev()
                .map(|rank| HandClass::new(rank, rank, false))
                .collect());
        }
        return Ok(vec![HandClass::new(high, low, false)]);
    }

    let suited = match suitedness {
        Some('s') | Some('S') => Some(true),
        Some('o') | Some('O') => Some(false),
        None => None,
        _ => return Err(RangeParseError::InvalidToken),
    };
    let lows: Vec<_> = if plus {
        ((low + 1)..high).chain(std::iter::once(low)).collect()
    } else {
        vec![low]
    };

    let mut classes = Vec::new();
    for low in lows {
        match suited {
            Some(value) => classes.push(HandClass::new(high, low, value)),
            None => {
                classes.push(HandClass::new(high, low, true));
                classes.push(HandClass::new(high, low, false));
            }
        }
    }
    Ok(classes)
}

fn parse_rank(rank: char) -> Option<u8> {
    match rank.to_ascii_uppercase() {
        'A' => Some(14),
        'K' => Some(13),
        'Q' => Some(12),
        'J' => Some(11),
        'T' => Some(10),
        '9' => Some(9),
        '8' => Some(8),
        '7' => Some(7),
        '6' => Some(6),
        '5' => Some(5),
        '4' => Some(4),
        '3' => Some(3),
        '2' => Some(2),
        _ => None,
    }
}

fn sampled_runouts(board: &[Card], used_mask: u64, max_runouts: usize) -> Vec<[Card; 5]> {
    debug_assert!((3..=5).contains(&board.len()));
    if board.len() == 5 {
        return vec![[board[0], board[1], board[2], board[3], board[4]]];
    }

    let available: Vec<_> = deck()
        .into_iter()
        .filter(|card| used_mask & card.mask() == 0)
        .collect();
    let needed = 5 - board.len();
    let mut runouts = Vec::new();

    match needed {
        1 => {
            for &turn_or_river in &available {
                let mut completed = [Card(0); 5];
                completed[..board.len()].copy_from_slice(board);
                completed[board.len()] = turn_or_river;
                runouts.push(completed);
            }
        }
        2 => {
            for first_index in 0..available.len() {
                for second_index in (first_index + 1)..available.len() {
                    let mut completed = [Card(0); 5];
                    completed[..board.len()].copy_from_slice(board);
                    completed[board.len()] = available[first_index];
                    completed[board.len() + 1] = available[second_index];
                    runouts.push(completed);
                }
            }
        }
        _ => unreachable!("postflop board should have 3 to 5 cards"),
    }

    if max_runouts == 0 || runouts.len() <= max_runouts {
        return runouts;
    }

    let mut sampled = Vec::with_capacity(max_runouts);
    for iteration in 0..max_runouts {
        sampled.push(runouts[sampled_index(iteration, runouts.len())]);
    }
    sampled
}

fn sampled_index(iteration: usize, len: usize) -> usize {
    let mut value = iteration as u64 + 0x517c_c1b7_2722_0a95;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) as usize) % len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn parses_exact_and_plus_ranges() {
        assert_eq!(parse_range("AA,AKs,AKo").unwrap().len(), 3);
        assert_eq!(parse_range("TT+").unwrap().len(), 5);
        assert_eq!(parse_range("AJs+").unwrap().len(), 3);
    }

    #[test]
    fn postflop_combos_remove_board_cards() {
        let classes = parse_range("AA,AKs,AKo").unwrap();
        let board = [c(14, 0), c(13, 1), c(2, 2)];
        let combos = postflop_combos(&classes, &board);
        let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());

        assert!(combos.iter().all(|combo| combo.mask & board_mask == 0));
    }

    #[test]
    fn flop_equity_reports_are_sorted() {
        let board = [c(14, 0), c(13, 1), c(2, 2)];
        let oop = postflop_combos(&parse_range("AA,AKs,AKo").unwrap(), &board);
        let ip = postflop_combos(&parse_range("QQ,JJ,AQs").unwrap(), &board);
        let reports = postflop_equity_reports(&board, &oop, &ip, 16);

        assert!(!reports.is_empty());
        assert!(reports[0].equity >= reports[reports.len() - 1].equity);
    }

    #[test]
    fn postflop_cfr_reports_ip_and_oop_strategy_nodes() {
        let board = [c(14, 0), c(13, 1), c(2, 2)];
        let oop = postflop_combos(&parse_range("AA,AKs,AKo").unwrap(), &board);
        let ip = postflop_combos(&parse_range("QQ,JJ,AQs").unwrap(), &board);
        let result = solve_postflop_check_bet(PostflopCfrConfig {
            board: board.to_vec(),
            oop_range: oop,
            ip_range: ip,
            pot: 100.0,
            bet: 75.0,
            iterations: 1_000,
            max_runouts: 8,
        });

        assert!(!result.strategies.is_empty());
        assert!(
            result
                .strategies
                .iter()
                .any(|strategy| strategy.role == PostflopRole::Ip)
        );
        assert!(
            result
                .strategies
                .iter()
                .any(|strategy| strategy.role == PostflopRole::Oop)
        );
    }
}
