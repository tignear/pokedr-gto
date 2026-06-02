use crate::cards::{Card, deck};
use crate::hand_eval::{HandValue, evaluate_seven};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Combo {
    pub first: Card,
    pub second: Card,
}

#[derive(Debug, Clone)]
pub struct RiverCombo {
    pub combo: Combo,
    pub mask: u64,
    pub value: HandValue,
}

#[derive(Debug, Clone)]
pub struct RiverBlockerReport {
    pub hero: RiverCombo,
    pub blocked_villain_combos: usize,
    pub blocked_top_combos: usize,
    pub total_villain_combos: usize,
    pub top_villain_combos: usize,
}

#[derive(Debug, Clone)]
pub struct RiverCfrConfig {
    pub board: [Card; 5],
    pub pot: f64,
    pub bet: f64,
    pub iterations: usize,
    pub top_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct RiverCfrResult {
    pub iterations: usize,
    pub expected_value_oop: f64,
    pub pot: f64,
    pub bet: f64,
    pub combo_count: usize,
    pub strategies: Vec<RiverComboStrategy>,
}

#[derive(Debug, Clone)]
pub struct RiverComboStrategy {
    pub combo: RiverCombo,
    pub role: RiverRole,
    pub node: RiverNode,
    pub actions: Vec<RiverActionFrequency>,
    pub blocked_villain_combos: usize,
    pub blocked_top_combos: usize,
    pub total_villain_combos: usize,
    pub top_villain_combos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiverRole {
    Oop,
    Ip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiverNode {
    FacingBet,
    AfterCheck,
}

#[derive(Debug, Clone)]
pub struct RiverActionFrequency {
    pub action: &'static str,
    pub frequency: f64,
}

impl Combo {
    pub fn new(first: Card, second: Card) -> Option<Self> {
        if first == second {
            return None;
        }

        Some(Self { first, second })
    }

    pub fn mask(self) -> u64 {
        self.first.mask() | self.second.mask()
    }

    pub fn label(self) -> String {
        format!("{}{}", card_label(self.first), card_label(self.second))
    }
}

impl RiverCombo {
    pub fn label(&self) -> String {
        self.combo.label()
    }
}

pub fn river_combos(board: [Card; 5]) -> Vec<RiverCombo> {
    let board_mask = board_mask(board);
    let cards = deck();
    let mut combos = Vec::with_capacity(1081);

    for first_index in 0..cards.len() {
        let first = cards[first_index];
        if board_mask & first.mask() != 0 {
            continue;
        }

        for &second in cards.iter().skip(first_index + 1) {
            let Some(combo) = Combo::new(first, second) else {
                continue;
            };
            let mask = combo.mask();
            if board_mask & mask != 0 {
                continue;
            }

            let value = evaluate_seven([
                combo.first,
                combo.second,
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ]);
            combos.push(RiverCombo { combo, mask, value });
        }
    }

    combos.sort_by(|left, right| {
        right
            .value
            .cmp(&left.value)
            .then_with(|| left.combo.first.0.cmp(&right.combo.first.0))
            .then_with(|| left.combo.second.0.cmp(&right.combo.second.0))
    });
    combos
}

pub fn river_blocker_reports(
    board: [Card; 5],
    hero_combos: &[RiverCombo],
    villain_combos: &[RiverCombo],
    top_fraction: f64,
) -> Vec<RiverBlockerReport> {
    let top_count = ((villain_combos.len() as f64) * top_fraction.clamp(0.0, 1.0))
        .ceil()
        .max(1.0) as usize;
    let top_count = top_count.min(villain_combos.len());
    let top_villain = &villain_combos[..top_count];
    let board_mask = board_mask(board);

    hero_combos
        .iter()
        .filter(|hero| hero.mask & board_mask == 0)
        .map(|hero| {
            let blocked_villain_combos = villain_combos
                .iter()
                .filter(|villain| hero.mask & villain.mask != 0)
                .count();
            let blocked_top_combos = top_villain
                .iter()
                .filter(|villain| hero.mask & villain.mask != 0)
                .count();

            RiverBlockerReport {
                hero: hero.clone(),
                blocked_villain_combos,
                blocked_top_combos,
                total_villain_combos: villain_combos.len(),
                top_villain_combos: top_count,
            }
        })
        .collect()
}

pub fn solve_river_check_bet(config: RiverCfrConfig) -> RiverCfrResult {
    let combos = river_combos(config.board);
    let deals = river_deals(&combos);
    let blocker_reports =
        river_blocker_reports(config.board, &combos, &combos, config.top_fraction);
    let blockers_by_mask: HashMap<_, _> = blocker_reports
        .into_iter()
        .map(|report| (report.hero.mask, report))
        .collect();
    let mut solver = RiverCfrTrainer {
        combos: &combos,
        pot: config.pot,
        bet: config.bet,
        nodes: HashMap::new(),
    };
    let mut utility_sum = 0.0;

    for iteration in 0..config.iterations {
        let (oop_index, ip_index) = deals[sampled_index(iteration, deals.len())];
        utility_sum += solver.cfr(RiverHistory::IpDecision, oop_index, ip_index, [1.0, 1.0]);
    }

    let strategies = solver.combo_strategies(&blockers_by_mask);
    RiverCfrResult {
        iterations: config.iterations,
        expected_value_oop: if config.iterations == 0 {
            0.0
        } else {
            utility_sum / config.iterations as f64
        },
        pot: config.pot,
        bet: config.bet,
        combo_count: combos.len(),
        strategies,
    }
}

#[derive(Debug, Clone, Copy)]
enum RiverHistory {
    IpDecision,
    OopFacingBet,
}

struct RiverCfrTrainer<'a> {
    combos: &'a [RiverCombo],
    pot: f64,
    bet: f64,
    nodes: HashMap<String, RiverCfrNode>,
}

#[derive(Debug, Clone)]
struct RiverCfrNode {
    role: RiverRole,
    node: RiverNode,
    combo_index: usize,
    action_labels: [&'static str; 2],
    regret_sum: [f64; 2],
    strategy_sum: [f64; 2],
}

impl RiverCfrTrainer<'_> {
    fn cfr(
        &mut self,
        history: RiverHistory,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        let player = match history {
            RiverHistory::IpDecision => 1,
            RiverHistory::OopFacingBet => 0,
        };
        let key = self.infoset_key(history, oop_index, ip_index);
        let strategy = self.strategy_for(&key, history, oop_index, ip_index);
        let mut action_values = [0.0; 2];
        let mut node_value = 0.0;

        for action in 0..2 {
            let mut next_reach = reach;
            next_reach[player] *= strategy[action];
            action_values[action] = match history {
                RiverHistory::IpDecision => {
                    if action == 0 {
                        self.showdown_utility(oop_index, ip_index, self.pot * 0.5)
                    } else {
                        self.cfr(RiverHistory::OopFacingBet, oop_index, ip_index, next_reach)
                    }
                }
                RiverHistory::OopFacingBet => {
                    if action == 0 {
                        -self.pot * 0.5
                    } else {
                        self.showdown_utility(oop_index, ip_index, self.pot * 0.5 + self.bet)
                    }
                }
            };
            node_value += strategy[action] * action_values[action];
        }

        let node = self
            .nodes
            .get_mut(&key)
            .expect("river CFR node should exist after strategy lookup");
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

    fn showdown_utility(&self, oop_index: usize, ip_index: usize, win_amount: f64) -> f64 {
        let oop = &self.combos[oop_index];
        let ip = &self.combos[ip_index];
        match oop.value.cmp(&ip.value) {
            std::cmp::Ordering::Greater => win_amount,
            std::cmp::Ordering::Less => -win_amount,
            std::cmp::Ordering::Equal => 0.0,
        }
    }

    fn infoset_key(&self, history: RiverHistory, oop_index: usize, ip_index: usize) -> String {
        match history {
            RiverHistory::IpDecision => format!("IP:{}:after-check", self.combos[ip_index].label()),
            RiverHistory::OopFacingBet => {
                format!("OOP:{}:facing-bet", self.combos[oop_index].label())
            }
        }
    }

    fn strategy_for(
        &mut self,
        key: &str,
        history: RiverHistory,
        oop_index: usize,
        ip_index: usize,
    ) -> [f64; 2] {
        let node = self
            .nodes
            .entry(key.to_string())
            .or_insert_with(|| RiverCfrNode {
                role: match history {
                    RiverHistory::IpDecision => RiverRole::Ip,
                    RiverHistory::OopFacingBet => RiverRole::Oop,
                },
                node: match history {
                    RiverHistory::IpDecision => RiverNode::AfterCheck,
                    RiverHistory::OopFacingBet => RiverNode::FacingBet,
                },
                combo_index: match history {
                    RiverHistory::IpDecision => ip_index,
                    RiverHistory::OopFacingBet => oop_index,
                },
                action_labels: match history {
                    RiverHistory::IpDecision => ["check", "bet"],
                    RiverHistory::OopFacingBet => ["fold", "call"],
                },
                regret_sum: [0.0, 0.0],
                strategy_sum: [0.0, 0.0],
            });
        node.strategy()
    }

    fn combo_strategies(
        &self,
        blockers_by_mask: &HashMap<u64, RiverBlockerReport>,
    ) -> Vec<RiverComboStrategy> {
        let mut strategies: Vec<_> = self
            .nodes
            .values()
            .map(|node| {
                let combo = self.combos[node.combo_index].clone();
                let blocker = blockers_by_mask
                    .get(&combo.mask)
                    .expect("blocker report should exist for river combo");
                RiverComboStrategy {
                    combo,
                    role: node.role,
                    node: node.node,
                    actions: node.average_strategy(),
                    blocked_villain_combos: blocker.blocked_villain_combos,
                    blocked_top_combos: blocker.blocked_top_combos,
                    total_villain_combos: blocker.total_villain_combos,
                    top_villain_combos: blocker.top_villain_combos,
                }
            })
            .collect();
        strategies.sort_by(|left, right| {
            role_key(left.role)
                .cmp(&role_key(right.role))
                .then_with(|| node_key(left.node).cmp(&node_key(right.node)))
                .then_with(|| right.combo.value.cmp(&left.combo.value))
                .then_with(|| right.blocked_top_combos.cmp(&left.blocked_top_combos))
                .then_with(|| left.combo.label().cmp(&right.combo.label()))
        });
        strategies
    }
}

impl RiverCfrNode {
    fn strategy(&self) -> [f64; 2] {
        let positive = [self.regret_sum[0].max(0.0), self.regret_sum[1].max(0.0)];
        let normalizer = positive[0] + positive[1];

        if normalizer > 0.0 {
            [positive[0] / normalizer, positive[1] / normalizer]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self) -> Vec<RiverActionFrequency> {
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
            .map(|(&action, frequency)| RiverActionFrequency { action, frequency })
            .collect()
    }
}

fn river_deals(combos: &[RiverCombo]) -> Vec<(usize, usize)> {
    let mut deals = Vec::new();

    for oop_index in 0..combos.len() {
        for ip_index in 0..combos.len() {
            if oop_index == ip_index || combos[oop_index].mask & combos[ip_index].mask != 0 {
                continue;
            }
            deals.push((oop_index, ip_index));
        }
    }

    deals
}

fn sampled_index(iteration: usize, len: usize) -> usize {
    let mut value = iteration as u64 + 0x9e37_79b9_7f4a_7c15;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) as usize) % len
}

fn role_key(role: RiverRole) -> u8 {
    match role {
        RiverRole::Ip => 0,
        RiverRole::Oop => 1,
    }
}

fn node_key(node: RiverNode) -> u8 {
    match node {
        RiverNode::AfterCheck => 0,
        RiverNode::FacingBet => 1,
    }
}

pub fn board_mask(board: [Card; 5]) -> u64 {
    board.iter().fold(0_u64, |mask, card| mask | card.mask())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn river_combos_exclude_board_cards() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);

        assert_eq!(combos.len(), 1081);
        assert!(
            combos
                .iter()
                .all(|combo| combo.mask & board_mask(board) == 0)
        );
    }

    #[test]
    fn river_combos_order_stronger_hands_first() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);

        assert!(combos[0].value >= combos[1].value);
        assert_eq!(combos[0].label(), "2cTc");
    }

    #[test]
    fn blocker_report_counts_top_range_blockers() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let combos = river_combos(board);
        let ten_spade = combos
            .iter()
            .find(|combo| combo.label() == "2cTc")
            .cloned()
            .expect("combo should be available");
        let reports = river_blocker_reports(board, &[ten_spade], &combos, 0.01);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].blocked_villain_combos > 0);
        assert!(reports[0].blocked_top_combos > 0);
    }

    #[test]
    fn river_cfr_reports_ip_and_oop_strategy_nodes() {
        let board = [c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(2, 1)];
        let result = solve_river_check_bet(RiverCfrConfig {
            board,
            pot: 100.0,
            bet: 75.0,
            iterations: 1_000,
            top_fraction: 0.01,
        });

        assert_eq!(result.combo_count, 1081);
        assert!(
            result
                .strategies
                .iter()
                .any(|strategy| strategy.role == RiverRole::Ip)
        );
        assert!(
            result
                .strategies
                .iter()
                .any(|strategy| strategy.role == RiverRole::Oop)
        );
    }
}
