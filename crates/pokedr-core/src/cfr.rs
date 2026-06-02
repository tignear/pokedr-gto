#[derive(Debug, Clone)]
pub struct KuhnCfrResult {
    pub iterations: usize,
    pub expected_value: f64,
    pub infosets: Vec<KuhnInfosetStrategy>,
}

#[derive(Debug, Clone)]
pub struct KuhnInfosetStrategy {
    pub key: String,
    pub check_or_fold: f64,
    pub bet_or_call: f64,
}

#[derive(Debug, Clone)]
pub struct KuhnCfrSolver {
    nodes: Vec<KuhnNode>,
}

#[derive(Debug, Clone)]
struct KuhnNode {
    key: String,
    regret_sum: [f64; 2],
    strategy_sum: [f64; 2],
}

impl KuhnCfrSolver {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn train(&mut self, iterations: usize) -> KuhnCfrResult {
        let cards = [0_u8, 1, 2];
        let mut utility_sum = 0.0;

        for iteration in 0..iterations {
            let deal = kuhn_deal(iteration, cards);
            utility_sum += self.cfr(deal, "", 1.0, 1.0);
        }

        KuhnCfrResult {
            iterations,
            expected_value: if iterations == 0 {
                0.0
            } else {
                utility_sum / iterations as f64
            },
            infosets: self.average_strategies(),
        }
    }

    fn cfr(&mut self, cards: [u8; 2], history: &str, p0: f64, p1: f64) -> f64 {
        if let Some(value) = terminal_value(cards, history) {
            return value;
        }

        let plays = history.len();
        let player = plays % 2;
        let opponent_reach = if player == 0 { p1 } else { p0 };
        let player_reach = if player == 0 { p0 } else { p1 };
        let key = infoset_key(cards[player], history);
        let node_index = self.node_index(&key);
        let strategy = self.nodes[node_index].strategy();
        let actions = legal_actions(history);
        let mut action_values = [0.0; 2];
        let mut node_value = 0.0;

        for action in actions {
            let next_history = next_history(history, action);
            let value = if player == 0 {
                -self.cfr(cards, &next_history, p0 * strategy[action], p1)
            } else {
                -self.cfr(cards, &next_history, p0, p1 * strategy[action])
            };
            action_values[action] = value;
            node_value += strategy[action] * value;
        }

        let node = &mut self.nodes[node_index];
        for action in actions {
            node.regret_sum[action] += opponent_reach * (action_values[action] - node_value);
            node.strategy_sum[action] += player_reach * strategy[action];
        }

        node_value
    }

    fn node_index(&mut self, key: &str) -> usize {
        if let Some(index) = self.nodes.iter().position(|node| node.key == key) {
            return index;
        }

        self.nodes.push(KuhnNode {
            key: key.to_string(),
            regret_sum: [0.0; 2],
            strategy_sum: [0.0; 2],
        });
        self.nodes.len() - 1
    }

    fn average_strategies(&self) -> Vec<KuhnInfosetStrategy> {
        let mut infosets: Vec<_> = self
            .nodes
            .iter()
            .map(|node| {
                let normalizer = node.strategy_sum[0] + node.strategy_sum[1];
                let average = if normalizer > 0.0 {
                    [
                        node.strategy_sum[0] / normalizer,
                        node.strategy_sum[1] / normalizer,
                    ]
                } else {
                    [0.5, 0.5]
                };
                KuhnInfosetStrategy {
                    key: node.key.clone(),
                    check_or_fold: average[0],
                    bet_or_call: average[1],
                }
            })
            .collect();
        infosets.sort_by(|left, right| left.key.cmp(&right.key));
        infosets
    }
}

impl Default for KuhnCfrSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl KuhnNode {
    fn strategy(&self) -> [f64; 2] {
        let positive = [self.regret_sum[0].max(0.0), self.regret_sum[1].max(0.0)];
        let normalizer = positive[0] + positive[1];

        if normalizer > 0.0 {
            [positive[0] / normalizer, positive[1] / normalizer]
        } else {
            [0.5, 0.5]
        }
    }
}

fn kuhn_deal(iteration: usize, mut cards: [u8; 3]) -> [u8; 2] {
    // Deterministic Fisher-Yates over the six possible deals keeps tests reproducible.
    let mut seed = iteration as u64 + 0x9e37_79b9_7f4a_7c15;
    for index in (1..cards.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let swap = (seed as usize) % (index + 1);
        cards.swap(index, swap);
    }
    [cards[0], cards[1]]
}

fn infoset_key(card: u8, history: &str) -> String {
    let rank = match card {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        _ => '?',
    };
    format!("{rank}{history}")
}

fn legal_actions(history: &str) -> [usize; 2] {
    match history {
        "" | "c" | "b" | "cb" => [0, 1],
        _ => [0, 1],
    }
}

fn next_history(history: &str, action: usize) -> String {
    let action = match action {
        0 => 'c',
        1 => 'b',
        _ => unreachable!("Kuhn poker has two actions"),
    };
    let mut next = String::with_capacity(history.len() + 1);
    next.push_str(history);
    next.push(action);
    next
}

fn terminal_value(cards: [u8; 2], history: &str) -> Option<f64> {
    let current_player = history.len() % 2;
    match history {
        "cc" => Some(showdown_value(cards, current_player, 1.0)),
        "bc" | "cbc" => Some(1.0),
        "bb" | "cbb" => Some(showdown_value(cards, current_player, 2.0)),
        _ => None,
    }
}

fn showdown_value(cards: [u8; 2], player: usize, pot_units: f64) -> f64 {
    if cards[player] > cards[1 - player] {
        pot_units
    } else {
        -pot_units
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kuhn_cfr_converges_near_known_game_value() {
        let mut solver = KuhnCfrSolver::new();
        let result = solver.train(100_000);

        assert!((result.expected_value + 1.0 / 18.0).abs() < 0.02);
    }

    #[test]
    fn kuhn_cfr_reports_infosets() {
        let mut solver = KuhnCfrSolver::new();
        let result = solver.train(1_000);

        assert!(!result.infosets.is_empty());
        assert!(result.infosets.iter().all(|strategy| {
            (strategy.check_or_fold + strategy.bet_or_call - 1.0).abs() < 1.0e-9
        }));
    }
}
