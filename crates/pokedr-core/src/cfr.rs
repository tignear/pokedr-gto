use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CfrResult {
    pub game: &'static str,
    pub iterations: usize,
    pub expected_value_p0: f64,
    pub infosets: Vec<InfosetStrategy>,
}

#[derive(Debug, Clone)]
pub struct InfosetStrategy {
    pub key: String,
    pub actions: Vec<ActionStrategy>,
}

#[derive(Debug, Clone)]
pub struct ActionStrategy {
    pub label: &'static str,
    pub probability: f64,
}

#[derive(Debug, Clone)]
pub struct KuhnCfrSolver {
    trainer: CfrTrainer<KuhnGame>,
}

#[derive(Debug, Clone)]
pub struct LeducCfrSolver {
    trainer: CfrTrainer<LeducGame>,
}

trait CfrGame: Clone {
    type State: Clone;

    fn name(&self) -> &'static str;
    fn root_state(&self, iteration: usize) -> Self::State;
    fn terminal_utilities(&self, state: &Self::State) -> Option<[f64; 2]>;
    fn current_player(&self, state: &Self::State) -> Option<usize>;
    fn chance_outcomes(&self, state: &Self::State) -> Vec<(usize, f64)>;
    fn infoset_key(&self, state: &Self::State, player: usize) -> String;
    fn actions(&self, state: &Self::State) -> Vec<usize>;
    fn action_label(&self, action: usize) -> &'static str;
    fn apply(&self, state: &Self::State, action: usize) -> Self::State;
}

#[derive(Debug, Clone)]
struct CfrTrainer<G> {
    game: G,
    nodes: HashMap<String, CfrNode>,
}

#[derive(Debug, Clone)]
struct CfrNode {
    action_labels: Vec<&'static str>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl KuhnCfrSolver {
    pub fn new() -> Self {
        Self {
            trainer: CfrTrainer::new(KuhnGame),
        }
    }

    pub fn train(&mut self, iterations: usize) -> CfrResult {
        self.trainer.train(iterations)
    }
}

impl Default for KuhnCfrSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LeducCfrSolver {
    pub fn new() -> Self {
        Self {
            trainer: CfrTrainer::new(LeducGame),
        }
    }

    pub fn train(&mut self, iterations: usize) -> CfrResult {
        self.trainer.train(iterations)
    }
}

impl Default for LeducCfrSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: CfrGame> CfrTrainer<G> {
    fn new(game: G) -> Self {
        Self {
            game,
            nodes: HashMap::new(),
        }
    }

    fn train(&mut self, iterations: usize) -> CfrResult {
        let mut utility_sum = 0.0;

        for iteration in 0..iterations {
            let root = self.game.root_state(iteration);
            utility_sum += self.cfr(&root, [1.0, 1.0])[0];
        }

        CfrResult {
            game: self.game.name(),
            iterations,
            expected_value_p0: if iterations == 0 {
                0.0
            } else {
                utility_sum / iterations as f64
            },
            infosets: self.average_strategies(),
        }
    }

    fn cfr(&mut self, state: &G::State, reach: [f64; 2]) -> [f64; 2] {
        if let Some(utilities) = self.game.terminal_utilities(state) {
            return utilities;
        }

        let Some(player) = self.game.current_player(state) else {
            let mut utilities = [0.0, 0.0];
            for (action, probability) in self.game.chance_outcomes(state) {
                let next = self.game.apply(state, action);
                let action_utilities = self.cfr(&next, reach);
                utilities[0] += probability * action_utilities[0];
                utilities[1] += probability * action_utilities[1];
            }
            return utilities;
        };

        let actions = self.game.actions(state);
        let key = self.game.infoset_key(state, player);
        let labels: Vec<_> = actions
            .iter()
            .map(|&action| self.game.action_label(action))
            .collect();
        let strategy = self.strategy_for(&key, &labels);
        let mut action_utilities = vec![[0.0, 0.0]; actions.len()];
        let mut node_utilities = [0.0, 0.0];

        for (index, &action) in actions.iter().enumerate() {
            let next = self.game.apply(state, action);
            let mut next_reach = reach;
            next_reach[player] *= strategy[index];
            action_utilities[index] = self.cfr(&next, next_reach);
            node_utilities[0] += strategy[index] * action_utilities[index][0];
            node_utilities[1] += strategy[index] * action_utilities[index][1];
        }

        let node = self
            .nodes
            .get_mut(&key)
            .expect("CFR node should exist after strategy lookup");
        let opponent_reach = reach[1 - player];
        for index in 0..actions.len() {
            node.regret_sum[index] +=
                opponent_reach * (action_utilities[index][player] - node_utilities[player]);
            node.strategy_sum[index] += reach[player] * strategy[index];
        }

        node_utilities
    }

    fn strategy_for(&mut self, key: &str, labels: &[&'static str]) -> Vec<f64> {
        let node = self
            .nodes
            .entry(key.to_string())
            .or_insert_with(|| CfrNode::new(labels));
        node.strategy()
    }

    fn average_strategies(&self) -> Vec<InfosetStrategy> {
        let mut infosets: Vec<_> = self
            .nodes
            .iter()
            .map(|(key, node)| InfosetStrategy {
                key: key.clone(),
                actions: node.average_strategy(),
            })
            .collect();
        infosets.sort_by(|left, right| left.key.cmp(&right.key));
        infosets
    }
}

impl CfrNode {
    fn new(labels: &[&'static str]) -> Self {
        Self {
            action_labels: labels.to_vec(),
            regret_sum: vec![0.0; labels.len()],
            strategy_sum: vec![0.0; labels.len()],
        }
    }

    fn strategy(&self) -> Vec<f64> {
        let positive: Vec<_> = self
            .regret_sum
            .iter()
            .map(|regret| regret.max(0.0))
            .collect();
        let normalizer: f64 = positive.iter().sum();

        if normalizer > 0.0 {
            positive.iter().map(|value| value / normalizer).collect()
        } else {
            vec![1.0 / positive.len() as f64; positive.len()]
        }
    }

    fn average_strategy(&self) -> Vec<ActionStrategy> {
        let normalizer: f64 = self.strategy_sum.iter().sum();

        self.action_labels
            .iter()
            .enumerate()
            .map(|(index, &label)| {
                let probability = if normalizer > 0.0 {
                    self.strategy_sum[index] / normalizer
                } else {
                    1.0 / self.action_labels.len() as f64
                };
                ActionStrategy { label, probability }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct KuhnGame;

#[derive(Debug, Clone)]
struct KuhnState {
    cards: [u8; 2],
    history: String,
}

impl CfrGame for KuhnGame {
    type State = KuhnState;

    fn name(&self) -> &'static str {
        "kuhn"
    }

    fn root_state(&self, iteration: usize) -> Self::State {
        let cards = sample_without_replacement(iteration, &[0, 1, 2], 2);
        KuhnState {
            cards: [cards[0], cards[1]],
            history: String::new(),
        }
    }

    fn terminal_utilities(&self, state: &Self::State) -> Option<[f64; 2]> {
        match state.history.as_str() {
            "cc" => Some(kuhn_showdown(state.cards, 1.0)),
            "bc" => Some([1.0, -1.0]),
            "cbc" => Some([-1.0, 1.0]),
            "bb" | "cbb" => Some(kuhn_showdown(state.cards, 2.0)),
            _ => None,
        }
    }

    fn current_player(&self, state: &Self::State) -> Option<usize> {
        Some(state.history.len() % 2)
    }

    fn chance_outcomes(&self, _state: &Self::State) -> Vec<(usize, f64)> {
        Vec::new()
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> String {
        format!("{}{}", kuhn_rank_label(state.cards[player]), state.history)
    }

    fn actions(&self, _state: &Self::State) -> Vec<usize> {
        vec![0, 1]
    }

    fn action_label(&self, action: usize) -> &'static str {
        match action {
            0 => "check/fold",
            1 => "bet/call",
            _ => "?",
        }
    }

    fn apply(&self, state: &Self::State, action: usize) -> Self::State {
        let mut next = state.clone();
        next.history.push(match action {
            0 => 'c',
            1 => 'b',
            _ => unreachable!("Kuhn poker has two actions"),
        });
        next
    }
}

#[derive(Debug, Clone)]
struct LeducGame;

#[derive(Debug, Clone)]
struct LeducState {
    private_cards: [u8; 2],
    public_card: Option<u8>,
    round: u8,
    history: String,
    current_player: Option<usize>,
    folded: Option<usize>,
    contributions: [f64; 2],
}

impl CfrGame for LeducGame {
    type State = LeducState;

    fn name(&self) -> &'static str {
        "leduc"
    }

    fn root_state(&self, iteration: usize) -> Self::State {
        let cards = sample_without_replacement(iteration, &[0, 1, 2, 3, 4, 5], 2);
        LeducState {
            private_cards: [cards[0], cards[1]],
            public_card: None,
            round: 0,
            history: String::new(),
            current_player: Some(0),
            folded: None,
            contributions: [1.0, 1.0],
        }
    }

    fn terminal_utilities(&self, state: &Self::State) -> Option<[f64; 2]> {
        if let Some(folded) = state.folded {
            let winner = 1 - folded;
            return Some(net_utilities(winner, state.contributions));
        }

        if state.round == 2 {
            let winner = leduc_winner(state.private_cards, state.public_card?);
            return Some(net_utilities(winner, state.contributions));
        }

        None
    }

    fn current_player(&self, state: &Self::State) -> Option<usize> {
        state.current_player
    }

    fn chance_outcomes(&self, state: &Self::State) -> Vec<(usize, f64)> {
        if state.current_player.is_some() || state.public_card.is_some() {
            return Vec::new();
        }

        let dead0 = state.private_cards[0];
        let dead1 = state.private_cards[1];
        let mut cards: Vec<_> = (0..6)
            .filter(|&card| card != dead0 && card != dead1)
            .collect();
        cards.sort_unstable();
        let probability = 1.0 / cards.len() as f64;
        cards
            .into_iter()
            .map(|card| (card as usize, probability))
            .collect()
    }

    fn infoset_key(&self, state: &Self::State, player: usize) -> String {
        format!(
            "{}|{}|{}|{}",
            leduc_card_label(state.private_cards[player]),
            state
                .public_card
                .map(leduc_card_label)
                .unwrap_or("-".to_string()),
            state.round,
            state.history
        )
    }

    fn actions(&self, _state: &Self::State) -> Vec<usize> {
        vec![0, 1]
    }

    fn action_label(&self, action: usize) -> &'static str {
        match action {
            0 => "check/fold",
            1 => "bet/call",
            _ => "?",
        }
    }

    fn apply(&self, state: &Self::State, action: usize) -> Self::State {
        if state.current_player.is_none() {
            let mut next = state.clone();
            next.public_card = Some(action as u8);
            next.round = 1;
            next.history.clear();
            next.current_player = Some(0);
            return next;
        }

        let player = state.current_player.expect("Leduc action needs a player");
        let mut next = state.clone();
        let action_char = match action {
            0 => 'c',
            1 => 'b',
            _ => unreachable!("Leduc poker has two actions"),
        };
        next.history.push(action_char);

        match next.history.as_str() {
            "cc" => {
                if next.round == 0 {
                    next.current_player = None;
                } else {
                    next.round = 2;
                    next.current_player = None;
                }
            }
            "bc" | "cbc" => {
                next.folded = Some(player);
                next.current_player = None;
            }
            "bb" | "cbb" => {
                next.contributions[player] += leduc_bet_size(next.round);
                if next.round == 0 {
                    next.current_player = None;
                    next.history.clear();
                } else {
                    next.round = 2;
                    next.current_player = None;
                }
            }
            _ => {
                if action == 1 {
                    next.contributions[player] += leduc_bet_size(next.round);
                }
                next.current_player = Some(1 - player);
            }
        }

        next
    }
}

fn sample_without_replacement(iteration: usize, deck: &[u8], count: usize) -> Vec<u8> {
    let mut cards = deck.to_vec();
    let mut seed = iteration as u64 + 0x9e37_79b9_7f4a_7c15;
    for index in (1..cards.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let swap = (seed as usize) % (index + 1);
        cards.swap(index, swap);
    }
    cards.truncate(count);
    cards
}

fn kuhn_rank_label(card: u8) -> char {
    match card {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        _ => '?',
    }
}

fn kuhn_showdown(cards: [u8; 2], win_amount: f64) -> [f64; 2] {
    if cards[0] > cards[1] {
        [win_amount, -win_amount]
    } else {
        [-win_amount, win_amount]
    }
}

fn leduc_bet_size(round: u8) -> f64 {
    if round == 0 { 2.0 } else { 4.0 }
}

fn leduc_card_label(card: u8) -> String {
    let rank = match card / 2 {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        _ => '?',
    };
    let suit = match card % 2 {
        0 => 'a',
        1 => 'b',
        _ => '?',
    };
    format!("{rank}{suit}")
}

fn leduc_rank(card: u8) -> u8 {
    card / 2
}

fn leduc_winner(private_cards: [u8; 2], public_card: u8) -> usize {
    let p0_pair = leduc_rank(private_cards[0]) == leduc_rank(public_card);
    let p1_pair = leduc_rank(private_cards[1]) == leduc_rank(public_card);

    match (p0_pair, p1_pair) {
        (true, false) => 0,
        (false, true) => 1,
        _ => {
            if leduc_rank(private_cards[0]) >= leduc_rank(private_cards[1]) {
                0
            } else {
                1
            }
        }
    }
}

fn net_utilities(winner: usize, contributions: [f64; 2]) -> [f64; 2] {
    let pot = contributions[0] + contributions[1];
    let mut utilities = [-contributions[0], -contributions[1]];
    utilities[winner] += pot;
    utilities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kuhn_cfr_converges_near_known_game_value() {
        let mut solver = KuhnCfrSolver::new();
        let result = solver.train(100_000);

        assert!((result.expected_value_p0 + 1.0 / 18.0).abs() < 0.02);
    }

    #[test]
    fn cfr_reports_normalized_infosets() {
        let mut solver = KuhnCfrSolver::new();
        let result = solver.train(1_000);

        assert!(!result.infosets.is_empty());
        assert!(result.infosets.iter().all(|strategy| {
            (strategy
                .actions
                .iter()
                .map(|action| action.probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1.0e-9
        }));
    }

    #[test]
    fn leduc_cfr_produces_public_card_infosets() {
        let mut solver = LeducCfrSolver::new();
        let result = solver.train(1_000);

        assert_eq!(result.game, "leduc");
        assert!(
            result
                .infosets
                .iter()
                .any(|infoset| infoset.key.contains('|'))
        );
        assert!(result.expected_value_p0.abs() < 10.0);
    }
}
