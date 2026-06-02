use std::collections::HashMap;

use crate::cards::{Card, deck};
use crate::hand_class::HandClass;
use crate::hand_eval::evaluate_seven;
use crate::postflop::{PostflopCombo, postflop_combos};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Flop,
    Turn,
    River,
}

impl Street {
    pub fn from_board(board: &[Card]) -> Result<Self, SubgameValidationError> {
        match board.len() {
            3 => Ok(Self::Flop),
            4 => Ok(Self::Turn),
            5 => Ok(Self::River),
            _ => Err(SubgameValidationError::InvalidBoardLength),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeState {
    pub oop: Vec<HandClass>,
    pub ip: Vec<HandClass>,
}

impl RangeState {
    pub fn new(oop: Vec<HandClass>, ip: Vec<HandClass>) -> Self {
        Self { oop, ip }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PotState {
    pub pot: f64,
    pub stacks: [f64; 2],
    pub committed: [f64; 2],
    pub current_bet: f64,
    pub min_raise: f64,
}

impl PotState {
    pub fn new(pot: f64, stacks: [f64; 2]) -> Self {
        Self {
            pot,
            stacks,
            committed: [0.0, 0.0],
            current_bet: 0.0,
            min_raise: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BetSize {
    PotFraction(f64),
    CurrentBetMultiple(f64),
    Chips(f64),
    AllIn,
}

#[derive(Debug, Clone)]
pub struct ActionAbstraction {
    pub bet_sizes: Vec<BetSize>,
    pub raise_sizes: Vec<BetSize>,
    pub reraise_sizes: Vec<BetSize>,
    pub allow_all_in: bool,
    pub max_raises: u8,
}

impl Default for ActionAbstraction {
    fn default() -> Self {
        Self {
            bet_sizes: vec![BetSize::PotFraction(0.75)],
            raise_sizes: vec![BetSize::CurrentBetMultiple(3.0)],
            reraise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
            allow_all_in: false,
            max_raises: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChancePolicy {
    Enumerate,
    Sample(usize),
}

impl ChancePolicy {
    pub fn max_runouts(self) -> usize {
        match self {
            Self::Enumerate => 0,
            Self::Sample(max_runouts) => max_runouts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubgameSpec {
    pub board: Vec<Card>,
    pub street: Street,
    pub pot: PotState,
    pub ranges: RangeState,
    pub actions: ActionAbstraction,
    pub chance: ChancePolicy,
}

impl SubgameSpec {
    pub fn postflop(
        board: Vec<Card>,
        pot: PotState,
        ranges: RangeState,
        actions: ActionAbstraction,
        chance: ChancePolicy,
    ) -> Result<Self, SubgameValidationError> {
        let street = Street::from_board(&board)?;
        let spec = Self {
            board,
            street,
            pot,
            ranges,
            actions,
            chance,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), SubgameValidationError> {
        if Street::from_board(&self.board)? != self.street {
            return Err(SubgameValidationError::StreetBoardMismatch);
        }
        if has_duplicate_cards(&self.board) {
            return Err(SubgameValidationError::DuplicateBoardCard);
        }
        if self.ranges.oop.is_empty() || self.ranges.ip.is_empty() {
            return Err(SubgameValidationError::EmptyRange);
        }
        validate_non_negative(self.pot.pot, SubgameValidationError::InvalidPot)?;
        validate_non_negative(self.pot.current_bet, SubgameValidationError::InvalidBet)?;
        validate_non_negative(self.pot.min_raise, SubgameValidationError::InvalidBet)?;
        for stack in self.pot.stacks {
            validate_non_negative(stack, SubgameValidationError::InvalidStack)?;
        }
        for committed in self.pot.committed {
            validate_non_negative(committed, SubgameValidationError::InvalidCommitment)?;
        }
        for size in self
            .actions
            .bet_sizes
            .iter()
            .chain(self.actions.raise_sizes.iter())
            .chain(self.actions.reraise_sizes.iter())
        {
            validate_bet_size(*size)?;
        }
        if matches!(self.chance, ChancePolicy::Sample(0)) {
            return Err(SubgameValidationError::InvalidRunoutSample);
        }
        Ok(())
    }

    pub fn build_action_tree(&self) -> Result<ActionTree, SubgameBuildError> {
        self.validate()?;
        let mut builder = ActionTreeBuilder {
            spec: self,
            nodes: Vec::new(),
        };
        let root = builder.build_decision(Player::Oop, self.pot, 0, Vec::new())?;
        Ok(ActionTree {
            root,
            nodes: builder.nodes,
        })
    }

    pub fn solve_cfr(&self, iterations: usize) -> Result<SubgameCfrResult, SubgameBuildError> {
        let tree = self.build_action_tree()?;
        let oop_range = postflop_combos(&self.ranges.oop, &self.board);
        let ip_range = postflop_combos(&self.ranges.ip, &self.board);
        if oop_range.is_empty() || ip_range.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }

        let deals = legal_deals(&oop_range, &ip_range);
        if deals.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }

        let mut trainer = SubgameCfrTrainer {
            spec: self,
            tree: &tree,
            oop_range: &oop_range,
            ip_range: &ip_range,
            nodes: HashMap::new(),
            equity_cache: HashMap::new(),
            average_weight: 1.0,
        };
        let mut utility_sum = 0.0;
        for iteration in 0..iterations {
            trainer.average_weight = iteration as f64 + 1.0;
            let (oop_index, ip_index) = deals[sampled_index(iteration, deals.len())];
            utility_sum += trainer.cfr(tree.root, oop_index, ip_index, [1.0, 1.0]);
        }

        Ok(SubgameCfrResult {
            iterations,
            expected_value_oop: if iterations == 0 {
                0.0
            } else {
                utility_sum / iterations as f64
            },
            root: tree.root,
            node_count: tree.nodes.len(),
            oop_combo_count: oop_range.len(),
            ip_combo_count: ip_range.len(),
            strategies: trainer.combo_strategies(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SubgameCfrResult {
    pub iterations: usize,
    pub expected_value_oop: f64,
    pub root: NodeId,
    pub node_count: usize,
    pub oop_combo_count: usize,
    pub ip_combo_count: usize,
    pub strategies: Vec<SubgameComboStrategy>,
}

#[derive(Debug, Clone)]
pub struct SubgameComboStrategy {
    pub node: NodeId,
    pub player: Player,
    pub board: Vec<Card>,
    pub history: Vec<ActionKind>,
    pub combo: PostflopCombo,
    pub equity: f64,
    pub actions: Vec<SubgameActionFrequency>,
}

#[derive(Debug, Clone)]
pub struct SubgameActionFrequency {
    pub action: ActionKind,
    pub frequency: f64,
}

#[derive(Debug, Clone)]
pub struct ActionTree {
    pub root: NodeId,
    pub nodes: Vec<ActionTreeNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
pub enum ActionTreeNode {
    Decision {
        player: Player,
        pot: PotState,
        raises: u8,
        history: Vec<ActionKind>,
        actions: Vec<ActionEdge>,
    },
    Terminal {
        terminal: TerminalKind,
        pot: PotState,
        history: Vec<ActionKind>,
    },
}

#[derive(Debug, Clone)]
pub struct ActionEdge {
    pub action: ActionKind,
    pub to: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Oop,
    Ip,
}

impl Player {
    pub fn index(self) -> usize {
        match self {
            Self::Oop => 0,
            Self::Ip => 1,
        }
    }

    pub fn other(self) -> Self {
        match self {
            Self::Oop => Self::Ip,
            Self::Ip => Self::Oop,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionKind {
    Check,
    Fold,
    Call,
    Bet(f64),
    Raise(f64),
    AllIn(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Fold { winner: Player },
    Showdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgameValidationError {
    InvalidBoardLength,
    StreetBoardMismatch,
    DuplicateBoardCard,
    EmptyRange,
    InvalidPot,
    InvalidStack,
    InvalidCommitment,
    InvalidBet,
    InvalidBetSize,
    InvalidRunoutSample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgameBuildError {
    Validation(SubgameValidationError),
    NoLegalDeals,
}

impl From<SubgameValidationError> for SubgameBuildError {
    fn from(error: SubgameValidationError) -> Self {
        Self::Validation(error)
    }
}

fn validate_non_negative(
    value: f64,
    error: SubgameValidationError,
) -> Result<(), SubgameValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(error)
    }
}

fn validate_bet_size(size: BetSize) -> Result<(), SubgameValidationError> {
    match size {
        BetSize::PotFraction(value)
        | BetSize::CurrentBetMultiple(value)
        | BetSize::Chips(value) => {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(SubgameValidationError::InvalidBetSize)
            }
        }
        BetSize::AllIn => Ok(()),
    }
}

fn resolve_bet_size(size: BetSize, pot: &PotState, current_bet: f64) -> f64 {
    match size {
        BetSize::PotFraction(fraction) => pot.pot * fraction,
        BetSize::CurrentBetMultiple(multiplier) => current_bet.max(1.0) * multiplier,
        BetSize::Chips(chips) => chips,
        BetSize::AllIn => unreachable!("all-in is resolved by actor stack"),
    }
}

fn has_duplicate_cards(board: &[Card]) -> bool {
    let mut mask = 0_u64;
    for card in board {
        let card_mask = card.mask();
        if mask & card_mask != 0 {
            return true;
        }
        mask |= card_mask;
    }
    false
}

struct ActionTreeBuilder<'a> {
    spec: &'a SubgameSpec,
    nodes: Vec<ActionTreeNode>,
}

impl ActionTreeBuilder<'_> {
    fn build_decision(
        &mut self,
        player: Player,
        pot: PotState,
        raises: u8,
        history: Vec<ActionKind>,
    ) -> Result<NodeId, SubgameBuildError> {
        let id = self.push(ActionTreeNode::Decision {
            player,
            pot,
            raises,
            history: history.clone(),
            actions: Vec::new(),
        });
        let actions = self.legal_actions(player, pot, raises)?;
        let mut edges = Vec::with_capacity(actions.len());

        for action in actions {
            let mut next_history = history.clone();
            next_history.push(action);
            let to = match action {
                ActionKind::Check if checked_through(&next_history) => {
                    self.push(ActionTreeNode::Terminal {
                        terminal: TerminalKind::Showdown,
                        pot,
                        history: next_history,
                    })
                }
                ActionKind::Call => {
                    let next_pot = apply_call(player, pot);
                    self.push(ActionTreeNode::Terminal {
                        terminal: TerminalKind::Showdown,
                        pot: next_pot,
                        history: next_history,
                    })
                }
                ActionKind::Fold => self.push(ActionTreeNode::Terminal {
                    terminal: TerminalKind::Fold {
                        winner: player.other(),
                    },
                    pot,
                    history: next_history,
                }),
                ActionKind::Check => {
                    self.build_decision(player.other(), pot, raises, next_history)?
                }
                ActionKind::Bet(amount) => self.build_decision(
                    player.other(),
                    apply_aggressive(player, pot, amount),
                    raises,
                    next_history,
                )?,
                ActionKind::Raise(amount) | ActionKind::AllIn(amount) => self.build_decision(
                    player.other(),
                    apply_aggressive(player, pot, amount),
                    raises + 1,
                    next_history,
                )?,
            };
            edges.push(ActionEdge { action, to });
        }

        if let ActionTreeNode::Decision { actions, .. } = &mut self.nodes[id.0] {
            *actions = edges;
        }
        Ok(id)
    }

    fn legal_actions(
        &self,
        player: Player,
        pot: PotState,
        raises: u8,
    ) -> Result<Vec<ActionKind>, SubgameBuildError> {
        if pot.current_bet <= pot.committed[player.index()] {
            let mut actions = vec![ActionKind::Check];
            for size in &self.spec.actions.bet_sizes {
                let amount = capped_commitment(
                    player,
                    pot,
                    resolve_tree_bet_size(*size, player, &pot, pot.pot),
                );
                if amount > pot.committed[player.index()] {
                    actions.push(sized_aggressive_action(*size, amount, false));
                }
            }
            if self.spec.actions.allow_all_in {
                let amount = all_in_commitment(player, pot);
                if amount > pot.committed[player.index()] {
                    actions.push(ActionKind::AllIn(amount));
                }
            }
            return Ok(actions);
        }

        let mut actions = vec![ActionKind::Fold, ActionKind::Call];
        if raises < self.spec.actions.max_raises {
            let sizes = if raises == 0 {
                &self.spec.actions.raise_sizes
            } else {
                &self.spec.actions.reraise_sizes
            };
            for size in sizes {
                let amount = capped_commitment(
                    player,
                    pot,
                    resolve_tree_bet_size(*size, player, &pot, pot.current_bet),
                );
                if amount > pot.current_bet {
                    actions.push(sized_aggressive_action(*size, amount, true));
                }
            }
            if self.spec.actions.allow_all_in {
                let amount = all_in_commitment(player, pot);
                if amount > pot.current_bet {
                    actions.push(ActionKind::AllIn(amount));
                }
            }
        }
        Ok(actions)
    }

    fn push(&mut self, node: ActionTreeNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }
}

fn checked_through(history: &[ActionKind]) -> bool {
    history.len() >= 2
        && matches!(
            history[history.len() - 2..],
            [ActionKind::Check, ActionKind::Check]
        )
}

fn sized_aggressive_action(size: BetSize, amount: f64, facing_bet: bool) -> ActionKind {
    match (size, facing_bet) {
        (BetSize::AllIn, _) => ActionKind::AllIn(amount),
        (_, true) => ActionKind::Raise(amount),
        (_, false) => ActionKind::Bet(amount),
    }
}

fn resolve_tree_bet_size(size: BetSize, player: Player, pot: &PotState, current_bet: f64) -> f64 {
    match size {
        BetSize::AllIn => all_in_commitment(player, *pot),
        _ => resolve_bet_size(size, pot, current_bet),
    }
}

fn all_in_commitment(player: Player, pot: PotState) -> f64 {
    pot.committed[player.index()] + pot.stacks[player.index()]
}

fn capped_commitment(player: Player, pot: PotState, amount: f64) -> f64 {
    amount.min(all_in_commitment(player, pot))
}

fn apply_aggressive(player: Player, mut pot: PotState, amount: f64) -> PotState {
    let index = player.index();
    let previous = pot.committed[index];
    let contribution = (amount - previous).max(0.0).min(pot.stacks[index]);
    pot.pot += contribution;
    pot.stacks[index] -= contribution;
    pot.committed[index] += contribution;
    pot.min_raise = (pot.committed[index] - pot.current_bet).max(pot.min_raise);
    pot.current_bet = pot.current_bet.max(pot.committed[index]);
    pot
}

fn apply_call(player: Player, mut pot: PotState) -> PotState {
    let index = player.index();
    let contribution = (pot.current_bet - pot.committed[index])
        .max(0.0)
        .min(pot.stacks[index]);
    pot.pot += contribution;
    pot.stacks[index] -= contribution;
    pot.committed[index] += contribution;
    pot
}

struct SubgameCfrTrainer<'a> {
    spec: &'a SubgameSpec,
    tree: &'a ActionTree,
    oop_range: &'a [PostflopCombo],
    ip_range: &'a [PostflopCombo],
    nodes: HashMap<String, SubgameCfrNode>,
    equity_cache: HashMap<(u64, u64, u64), f64>,
    average_weight: f64,
}

#[derive(Debug, Clone)]
struct SubgameCfrNode {
    node: NodeId,
    player: Player,
    board: Vec<Card>,
    history: Vec<ActionKind>,
    combo: PostflopCombo,
    equity: f64,
    actions: Vec<ActionKind>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl SubgameCfrTrainer<'_> {
    fn cfr(&mut self, node_id: NodeId, oop_index: usize, ip_index: usize, reach: [f64; 2]) -> f64 {
        match &self.tree.nodes[node_id.0] {
            ActionTreeNode::Terminal { terminal, pot, .. } => {
                self.terminal_utility(*terminal, *pot, oop_index, ip_index)
            }
            ActionTreeNode::Decision {
                player,
                history,
                actions,
                ..
            } => {
                let player_index = player.index();
                let key = self.infoset_key(node_id, *player, history, oop_index, ip_index);
                let strategy = self.strategy_for(
                    &key, node_id, *player, history, actions, oop_index, ip_index,
                );
                let mut action_values = vec![0.0; strategy.len()];
                let mut node_value = 0.0;

                for (index, edge) in actions.iter().enumerate() {
                    let mut next_reach = reach;
                    next_reach[player_index] *= strategy[index];
                    action_values[index] = self.cfr(edge.to, oop_index, ip_index, next_reach);
                    node_value += strategy[index] * action_values[index];
                }

                let node = self
                    .nodes
                    .get_mut(&key)
                    .expect("subgame CFR node should exist after strategy lookup");
                let opponent_reach = reach[player.other().index()];
                for action in 0..strategy.len() {
                    let regret = if *player == Player::Oop {
                        action_values[action] - node_value
                    } else {
                        node_value - action_values[action]
                    };
                    node.regret_sum[action] =
                        (node.regret_sum[action] + opponent_reach * regret).max(0.0);
                    node.strategy_sum[action] +=
                        self.average_weight * reach[player_index] * strategy[action];
                }

                node_value
            }
        }
    }

    fn terminal_utility(
        &mut self,
        terminal: TerminalKind,
        pot: PotState,
        oop_index: usize,
        ip_index: usize,
    ) -> f64 {
        match terminal {
            TerminalKind::Fold { winner } => {
                if winner == Player::Oop {
                    pot.pot - pot.committed[Player::Oop.index()]
                } else {
                    -pot.committed[Player::Oop.index()]
                }
            }
            TerminalKind::Showdown => {
                let equity = self.combo_equity(oop_index, ip_index);
                equity * pot.pot - pot.committed[Player::Oop.index()]
            }
        }
    }

    fn infoset_key(
        &self,
        node_id: NodeId,
        player: Player,
        history: &[ActionKind],
        oop_index: usize,
        ip_index: usize,
    ) -> String {
        let combo = match player {
            Player::Oop => self.oop_range[oop_index].label(),
            Player::Ip => self.ip_range[ip_index].label(),
        };
        format!(
            "{:?}:{}:{}:{combo}:{}",
            player,
            node_id.0,
            board_label(&self.spec.board),
            history_label(history)
        )
    }

    fn strategy_for(
        &mut self,
        key: &str,
        node_id: NodeId,
        player: Player,
        history: &[ActionKind],
        actions: &[ActionEdge],
        oop_index: usize,
        ip_index: usize,
    ) -> Vec<f64> {
        if let Some(node) = self.nodes.get(key) {
            return node.strategy();
        }

        let combo = match player {
            Player::Oop => self.oop_range[oop_index].clone(),
            Player::Ip => self.ip_range[ip_index].clone(),
        };
        let oop_equity = self.combo_equity(oop_index, ip_index);
        let equity = match player {
            Player::Oop => oop_equity,
            Player::Ip => 1.0 - oop_equity,
        };
        let node = SubgameCfrNode {
            node: node_id,
            player,
            board: self.spec.board.clone(),
            history: history.to_vec(),
            combo,
            equity,
            actions: actions.iter().map(|edge| edge.action).collect(),
            regret_sum: vec![0.0; actions.len()],
            strategy_sum: vec![0.0; actions.len()],
        };
        let strategy = node.strategy();
        self.nodes.insert(key.to_string(), node);
        strategy
    }

    fn combo_equity(&mut self, oop_index: usize, ip_index: usize) -> f64 {
        let oop = &self.oop_range[oop_index];
        let ip = &self.ip_range[ip_index];
        let key = (board_mask(&self.spec.board), oop.mask, ip.mask);
        if let Some(&equity) = self.equity_cache.get(&key) {
            return equity;
        }
        let equity = combo_equity(&self.spec.board, oop, ip, self.spec.chance.max_runouts());
        self.equity_cache.insert(key, equity);
        equity
    }

    fn combo_strategies(&mut self) -> Vec<SubgameComboStrategy> {
        let mut strategies: Vec<_> = self
            .nodes
            .values()
            .map(|node| SubgameComboStrategy {
                node: node.node,
                player: node.player,
                board: node.board.clone(),
                history: node.history.clone(),
                combo: node.combo.clone(),
                equity: node.equity,
                actions: node.average_strategy(),
            })
            .collect();
        strategies.sort_by(|left, right| {
            left.node
                .0
                .cmp(&right.node.0)
                .then_with(|| player_key(left.player).cmp(&player_key(right.player)))
                .then_with(|| right.equity.total_cmp(&left.equity))
                .then_with(|| left.combo.label().cmp(&right.combo.label()))
        });
        strategies
    }
}

impl SubgameCfrNode {
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
            vec![1.0 / self.regret_sum.len() as f64; self.regret_sum.len()]
        }
    }

    fn average_strategy(&self) -> Vec<SubgameActionFrequency> {
        let normalizer: f64 = self.strategy_sum.iter().sum();
        self.actions
            .iter()
            .enumerate()
            .map(|(index, &action)| {
                let frequency = if normalizer > 0.0 {
                    self.strategy_sum[index] / normalizer
                } else {
                    1.0 / self.strategy_sum.len() as f64
                };
                SubgameActionFrequency { action, frequency }
            })
            .collect()
    }
}

fn legal_deals(oop_range: &[PostflopCombo], ip_range: &[PostflopCombo]) -> Vec<(usize, usize)> {
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

fn combo_equity(
    board: &[Card],
    oop: &PostflopCombo,
    ip: &PostflopCombo,
    max_runouts: usize,
) -> f64 {
    if oop.mask & ip.mask != 0 {
        return 0.0;
    }
    let mut win = 0.0;
    let mut tie = 0.0;
    let mut total = 0.0;
    for completed_board in
        sampled_runouts(board, board_mask(board) | oop.mask | ip.mask, max_runouts)
    {
        let oop_value = evaluate_seven([
            oop.combo.first,
            oop.combo.second,
            completed_board[0],
            completed_board[1],
            completed_board[2],
            completed_board[3],
            completed_board[4],
        ]);
        let ip_value = evaluate_seven([
            ip.combo.first,
            ip.combo.second,
            completed_board[0],
            completed_board[1],
            completed_board[2],
            completed_board[3],
            completed_board[4],
        ]);
        match oop_value.cmp(&ip_value) {
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
            for &river in &available {
                let mut completed = [Card(0); 5];
                completed[..board.len()].copy_from_slice(board);
                completed[board.len()] = river;
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
    (0..max_runouts)
        .map(|iteration| runouts[sampled_index(iteration, runouts.len())])
        .collect()
}

fn sampled_index(iteration: usize, len: usize) -> usize {
    let mut value = iteration as u64 + 0x517c_c1b7_2722_0a95;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) as usize) % len
}

fn board_mask(board: &[Card]) -> u64 {
    board.iter().fold(0_u64, |mask, card| mask | card.mask())
}

fn board_label(board: &[Card]) -> String {
    board.iter().map(|card| format!("{:02}", card.0)).collect()
}

fn history_label(history: &[ActionKind]) -> String {
    history
        .iter()
        .map(|action| match action {
            ActionKind::Check => "x".to_string(),
            ActionKind::Fold => "f".to_string(),
            ActionKind::Call => "c".to_string(),
            ActionKind::Bet(amount) => format!("b{amount:.2}"),
            ActionKind::Raise(amount) => format!("r{amount:.2}"),
            ActionKind::AllIn(amount) => format!("a{amount:.2}"),
        })
        .collect::<Vec<_>>()
        .join("-")
}

fn player_key(player: Player) -> u8 {
    match player {
        Player::Oop => 0,
        Player::Ip => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postflop::parse_range;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn rejects_duplicate_board_cards() {
        let spec = SubgameSpec {
            board: vec![c(14, 0), c(14, 0), c(2, 1)],
            street: Street::Flop,
            pot: PotState::new(100.0, [1_000.0, 1_000.0]),
            ranges: RangeState::new(parse_range("AA").unwrap(), parse_range("KK").unwrap()),
            actions: ActionAbstraction::default(),
            chance: ChancePolicy::Sample(8),
        };

        assert_eq!(
            spec.validate(),
            Err(SubgameValidationError::DuplicateBoardCard)
        );
    }

    #[test]
    fn builds_action_tree_from_abstraction() {
        let spec = SubgameSpec::postflop(
            vec![c(14, 0), c(13, 1), c(2, 2)],
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(
                parse_range("AA,AKs").unwrap(),
                parse_range("QQ,AQs").unwrap(),
            ),
            ActionAbstraction {
                bet_sizes: vec![BetSize::PotFraction(0.5), BetSize::PotFraction(1.0)],
                raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
                reraise_sizes: vec![BetSize::CurrentBetMultiple(2.0)],
                allow_all_in: true,
                max_raises: 2,
            },
            ChancePolicy::Sample(8),
        )
        .unwrap();

        let tree = spec.build_action_tree().unwrap();
        let ActionTreeNode::Decision {
            player, actions, ..
        } = &tree.nodes[tree.root.0]
        else {
            panic!("root must be a decision");
        };

        assert_eq!(*player, Player::Oop);
        assert!(actions.iter().any(|edge| edge.action == ActionKind::Check));
        assert!(
            actions
                .iter()
                .any(|edge| edge.action == ActionKind::Bet(50.0))
        );
        assert!(
            actions
                .iter()
                .any(|edge| edge.action == ActionKind::Bet(100.0))
        );
        assert!(tree.nodes.iter().any(|node| matches!(
            node,
            ActionTreeNode::Decision { actions, .. }
                if actions.iter().any(|edge| edge.action == ActionKind::Raise(250.0))
        )));
        assert!(tree.nodes.len() > actions.len());
    }

    #[test]
    fn solves_cfr_on_generated_action_tree() {
        let spec = SubgameSpec::postflop(
            vec![c(14, 0), c(13, 1), c(2, 2)],
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(
                parse_range("AA,AKs,AKo").unwrap(),
                parse_range("QQ,JJ,AQs").unwrap(),
            ),
            ActionAbstraction {
                bet_sizes: vec![BetSize::PotFraction(0.5), BetSize::PotFraction(1.0)],
                raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
                reraise_sizes: vec![BetSize::CurrentBetMultiple(2.0)],
                allow_all_in: false,
                max_raises: 2,
            },
            ChancePolicy::Sample(8),
        )
        .unwrap();

        let result = spec.solve_cfr(500).unwrap();

        assert_eq!(result.iterations, 500);
        assert!(result.node_count > 4);
        assert!(!result.strategies.is_empty());
        assert!(result.strategies.iter().any(|strategy| {
            strategy
                .actions
                .iter()
                .any(|action| action.action == ActionKind::Bet(50.0))
        }));
        assert!(result.strategies.iter().any(|strategy| {
            strategy
                .actions
                .iter()
                .any(|action| action.action == ActionKind::Raise(250.0))
        }));
    }
}
