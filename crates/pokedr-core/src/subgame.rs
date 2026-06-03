use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::cards::{Card, deck};
use crate::hand_class::HandClass;
use crate::hand_eval::evaluate_seven;
use crate::postflop::PostflopCombo;
use crate::river::Combo;
use rayon::prelude::*;

type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.0;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        self.0 = hash;
    }

    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0 ^ value).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

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
    pub oop: Vec<RangeEntry>,
    pub ip: Vec<RangeEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct RangeEntry {
    pub class: HandClass,
    pub combo: Option<Combo>,
    pub weight: f64,
}

impl RangeState {
    pub fn new(oop: Vec<HandClass>, ip: Vec<HandClass>) -> Self {
        Self {
            oop: uniform_range(oop),
            ip: uniform_range(ip),
        }
    }

    pub fn weighted(oop: Vec<(HandClass, f64)>, ip: Vec<(HandClass, f64)>) -> Self {
        Self {
            oop: weighted_range(oop),
            ip: weighted_range(ip),
        }
    }

    pub fn weighted_combos(oop: Vec<(Combo, f64)>, ip: Vec<(Combo, f64)>) -> Self {
        Self {
            oop: weighted_combo_range(oop),
            ip: weighted_combo_range(ip),
        }
    }
}

fn uniform_range(classes: Vec<HandClass>) -> Vec<RangeEntry> {
    classes
        .into_iter()
        .map(|class| RangeEntry {
            class,
            combo: None,
            weight: 1.0,
        })
        .collect()
}

fn weighted_range(entries: Vec<(HandClass, f64)>) -> Vec<RangeEntry> {
    entries
        .into_iter()
        .filter_map(|(class, weight)| {
            if weight.is_finite() && weight > 0.0 {
                Some(RangeEntry {
                    class,
                    combo: None,
                    weight,
                })
            } else {
                None
            }
        })
        .collect()
}

fn weighted_combo_range(entries: Vec<(Combo, f64)>) -> Vec<RangeEntry> {
    entries
        .into_iter()
        .filter_map(|(combo, weight)| {
            if weight.is_finite() && weight > 0.0 {
                Some(RangeEntry {
                    class: combo_class(combo),
                    combo: Some(combo),
                    weight,
                })
            } else {
                None
            }
        })
        .collect()
}

fn combo_class(combo: Combo) -> HandClass {
    let first_rank = combo.first.rank();
    let second_rank = combo.second.rank();
    HandClass::new(
        first_rank.max(second_rank),
        first_rank.min(second_rank),
        combo.first.suit() == combo.second.suit() && first_rank != second_rank,
    )
}

fn weighted_postflop_combos(entries: &[RangeEntry], board: &[Card]) -> Vec<WeightedPostflopCombo> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let mut combos = Vec::new();
    let mut seen = HashSet::new();

    for entry in entries {
        if !(entry.weight.is_finite() && entry.weight > 0.0) {
            continue;
        }
        if let Some(combo) = entry.combo {
            push_weighted_combo(
                &mut combos,
                &mut seen,
                board_mask,
                combo,
                entry.class,
                entry.weight,
            );
        } else {
            for [first, second] in entry.class.combos() {
                let Some(combo) = Combo::new(first, second) else {
                    continue;
                };
                push_weighted_combo(
                    &mut combos,
                    &mut seen,
                    board_mask,
                    combo,
                    entry.class,
                    entry.weight,
                );
            }
        }
    }

    combos
}

fn push_weighted_combo(
    combos: &mut Vec<WeightedPostflopCombo>,
    seen: &mut HashSet<u64>,
    board_mask: u64,
    combo: Combo,
    class: HandClass,
    weight: f64,
) {
    let mask = combo.mask();
    if mask & board_mask != 0 || !seen.insert(mask) {
        return;
    }
    combos.push(WeightedPostflopCombo {
        combo: PostflopCombo { combo, class, mask },
        weight,
    });
}

#[derive(Debug, Clone, Copy)]
pub struct PotState {
    pub pot: f64,
    pub stacks: [f64; 2],
    pub committed: [f64; 2],
    pub invested: [f64; 2],
    pub current_bet: f64,
    pub min_raise: f64,
}

impl PotState {
    pub fn new(pot: f64, stacks: [f64; 2]) -> Self {
        Self {
            pot,
            stacks,
            committed: [0.0, 0.0],
            invested: [0.0, 0.0],
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
    pub root_player: Player,
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
            root_player: Player::Oop,
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
        let root = builder.build_decision(self.root_player, self.pot, 0, Vec::new())?;
        Ok(ActionTree {
            root,
            nodes: builder.nodes,
        })
    }

    pub fn with_root_player(mut self, root_player: Player) -> Self {
        self.root_player = root_player;
        self
    }

    pub fn solve_cfr(&self, iterations: usize) -> Result<SubgameCfrResult, SubgameBuildError> {
        self.solve_cfr_with_request(SubgameSolveRequest {
            iterations,
            focused_oop_combo_mask: None,
            focused_ip_combo_mask: None,
            focused_sampling_rate: 0.0,
            variant: CfrVariant::default(),
        })
    }

    pub fn solve_cfr_with_request(
        &self,
        request: SubgameSolveRequest,
    ) -> Result<SubgameCfrResult, SubgameBuildError> {
        let tree = self.build_action_tree()?;
        let oop_range = weighted_postflop_combos(&self.ranges.oop, &self.board);
        let ip_range = weighted_postflop_combos(&self.ranges.ip, &self.board);
        if oop_range.is_empty() || ip_range.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }

        let deals = legal_deals(&oop_range, &ip_range);
        if deals.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }
        let focused_deals = request
            .focused_oop_combo_mask
            .or(request.focused_ip_combo_mask)
            .map(|_| {
                focused_deals(
                    &oop_range,
                    &ip_range,
                    &deals,
                    request.focused_oop_combo_mask,
                    request.focused_ip_combo_mask,
                )
            })
            .unwrap_or_default();
        let deal_equity = precompute_sampled_deal_equity(
            &self.board,
            &oop_range,
            &ip_range,
            &deals,
            &focused_deals,
            request.iterations,
            request.focused_sampling_rate,
            self.chance.max_runouts(),
        );

        let mut trainer = SubgameCfrTrainer {
            spec: self,
            tree: &tree,
            oop_range: &oop_range,
            ip_range: &ip_range,
            nodes: HashMap::new(),
            deal_equity,
            average_weight: 1.0,
            iteration: 1,
            variant: request.variant,
        };
        let mut utility_sum = 0.0;
        for iteration in 0..request.iterations {
            trainer.iteration = iteration + 1;
            trainer.average_weight = average_strategy_weight(request.variant, iteration + 1);
            let focused = !focused_deals.is_empty()
                && should_sample_focused(iteration, request.focused_sampling_rate);
            let deal_pool = if focused { &focused_deals } else { &deals };
            let (oop_index, ip_index) = sample_weighted_deal(iteration, deal_pool);
            utility_sum += trainer.cfr(tree.root, oop_index, ip_index, [1.0, 1.0]);
        }

        Ok(SubgameCfrResult {
            iterations: request.iterations,
            expected_value_oop: if request.iterations == 0 {
                0.0
            } else {
                utility_sum / request.iterations as f64
            },
            root: tree.root,
            node_count: tree.nodes.len(),
            oop_combo_count: oop_range.len(),
            ip_combo_count: ip_range.len(),
            strategies: trainer.combo_strategies(),
        })
    }

    pub fn solve_multistreet_cfr_with_request(
        &self,
        request: SubgameSolveRequest,
    ) -> Result<SubgameCfrResult, SubgameBuildError> {
        self.validate()?;
        let oop_range = weighted_postflop_combos(&self.ranges.oop, &self.board);
        let ip_range = weighted_postflop_combos(&self.ranges.ip, &self.board);
        if oop_range.is_empty() || ip_range.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }

        let deals = legal_deals(&oop_range, &ip_range);
        if deals.is_empty() {
            return Err(SubgameBuildError::NoLegalDeals);
        }
        let focused_deals = request
            .focused_oop_combo_mask
            .or(request.focused_ip_combo_mask)
            .map(|_| {
                focused_deals(
                    &oop_range,
                    &ip_range,
                    &deals,
                    request.focused_oop_combo_mask,
                    request.focused_ip_combo_mask,
                )
            })
            .unwrap_or_default();
        let shard_count = rayon::current_num_threads().min(request.iterations.max(1));
        let shard_size = request.iterations.div_ceil(shard_count);
        let (trainer, utility_sum) = (0..shard_count)
            .into_par_iter()
            .map(|shard| {
                let start = shard * shard_size;
                let end = ((shard + 1) * shard_size).min(request.iterations);
                let mut trainer = MultistreetCfrTrainer {
                    spec: self,
                    oop_range: &oop_range,
                    ip_range: &ip_range,
                    node_index: FastHashMap::default(),
                    nodes: Vec::new(),
                    equity_cache: FastHashMap::default(),
                    average_weight: 1.0,
                    iteration: 1,
                    variant: request.variant,
                };
                let mut utility_sum = 0.0;
                for iteration in start..end {
                    trainer.iteration = iteration + 1;
                    trainer.average_weight = average_strategy_weight(request.variant, iteration + 1);
                    let focused = !focused_deals.is_empty()
                        && should_sample_focused(iteration, request.focused_sampling_rate);
                    let deal_pool = if focused { &focused_deals } else { &deals };
                    let (oop_index, ip_index) = sample_weighted_deal(iteration, deal_pool);
                    utility_sum += trainer.cfr(
                        self.root_player,
                        self.board.clone(),
                        self.pot,
                        0,
                        ActionHistory::new(),
                        oop_index,
                        ip_index,
                        [1.0, 1.0],
                    );
                }
                (trainer, utility_sum)
            })
            .reduce(
                || {
                    let trainer = MultistreetCfrTrainer {
                        spec: self,
                        oop_range: &oop_range,
                        ip_range: &ip_range,
                        node_index: FastHashMap::default(),
                        nodes: Vec::new(),
                        equity_cache: FastHashMap::default(),
                        average_weight: 1.0,
                        iteration: request.iterations,
                        variant: request.variant,
                    };
                    (trainer, 0.0)
                },
                |(mut left, left_utility), (right, right_utility)| {
                    left.merge_from(right);
                    (left, left_utility + right_utility)
                },
            );

        Ok(SubgameCfrResult {
            iterations: request.iterations,
            expected_value_oop: if request.iterations == 0 {
                0.0
            } else {
                utility_sum / request.iterations as f64
            },
            root: NodeId(0),
            node_count: trainer.current_board_node_count(),
            oop_combo_count: oop_range.len(),
            ip_combo_count: ip_range.len(),
            strategies: trainer.combo_strategies_for_board(&self.board),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubgameSolveRequest {
    pub iterations: usize,
    pub focused_oop_combo_mask: Option<u64>,
    pub focused_ip_combo_mask: Option<u64>,
    pub focused_sampling_rate: f64,
    pub variant: CfrVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfrVariant {
    CfrPlus,
    DcfrPlus,
}

impl Default for CfrVariant {
    fn default() -> Self {
        Self::CfrPlus
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
    pot.invested[index] += contribution;
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
    pot.invested[index] += contribution;
    pot
}

fn advance_street(mut pot: PotState) -> PotState {
    pot.committed = [0.0, 0.0];
    pot.current_bet = 0.0;
    pot.min_raise = 0.0;
    pot
}

struct SubgameCfrTrainer<'a> {
    spec: &'a SubgameSpec,
    tree: &'a ActionTree,
    oop_range: &'a [WeightedPostflopCombo],
    ip_range: &'a [WeightedPostflopCombo],
    nodes: HashMap<SubgameInfoKey, SubgameCfrNode>,
    deal_equity: HashMap<(usize, usize), f64>,
    average_weight: f64,
    iteration: usize,
    variant: CfrVariant,
}

#[derive(Debug, Clone)]
struct WeightedPostflopCombo {
    combo: PostflopCombo,
    weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SubgameInfoKey {
    node: NodeId,
    player: Player,
    combo_mask: u64,
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

struct MultistreetCfrTrainer<'a> {
    spec: &'a SubgameSpec,
    oop_range: &'a [WeightedPostflopCombo],
    ip_range: &'a [WeightedPostflopCombo],
    node_index: FastHashMap<MultistreetInfoKey, usize>,
    nodes: Vec<MultistreetCfrNode>,
    equity_cache: FastHashMap<(u64, usize, usize), f64>,
    average_weight: f64,
    iteration: usize,
    variant: CfrVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HistoryKey {
    len: u8,
    code: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MultistreetInfoKey {
    board_mask: u64,
    history: HistoryKey,
    player: Player,
    combo_mask: u64,
}

impl Hash for MultistreetInfoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.board_mask);
        state.write_u8(self.history.len);
        state.write_u64(self.history.code);
        state.write_u8(self.player.index() as u8);
        state.write_u64(self.combo_mask);
    }
}

#[derive(Debug, Clone)]
struct MultistreetCfrNode {
    board: Vec<Card>,
    history: Vec<ActionKind>,
    player: Player,
    combo: PostflopCombo,
    equity: f64,
    actions: Vec<ActionKind>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

struct ActionList {
    actions: [ActionKind; 8],
    len: usize,
}

impl ActionList {
    fn new() -> Self {
        Self {
            actions: [ActionKind::Check; 8],
            len: 0,
        }
    }

    fn push(&mut self, action: ActionKind) {
        if self.len < self.actions.len() {
            self.actions[self.len] = action;
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[ActionKind] {
        &self.actions[..self.len]
    }
}

#[derive(Clone, Copy)]
struct ActionHistory {
    actions: [ActionKind; 8],
    len: usize,
    code: u64,
}

impl ActionHistory {
    fn new() -> Self {
        Self {
            actions: [ActionKind::Check; 8],
            len: 0,
            code: 0,
        }
    }

    fn push(&mut self, action: ActionKind, action_index: usize) {
        if self.len < self.actions.len() {
            self.actions[self.len] = action;
            self.code |= ((action_index as u64) + 1) << (self.len * 4);
            self.len += 1;
        }
    }

    fn as_slice(&self) -> &[ActionKind] {
        &self.actions[..self.len]
    }

    fn to_vec(self) -> Vec<ActionKind> {
        self.as_slice().to_vec()
    }

    fn checked_through(self) -> bool {
        self.len >= 2
            && matches!(
                self.actions[self.len - 2..self.len],
                [ActionKind::Check, ActionKind::Check]
            )
    }
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
                let key = self.infoset_key(node_id, *player, oop_index, ip_index);
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
                    let instant_regret = opponent_reach * regret;
                    node.update_regret(action, instant_regret, self.iteration, self.variant);
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
                let oop_added = pot.invested[Player::Oop.index()]
                    - self.spec.pot.invested[Player::Oop.index()];
                if winner == Player::Oop {
                    pot.pot - oop_added
                } else {
                    -oop_added
                }
            }
            TerminalKind::Showdown => {
                let equity = self.combo_equity(oop_index, ip_index);
                let oop_added = pot.invested[Player::Oop.index()]
                    - self.spec.pot.invested[Player::Oop.index()];
                equity * pot.pot - oop_added
            }
        }
    }

    fn infoset_key(
        &self,
        node_id: NodeId,
        player: Player,
        oop_index: usize,
        ip_index: usize,
    ) -> SubgameInfoKey {
        let combo_mask = match player {
            Player::Oop => self.oop_range[oop_index].combo.mask,
            Player::Ip => self.ip_range[ip_index].combo.mask,
        };
        SubgameInfoKey {
            node: node_id,
            player,
            combo_mask,
        }
    }

    fn strategy_for(
        &mut self,
        key: &SubgameInfoKey,
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
            Player::Oop => self.oop_range[oop_index].combo.clone(),
            Player::Ip => self.ip_range[ip_index].combo.clone(),
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
        self.nodes.insert(*key, node);
        strategy
    }

    fn combo_equity(&self, oop_index: usize, ip_index: usize) -> f64 {
        self.deal_equity
            .get(&(oop_index, ip_index))
            .copied()
            .unwrap_or_else(|| {
                combo_equity(
                    &self.spec.board,
                    &self.oop_range[oop_index],
                    &self.ip_range[ip_index],
                    self.spec.chance.max_runouts(),
                )
            })
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

impl MultistreetCfrTrainer<'_> {
    fn cfr(
        &mut self,
        player: Player,
        board: Vec<Card>,
        pot: PotState,
        raises: u8,
        history: ActionHistory,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        let player_index = player.index();
        let actions = self.legal_actions(player, pot, raises);
        let key = self.infoset_key(&board, history, player, oop_index, ip_index);
        let (strategy, node_index) = self.strategy_for(
            &key,
            player,
            board.as_slice(),
            history,
            actions.as_slice(),
            oop_index,
            ip_index,
        );
        let mut action_values = [0.0; 8];
        let mut node_value = 0.0;

        for (index, action) in actions.as_slice().iter().enumerate() {
            let mut next_reach = reach;
            next_reach[player_index] *= strategy[index];
            action_values[index] = self.apply_action(
                player,
                board.clone(),
                pot,
                raises,
                history,
                *action,
                index,
                oop_index,
                ip_index,
                next_reach,
            );
            node_value += strategy[index] * action_values[index];
        }

        let node = &mut self.nodes[node_index];
        let opponent_reach = reach[player.other().index()];
        for action in 0..actions.len {
            let regret = if player == Player::Oop {
                action_values[action] - node_value
            } else {
                node_value - action_values[action]
            };
            node.update_regret(action, opponent_reach * regret, self.iteration, self.variant);
            node.strategy_sum[action] +=
                self.average_weight * reach[player_index] * strategy[action];
        }

        node_value
    }

    fn apply_action(
        &mut self,
        player: Player,
        board: Vec<Card>,
        pot: PotState,
        raises: u8,
        mut history: ActionHistory,
        action: ActionKind,
        action_index: usize,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        history.push(action, action_index);
        match action {
            ActionKind::Check if history.checked_through() => {
                self.finish_betting_round(board, pot, oop_index, ip_index, reach)
            }
            ActionKind::Call => {
                let next_pot = apply_call(player, pot);
                self.finish_betting_round(board, next_pot, oop_index, ip_index, reach)
            }
            ActionKind::Fold => self.fold_utility(player.other(), pot),
            ActionKind::Check => self.cfr(
                player.other(),
                board,
                pot,
                raises,
                history,
                oop_index,
                ip_index,
                reach,
            ),
            ActionKind::Bet(amount) => self.cfr(
                player.other(),
                board,
                apply_aggressive(player, pot, amount),
                raises,
                history,
                oop_index,
                ip_index,
                reach,
            ),
            ActionKind::Raise(amount) | ActionKind::AllIn(amount) => self.cfr(
                player.other(),
                board,
                apply_aggressive(player, pot, amount),
                raises + 1,
                history,
                oop_index,
                ip_index,
                reach,
            ),
        }
    }

    fn finish_betting_round(
        &mut self,
        board: Vec<Card>,
        pot: PotState,
        oop_index: usize,
        ip_index: usize,
        reach: [f64; 2],
    ) -> f64 {
        if board.len() == 5 {
            return self.showdown_utility(board.as_slice(), pot, oop_index, ip_index);
        }

        let next_boards: Vec<_> = self
            .next_public_boards(board.as_slice())
            .into_iter()
            .filter(|next_board| self.board_legal_for_deal(next_board, oop_index, ip_index))
            .collect();
        if next_boards.is_empty() {
            return self.showdown_utility(board.as_slice(), pot, oop_index, ip_index);
        }
        let next_board_count = next_boards.len();
        let chance_probability = 1.0 / next_board_count as f64;
        let next_pot = advance_street(pot);
        next_boards
            .into_iter()
            .map(|next_board| {
                let next_reach = [
                    reach[Player::Oop.index()] * chance_probability,
                    reach[Player::Ip.index()] * chance_probability,
                ];
                self.cfr(
                    Player::Oop,
                    next_board,
                    next_pot,
                    0,
                    ActionHistory::new(),
                    oop_index,
                    ip_index,
                    next_reach,
                )
            })
            .sum::<f64>()
            * chance_probability
    }

    fn next_public_boards(&self, board: &[Card]) -> Vec<Vec<Card>> {
        let used_mask = board_mask(board);
        let available: Vec<_> = deck()
            .into_iter()
            .filter(|card| used_mask & card.mask() == 0)
            .collect();
        let limit = self.spec.chance.max_runouts().max(1).min(available.len());
        let mut boards = Vec::with_capacity(limit);
        let mut seen = HashSet::new();
        for sample in 0..available.len() {
            if boards.len() == limit {
                break;
            }
            let card = available[sampled_index(self.iteration + sample + board.len(), available.len())];
            if seen.insert(card) {
                let mut next = board.to_vec();
                next.push(card);
                boards.push(next);
            }
        }
        boards
    }

    fn board_legal_for_deal(&self, board: &[Card], oop_index: usize, ip_index: usize) -> bool {
        let public_mask = board_mask(board);
        public_mask & self.oop_range[oop_index].combo.mask == 0
            && public_mask & self.ip_range[ip_index].combo.mask == 0
    }

    fn legal_actions(&self, player: Player, pot: PotState, raises: u8) -> ActionList {
        if pot.current_bet <= pot.committed[player.index()] {
            let mut actions = ActionList::new();
            actions.push(ActionKind::Check);
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
            return actions;
        }

        let mut actions = ActionList::new();
        actions.push(ActionKind::Fold);
        actions.push(ActionKind::Call);
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
        actions
    }

    fn infoset_key(
        &self,
        board: &[Card],
        history: ActionHistory,
        player: Player,
        oop_index: usize,
        ip_index: usize,
    ) -> MultistreetInfoKey {
        let combo_mask = match player {
            Player::Oop => self.oop_range[oop_index].combo.mask,
            Player::Ip => self.ip_range[ip_index].combo.mask,
        };
        MultistreetInfoKey {
            board_mask: board_mask(board),
            history: history_key(history),
            player,
            combo_mask,
        }
    }

    fn strategy_for(
        &mut self,
        key: &MultistreetInfoKey,
        player: Player,
        board: &[Card],
        history: ActionHistory,
        actions: &[ActionKind],
        oop_index: usize,
        ip_index: usize,
    ) -> ([f64; 8], usize) {
        if let Some(&index) = self.node_index.get(key) {
            return (self.nodes[index].strategy(), index);
        }
        let node = self.new_node(player, board, history, actions, oop_index, ip_index);
        let strategy = node.strategy();
        let index = self.nodes.len();
        self.nodes.push(node);
        self.node_index.insert(*key, index);
        (strategy, index)
    }

    fn new_node(
        &mut self,
        player: Player,
        board: &[Card],
        history: ActionHistory,
        actions: &[ActionKind],
        oop_index: usize,
        ip_index: usize,
    ) -> MultistreetCfrNode {
        let combo = match player {
            Player::Oop => self.oop_range[oop_index].combo.clone(),
            Player::Ip => self.ip_range[ip_index].combo.clone(),
        };
        let oop_equity = self.combo_equity_on_board(board, oop_index, ip_index);
        let equity = match player {
            Player::Oop => oop_equity,
            Player::Ip => 1.0 - oop_equity,
        };
        MultistreetCfrNode {
            board: board.to_vec(),
            history: history.to_vec(),
            player,
            combo,
            equity,
            actions: actions.to_vec(),
            regret_sum: vec![0.0; actions.len()],
            strategy_sum: vec![0.0; actions.len()],
        }
    }

    fn combo_equity_on_board(&mut self, board: &[Card], oop_index: usize, ip_index: usize) -> f64 {
        let key = (board_mask(board), oop_index, ip_index);
        if let Some(equity) = self.equity_cache.get(&key) {
            return *equity;
        }
        let equity = combo_equity(
            board,
            &self.oop_range[oop_index],
            &self.ip_range[ip_index],
            self.spec.chance.max_runouts(),
        );
        self.equity_cache.insert(key, equity);
        equity
    }

    fn fold_utility(&self, winner: Player, pot: PotState) -> f64 {
        let oop_added =
            pot.invested[Player::Oop.index()] - self.spec.pot.invested[Player::Oop.index()];
        if winner == Player::Oop {
            pot.pot - oop_added
        } else {
            -oop_added
        }
    }

    fn showdown_utility(
        &mut self,
        board: &[Card],
        pot: PotState,
        oop_index: usize,
        ip_index: usize,
    ) -> f64 {
        let equity = self.combo_equity_on_board(board, oop_index, ip_index);
        let oop_added =
            pot.invested[Player::Oop.index()] - self.spec.pot.invested[Player::Oop.index()];
        equity * pot.pot - oop_added
    }

    fn merge_from(&mut self, other: MultistreetCfrTrainer<'_>) {
        for (key, other_index) in other.node_index {
            let node_to_add = &other.nodes[other_index];
            if let Some(&index) = self.node_index.get(&key) {
                self.nodes[index].add(node_to_add);
            } else {
                let index = self.nodes.len();
                self.nodes.push(node_to_add.zeroed_like());
                self.nodes[index].add(node_to_add);
                self.node_index.insert(key, index);
            }
        }
    }

    fn current_board_node_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.board == self.spec.board)
            .count()
    }

    fn combo_strategies_for_board(&self, board: &[Card]) -> Vec<SubgameComboStrategy> {
        let mut strategies: Vec<_> = self
            .nodes
            .iter()
            .filter(|node| node.board == board)
            .map(|node| SubgameComboStrategy {
                node: NodeId(0),
                player: node.player,
                board: node.board.clone(),
                history: node.history.clone(),
                combo: node.combo.clone(),
                equity: node.equity,
                actions: node.average_strategy(),
            })
            .collect();
        strategies.sort_by(|left, right| {
            player_key(left.player)
                .cmp(&player_key(right.player))
                .then_with(|| left.history.len().cmp(&right.history.len()))
                .then_with(|| right.equity.total_cmp(&left.equity))
                .then_with(|| left.combo.label().cmp(&right.combo.label()))
        });
        strategies
    }
}

impl MultistreetCfrNode {
    fn update_regret(
        &mut self,
        action: usize,
        instant_regret: f64,
        iteration: usize,
        variant: CfrVariant,
    ) {
        let discounted =
            self.regret_sum[action] * regret_discount(self.regret_sum[action], iteration, variant);
        self.regret_sum[action] = (discounted + instant_regret).max(0.0);
    }

    fn strategy(&self) -> [f64; 8] {
        let mut strategy = [0.0; 8];
        let mut normalizer = 0.0;
        for regret in &self.regret_sum {
            normalizer += regret.max(0.0);
        }
        if normalizer > 0.0 {
            for (index, regret) in self.regret_sum.iter().enumerate() {
                strategy[index] = regret.max(0.0) / normalizer;
            }
        } else {
            let frequency = 1.0 / self.regret_sum.len() as f64;
            for value in strategy.iter_mut().take(self.regret_sum.len()) {
                *value = frequency;
            }
        }
        strategy
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

    fn zeroed_like(&self) -> Self {
        Self {
            board: self.board.clone(),
            history: self.history.clone(),
            player: self.player,
            combo: self.combo.clone(),
            equity: self.equity,
            actions: self.actions.clone(),
            regret_sum: vec![0.0; self.regret_sum.len()],
            strategy_sum: vec![0.0; self.strategy_sum.len()],
        }
    }

    fn add(&mut self, other: &Self) {
        for (left, right) in self.regret_sum.iter_mut().zip(other.regret_sum.iter()) {
            *left += right;
        }
        for (left, right) in self.strategy_sum.iter_mut().zip(other.strategy_sum.iter()) {
            *left += right;
        }
    }
}

fn history_key(history: ActionHistory) -> HistoryKey {
    HistoryKey {
        len: history.len as u8,
        code: history.code,
    }
}

impl SubgameCfrNode {
    fn update_regret(
        &mut self,
        action: usize,
        instant_regret: f64,
        iteration: usize,
        variant: CfrVariant,
    ) {
        let discounted =
            self.regret_sum[action] * regret_discount(self.regret_sum[action], iteration, variant);
        self.regret_sum[action] = (discounted + instant_regret).max(0.0);
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

fn regret_discount(regret: f64, iteration: usize, variant: CfrVariant) -> f64 {
    match variant {
        CfrVariant::CfrPlus => 1.0,
        CfrVariant::DcfrPlus => {
            let t = iteration as f64;
            let alpha = if regret >= 0.0 { 1.5 } else { 0.0 };
            t.powf(alpha) / (t.powf(alpha) + 1.0)
        }
    }
}

fn average_strategy_weight(variant: CfrVariant, iteration: usize) -> f64 {
    match variant {
        CfrVariant::CfrPlus => iteration as f64,
        CfrVariant::DcfrPlus => {
            let t = iteration as f64;
            t * t
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LegalDeal {
    oop_index: usize,
    ip_index: usize,
    weight: f64,
    cumulative_weight: f64,
}

fn legal_deals(
    oop_range: &[WeightedPostflopCombo],
    ip_range: &[WeightedPostflopCombo],
) -> Vec<LegalDeal> {
    let mut deals = Vec::new();
    for oop_index in 0..oop_range.len() {
        for ip_index in 0..ip_range.len() {
            if oop_range[oop_index].combo.mask & ip_range[ip_index].combo.mask == 0 {
                deals.push(LegalDeal {
                    oop_index,
                    ip_index,
                    weight: oop_range[oop_index].weight * ip_range[ip_index].weight,
                    cumulative_weight: 0.0,
                });
            }
        }
    }
    with_cumulative_weights(deals)
}

fn precompute_sampled_deal_equity(
    board: &[Card],
    oop_range: &[WeightedPostflopCombo],
    ip_range: &[WeightedPostflopCombo],
    deals: &[LegalDeal],
    focused_deals: &[LegalDeal],
    iterations: usize,
    focused_sampling_rate: f64,
    max_runouts: usize,
) -> HashMap<(usize, usize), f64> {
    let sampled_pairs = sampled_deal_pairs(iterations, deals, focused_deals, focused_sampling_rate);
    sampled_pairs
        .par_iter()
        .map(|&(oop_index, ip_index)| {
            let equity = combo_equity(
                board,
                &oop_range[oop_index],
                &ip_range[ip_index],
                max_runouts,
            );
            ((oop_index, ip_index), equity)
        })
        .collect()
}

fn sampled_deal_pairs(
    iterations: usize,
    deals: &[LegalDeal],
    focused_deals: &[LegalDeal],
    focused_sampling_rate: f64,
) -> Vec<(usize, usize)> {
    let mut seen = HashSet::new();
    let mut sampled = Vec::new();
    for iteration in 0..iterations {
        let focused =
            !focused_deals.is_empty() && should_sample_focused(iteration, focused_sampling_rate);
        let deal_pool = if focused { focused_deals } else { deals };
        let pair = sample_weighted_deal(iteration, deal_pool);
        if seen.insert(pair) {
            sampled.push(pair);
        }
    }
    sampled
}

fn focused_deals(
    oop_range: &[WeightedPostflopCombo],
    ip_range: &[WeightedPostflopCombo],
    deals: &[LegalDeal],
    focused_oop_mask: Option<u64>,
    focused_ip_mask: Option<u64>,
) -> Vec<LegalDeal> {
    let focused = deals
        .iter()
        .copied()
        .filter(|deal| {
            focused_oop_mask
                .map(|mask| oop_range[deal.oop_index].combo.mask == mask)
                .unwrap_or(true)
                && focused_ip_mask
                    .map(|mask| ip_range[deal.ip_index].combo.mask == mask)
                    .unwrap_or(true)
        })
        .map(|deal| LegalDeal {
            cumulative_weight: 0.0,
            ..deal
        })
        .collect();
    with_cumulative_weights(focused)
}

fn with_cumulative_weights(mut deals: Vec<LegalDeal>) -> Vec<LegalDeal> {
    let mut cumulative = 0.0;
    for deal in &mut deals {
        cumulative += deal.weight.max(0.0);
        deal.cumulative_weight = cumulative;
    }
    deals
}

fn sample_weighted_deal(iteration: usize, deals: &[LegalDeal]) -> (usize, usize) {
    let total_weight = deals
        .last()
        .map(|deal| deal.cumulative_weight)
        .unwrap_or(0.0);
    if total_weight <= 0.0 {
        let deal = deals[sampled_index(iteration, deals.len())];
        return (deal.oop_index, deal.ip_index);
    }

    let threshold = deterministic_unit(iteration) * total_weight;
    let mut low = 0;
    let mut high = deals.len();
    while low < high {
        let mid = (low + high) / 2;
        if deals[mid].cumulative_weight < threshold {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let deal = deals[low.min(deals.len() - 1)];
    (deal.oop_index, deal.ip_index)
}

fn should_sample_focused(iteration: usize, focused_sampling_rate: f64) -> bool {
    if focused_sampling_rate <= 0.0 {
        return false;
    }
    if focused_sampling_rate >= 1.0 {
        return true;
    }
    let threshold = (focused_sampling_rate * 10_000.0).round() as usize;
    sampled_index(iteration, 10_000) < threshold
}

fn combo_equity(
    board: &[Card],
    oop: &WeightedPostflopCombo,
    ip: &WeightedPostflopCombo,
    max_runouts: usize,
) -> f64 {
    if oop.combo.mask & ip.combo.mask != 0 {
        return 0.0;
    }
    let mut win = 0.0;
    let mut tie = 0.0;
    let mut total = 0.0;
    for completed_board in sampled_runouts(
        board,
        board_mask(board) | oop.combo.mask | ip.combo.mask,
        max_runouts,
    ) {
        let oop_value = evaluate_seven([
            oop.combo.combo.first,
            oop.combo.combo.second,
            completed_board[0],
            completed_board[1],
            completed_board[2],
            completed_board[3],
            completed_board[4],
        ]);
        let ip_value = evaluate_seven([
            ip.combo.combo.first,
            ip.combo.combo.second,
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
    (deterministic_u64(iteration) as usize) % len
}

fn deterministic_unit(iteration: usize) -> f64 {
    deterministic_u64(iteration) as f64 / u64::MAX as f64
}

fn deterministic_u64(iteration: usize) -> u64 {
    let mut value = iteration as u64 + 0x517c_c1b7_2722_0a95;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn board_mask(board: &[Card]) -> u64 {
    board.iter().fold(0_u64, |mask, card| mask | card.mask())
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
    use crate::postflop::{parse_range, postflop_combos};

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn rejects_duplicate_board_cards() {
        let spec = SubgameSpec {
            board: vec![c(14, 0), c(14, 0), c(2, 1)],
            street: Street::Flop,
            root_player: Player::Oop,
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
    fn action_tree_uses_configured_root_player() {
        let spec = SubgameSpec::postflop(
            vec![c(14, 0), c(13, 1), c(2, 2)],
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(
                parse_range("AA,AKs").unwrap(),
                parse_range("QQ,AQs").unwrap(),
            ),
            ActionAbstraction::default(),
            ChancePolicy::Sample(8),
        )
        .unwrap()
        .with_root_player(Player::Ip);

        let tree = spec.build_action_tree().unwrap();
        let ActionTreeNode::Decision { player, .. } = &tree.nodes[tree.root.0] else {
            panic!("root must be a decision");
        };

        assert_eq!(*player, Player::Ip);
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

    #[test]
    fn focused_sampling_visits_requested_oop_combo() {
        let board = vec![c(14, 0), c(13, 1), c(2, 2)];
        let spec = SubgameSpec::postflop(
            board,
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(
                parse_range("AA,AKs,AKo,AQs").unwrap(),
                parse_range("QQ,JJ,AQs,KQs,QJs").unwrap(),
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
        let focused_mask = Card::new(14, 1).mask() | Card::new(12, 1).mask();

        let result = spec
            .solve_cfr_with_request(SubgameSolveRequest {
                iterations: 8,
                focused_oop_combo_mask: Some(focused_mask),
                focused_ip_combo_mask: None,
                focused_sampling_rate: 1.0,
                variant: CfrVariant::default(),
            })
            .unwrap();

        assert!(result.strategies.iter().any(|strategy| {
            strategy.node == result.root
                && strategy.player == Player::Oop
                && strategy.combo.mask == focused_mask
        }));
    }

    #[test]
    fn focused_sampling_visits_requested_ip_combo() {
        let board = vec![c(14, 0), c(13, 1), c(2, 2)];
        let ip_classes = parse_range("QQ,JJ,AQs,KQs,QJs").unwrap();
        let focused_mask = postflop_combos(&ip_classes, &board)[0].mask;
        let spec = SubgameSpec::postflop(
            board,
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(parse_range("AA,AKs,AKo,AQs").unwrap(), ip_classes),
            ActionAbstraction::default(),
            ChancePolicy::Sample(8),
        )
        .unwrap()
        .with_root_player(Player::Ip);

        let result = spec
            .solve_cfr_with_request(SubgameSolveRequest {
                iterations: 8,
                focused_oop_combo_mask: None,
                focused_ip_combo_mask: Some(focused_mask),
                focused_sampling_rate: 1.0,
                variant: CfrVariant::default(),
            })
            .unwrap();

        assert!(result.strategies.iter().any(|strategy| {
            strategy.node == result.root
                && strategy.player == Player::Ip
                && strategy.combo.mask == focused_mask
        }));
    }

    #[test]
    fn weighted_range_entries_bias_chance_sampling() {
        let board = vec![c(2, 0), c(7, 1), c(9, 2)];
        let aa = HandClass::new(14, 14, false);
        let t3o = HandClass::new(10, 3, false);
        let oop_range =
            weighted_postflop_combos(&weighted_range(vec![(aa, 100.0), (t3o, 1.0)]), &board);
        let ip_range = weighted_postflop_combos(
            &weighted_range(vec![(HandClass::new(5, 4, true), 1.0)]),
            &board,
        );
        let deals = legal_deals(&oop_range, &ip_range);

        let mut aa_samples = 0;
        let mut t3o_samples = 0;
        for iteration in 0..128 {
            let (oop_index, _) = sample_weighted_deal(iteration, &deals);
            match oop_range[oop_index].combo.class {
                class if class == aa => aa_samples += 1,
                class if class == t3o => t3o_samples += 1,
                _ => {}
            }
        }

        assert!(
            aa_samples > t3o_samples * 20,
            "weighted sampling ignored range weights: AA={aa_samples} T3o={t3o_samples}"
        );
    }
}
