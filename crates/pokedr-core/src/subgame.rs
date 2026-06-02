use crate::cards::Card;
use crate::hand_class::HandClass;
use crate::postflop::{PostflopCfrConfig, postflop_combos};

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

    pub fn to_postflop_config(
        &self,
        iterations: usize,
    ) -> Result<PostflopCfrConfig, SubgameBuildError> {
        self.validate()?;
        let oop_range = postflop_combos(&self.ranges.oop, &self.board);
        let ip_range = postflop_combos(&self.ranges.ip, &self.board);
        if oop_range.is_empty() || ip_range.is_empty() {
            return Err(SubgameBuildError::EmptyComboRange);
        }

        let bet = resolve_first(
            &self.actions.bet_sizes,
            &self.pot,
            self.pot.current_bet.max(self.pot.min_raise),
        )?;
        let raise = resolve_first(&self.actions.raise_sizes, &self.pot, bet)?;
        let reraise = resolve_first(&self.actions.reraise_sizes, &self.pot, raise)?;

        Ok(PostflopCfrConfig {
            board: self.board.clone(),
            oop_range,
            ip_range,
            pot: self.pot.pot,
            bet,
            raise,
            reraise,
            iterations,
            max_runouts: self.chance.max_runouts(),
        })
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
    EmptyComboRange,
    UnsupportedAllInSize,
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

fn resolve_first(
    sizes: &[BetSize],
    pot: &PotState,
    current_bet: f64,
) -> Result<f64, SubgameBuildError> {
    let Some(size) = sizes.first() else {
        return Err(SubgameBuildError::Validation(
            SubgameValidationError::InvalidBetSize,
        ));
    };
    resolve_bet_size(*size, pot, current_bet)
}

fn resolve_bet_size(
    size: BetSize,
    pot: &PotState,
    current_bet: f64,
) -> Result<f64, SubgameBuildError> {
    match size {
        BetSize::PotFraction(fraction) => Ok(pot.pot * fraction),
        BetSize::CurrentBetMultiple(multiplier) => Ok(current_bet.max(1.0) * multiplier),
        BetSize::Chips(chips) => Ok(chips),
        BetSize::AllIn => Err(SubgameBuildError::UnsupportedAllInSize),
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
                    resolve_tree_bet_size(*size, player, &pot, pot.pot)?,
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
                    resolve_tree_bet_size(*size, player, &pot, pot.current_bet)?,
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

fn resolve_tree_bet_size(
    size: BetSize,
    player: Player,
    pot: &PotState,
    current_bet: f64,
) -> Result<f64, SubgameBuildError> {
    match size {
        BetSize::AllIn => Ok(all_in_commitment(player, *pot)),
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
    fn builds_compatible_postflop_config() {
        let board = vec![c(14, 0), c(13, 1), c(2, 2)];
        let spec = SubgameSpec::postflop(
            board.clone(),
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(
                parse_range("AA,AKs,AKo").unwrap(),
                parse_range("QQ,JJ,AQs").unwrap(),
            ),
            ActionAbstraction::default(),
            ChancePolicy::Sample(8),
        )
        .unwrap();

        let config = spec.to_postflop_config(500).unwrap();

        assert_eq!(config.board, board);
        assert_eq!(config.iterations, 500);
        assert_eq!(config.max_runouts, 8);
        assert_eq!(config.pot, 100.0);
        assert_eq!(config.bet, 75.0);
        assert_eq!(config.raise, 225.0);
        assert_eq!(config.reraise, 562.5);
        assert!(!config.oop_range.is_empty());
        assert!(!config.ip_range.is_empty());
    }

    #[test]
    fn adapter_rejects_all_in_sizes_until_tree_supports_stacks() {
        let spec = SubgameSpec::postflop(
            vec![c(14, 0), c(13, 1), c(2, 2)],
            PotState::new(100.0, [1_000.0, 1_000.0]),
            RangeState::new(parse_range("AA").unwrap(), parse_range("KK").unwrap()),
            ActionAbstraction {
                bet_sizes: vec![BetSize::AllIn],
                ..ActionAbstraction::default()
            },
            ChancePolicy::Sample(8),
        )
        .unwrap();

        assert!(matches!(
            spec.to_postflop_config(500),
            Err(SubgameBuildError::UnsupportedAllInSize)
        ));
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
}
