use crate::cards::{Board, Card};
use crate::isomorphism::{SuitPermutation, next_card_isomorphism};
use crate::range::RangeSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Player {
    Oop,
    Ip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Street {
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionKind {
    Check,
    Bet { amount: u32 },
    Call { amount: u32 },
    Fold,
    Raise { to: u32 },
    AllIn { to: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChanceSpec {
    pub next_street: Street,
    pub cards: Vec<Card>,
    pub child_multiplicities: Vec<usize>,
    pub child_permutation_codes: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicNodeKind {
    Decision {
        player: Player,
        actions: Vec<ActionKind>,
    },
    Chance(ChanceSpec),
    Terminal {
        reason: TerminalReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalReason {
    Fold,
    Showdown,
    AllIn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicNode {
    pub id: usize,
    pub state: PublicState,
    pub kind: PublicNodeKind,
    pub children: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicState {
    pub street: Street,
    pub board: Board,
    pub pot: u32,
    pub oop_stack: u32,
    pub ip_stack: u32,
    pub oop_street_commit: u32,
    pub ip_street_commit: u32,
    pub last_raise_size: u32,
    pub raises_this_street: u8,
    pub checks_this_street: u8,
    pub can_donk: bool,
    pub player: Player,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spot {
    pub board: Board,
    pub pot: u32,
    pub effective_stack: u32,
    pub oop_range: RangeSpec,
    pub ip_range: RangeSpec,
    pub first_player: Player,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreetTemplate {
    pub first_bet_sizes: Vec<BetSizeSpec>,
    pub donk_bet_sizes: Vec<BetSizeSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaisePolicy {
    pub raise_multiplier: f32,
    pub raise_sizes: Vec<RaiseSizeSpec>,
    pub max_raises_per_street: u8,
    pub shove_spr_threshold: f32,
    pub shove_commit_fraction: f32,
    pub add_all_in_threshold: f32,
    pub force_all_in_threshold: f32,
    pub merging_threshold: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionAbstraction {
    pub min_bet: u32,
    pub flop: StreetTemplate,
    pub turn: StreetTemplate,
    pub river: StreetTemplate,
    pub raise: RaisePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BetSizeSpec {
    PotFraction(f32),
    Geometric { streets: u8, max_pot_fraction: f32 },
    AllIn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RaiseSizeSpec {
    PotFraction(f32),
    PreviousBetMultiplier(f32),
    Geometric { streets: u8, max_pot_fraction: f32 },
    AllIn,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeTemplate {
    pub action_abstraction: ActionAbstraction,
    pub chance_expansion: ChanceExpansion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChanceExpansion {
    TemplateOnly,
    Isomorphic,
    Enumerate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicTree {
    pub spot: SpotSummary,
    pub nodes: Vec<PublicNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpotSummary {
    pub board: Board,
    pub pot: u32,
    pub effective_stack: u32,
    pub first_player: Player,
    pub oop_range_combos: usize,
    pub ip_range_combos: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeStats {
    pub nodes: usize,
    pub decisions: usize,
    pub chances: usize,
    pub terminals: usize,
    pub max_depth: usize,
    pub decisions_by_street: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeBuildError {
    BoardMustStartOnFlop,
    InvalidStack,
    InvalidSizing,
}

pub struct TreeBuilder {
    template: TreeTemplate,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Self::Oop => Self::Ip,
            Self::Ip => Self::Oop,
        }
    }
}

impl Street {
    pub fn index(self) -> usize {
        match self {
            Self::Flop => 0,
            Self::Turn => 1,
            Self::River => 2,
        }
    }

    pub fn from_board_len(cards: usize) -> Option<Self> {
        match cards {
            3 => Some(Self::Flop),
            4 => Some(Self::Turn),
            5 => Some(Self::River),
            _ => None,
        }
    }

    pub fn next(self) -> Option<Self> {
        match self {
            Self::Flop => Some(Self::Turn),
            Self::Turn => Some(Self::River),
            Self::River => None,
        }
    }
}

impl ActionAbstraction {
    pub fn conservative_default() -> Self {
        Self {
            min_bet: 100,
            flop: StreetTemplate {
                first_bet_sizes: vec![
                    BetSizeSpec::PotFraction(0.33),
                    BetSizeSpec::PotFraction(0.75),
                ],
                donk_bet_sizes: vec![BetSizeSpec::PotFraction(0.50)],
            },
            turn: StreetTemplate {
                first_bet_sizes: vec![
                    BetSizeSpec::PotFraction(0.50),
                    BetSizeSpec::PotFraction(1.00),
                ],
                donk_bet_sizes: Vec::new(),
            },
            river: StreetTemplate {
                first_bet_sizes: vec![
                    BetSizeSpec::PotFraction(0.50),
                    BetSizeSpec::PotFraction(1.00),
                ],
                donk_bet_sizes: vec![BetSizeSpec::PotFraction(0.75)],
            },
            raise: RaisePolicy {
                raise_multiplier: 3.0,
                raise_sizes: vec![RaiseSizeSpec::PreviousBetMultiplier(3.0)],
                max_raises_per_street: 2,
                shove_spr_threshold: 1.5,
                shove_commit_fraction: 0.70,
                add_all_in_threshold: 0.0,
                force_all_in_threshold: 0.0,
                merging_threshold: 0.0,
            },
        }
    }

    pub fn postflop_solver_basic() -> Self {
        let first_bets = vec![
            BetSizeSpec::PotFraction(0.60),
            BetSizeSpec::Geometric {
                streets: 0,
                max_pot_fraction: f32::INFINITY,
            },
            BetSizeSpec::AllIn,
        ];
        Self {
            min_bet: 100,
            flop: StreetTemplate {
                first_bet_sizes: first_bets.clone(),
                donk_bet_sizes: Vec::new(),
            },
            turn: StreetTemplate {
                first_bet_sizes: first_bets.clone(),
                donk_bet_sizes: Vec::new(),
            },
            river: StreetTemplate {
                first_bet_sizes: first_bets,
                donk_bet_sizes: vec![BetSizeSpec::PotFraction(0.50)],
            },
            raise: RaisePolicy {
                raise_multiplier: 2.5,
                raise_sizes: vec![RaiseSizeSpec::PreviousBetMultiplier(2.5)],
                max_raises_per_street: u8::MAX,
                shove_spr_threshold: 0.0,
                shove_commit_fraction: 1.0,
                add_all_in_threshold: 1.5,
                force_all_in_threshold: 0.15,
                merging_threshold: 0.1,
            },
        }
    }
}

impl TreeTemplate {
    pub fn conservative_default() -> Self {
        Self {
            action_abstraction: ActionAbstraction::conservative_default(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        }
    }
}

impl TreeBuilder {
    pub fn new(template: TreeTemplate) -> Result<Self, TreeBuildError> {
        let abstraction = &template.action_abstraction;
        if abstraction.min_bet == 0
            || abstraction.raise.raise_multiplier <= 1.0
            || !abstraction.raise.raise_multiplier.is_finite()
            || abstraction
                .raise
                .raise_sizes
                .iter()
                .any(|size| !size.valid())
        {
            return Err(TreeBuildError::InvalidSizing);
        }
        Ok(Self { template })
    }

    pub fn build(&self, spot: Spot) -> Result<PublicTree, TreeBuildError> {
        if spot.board.cards().len() != 3 {
            return Err(TreeBuildError::BoardMustStartOnFlop);
        }
        if spot.pot == 0 || spot.effective_stack == 0 {
            return Err(TreeBuildError::InvalidStack);
        }
        let root = PublicState {
            street: Street::Flop,
            board: spot.board.clone(),
            pot: spot.pot,
            oop_stack: spot.effective_stack,
            ip_stack: spot.effective_stack,
            oop_street_commit: 0,
            ip_street_commit: 0,
            last_raise_size: 0,
            raises_this_street: 0,
            checks_this_street: 0,
            can_donk: false,
            player: spot.first_player,
        };
        let summary = SpotSummary {
            board: spot.board,
            pot: spot.pot,
            effective_stack: spot.effective_stack,
            first_player: spot.first_player,
            oop_range_combos: spot.oop_range.combos().len(),
            ip_range_combos: spot.ip_range.combos().len(),
        };
        let mut tree = PublicTree {
            spot: summary,
            nodes: Vec::new(),
        };
        self.build_state(&mut tree, root, 0, &spot.oop_range, &spot.ip_range);
        Ok(tree)
    }

    fn build_state(
        &self,
        tree: &mut PublicTree,
        state: PublicState,
        depth: usize,
        oop_range: &RangeSpec,
        ip_range: &RangeSpec,
    ) -> usize {
        let id = tree.nodes.len();
        tree.nodes.push(PublicNode {
            id,
            state: state.clone(),
            kind: PublicNodeKind::Terminal {
                reason: TerminalReason::Showdown,
            },
            children: Vec::new(),
        });

        let actions = self.legal_actions(&state);
        if actions.is_empty() {
            tree.nodes[id].kind = PublicNodeKind::Terminal {
                reason: TerminalReason::Showdown,
            };
            return id;
        }
        tree.nodes[id].kind = PublicNodeKind::Decision {
            player: state.player,
            actions: actions.clone(),
        };
        for action in actions {
            match self.apply_action(&state, action) {
                Transition::State(next) => {
                    let child = self.build_state(tree, next, depth + 1, oop_range, ip_range);
                    tree.nodes[id].children.push(child);
                }
                Transition::Terminal(terminal_state, reason) => {
                    let child = self.push_terminal(tree, terminal_state, reason);
                    tree.nodes[id].children.push(child);
                }
                Transition::Chance(next_state) => {
                    let child = self.push_chance(tree, next_state, depth + 1, oop_range, ip_range);
                    tree.nodes[id].children.push(child);
                }
            }
        }
        id
    }

    fn push_terminal(
        &self,
        tree: &mut PublicTree,
        state: PublicState,
        reason: TerminalReason,
    ) -> usize {
        let id = tree.nodes.len();
        tree.nodes.push(PublicNode {
            id,
            state,
            kind: PublicNodeKind::Terminal { reason },
            children: Vec::new(),
        });
        id
    }

    fn push_chance(
        &self,
        tree: &mut PublicTree,
        state: PublicState,
        depth: usize,
        oop_range: &RangeSpec,
        ip_range: &RangeSpec,
    ) -> usize {
        let id = tree.nodes.len();
        let next_street = state.street;
        let remaining = state.board.remaining_deck();
        let (cards, child_multiplicities, child_permutation_codes) =
            match self.template.chance_expansion {
                ChanceExpansion::TemplateOnly => (
                    remaining.iter().copied().take(1).collect::<Vec<_>>(),
                    vec![1],
                    vec![vec![SuitPermutation::identity().code()]],
                ),
                ChanceExpansion::Isomorphic => {
                    let iso = next_card_isomorphism(&state.board, oop_range, ip_range);
                    (
                        iso.classes
                            .iter()
                            .filter_map(|class| class.representative.first().copied())
                            .collect::<Vec<_>>(),
                        iso.classes
                            .iter()
                            .map(|class| class.multiplicity)
                            .collect::<Vec<_>>(),
                        iso.classes
                            .iter()
                            .map(|class| {
                                class
                                    .members
                                    .iter()
                                    .map(|member| member.permutation_to_representative.code())
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>(),
                    )
                }
                ChanceExpansion::Enumerate => (
                    remaining.clone(),
                    vec![1; remaining.len()],
                    remaining
                        .iter()
                        .map(|_| vec![SuitPermutation::identity().code()])
                        .collect::<Vec<_>>(),
                ),
            };
        tree.nodes.push(PublicNode {
            id,
            state: state.clone(),
            kind: PublicNodeKind::Chance(ChanceSpec {
                next_street,
                cards: cards.clone(),
                child_multiplicities,
                child_permutation_codes,
            }),
            children: Vec::new(),
        });
        for card in cards {
            let mut child_state = state.clone();
            child_state.board = child_state
                .board
                .push(card)
                .expect("chance card must not duplicate board");
            let child = self.build_state(tree, child_state, depth + 1, oop_range, ip_range);
            tree.nodes[id].children.push(child);
        }
        id
    }

    fn legal_actions(&self, state: &PublicState) -> Vec<ActionKind> {
        let to_call = to_call(state);
        let stack = stack_for(state, state.player);
        if stack == 0 {
            return Vec::new();
        }
        if to_call > 0 {
            let mut actions = vec![
                ActionKind::Fold,
                ActionKind::Call {
                    amount: to_call.min(stack),
                },
            ];
            if state.raises_this_street
                < self.template.action_abstraction.raise.max_raises_per_street
                && stack > to_call
            {
                for raise in self.raise_actions(state) {
                    push_unique_action(&mut actions, raise);
                }
                if let Some(all_in) = self.threshold_all_in_raise_response(state) {
                    push_unique_action(&mut actions, all_in);
                }
                let opponent_commit = commit_for(state, state.player.other());
                actions = merge_bet_actions(
                    sort_and_dedup_response_actions(actions),
                    state.pot + to_call.min(stack),
                    opponent_commit,
                    self.template.action_abstraction.raise.merging_threshold,
                );
            }
            return actions;
        }

        let mut actions = vec![ActionKind::Check];
        let street_template = match state.street {
            Street::Flop => &self.template.action_abstraction.flop,
            Street::Turn => &self.template.action_abstraction.turn,
            Street::River => &self.template.action_abstraction.river,
        };
        let sizes = if state.can_donk && !street_template.donk_bet_sizes.is_empty() {
            &street_template.donk_bet_sizes
        } else {
            &street_template.first_bet_sizes
        };
        for size in sizes {
            push_unique_action(&mut actions, self.bet_action(state, *size));
        }
        if self.add_threshold_all_in_allowed(state) {
            push_unique_action(&mut actions, ActionKind::AllIn { to: stack });
        }
        actions = sort_and_dedup_no_call_actions(actions);
        merge_bet_actions(
            actions,
            state.pot,
            0,
            self.template.action_abstraction.raise.merging_threshold,
        )
    }

    fn bet_action(&self, state: &PublicState, size: BetSizeSpec) -> ActionKind {
        let stack = stack_for(state, state.player);
        let amount = match size {
            BetSizeSpec::PotFraction(fraction) => sized_amount(
                state.pot,
                fraction,
                self.template.action_abstraction.min_bet,
                stack,
            ),
            BetSizeSpec::Geometric {
                streets,
                max_pot_fraction,
            } => geometric_bet_amount(
                state.pot,
                stack,
                streets_for_geometric(state.street, streets),
                max_pot_fraction,
                self.template.action_abstraction.min_bet,
            ),
            BetSizeSpec::AllIn => stack,
        };
        self.force_all_in_if_close(state, bet_or_all_in(amount, stack))
    }

    fn raise_actions(&self, state: &PublicState) -> Vec<ActionKind> {
        let sizes = &self.template.action_abstraction.raise.raise_sizes;
        let mut actions = Vec::with_capacity(sizes.len().max(1));
        if sizes.is_empty() {
            if let Some(action) = self.raise_action_for_size(
                state,
                RaiseSizeSpec::PreviousBetMultiplier(
                    self.template.action_abstraction.raise.raise_multiplier,
                ),
            ) {
                actions.push(action);
            }
            return actions;
        }
        for size in sizes {
            if let Some(action) = self.raise_action_for_size(state, *size) {
                actions.push(action);
            }
        }
        actions
    }

    fn raise_action_for_size(
        &self,
        state: &PublicState,
        size: RaiseSizeSpec,
    ) -> Option<ActionKind> {
        let actor_commit = commit_for(state, state.player);
        let opponent_commit = commit_for(state, state.player.other());
        let stack = stack_for(state, state.player);
        let to_call = opponent_commit.saturating_sub(actor_commit);
        let min_raise_to = opponent_commit
            + state
                .last_raise_size
                .max(self.template.action_abstraction.min_bet);
        let max_to = actor_commit + stack;
        let pot_after_call = state.pot + to_call.min(stack);
        let target_to = match size {
            RaiseSizeSpec::PotFraction(fraction) => {
                opponent_commit + ((pot_after_call as f32) * fraction).round() as u32
            }
            RaiseSizeSpec::PreviousBetMultiplier(multiplier) => {
                ((opponent_commit as f32) * multiplier).round() as u32
            }
            RaiseSizeSpec::Geometric {
                streets,
                max_pot_fraction,
            } => {
                let additional = geometric_bet_amount(
                    pot_after_call,
                    stack.saturating_sub(to_call),
                    streets_for_raise_geometric(state.street, streets, state.raises_this_street),
                    max_pot_fraction,
                    self.template.action_abstraction.min_bet,
                );
                opponent_commit + additional
            }
            RaiseSizeSpec::AllIn => max_to,
        }
        .max(min_raise_to);
        if target_to >= max_to {
            return Some(ActionKind::AllIn { to: max_to });
        }
        let additional = target_to - actor_commit;
        let remaining_after_raise = stack.saturating_sub(additional);
        let spr_after_raise = if pot_after_call == 0 {
            f32::INFINITY
        } else {
            remaining_after_raise as f32 / pot_after_call as f32
        };
        if spr_after_raise <= self.template.action_abstraction.raise.shove_spr_threshold
            || additional as f32
                >= stack as f32 * self.template.action_abstraction.raise.shove_commit_fraction
        {
            return Some(ActionKind::AllIn { to: max_to });
        }
        Some(self.force_all_in_if_close(state, ActionKind::Raise { to: target_to }))
    }

    fn add_threshold_all_in_allowed(&self, state: &PublicState) -> bool {
        let threshold = self.template.action_abstraction.raise.add_all_in_threshold;
        threshold > 0.0 && stack_for(state, state.player) as f32 <= state.pot as f32 * threshold
    }

    fn threshold_all_in_raise_response(&self, state: &PublicState) -> Option<ActionKind> {
        let threshold = self.template.action_abstraction.raise.add_all_in_threshold;
        if threshold <= 0.0 {
            return None;
        }
        let actor_commit = commit_for(state, state.player);
        let opponent_commit = commit_for(state, state.player.other());
        let stack = stack_for(state, state.player);
        let to_call = opponent_commit.saturating_sub(actor_commit);
        if stack <= to_call {
            return None;
        }
        let max_to = actor_commit + stack;
        let pot_after_call = state.pot.saturating_add(to_call.min(stack));
        let threshold_to = opponent_commit + ((pot_after_call as f32) * threshold).round() as u32;
        if max_to <= threshold_to {
            Some(ActionKind::AllIn { to: max_to })
        } else {
            None
        }
    }

    fn force_all_in_if_close(&self, state: &PublicState, action: ActionKind) -> ActionKind {
        let threshold = self
            .template
            .action_abstraction
            .raise
            .force_all_in_threshold;
        if threshold <= 0.0 {
            return action;
        }
        let actor_commit = commit_for(state, state.player);
        let opponent_commit = commit_for(state, state.player.other());
        let stack = stack_for(state, state.player);
        let (additional, pot_if_called) = match action {
            ActionKind::Bet { amount } => {
                (amount, state.pot.saturating_add(amount.saturating_mul(2)))
            }
            ActionKind::Raise { to } => {
                let to_call = opponent_commit.saturating_sub(actor_commit).min(stack);
                let raise_delta = to.saturating_sub(opponent_commit);
                (
                    to.saturating_sub(actor_commit),
                    state
                        .pot
                        .saturating_add(to_call)
                        .saturating_add(raise_delta.saturating_mul(2)),
                )
            }
            ActionKind::AllIn { .. } => return action,
            _ => return action,
        };
        let remaining = stack.saturating_sub(additional);
        let close_threshold = (pot_if_called as f32 * threshold).round() as u32;
        if remaining <= close_threshold {
            ActionKind::AllIn {
                to: actor_commit + stack,
            }
        } else {
            action
        }
    }

    fn apply_action(&self, state: &PublicState, action: ActionKind) -> Transition {
        match action {
            ActionKind::Fold => Transition::Terminal(state.clone(), TerminalReason::Fold),
            ActionKind::Check => {
                if state.checks_this_street >= 1 {
                    return self.close_street(state, false);
                }
                let mut next = state.clone();
                next.checks_this_street += 1;
                next.can_donk = false;
                next.player = next.player.other();
                Transition::State(next)
            }
            ActionKind::Call { amount } => {
                let mut next = state.clone();
                let can_donk_next_street = state.player == Player::Oop && to_call(state) > 0;
                commit_chips(&mut next, state.player, amount);
                self.close_street(&next, can_donk_next_street)
            }
            ActionKind::Bet { amount } => {
                let mut next = state.clone();
                commit_chips(&mut next, state.player, amount);
                next.can_donk = false;
                next.last_raise_size = amount;
                next.player = next.player.other();
                Transition::State(next)
            }
            ActionKind::Raise { to } => {
                let mut next = state.clone();
                let current = commit_for(&next, state.player);
                let opponent = commit_for(&next, state.player.other());
                commit_chips(&mut next, state.player, to.saturating_sub(current));
                next.last_raise_size = to.saturating_sub(opponent);
                next.raises_this_street += 1;
                next.can_donk = false;
                next.player = next.player.other();
                Transition::State(next)
            }
            ActionKind::AllIn { to } => {
                let mut next = state.clone();
                let current = commit_for(&next, state.player);
                commit_chips(&mut next, state.player, to.saturating_sub(current));
                let opponent_commit = commit_for(&next, state.player.other());
                let actor_commit = commit_for(&next, state.player);
                if opponent_commit == actor_commit {
                    Transition::Terminal(next, TerminalReason::AllIn)
                } else {
                    next.raises_this_street += 1;
                    next.can_donk = false;
                    next.player = next.player.other();
                    Transition::State(next)
                }
            }
        }
    }

    fn close_street(&self, state: &PublicState, can_donk_next_street: bool) -> Transition {
        if state.oop_stack == 0 || state.ip_stack == 0 {
            return Transition::Terminal(state.clone(), TerminalReason::AllIn);
        }
        let Some(next_street) = state.street.next() else {
            return Transition::Terminal(state.clone(), TerminalReason::Showdown);
        };
        let mut next = state.clone();
        next.street = next_street;
        next.oop_street_commit = 0;
        next.ip_street_commit = 0;
        next.last_raise_size = 0;
        next.raises_this_street = 0;
        next.checks_this_street = 0;
        next.can_donk = can_donk_next_street;
        next.player = Player::Oop;
        Transition::Chance(next)
    }
}

impl PublicTree {
    pub fn stats(&self) -> TreeStats {
        let mut stats = TreeStats {
            nodes: self.nodes.len(),
            decisions: 0,
            chances: 0,
            terminals: 0,
            max_depth: 0,
            decisions_by_street: [0; 3],
        };
        fn visit(tree: &PublicTree, node: usize, depth: usize, stats: &mut TreeStats) {
            stats.max_depth = stats.max_depth.max(depth);
            match tree.nodes[node].kind {
                PublicNodeKind::Decision { .. } => {
                    stats.decisions += 1;
                    stats.decisions_by_street[tree.nodes[node].state.street.index()] += 1;
                }
                PublicNodeKind::Chance(_) => stats.chances += 1,
                PublicNodeKind::Terminal { .. } => stats.terminals += 1,
            }
            for child in &tree.nodes[node].children {
                visit(tree, *child, depth + 1, stats);
            }
        }
        if !self.nodes.is_empty() {
            visit(self, 0, 0, &mut stats);
        }
        stats
    }
}

enum Transition {
    State(PublicState),
    Chance(PublicState),
    Terminal(PublicState, TerminalReason),
}

fn to_call(state: &PublicState) -> u32 {
    let actor = commit_for(state, state.player);
    let opponent = commit_for(state, state.player.other());
    opponent.saturating_sub(actor)
}

fn commit_for(state: &PublicState, player: Player) -> u32 {
    match player {
        Player::Oop => state.oop_street_commit,
        Player::Ip => state.ip_street_commit,
    }
}

fn stack_for(state: &PublicState, player: Player) -> u32 {
    match player {
        Player::Oop => state.oop_stack,
        Player::Ip => state.ip_stack,
    }
}

fn commit_chips(state: &mut PublicState, player: Player, amount: u32) {
    let amount = amount.min(stack_for(state, player));
    state.pot += amount;
    match player {
        Player::Oop => {
            state.oop_stack -= amount;
            state.oop_street_commit += amount;
        }
        Player::Ip => {
            state.ip_stack -= amount;
            state.ip_street_commit += amount;
        }
    }
}

fn sized_amount(pot: u32, fraction: f32, min_bet: u32, stack: u32) -> u32 {
    let amount = ((pot as f32) * fraction).round() as u32;
    amount.max(min_bet).min(stack)
}

fn geometric_bet_amount(
    pot: u32,
    stack: u32,
    streets: u8,
    max_pot_fraction: f32,
    min_bet: u32,
) -> u32 {
    let streets = streets.max(1) as f32;
    let spr = if pot == 0 {
        f32::INFINITY
    } else {
        stack as f32 / pot as f32
    };
    let ratio = ((2.0 * spr + 1.0).powf(1.0 / streets) - 1.0) / 2.0;
    sized_amount(pot, ratio.min(max_pot_fraction), min_bet, stack)
}

fn streets_for_geometric(street: Street, configured: u8) -> u8 {
    if configured > 0 {
        return configured;
    }
    match street {
        Street::Flop => 3,
        Street::Turn => 2,
        Street::River => 1,
    }
}

fn streets_for_raise_geometric(street: Street, configured: u8, raises_this_street: u8) -> u8 {
    let base = if configured > 0 {
        configured
    } else {
        streets_for_geometric(street, 0)
    };
    base.saturating_sub(raises_this_street).max(1)
}

impl RaiseSizeSpec {
    fn valid(self) -> bool {
        match self {
            Self::PotFraction(value) | Self::PreviousBetMultiplier(value) => {
                value.is_finite() && value > 0.0
            }
            Self::Geometric {
                max_pot_fraction, ..
            } => max_pot_fraction.is_finite() || max_pot_fraction.is_infinite(),
            Self::AllIn => true,
        }
    }
}

fn bet_or_all_in(amount: u32, stack: u32) -> ActionKind {
    if amount >= stack {
        ActionKind::AllIn { to: stack }
    } else {
        ActionKind::Bet { amount }
    }
}

fn push_unique_action(actions: &mut Vec<ActionKind>, action: ActionKind) {
    if !actions.contains(&action) {
        actions.push(action);
    }
}

fn sort_and_dedup_response_actions(mut actions: Vec<ActionKind>) -> Vec<ActionKind> {
    fn key(action: &ActionKind) -> (u8, u32) {
        match *action {
            ActionKind::Fold => (0, 0),
            ActionKind::Call { amount } => (1, amount),
            ActionKind::Raise { to } => (2, to),
            ActionKind::AllIn { to } => (2, to),
            ActionKind::Check => (3, 0),
            ActionKind::Bet { amount } => (4, amount),
        }
    }
    actions.sort_by_key(key);
    actions.dedup();
    actions
}

fn sort_and_dedup_no_call_actions(mut actions: Vec<ActionKind>) -> Vec<ActionKind> {
    fn key(action: &ActionKind) -> (u8, u32) {
        match *action {
            ActionKind::Check => (0, 0),
            ActionKind::Bet { amount } => (1, amount),
            ActionKind::Raise { to } => (1, to),
            ActionKind::AllIn { to } => (1, to),
            ActionKind::Call { amount } => (2, amount),
            ActionKind::Fold => (3, 0),
        }
    }
    actions.sort_by_key(key);
    actions.dedup();
    actions
}

fn merge_bet_actions(
    actions: Vec<ActionKind>,
    pot: u32,
    offset: u32,
    threshold: f32,
) -> Vec<ActionKind> {
    if threshold <= 0.0 || pot == 0 {
        return actions;
    }
    let amount = |action: ActionKind| match action {
        ActionKind::Bet { amount }
        | ActionKind::Raise { to: amount }
        | ActionKind::AllIn { to: amount } => Some(amount),
        _ => None,
    };
    let mut current = u32::MAX;
    let mut merged = Vec::with_capacity(actions.len());
    for action in actions.iter().rev() {
        if let Some(action_amount) = amount(*action) {
            let ratio = action_amount.saturating_sub(offset) as f32 / pot as f32;
            let current_ratio = current.saturating_sub(offset) as f32 / pot as f32;
            let threshold_ratio = (current_ratio - threshold) / (1.0 + threshold);
            if ratio < threshold_ratio {
                merged.push(*action);
                current = action_amount;
            }
        } else {
            merged.push(*action);
        }
    }
    merged.reverse();
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn default_spot() -> Spot {
        Spot {
            board: Board::from_str("As7h2c").unwrap(),
            pot: 650,
            effective_stack: 9700,
            oop_range: RangeSpec::full_deck_uniform(),
            ip_range: RangeSpec::full_deck_uniform(),
            first_player: Player::Oop,
        }
    }

    #[test]
    fn builds_nontrivial_flop_tree_with_chance_nodes() {
        let builder = TreeBuilder::new(TreeTemplate::conservative_default()).unwrap();
        let tree = builder.build(default_spot()).unwrap();
        let stats = tree.stats();
        assert!(stats.decisions > 400, "{stats:?}");
        assert!(stats.chances > 0, "{stats:?}");
        assert!(stats.terminals > stats.decisions, "{stats:?}");
    }

    #[test]
    fn isomorphic_chance_children_preserve_concrete_card_count() {
        let builder = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::Isomorphic,
        })
        .unwrap();
        let tree = builder
            .build(Spot {
                board: Board::from_str("AsKsQs").unwrap(),
                pot: 200,
                effective_stack: 900,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let chance = tree
            .nodes
            .iter()
            .find(|node| matches!(node.kind, PublicNodeKind::Chance(_)))
            .expect("tree should contain a turn chance");
        let PublicNodeKind::Chance(chance) = &chance.kind else {
            unreachable!();
        };

        assert_eq!(chance.next_street, Street::Turn);
        assert_eq!(chance.cards.len(), chance.child_multiplicities.len());
        assert_eq!(chance.cards.len(), chance.child_permutation_codes.len());
        assert_eq!(chance.child_multiplicities.iter().sum::<usize>(), 49);
        assert_eq!(
            chance
                .child_permutation_codes
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
            49
        );
        assert!(chance.cards.len() < 49);
    }

    #[test]
    fn first_raise_response_has_single_raise_size_plus_call_fold() {
        let builder = TreeBuilder::new(TreeTemplate::conservative_default()).unwrap();
        let tree = builder.build(default_spot()).unwrap();
        let root = &tree.nodes[0];
        let PublicNodeKind::Decision { actions, .. } = &root.kind else {
            panic!("root must be decision");
        };
        let bet_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Bet { .. }))
            .and_then(|index| root.children.get(index))
            .map(|child| *child)
            .expect("root should include a bet");
        let response = &tree.nodes[bet_child];
        let PublicNodeKind::Decision { actions, .. } = &response.kind else {
            panic!("bet response must be decision");
        };
        let raises = actions
            .iter()
            .filter(|action| matches!(action, ActionKind::Raise { .. } | ActionKind::AllIn { .. }))
            .count();
        assert_eq!(raises, 1, "{actions:?}");
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Fold))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Call { .. }))
        );
    }

    #[test]
    fn postflop_basic_empty_turn_donk_uses_regular_bet_sizes() {
        let builder = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap();
        let tree = builder
            .build(Spot {
                board: Board::from_str("Td9d6h").unwrap(),
                pot: 200,
                effective_stack: 900,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let root = &tree.nodes[0];
        let PublicNodeKind::Decision { actions, .. } = &root.kind else {
            panic!("root must be decision");
        };
        let check_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Check))
            .and_then(|index| root.children.get(index))
            .copied()
            .expect("root should include check");
        let after_check = &tree.nodes[check_child];
        let PublicNodeKind::Decision { actions, .. } = &after_check.kind else {
            panic!("after-check node must be decision");
        };
        let bet_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Bet { amount: 120 }))
            .and_then(|index| after_check.children.get(index))
            .copied()
            .expect("IP should include 60% bet after OOP check");
        let response = &tree.nodes[bet_child];
        let PublicNodeKind::Decision { actions, .. } = &response.kind else {
            panic!("bet response must be decision");
        };
        let call_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Call { amount: 120 }))
            .and_then(|index| response.children.get(index))
            .copied()
            .expect("bet response should include call");
        let chance = &tree.nodes[call_child];
        let turn_child = *chance
            .children
            .first()
            .expect("turn chance should have template child");
        let turn = &tree.nodes[turn_child];
        let PublicNodeKind::Decision { actions, .. } = &turn.kind else {
            panic!("turn child must be decision");
        };
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Check))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Bet { .. })),
            "{actions:?}"
        );
    }

    #[test]
    fn postflop_basic_raise_response_keeps_threshold_all_in_after_call_pot() {
        let builder = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap();
        let tree = builder
            .build(Spot {
                board: Board::from_str("Td9d6h").unwrap(),
                pot: 200,
                effective_stack: 900,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let root = &tree.nodes[0];
        let PublicNodeKind::Decision { actions, .. } = &root.kind else {
            panic!("root must be decision");
        };
        let check_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Check))
            .and_then(|index| root.children.get(index))
            .copied()
            .expect("root should include check");
        let after_check = &tree.nodes[check_child];
        let PublicNodeKind::Decision { actions, .. } = &after_check.kind else {
            panic!("after-check node must be decision");
        };
        let check_check_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Check))
            .and_then(|index| after_check.children.get(index))
            .copied()
            .expect("IP should include check after OOP check");
        let chance = &tree.nodes[check_check_child];
        let turn_child = *chance
            .children
            .first()
            .expect("turn chance should have template child");
        let turn = &tree.nodes[turn_child];
        let PublicNodeKind::Decision { actions, .. } = &turn.kind else {
            panic!("turn child must be decision");
        };
        let geometric_bet_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Bet { amount: 216 }))
            .and_then(|index| turn.children.get(index))
            .copied()
            .expect("turn should include geometric bet");
        let response = &tree.nodes[geometric_bet_child];
        let PublicNodeKind::Decision { actions, .. } = &response.kind else {
            panic!("turn bet response must be decision");
        };
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Raise { to: 540 })),
            "{actions:?}"
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::AllIn { to: 900 })),
            "{actions:?}"
        );
    }

    #[test]
    fn all_in_open_keeps_opponent_response() {
        let builder = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap();
        let tree = builder
            .build(Spot {
                board: Board::from_str("Td9d6h").unwrap(),
                pot: 200,
                effective_stack: 900,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let root = &tree.nodes[0];
        let PublicNodeKind::Decision { actions, .. } = &root.kind else {
            panic!("root must be decision");
        };
        let all_in_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::AllIn { to: 900 }))
            .and_then(|index| root.children.get(index))
            .copied()
            .expect("root should include all-in");
        let response = &tree.nodes[all_in_child];
        let PublicNodeKind::Decision { player, actions } = &response.kind else {
            panic!("all-in open must give the opponent a response");
        };
        assert_eq!(*player, Player::Ip);
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Fold))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, ActionKind::Call { amount: 900 }))
        );
    }

    #[test]
    fn all_in_call_terminal_keeps_called_pot_and_commits() {
        let builder = TreeBuilder::new(TreeTemplate {
            action_abstraction: ActionAbstraction::postflop_solver_basic(),
            chance_expansion: ChanceExpansion::TemplateOnly,
        })
        .unwrap();
        let tree = builder
            .build(Spot {
                board: Board::from_str("Td9d6h").unwrap(),
                pot: 200,
                effective_stack: 900,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let root = &tree.nodes[0];
        let PublicNodeKind::Decision { actions, .. } = &root.kind else {
            panic!("root must be decision");
        };
        let all_in_child = actions
            .iter()
            .position(|action| matches!(action, ActionKind::AllIn { to: 900 }))
            .and_then(|index| root.children.get(index))
            .copied()
            .expect("root should include all-in");
        let response = &tree.nodes[all_in_child];
        let PublicNodeKind::Decision { actions, .. } = &response.kind else {
            panic!("all-in open must give the opponent a response");
        };
        let call_terminal = actions
            .iter()
            .position(|action| matches!(action, ActionKind::Call { amount: 900 }))
            .and_then(|index| response.children.get(index))
            .copied()
            .expect("all-in response should include call");
        let terminal = &tree.nodes[call_terminal];
        assert!(matches!(
            terminal.kind,
            PublicNodeKind::Terminal {
                reason: TerminalReason::AllIn
            }
        ));
        assert_eq!(terminal.state.pot, 2000);
        assert_eq!(terminal.state.oop_stack, 0);
        assert_eq!(terminal.state.ip_stack, 0);
        assert_eq!(terminal.state.oop_street_commit, 900);
        assert_eq!(terminal.state.ip_street_commit, 900);
    }
}
