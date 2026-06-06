use crate::cards::{Board, Card};
use crate::range::RangeSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub first_bet_fractions: Vec<f32>,
    pub donk_bet_fractions: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaisePolicy {
    pub raise_multiplier: f32,
    pub max_raises_per_street: u8,
    pub shove_spr_threshold: f32,
    pub shove_commit_fraction: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionAbstraction {
    pub min_bet: u32,
    pub flop: StreetTemplate,
    pub turn: StreetTemplate,
    pub river: StreetTemplate,
    pub raise: RaisePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeTemplate {
    pub action_abstraction: ActionAbstraction,
    pub chance_expansion: ChanceExpansion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChanceExpansion {
    TemplateOnly,
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
                first_bet_fractions: vec![0.33, 0.75],
                donk_bet_fractions: vec![0.50],
            },
            turn: StreetTemplate {
                first_bet_fractions: vec![0.50, 1.00],
                donk_bet_fractions: vec![0.75],
            },
            river: StreetTemplate {
                first_bet_fractions: vec![0.50, 1.00],
                donk_bet_fractions: vec![0.75],
            },
            raise: RaisePolicy {
                raise_multiplier: 3.0,
                max_raises_per_street: 2,
                shove_spr_threshold: 1.5,
                shove_commit_fraction: 0.70,
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
        self.build_state(&mut tree, root, 0);
        Ok(tree)
    }

    fn build_state(&self, tree: &mut PublicTree, state: PublicState, depth: usize) -> usize {
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
                    let child = self.build_state(tree, next, depth + 1);
                    tree.nodes[id].children.push(child);
                }
                Transition::Terminal(reason) => {
                    let child = self.push_terminal(tree, state.clone(), reason);
                    tree.nodes[id].children.push(child);
                }
                Transition::Chance(next_state) => {
                    let child = self.push_chance(tree, next_state, depth + 1);
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

    fn push_chance(&self, tree: &mut PublicTree, state: PublicState, depth: usize) -> usize {
        let id = tree.nodes.len();
        let next_street = state.street;
        let cards = state.board.remaining_deck();
        tree.nodes.push(PublicNode {
            id,
            state: state.clone(),
            kind: PublicNodeKind::Chance(ChanceSpec { next_street, cards }),
            children: Vec::new(),
        });
        let cards = match self.template.chance_expansion {
            ChanceExpansion::TemplateOnly => state
                .board
                .remaining_deck()
                .into_iter()
                .take(1)
                .collect::<Vec<_>>(),
            ChanceExpansion::Enumerate => state.board.remaining_deck(),
        };
        for card in cards {
            let mut child_state = state.clone();
            child_state.board = child_state
                .board
                .push(card)
                .expect("chance card must not duplicate board");
            let child = self.build_state(tree, child_state, depth + 1);
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
                && let Some(raise) = self.raise_action(state)
            {
                actions.push(raise);
            }
            return actions;
        }

        let mut actions = vec![ActionKind::Check];
        let fractions = match state.street {
            Street::Flop => &self.template.action_abstraction.flop.first_bet_fractions,
            Street::Turn => &self.template.action_abstraction.turn.first_bet_fractions,
            Street::River => &self.template.action_abstraction.river.first_bet_fractions,
        };
        for fraction in fractions {
            let amount = sized_amount(
                state.pot,
                *fraction,
                self.template.action_abstraction.min_bet,
                stack,
            );
            push_unique_action(&mut actions, bet_or_all_in(amount, stack));
        }
        actions
    }

    fn raise_action(&self, state: &PublicState) -> Option<ActionKind> {
        let actor_commit = commit_for(state, state.player);
        let opponent_commit = commit_for(state, state.player.other());
        let stack = stack_for(state, state.player);
        let to_call = opponent_commit.saturating_sub(actor_commit);
        let min_raise_to = opponent_commit
            + state
                .last_raise_size
                .max(self.template.action_abstraction.min_bet);
        let geometric_to = ((opponent_commit as f32)
            * self.template.action_abstraction.raise.raise_multiplier)
            .round() as u32;
        let target_to = min_raise_to.max(geometric_to);
        let max_to = actor_commit + stack;
        if target_to >= max_to {
            return Some(ActionKind::AllIn { to: max_to });
        }
        let additional = target_to - actor_commit;
        let pot_after_call = state.pot + to_call.min(stack);
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
        Some(ActionKind::Raise { to: target_to })
    }

    fn apply_action(&self, state: &PublicState, action: ActionKind) -> Transition {
        match action {
            ActionKind::Fold => Transition::Terminal(TerminalReason::Fold),
            ActionKind::Check => {
                if state.checks_this_street >= 1 {
                    return self.close_street(state);
                }
                let mut next = state.clone();
                next.checks_this_street += 1;
                next.player = next.player.other();
                Transition::State(next)
            }
            ActionKind::Call { amount } => {
                let mut next = state.clone();
                commit_chips(&mut next, state.player, amount);
                self.close_street(&next)
            }
            ActionKind::Bet { amount } => {
                let mut next = state.clone();
                commit_chips(&mut next, state.player, amount);
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
                next.player = next.player.other();
                Transition::State(next)
            }
            ActionKind::AllIn { to } => {
                let mut next = state.clone();
                let current = commit_for(&next, state.player);
                commit_chips(&mut next, state.player, to.saturating_sub(current));
                if to_call(&next) == 0 {
                    Transition::Terminal(TerminalReason::AllIn)
                } else {
                    next.raises_this_street += 1;
                    next.player = next.player.other();
                    Transition::State(next)
                }
            }
        }
    }

    fn close_street(&self, state: &PublicState) -> Transition {
        if state.oop_stack == 0 || state.ip_stack == 0 {
            return Transition::Terminal(TerminalReason::AllIn);
        }
        let Some(next_street) = state.street.next() else {
            return Transition::Terminal(TerminalReason::Showdown);
        };
        let mut next = state.clone();
        next.street = next_street;
        next.oop_street_commit = 0;
        next.ip_street_commit = 0;
        next.last_raise_size = 0;
        next.raises_this_street = 0;
        next.checks_this_street = 0;
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
        };
        fn visit(tree: &PublicTree, node: usize, depth: usize, stats: &mut TreeStats) {
            stats.max_depth = stats.max_depth.max(depth);
            match tree.nodes[node].kind {
                PublicNodeKind::Decision { .. } => stats.decisions += 1,
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
    Terminal(TerminalReason),
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
        assert!(stats.decisions > 500, "{stats:?}");
        assert!(stats.chances > 0, "{stats:?}");
        assert!(stats.terminals > stats.decisions, "{stats:?}");
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
}
