use std::cmp::Ordering;

use crate::cards::{Board, Card};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerAction {
    Fold,
    Check,
    Call { amount: u32 },
    Bet { amount: u32 },
    Raise { amount: u32 },
    AllIn { amount: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingSource {
    Standard,
    Observed,
    Forced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionCandidate {
    pub action: PlayerAction,
    pub source: SizingSource,
}

#[derive(Debug, Clone)]
pub struct ActionSetConfig {
    pub max_aggressive_actions: usize,
    pub merge_log_ratio: f32,
    pub all_in_threshold: f32,
    pub flop_bet_fractions: Vec<f32>,
    pub turn_bet_fractions: Vec<f32>,
    pub river_bet_fractions: Vec<f32>,
    pub raise_fractions: Vec<f32>,
}

impl Default for ActionSetConfig {
    fn default() -> Self {
        Self {
            max_aggressive_actions: 4,
            merge_log_ratio: 0.20,
            all_in_threshold: 0.90,
            flop_bet_fractions: vec![0.33, 0.75],
            turn_bet_fractions: vec![0.50, 1.00],
            river_bet_fractions: vec![0.33, 0.75, 1.25],
            raise_fractions: vec![0.50, 1.00],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionSetRequest {
    pub street: Street,
    pub pot: u32,
    pub stack: u32,
    pub to_call: u32,
    pub min_aggressive_amount: u32,
    pub observed_aggressive_amounts: Vec<u32>,
}

impl ActionSetRequest {
    pub fn can_check(&self) -> bool {
        self.to_call == 0
    }

    pub fn can_raise_or_bet(&self) -> bool {
        self.stack > self.to_call && self.min_aggressive_amount > self.to_call
    }
}

pub struct ActionSetBuilder {
    config: ActionSetConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    Hero,
    Villain,
}

#[derive(Debug, Clone)]
pub struct PublicState {
    pub street: Street,
    pub board: Board,
    pub pot: u32,
    pub hero_invested: u32,
    pub villain_invested: u32,
    pub effective_stack: u32,
    pub to_call: u32,
    pub min_aggressive_amount: u32,
    pub acting_player: Player,
    pub raises_this_street: u8,
    pub checks_this_street: u8,
}

#[derive(Debug, Clone)]
pub struct SubgameTreeConfig {
    pub action_set: ActionSetConfig,
    pub max_raises_per_street: u8,
}

impl Default for SubgameTreeConfig {
    fn default() -> Self {
        Self {
            action_set: ActionSetConfig::default(),
            max_raises_per_street: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Fold,
    Showdown,
}

#[derive(Debug, Clone)]
pub enum PublicNodeKind {
    Decision {
        state: PublicState,
        actions: Vec<ActionCandidate>,
    },
    Chance {
        street: Street,
        board: Board,
        cards: Vec<Card>,
    },
    Terminal {
        kind: TerminalKind,
        board: Board,
        pot: u32,
        hero_invested: u32,
        villain_invested: u32,
    },
}

#[derive(Debug, Clone)]
pub struct PublicNode {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub kind: PublicNodeKind,
}

#[derive(Debug, Clone)]
pub struct SubgameTree {
    nodes: Vec<PublicNode>,
}

impl ActionSetBuilder {
    pub fn new(config: ActionSetConfig) -> Self {
        Self { config }
    }

    pub fn build(&self, request: &ActionSetRequest) -> Vec<ActionCandidate> {
        assert!(request.pot > 0, "pot must be non-empty");
        assert!(request.stack > 0, "stack must be non-empty");
        assert!(
            request.to_call <= request.stack,
            "to_call must fit in stack"
        );

        let mut actions = Vec::new();
        if request.can_check() {
            actions.push(ActionCandidate {
                action: PlayerAction::Check,
                source: SizingSource::Forced,
            });
        } else {
            actions.push(ActionCandidate {
                action: PlayerAction::Fold,
                source: SizingSource::Forced,
            });
            actions.push(ActionCandidate {
                action: PlayerAction::Call {
                    amount: request.to_call,
                },
                source: SizingSource::Forced,
            });
        }

        if request.can_raise_or_bet() {
            actions.extend(self.aggressive_actions(request));
        }
        actions
    }

    fn aggressive_actions(&self, request: &ActionSetRequest) -> Vec<ActionCandidate> {
        let mut sizings = self.standard_sizings(request);
        for observed in request.observed_aggressive_amounts.iter().copied() {
            if let Some(amount) =
                legal_aggressive_amount(request, observed, self.config.all_in_threshold)
            {
                insert_or_replace_near(
                    &mut sizings,
                    AggressiveSizing {
                        amount,
                        source: SizingSource::Observed,
                    },
                    self.config.merge_log_ratio,
                );
            }
        }

        if !request.can_check() {
            let all_in = request.stack;
            insert_or_replace_near(
                &mut sizings,
                AggressiveSizing {
                    amount: all_in,
                    source: SizingSource::Forced,
                },
                self.config.merge_log_ratio,
            );
        }

        sizings.sort_by_key(|sizing| sizing.amount);
        sizings.dedup_by_key(|sizing| sizing.amount);
        self.prune_sizings(request, sizings)
            .into_iter()
            .map(|sizing| ActionCandidate {
                action: aggressive_action(request, sizing.amount),
                source: sizing.source,
            })
            .collect()
    }

    fn standard_sizings(&self, request: &ActionSetRequest) -> Vec<AggressiveSizing> {
        let fractions = if request.can_check() {
            match request.street {
                Street::Flop => &self.config.flop_bet_fractions,
                Street::Turn => &self.config.turn_bet_fractions,
                Street::River => &self.config.river_bet_fractions,
            }
        } else {
            &self.config.raise_fractions
        };

        let base = if request.can_check() {
            request.pot
        } else {
            request.pot.saturating_add(request.to_call)
        };

        let mut sizings = Vec::new();
        for fraction in fractions {
            let raw = (base as f32 * fraction).round() as u32;
            let amount = if request.can_check() {
                raw
            } else {
                request.to_call.saturating_add(raw)
            };
            if let Some(amount) =
                legal_aggressive_amount(request, amount, self.config.all_in_threshold)
            {
                insert_or_replace_near(
                    &mut sizings,
                    AggressiveSizing {
                        amount,
                        source: SizingSource::Standard,
                    },
                    self.config.merge_log_ratio,
                );
            }
        }
        sizings
    }

    fn prune_sizings(
        &self,
        request: &ActionSetRequest,
        mut sizings: Vec<AggressiveSizing>,
    ) -> Vec<AggressiveSizing> {
        let max = self.config.max_aggressive_actions.max(1);
        if sizings.len() <= max {
            return sizings;
        }

        sizings
            .sort_by(|left, right| sizing_rank(request, *left).cmp(&sizing_rank(request, *right)));
        sizings.truncate(max);
        sizings.sort_by_key(|sizing| sizing.amount);
        sizings
    }
}

impl Player {
    pub fn next(self) -> Self {
        match self {
            Self::Hero => Self::Villain,
            Self::Villain => Self::Hero,
        }
    }
}

impl SubgameTree {
    pub fn build(root: PublicState, config: SubgameTreeConfig) -> Self {
        let builder = ActionSetBuilder::new(config.action_set.clone());
        let mut tree = Self { nodes: Vec::new() };
        tree.expand_state(None, root, &config, &builder);
        tree
    }

    pub fn nodes(&self) -> &[PublicNode] {
        &self.nodes
    }

    pub fn decision_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node.kind, PublicNodeKind::Decision { .. }))
            .count()
    }

    pub fn chance_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node.kind, PublicNodeKind::Chance { .. }))
            .count()
    }

    pub fn terminal_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node.kind, PublicNodeKind::Terminal { .. }))
            .count()
    }

    fn push_node(&mut self, parent: Option<usize>, kind: PublicNodeKind) -> usize {
        let index = self.nodes.len();
        self.nodes.push(PublicNode {
            parent,
            children: Vec::new(),
            kind,
        });
        if let Some(parent) = parent {
            self.nodes[parent].children.push(index);
        }
        index
    }

    fn expand_state(
        &mut self,
        parent: Option<usize>,
        state: PublicState,
        config: &SubgameTreeConfig,
        builder: &ActionSetBuilder,
    ) -> usize {
        if state.effective_stack == 0 {
            return self.push_node(
                parent,
                PublicNodeKind::Terminal {
                    kind: TerminalKind::Showdown,
                    board: state.board,
                    pot: state.pot,
                    hero_invested: state.hero_invested,
                    villain_invested: state.villain_invested,
                },
            );
        }
        if state.street == Street::River && state.to_call == 0 && state.checks_this_street >= 2 {
            return self.push_node(
                parent,
                PublicNodeKind::Terminal {
                    kind: TerminalKind::Showdown,
                    board: state.board,
                    pot: state.pot,
                    hero_invested: state.hero_invested,
                    villain_invested: state.villain_invested,
                },
            );
        }

        let mut actions = builder.build(&ActionSetRequest {
            street: state.street,
            pot: state.pot,
            stack: state.effective_stack,
            to_call: state.to_call,
            min_aggressive_amount: state.min_aggressive_amount,
            observed_aggressive_amounts: Vec::new(),
        });
        if state.raises_this_street >= config.max_raises_per_street {
            actions.retain(|candidate| !is_aggressive_action(candidate.action));
        }
        let node = self.push_node(
            parent,
            PublicNodeKind::Decision {
                state: state.clone(),
                actions: actions.clone(),
            },
        );
        for action in actions {
            self.expand_action(node, &state, action.action, config, builder);
        }
        node
    }

    fn expand_action(
        &mut self,
        parent: usize,
        state: &PublicState,
        action: PlayerAction,
        config: &SubgameTreeConfig,
        builder: &ActionSetBuilder,
    ) {
        match action {
            PlayerAction::Fold => {
                self.push_node(
                    Some(parent),
                    PublicNodeKind::Terminal {
                        kind: TerminalKind::Fold,
                        board: state.board.clone(),
                        pot: state.pot,
                        hero_invested: state.hero_invested,
                        villain_invested: state.villain_invested,
                    },
                );
            }
            PlayerAction::Check | PlayerAction::Call { .. } => {
                let advance = should_advance_street(state, action);
                let next_state = state_after_passive_action(state, action);
                if advance {
                    self.expand_chance(Some(parent), next_state, config, builder);
                } else {
                    self.expand_state(Some(parent), next_state, config, builder);
                }
            }
            PlayerAction::Bet { amount }
            | PlayerAction::Raise { amount }
            | PlayerAction::AllIn { amount } => {
                let next_state = state_after_aggressive_action(state, amount);
                self.expand_state(Some(parent), next_state, config, builder);
            }
        }
    }

    fn expand_chance(
        &mut self,
        parent: Option<usize>,
        state: PublicState,
        config: &SubgameTreeConfig,
        builder: &ActionSetBuilder,
    ) {
        if state.street == Street::River {
            self.push_node(
                parent,
                PublicNodeKind::Terminal {
                    kind: TerminalKind::Showdown,
                    board: state.board,
                    pot: state.pot,
                    hero_invested: state.hero_invested,
                    villain_invested: state.villain_invested,
                },
            );
            return;
        }

        let cards = remaining_cards(state.board.deck_mask());
        let chance = self.push_node(
            parent,
            PublicNodeKind::Chance {
                street: next_street(state.street),
                board: state.board.clone(),
                cards: cards.clone(),
            },
        );
        for card in cards {
            let mut child_state = state.clone();
            child_state.street = next_street(state.street);
            child_state.board = state.board.with_card(card);
            child_state.to_call = 0;
            child_state.min_aggressive_amount = child_state.pot.max(1);
            child_state.raises_this_street = 0;
            child_state.checks_this_street = 0;
            self.expand_state(Some(chance), child_state, config, builder);
        }
    }
}

pub fn ordered_runouts(board: &Board) -> Vec<(Card, Card)> {
    assert_eq!(board.cards().len(), 3, "ordered runouts require a flop");
    ordered_runouts_from_dead_mask(board.deck_mask())
}

pub fn ordered_runouts_from_dead_mask(dead_mask: u64) -> Vec<(Card, Card)> {
    let cards = remaining_cards(dead_mask);
    let mut runouts = Vec::with_capacity(cards.len() * (cards.len() - 1));
    for &turn in &cards {
        for &river in &cards {
            if turn != river {
                runouts.push((turn, river));
            }
        }
    }
    runouts
}

pub fn unordered_runouts(board: &Board) -> Vec<(Card, Card)> {
    assert_eq!(board.cards().len(), 3, "unordered runouts require a flop");
    unordered_runouts_from_dead_mask(board.deck_mask())
}

pub fn unordered_runouts_from_dead_mask(dead_mask: u64) -> Vec<(Card, Card)> {
    let cards = remaining_cards(dead_mask);
    let mut runouts = Vec::with_capacity(cards.len() * (cards.len() - 1) / 2);
    for first in 0..cards.len() {
        for second in first + 1..cards.len() {
            runouts.push((cards[first], cards[second]));
        }
    }
    runouts
}

fn remaining_cards(dead_mask: u64) -> Vec<Card> {
    (0..Card::COUNT as u8)
        .map(Card::from_index)
        .filter(|card| card.deck_mask() & dead_mask == 0)
        .collect()
}

fn next_street(street: Street) -> Street {
    match street {
        Street::Flop => Street::Turn,
        Street::Turn => Street::River,
        Street::River => Street::River,
    }
}

fn should_advance_street(state: &PublicState, action: PlayerAction) -> bool {
    match action {
        PlayerAction::Call { .. } => true,
        PlayerAction::Check => state.to_call == 0 && state.checks_this_street + 1 >= 2,
        _ => false,
    }
}

fn is_aggressive_action(action: PlayerAction) -> bool {
    matches!(
        action,
        PlayerAction::Bet { .. } | PlayerAction::Raise { .. } | PlayerAction::AllIn { .. }
    )
}

fn state_after_passive_action(state: &PublicState, action: PlayerAction) -> PublicState {
    let call_amount = match action {
        PlayerAction::Call { amount } => amount,
        _ => 0,
    };
    let mut next = state.clone();
    next.pot = next.pot.saturating_add(call_amount);
    add_investment(&mut next, state.acting_player, call_amount);
    next.effective_stack = next.effective_stack.saturating_sub(call_amount);
    next.to_call = 0;
    next.acting_player = state.acting_player.next();
    next.checks_this_street = match action {
        PlayerAction::Check => next.checks_this_street.saturating_add(1),
        PlayerAction::Call { .. } => 0,
        _ => next.checks_this_street,
    };
    next
}

fn state_after_aggressive_action(state: &PublicState, amount: u32) -> PublicState {
    let contribution = amount.min(state.effective_stack);
    let mut next = state.clone();
    next.pot = next.pot.saturating_add(contribution);
    add_investment(&mut next, state.acting_player, contribution);
    next.to_call = match state.acting_player {
        Player::Hero => next.hero_invested.saturating_sub(next.villain_invested),
        Player::Villain => next.villain_invested.saturating_sub(next.hero_invested),
    };
    next.min_aggressive_amount = contribution.saturating_mul(2).max(1);
    next.acting_player = state.acting_player.next();
    next.raises_this_street = next.raises_this_street.saturating_add(1);
    next.checks_this_street = 0;
    next
}

fn add_investment(state: &mut PublicState, player: Player, amount: u32) {
    match player {
        Player::Hero => state.hero_invested = state.hero_invested.saturating_add(amount),
        Player::Villain => state.villain_invested = state.villain_invested.saturating_add(amount),
    }
}

#[derive(Debug, Clone, Copy)]
struct AggressiveSizing {
    amount: u32,
    source: SizingSource,
}

fn legal_aggressive_amount(
    request: &ActionSetRequest,
    amount: u32,
    all_in_threshold: f32,
) -> Option<u32> {
    if request.min_aggressive_amount > request.stack {
        return (request.stack > request.to_call).then_some(request.stack);
    }
    if amount >= all_in_absorb_threshold(request, all_in_threshold) {
        return Some(request.stack);
    }
    let amount = amount.clamp(request.min_aggressive_amount, request.stack);
    if amount <= request.to_call {
        None
    } else {
        Some(amount)
    }
}

fn all_in_absorb_threshold(request: &ActionSetRequest, all_in_threshold: f32) -> u32 {
    let threshold = request.stack as f32 * all_in_threshold;
    threshold.ceil() as u32
}

fn aggressive_action(request: &ActionSetRequest, amount: u32) -> PlayerAction {
    if amount == request.stack {
        PlayerAction::AllIn { amount }
    } else if request.can_check() {
        PlayerAction::Bet { amount }
    } else {
        PlayerAction::Raise { amount }
    }
}

fn insert_or_replace_near(
    sizings: &mut Vec<AggressiveSizing>,
    incoming: AggressiveSizing,
    merge_log_ratio: f32,
) {
    if let Some(index) = nearest_index(sizings, incoming.amount, merge_log_ratio) {
        let current = sizings[index];
        if should_replace(current, incoming) {
            sizings[index] = incoming;
        }
    } else {
        sizings.push(incoming);
    }
}

fn nearest_index(sizings: &[AggressiveSizing], amount: u32, merge_log_ratio: f32) -> Option<usize> {
    sizings
        .iter()
        .enumerate()
        .filter_map(|(index, sizing)| {
            let distance = log_ratio_distance(sizing.amount, amount);
            (distance <= merge_log_ratio).then_some((index, distance))
        })
        .min_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
}

fn log_ratio_distance(left: u32, right: u32) -> f32 {
    ((left.max(1) as f32) / (right.max(1) as f32)).ln().abs()
}

fn should_replace(current: AggressiveSizing, incoming: AggressiveSizing) -> bool {
    source_priority(incoming.source) >= source_priority(current.source)
}

fn source_priority(source: SizingSource) -> u8 {
    match source {
        SizingSource::Standard => 0,
        SizingSource::Observed => 1,
        SizingSource::Forced => 2,
    }
}

fn sizing_rank(request: &ActionSetRequest, sizing: AggressiveSizing) -> (u8, u32) {
    let source_rank = match sizing.source {
        SizingSource::Forced => 0,
        SizingSource::Observed => 1,
        SizingSource::Standard => 2,
    };
    let middle = if request.can_check() {
        request.pot
    } else {
        request.to_call + request.pot
    };
    let distance = sizing.amount.abs_diff(middle);
    (source_rank, distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{Card, Rank, Suit};

    fn builder() -> ActionSetBuilder {
        ActionSetBuilder::new(ActionSetConfig::default())
    }

    #[test]
    fn check_spot_includes_standard_bets_without_forced_all_in() {
        let actions = builder().build(&ActionSetRequest {
            street: Street::Flop,
            pot: 100,
            stack: 500,
            to_call: 0,
            min_aggressive_amount: 10,
            observed_aggressive_amounts: vec![],
        });

        assert_eq!(actions[0].action, PlayerAction::Check);
        assert!(
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Bet { amount: 33 })
        );
        assert!(
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Bet { amount: 75 })
        );
        assert!(
            !actions
                .iter()
                .any(|candidate| matches!(candidate.action, PlayerAction::AllIn { .. }))
        );
    }

    #[test]
    fn observed_sizing_replaces_near_standard_sizing() {
        let actions = builder().build(&ActionSetRequest {
            street: Street::Turn,
            pot: 100,
            stack: 500,
            to_call: 0,
            min_aggressive_amount: 10,
            observed_aggressive_amounts: vec![55],
        });

        assert!(actions.iter().any(|candidate| candidate.action
            == PlayerAction::Bet { amount: 55 }
            && candidate.source == SizingSource::Observed));
        assert!(
            !actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Bet { amount: 50 })
        );
    }

    #[test]
    fn facing_bet_keeps_fold_call_observed_raise_and_all_in() {
        let actions = builder().build(&ActionSetRequest {
            street: Street::River,
            pot: 180,
            stack: 420,
            to_call: 80,
            min_aggressive_amount: 200,
            observed_aggressive_amounts: vec![260],
        });

        assert!(
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Fold)
        );
        assert!(
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Call { amount: 80 })
        );
        assert!(actions.iter().any(|candidate| candidate.action
            == PlayerAction::Raise { amount: 260 }
            && candidate.source == SizingSource::Observed));
        assert!(
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::AllIn { amount: 420 })
        );
    }

    #[test]
    fn aggressive_action_cap_preserves_observed_without_forced_check_spot_all_in() {
        let config = ActionSetConfig {
            max_aggressive_actions: 2,
            ..ActionSetConfig::default()
        };
        let actions = ActionSetBuilder::new(config).build(&ActionSetRequest {
            street: Street::River,
            pot: 100,
            stack: 1000,
            to_call: 0,
            min_aggressive_amount: 10,
            observed_aggressive_amounts: vec![220],
        });

        let aggressive: Vec<_> = actions
            .iter()
            .filter(|candidate| !matches!(candidate.action, PlayerAction::Check))
            .collect();
        assert_eq!(aggressive.len(), 2);
        assert!(
            aggressive
                .iter()
                .any(|candidate| candidate.action == PlayerAction::Bet { amount: 220 })
        );
        assert!(
            !aggressive
                .iter()
                .any(|candidate| matches!(candidate.action, PlayerAction::AllIn { .. }))
        );
        assert!(
            aggressive
                .iter()
                .any(|candidate| candidate.source == SizingSource::Standard)
        );
    }

    fn flop() -> Board {
        Board::new(vec![
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Seven, Suit::Hearts),
            Card::new(Rank::Two, Suit::Clubs),
        ])
    }

    fn root_state(street: Street) -> PublicState {
        PublicState {
            street,
            board: flop(),
            pot: 100,
            hero_invested: 50,
            villain_invested: 50,
            effective_stack: 300,
            to_call: 0,
            min_aggressive_amount: 50,
            acting_player: Player::Hero,
            raises_this_street: 0,
            checks_this_street: 0,
        }
    }

    fn compact_tree_config() -> SubgameTreeConfig {
        SubgameTreeConfig {
            action_set: ActionSetConfig {
                max_aggressive_actions: 2,
                flop_bet_fractions: vec![0.5],
                turn_bet_fractions: vec![0.5],
                river_bet_fractions: vec![0.5],
                raise_fractions: vec![1.0],
                ..ActionSetConfig::default()
            },
            max_raises_per_street: 1,
        }
    }

    #[test]
    fn flop_runout_counts_cover_all_turn_river_cards() {
        assert_eq!(ordered_runouts(&flop()).len(), 49 * 48);
        assert_eq!(unordered_runouts(&flop()).len(), 49 * 48 / 2);
    }

    #[test]
    fn combo_specific_runout_counts_exclude_private_cards() {
        let board = flop();
        let hero_combo = Card::new(Rank::King, Suit::Spades).deck_mask()
            | Card::new(Rank::Queen, Suit::Spades).deck_mask();
        let dead_mask = board.deck_mask() | hero_combo;

        assert_eq!(ordered_runouts_from_dead_mask(dead_mask).len(), 47 * 46);
        assert_eq!(
            unordered_runouts_from_dead_mask(dead_mask).len(),
            47 * 46 / 2
        );
    }

    #[test]
    fn subgame_tree_builds_finite_public_nodes() {
        let tree = SubgameTree::build(root_state(Street::Flop), compact_tree_config());

        assert!(!tree.nodes().is_empty());
        assert!(tree.decision_count() > 0);
        assert!(tree.chance_count() > 0);
        assert!(tree.terminal_count() > 0);
        assert!(tree.nodes().len() < 500_000);
    }

    #[test]
    fn check_then_check_advances_to_turn_chance() {
        let tree = SubgameTree::build(root_state(Street::Flop), compact_tree_config());

        let root_check = tree.nodes()[0].children[0];
        let PublicNodeKind::Decision {
            state: checked_state,
            ..
        } = &tree.nodes()[root_check].kind
        else {
            panic!("first check should keep action on the flop");
        };
        assert_eq!(checked_state.street, Street::Flop);
        assert_eq!(checked_state.checks_this_street, 1);

        let second_check = tree.nodes()[root_check].children[0];
        let PublicNodeKind::Chance { street, cards, .. } = &tree.nodes()[second_check].kind else {
            panic!("second check should advance through chance");
        };
        assert_eq!(*street, Street::Turn);
        assert_eq!(cards.len(), 49);
    }

    #[test]
    fn bet_call_advances_to_turn_chance() {
        let tree = SubgameTree::build(root_state(Street::Flop), compact_tree_config());
        let root = &tree.nodes()[0];

        let facing_bet = root
            .children
            .iter()
            .copied()
            .find(|&child| {
                matches!(
                    &tree.nodes()[child].kind,
                    PublicNodeKind::Decision { state, .. } if state.to_call > 0
                )
            })
            .expect("root should contain at least one bet branch");

        let call_child = tree.nodes()[facing_bet]
            .children
            .iter()
            .copied()
            .find(|&child| matches!(tree.nodes()[child].kind, PublicNodeKind::Chance { .. }))
            .expect("call should advance through chance");

        let PublicNodeKind::Chance { street, cards, .. } = &tree.nodes()[call_child].kind else {
            unreachable!();
        };
        assert_eq!(*street, Street::Turn);
        assert_eq!(cards.len(), 49);
    }

    #[test]
    fn large_bet_keeps_next_actor_able_to_call() {
        let state = PublicState {
            effective_stack: 100,
            ..root_state(Street::Flop)
        };
        let next = state_after_aggressive_action(&state, 75);

        assert_eq!(next.to_call, 75);
        assert_eq!(next.effective_stack, 100);
    }

    #[test]
    fn raise_to_call_is_investment_difference() {
        let state = PublicState {
            hero_invested: 125,
            villain_invested: 50,
            to_call: 75,
            acting_player: Player::Villain,
            ..root_state(Street::Flop)
        };
        let next = state_after_aggressive_action(&state, 175);

        assert_eq!(next.villain_invested, 225);
        assert_eq!(next.to_call, 100);
    }

    #[test]
    fn river_check_check_ends_at_showdown() {
        let tree = SubgameTree::build(root_state(Street::River), compact_tree_config());

        let root_check = tree.nodes()[0].children[0];
        let second_check = tree.nodes()[root_check].children[0];
        let PublicNodeKind::Terminal { kind, .. } = tree.nodes()[second_check].kind else {
            panic!("river check-check should finish the public tree");
        };
        assert_eq!(kind, TerminalKind::Showdown);
    }

    #[test]
    fn flop_tree_reaches_river_decisions() {
        let tree = SubgameTree::build(root_state(Street::Flop), compact_tree_config());

        let river_decisions = tree
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    &node.kind,
                    PublicNodeKind::Decision { state, .. } if state.street == Street::River
                )
            })
            .count();

        assert!(river_decisions > 0);
    }
}
