use std::time::Instant;

use pokedr_core::{
    cards::{Board, Card as PokedrCard},
    dense_cfr::DenseCfrState,
    postflop::{Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
    range::{COMBO_COUNT, ComboIndexer},
};

use crate::{
    PokedrAgentConfig, fixed_flop_root_weights, format_pokedr_cards, root_board,
    solve_public_tree_cfr,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FixedRiverSolveSummary {
    pub board: String,
    pub iterations: usize,
    pub decisions: usize,
    pub chance: usize,
    pub terminals: usize,
    pub public_infosets: usize,
    pub private_infosets: usize,
    pub max_actions: usize,
    pub elapsed_secs: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedRiverBatchSolveSummary {
    pub boards: Vec<FixedRiverSolveSummary>,
    pub iterations: usize,
    pub total_elapsed_secs: f32,
}

#[derive(Debug, Clone)]
pub struct RiverSubgameInput {
    pub public_state: PublicState,
    pub oop_weights: Vec<f32>,
    pub ip_weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RiverSubgameResult {
    pub summary: FixedRiverSolveSummary,
    pub state: DenseCfrState,
}

#[derive(Debug, Clone)]
pub struct RiverBatchSolver {
    config: PokedrAgentConfig,
}

impl RiverBatchSolver {
    pub fn new(config: PokedrAgentConfig) -> Self {
        Self { config }
    }

    pub fn solve_fixed_board(&self, board_cards: [PokedrCard; 5]) -> RiverSubgameResult {
        self.solve_subgame(RiverSubgameInput::with_default_ranges(board_cards))
    }

    pub fn solve_fixed_boards(&self, boards: &[[PokedrCard; 5]]) -> FixedRiverBatchSolveSummary {
        let started = Instant::now();
        let summaries = boards
            .iter()
            .copied()
            .map(|board| self.solve_fixed_board(board).summary)
            .collect::<Vec<_>>();
        FixedRiverBatchSolveSummary {
            boards: summaries,
            iterations: self.config.cfr_iterations.max(1),
            total_elapsed_secs: started.elapsed().as_secs_f32(),
        }
    }

    pub fn solve_subgames(&self, inputs: Vec<RiverSubgameInput>) -> Vec<RiverSubgameResult> {
        inputs
            .into_iter()
            .map(|input| self.solve_subgame(input))
            .collect()
    }

    pub fn solve_subgame(&self, input: RiverSubgameInput) -> RiverSubgameResult {
        input.validate();
        let started = Instant::now();
        let tree = SubgameTree::build(
            input.public_state,
            SubgameTreeConfig {
                action_set: self.config.action_set.clone(),
                max_raises_per_street: self.config.max_raises_per_street,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&tree);
        let state = solve_public_tree_cfr(
            &tree,
            &layout,
            &self.config,
            &input.oop_weights,
            &input.ip_weights,
        );
        RiverSubgameResult {
            summary: FixedRiverSolveSummary {
                board: format_pokedr_cards(root_board(&tree).cards()),
                iterations: self.config.cfr_iterations.max(1),
                decisions: tree.decision_count(),
                chance: tree.chance_count(),
                terminals: tree.terminal_count(),
                public_infosets: layout.infoset_count(),
                private_infosets: state.infosets(),
                max_actions: layout.max_actions(),
                elapsed_secs: started.elapsed().as_secs_f32(),
            },
            state,
        }
    }
}

impl RiverSubgameInput {
    pub fn with_default_ranges(board_cards: [PokedrCard; 5]) -> Self {
        let public_state = default_river_public_state(board_cards);
        let indexer = ComboIndexer::new();
        let root_dead = public_state.board.deck_mask();
        let (oop_weights, ip_weights) = fixed_flop_root_weights(&indexer, root_dead);
        Self {
            public_state,
            oop_weights,
            ip_weights,
        }
    }

    pub fn validate(&self) {
        assert_eq!(self.public_state.street, Street::River);
        assert_eq!(self.public_state.board.cards().len(), 5);
        assert_eq!(self.oop_weights.len(), COMBO_COUNT);
        assert_eq!(self.ip_weights.len(), COMBO_COUNT);
    }
}

fn default_river_public_state(board_cards: [PokedrCard; 5]) -> PublicState {
    PublicState {
        street: Street::River,
        board: Board::new(board_cards.to_vec()),
        pot: 100,
        hero_invested: 50,
        villain_invested: 50,
        effective_stack: 100,
        to_call: 0,
        min_aggressive_amount: 50,
        acting_player: Player::Hero,
        raises_this_street: 0,
        checks_this_street: 0,
    }
}
