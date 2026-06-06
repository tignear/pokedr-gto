use std::{cell::RefCell, collections::HashMap, time::Instant};

use pokedr_core::{
    cards::{Board, Card as PokedrCard},
    dense_cfr::{BatchedPrivateCfrConfig, BatchedPrivateCfrState, DenseCfrState},
    postflop::{Player, PublicState, Street, SubgameTree, SubgameTreeConfig},
    postflop_dense::PostflopDenseLayout,
    range::{COMBO_COUNT, ComboIndexer},
};

use crate::{
    PokedrAgentConfig, ShowdownMatrixCache, active_weighted_combos,
    cpu_infoset_profile_hero_payoff_matrix, fixed_flop_root_weights, format_pokedr_cards,
    root_board, showdown_matrix_cache_capacity, solve_public_tree_cfr,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiverShapeBatchPlanSummary {
    pub shape_groups: usize,
    pub total_inputs: usize,
    pub largest_group: usize,
    pub public_infosets: usize,
    pub max_actions: usize,
    pub combos: usize,
    pub action_slots: usize,
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
    pub oop_cfv: Vec<f32>,
    pub ip_cfv: Vec<f32>,
    pub state: DenseCfrState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RiverSubgameShapeKey {
    pot: u32,
    hero_invested: u32,
    villain_invested: u32,
    effective_stack: u32,
    to_call: u32,
    min_aggressive_amount: u32,
    acting_player: u8,
    raises_this_street: u8,
    checks_this_street: u8,
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
        let inputs = boards
            .iter()
            .copied()
            .map(RiverSubgameInput::with_default_ranges)
            .collect::<Vec<_>>();
        let summaries = self
            .solve_subgames(inputs)
            .into_iter()
            .map(|result| result.summary)
            .collect();
        FixedRiverBatchSolveSummary {
            boards: summaries,
            iterations: self.config.cfr_iterations.max(1),
            total_elapsed_secs: started.elapsed().as_secs_f32(),
        }
    }

    pub fn solve_subgames(&self, inputs: Vec<RiverSubgameInput>) -> Vec<RiverSubgameResult> {
        let mut groups: HashMap<RiverSubgameShapeKey, Vec<(usize, RiverSubgameInput)>> =
            HashMap::new();
        for (index, input) in inputs.into_iter().enumerate() {
            input.validate();
            groups
                .entry(input.shape_key())
                .or_default()
                .push((index, input));
        }

        let mut indexed_results = Vec::new();
        for group in groups.into_values() {
            indexed_results.extend(self.solve_shape_group(group));
        }
        indexed_results.sort_by_key(|(index, _)| *index);
        indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect()
    }

    pub fn shape_batch_plan_summary(
        &self,
        inputs: &[RiverSubgameInput],
    ) -> RiverShapeBatchPlanSummary {
        assert!(!inputs.is_empty(), "river batch inputs must be non-empty");
        for input in inputs {
            input.validate();
        }
        let mut groups: HashMap<RiverSubgameShapeKey, usize> = HashMap::new();
        for input in inputs {
            *groups.entry(input.shape_key()).or_default() += 1;
        }
        let largest_group = groups.values().copied().max().unwrap_or(0);
        let template_tree = SubgameTree::build(
            inputs[0].public_state.clone(),
            SubgameTreeConfig {
                action_set: self.config.action_set.clone(),
                max_raises_per_street: self.config.max_raises_per_street,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&template_tree);
        let batch_config = BatchedPrivateCfrConfig {
            batches: largest_group.max(1),
            public_infosets: layout.infoset_count(),
            combos: COMBO_COUNT,
            actions: layout.max_actions(),
            variant: self.config.cfr_variant,
        };
        let batch_state = BatchedPrivateCfrState::new(batch_config.clone(), layout.legal_actions());
        RiverShapeBatchPlanSummary {
            shape_groups: groups.len(),
            total_inputs: inputs.len(),
            largest_group,
            public_infosets: batch_config.public_infosets,
            max_actions: batch_config.actions,
            combos: batch_config.combos,
            action_slots: batch_state.config().action_slots(),
        }
    }

    pub fn solve_subgame(&self, input: RiverSubgameInput) -> RiverSubgameResult {
        input.validate();
        let template_tree = SubgameTree::build(
            input.public_state.clone(),
            SubgameTreeConfig {
                action_set: self.config.action_set.clone(),
                max_raises_per_street: self.config.max_raises_per_street,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&template_tree);
        self.solve_subgame_with_layout(input, &template_tree, &layout)
    }

    fn solve_shape_group(
        &self,
        mut group: Vec<(usize, RiverSubgameInput)>,
    ) -> Vec<(usize, RiverSubgameResult)> {
        debug_assert!(!group.is_empty());
        group.sort_by_key(|(index, _)| *index);
        let template_state = group[0].1.public_state.clone();
        let template_tree = SubgameTree::build(
            template_state,
            SubgameTreeConfig {
                action_set: self.config.action_set.clone(),
                max_raises_per_street: self.config.max_raises_per_street,
            },
        );
        let layout = PostflopDenseLayout::from_tree(&template_tree);
        group
            .into_iter()
            .map(|(index, input)| {
                let result = self.solve_subgame_with_layout(input, &template_tree, &layout);
                (index, result)
            })
            .collect()
    }

    fn solve_subgame_with_layout(
        &self,
        input: RiverSubgameInput,
        template_tree: &SubgameTree,
        layout: &PostflopDenseLayout,
    ) -> RiverSubgameResult {
        let started = Instant::now();
        let tree = template_tree.with_replaced_board(input.public_state.board.clone());
        let state = solve_public_tree_cfr(
            &tree,
            layout,
            &self.config,
            &input.oop_weights,
            &input.ip_weights,
        );
        let (oop_cfv, ip_cfv) = river_root_average_profile_cfvs(
            &tree,
            layout,
            &state,
            &input.oop_weights,
            &input.ip_weights,
            self.config.max_showdown_runouts,
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
            oop_cfv,
            ip_cfv,
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

    pub fn shape_key(&self) -> RiverSubgameShapeKey {
        RiverSubgameShapeKey {
            pot: self.public_state.pot,
            hero_invested: self.public_state.hero_invested,
            villain_invested: self.public_state.villain_invested,
            effective_stack: self.public_state.effective_stack,
            to_call: self.public_state.to_call,
            min_aggressive_amount: self.public_state.min_aggressive_amount,
            acting_player: match self.public_state.acting_player {
                Player::Hero => 0,
                Player::Villain => 1,
            },
            raises_this_street: self.public_state.raises_this_street,
            checks_this_street: self.public_state.checks_this_street,
        }
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

fn river_root_average_profile_cfvs(
    tree: &SubgameTree,
    layout: &PostflopDenseLayout,
    state: &DenseCfrState,
    oop_weights: &[f32],
    ip_weights: &[f32],
    max_showdown_runouts: usize,
) -> (Vec<f32>, Vec<f32>) {
    let indexer = ComboIndexer::new();
    let root_dead = root_board(tree).deck_mask();
    let oop_combos = active_weighted_combos(&indexer, root_dead, oop_weights);
    let ip_combos = active_weighted_combos(&indexer, root_dead, ip_weights);
    let matrix_cache = RefCell::new(ShowdownMatrixCache::new(showdown_matrix_cache_capacity()));
    let pair_values = cpu_infoset_profile_hero_payoff_matrix(
        tree,
        layout,
        0,
        None,
        None,
        &oop_combos,
        &ip_combos,
        state,
        &matrix_cache,
        max_showdown_runouts,
    );

    let mut oop_cfv = vec![0.0f64; COMBO_COUNT];
    let mut ip_cfv = vec![0.0f64; COMBO_COUNT];
    for (oop_local, oop_combo) in oop_combos.iter().enumerate() {
        for (ip_local, ip_combo) in ip_combos.iter().enumerate() {
            if oop_combo.mask & ip_combo.mask != 0 {
                continue;
            }
            let pair = oop_local * ip_combos.len() + ip_local;
            let hero_payoff = pair_values[pair] as f64;
            oop_cfv[oop_combo.index] += ip_combo.weight as f64 * hero_payoff;
            ip_cfv[ip_combo.index] += oop_combo.weight as f64 * -hero_payoff;
        }
    }
    (
        oop_cfv.into_iter().map(|value| value as f32).collect(),
        ip_cfv.into_iter().map(|value| value as f32).collect(),
    )
}
