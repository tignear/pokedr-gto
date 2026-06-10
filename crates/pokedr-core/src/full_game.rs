use crate::Board;
use crate::isomorphism::full_deck_flop_isomorphism_survey;
use crate::plan::CfrStorageConfig;
use crate::range::RangeSpec;
use crate::tree::{
    ActionAbstraction, ChanceExpansion, Player, PublicNodeKind, Spot, Street, TerminalReason,
    TreeBuildError, TreeBuilder, TreeTemplate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct HuFullGameConfig {
    pub small_blind: u32,
    pub big_blind: u32,
    pub ante: u32,
    pub sb_stack: u32,
    pub bb_stack: u32,
    pub sb_range: RangeSpec,
    pub bb_range: RangeSpec,
    pub preflop: HuPreflopActionTemplate,
    pub postflop: ActionAbstraction,
    pub storage: CfrStorageConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HuPreflopActionTemplate {
    pub open_raise_sizes_bb: &'static [f32],
    pub limp_iso_sizes_bb: &'static [f32],
    pub facing_raise_size_multipliers: &'static [f32],
    pub max_raises: u8,
    pub add_all_in_spr_threshold: f32,
    pub force_all_in_remaining_stack_fraction: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HuFullGamePlan {
    pub preflop: HuPreflopPlan,
    pub flop_classes: usize,
    pub flop_concrete: usize,
    pub representative_subgames: u128,
    pub postflop: Vec<BoundaryPostflopPlan>,
    pub streaming: HuFullGameStreamingPlan,
    pub total_action_slots: u128,
    pub total_storage_bytes: u128,
    pub compact_strategy_storage_bytes: u128,
    pub terminal_cfv_calls_per_iteration: u128,
    pub terminal_pair_upper_bound_per_iteration: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuPreflopPlan {
    pub nodes: usize,
    pub decisions: usize,
    pub fold_terminals: usize,
    pub all_in_terminals: usize,
    pub postflop_boundaries: Vec<HuPostflopBoundaryGroup>,
    pub action_slots: u128,
    pub storage_bytes: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuPostflopBoundaryGroup {
    pub pot: u32,
    pub effective_stack: u32,
    pub sb_commit: u32,
    pub bb_commit: u32,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepresentativePostflopPlan {
    pub nodes: u128,
    pub decisions: u128,
    pub decisions_by_street: [u128; 3],
    pub chances: u128,
    pub terminals: u128,
    pub action_slots: u128,
    pub action_slots_by_street: [u128; 3],
    pub storage_bytes: u128,
    pub terminal_cfv_calls: u128,
    pub terminal_pair_upper_bound: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuFullGameStreamingPlan {
    pub chunk_target_bytes: u128,
    pub postflop_chunks: u128,
    pub max_postflop_chunk_bytes: u128,
    pub max_resident_bytes: u128,
    pub disk_state_bytes: u128,
    pub read_write_bytes_per_iteration: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPostflopPlan {
    pub boundary: HuPostflopBoundaryGroup,
    pub representative_flop: RepresentativePostflopPlan,
    pub representative_subgames: u128,
    pub chunks_per_representative: u128,
    pub total_chunks: u128,
    pub max_chunk_bytes: u128,
    pub action_slots: u128,
    pub storage_bytes: u128,
    pub terminal_cfv_calls: u128,
    pub terminal_pair_upper_bound: u128,
}

impl Default for HuPreflopActionTemplate {
    fn default() -> Self {
        Self {
            open_raise_sizes_bb: &[2.0, 2.5],
            limp_iso_sizes_bb: &[3.0, 4.0],
            facing_raise_size_multipliers: &[2.5],
            max_raises: 2,
            add_all_in_spr_threshold: 1.5,
            force_all_in_remaining_stack_fraction: 0.15,
        }
    }
}

impl HuFullGameConfig {
    pub fn hu_spin_15bb(
        sb_range: RangeSpec,
        bb_range: RangeSpec,
        postflop: ActionAbstraction,
    ) -> Self {
        Self {
            small_blind: 50,
            big_blind: 100,
            ante: 0,
            sb_stack: 1500,
            bb_stack: 1500,
            sb_range,
            bb_range,
            preflop: HuPreflopActionTemplate::default(),
            postflop,
            storage: CfrStorageConfig::default(),
        }
    }
}

impl HuFullGamePlan {
    pub fn storage_gib(&self) -> f64 {
        bytes_to_gib(self.total_storage_bytes)
    }

    pub fn compact_strategy_storage_gib(&self) -> f64 {
        bytes_to_gib(self.compact_strategy_storage_bytes)
    }
}

impl HuPreflopPlan {
    pub fn storage_gib(&self) -> f64 {
        bytes_to_gib(self.storage_bytes)
    }
}

impl HuFullGameStreamingPlan {
    pub fn max_postflop_chunk_mib(&self) -> f64 {
        bytes_to_mib(self.max_postflop_chunk_bytes)
    }

    pub fn max_resident_gib(&self) -> f64 {
        bytes_to_gib(self.max_resident_bytes)
    }

    pub fn disk_state_gib(&self) -> f64 {
        bytes_to_gib(self.disk_state_bytes)
    }

    pub fn read_write_gib_per_iteration(&self) -> f64 {
        bytes_to_gib(self.read_write_bytes_per_iteration)
    }
}

impl RepresentativePostflopPlan {
    pub fn storage_gib(&self) -> f64 {
        bytes_to_gib(self.storage_bytes)
    }
}

pub fn plan_hu_full_game(config: &HuFullGameConfig) -> Result<HuFullGamePlan, String> {
    validate_hu_config(config)?;
    let preflop = plan_preflop(config);
    let flop_survey = full_deck_flop_isomorphism_survey(&config.bb_range, &config.sb_range)?;
    let mut postflop = Vec::with_capacity(preflop.postflop_boundaries.len());
    let mut representative_subgames = 0u128;
    let mut postflop_action_slots = 0u128;
    let mut postflop_storage_bytes = 0u128;
    let mut postflop_chunks = 0u128;
    let mut max_postflop_chunk_bytes = 0u128;
    let mut terminal_cfv_calls_per_iteration = 0u128;
    let mut terminal_pair_upper_bound_per_iteration = 0u128;
    for boundary in &preflop.postflop_boundaries {
        let representative_flop = plan_representative_postflop(config, boundary)?;
        let boundary_subgames = boundary.count as u128 * flop_survey.classes.len() as u128;
        let action_slots = representative_flop.action_slots * boundary_subgames;
        let storage_bytes = representative_flop.storage_bytes * boundary_subgames;
        let terminal_cfv_calls = representative_flop.terminal_cfv_calls * boundary_subgames;
        let terminal_pair_upper_bound =
            representative_flop.terminal_pair_upper_bound * boundary_subgames;
        representative_subgames += boundary_subgames;
        postflop_action_slots += action_slots;
        postflop_storage_bytes += storage_bytes;
        terminal_cfv_calls_per_iteration += terminal_cfv_calls;
        terminal_pair_upper_bound_per_iteration += terminal_pair_upper_bound;
        let chunks_per_representative = ceil_div(
            representative_flop.storage_bytes,
            config
                .storage
                .chunk_target_bytes
                .max(config.storage.regret_bytes + config.storage.strategy_sum_bytes),
        )
        .max(if representative_flop.storage_bytes == 0 {
            0
        } else {
            1
        });
        let max_chunk_bytes = if chunks_per_representative == 0 {
            0
        } else {
            ceil_div(representative_flop.storage_bytes, chunks_per_representative)
        };
        let total_chunks = chunks_per_representative * boundary_subgames;
        postflop_chunks += total_chunks;
        max_postflop_chunk_bytes = max_postflop_chunk_bytes.max(max_chunk_bytes);
        postflop.push(BoundaryPostflopPlan {
            boundary: boundary.clone(),
            representative_flop,
            representative_subgames: boundary_subgames,
            chunks_per_representative,
            total_chunks,
            max_chunk_bytes,
            action_slots,
            storage_bytes,
            terminal_cfv_calls,
            terminal_pair_upper_bound,
        });
    }

    let total_action_slots = preflop.action_slots + postflop_action_slots;
    let total_storage_bytes = preflop.storage_bytes + postflop_storage_bytes;
    let compact_strategy_storage_bytes = total_action_slots * (config.storage.regret_bytes + 2);
    let streaming = HuFullGameStreamingPlan {
        chunk_target_bytes: config.storage.chunk_target_bytes,
        postflop_chunks,
        max_postflop_chunk_bytes,
        max_resident_bytes: preflop.storage_bytes + max_postflop_chunk_bytes,
        disk_state_bytes: postflop_storage_bytes,
        read_write_bytes_per_iteration: postflop_storage_bytes * 2,
    };

    Ok(HuFullGamePlan {
        preflop,
        flop_classes: flop_survey.classes.len(),
        flop_concrete: flop_survey.concrete_flops,
        representative_subgames,
        postflop,
        streaming,
        total_action_slots,
        total_storage_bytes,
        compact_strategy_storage_bytes,
        terminal_cfv_calls_per_iteration,
        terminal_pair_upper_bound_per_iteration,
    })
}

fn validate_hu_config(config: &HuFullGameConfig) -> Result<(), String> {
    if config.small_blind == 0 || config.big_blind == 0 || config.big_blind <= config.small_blind {
        return Err("HU full-game planner requires positive blinds with BB > SB".to_string());
    }
    if config.sb_stack <= config.small_blind || config.bb_stack <= config.big_blind {
        return Err("HU full-game planner requires stacks larger than posted blinds".to_string());
    }
    if config.preflop.max_raises == 0 {
        return Err("HU preflop template must allow at least one raise".to_string());
    }
    Ok(())
}

fn plan_preflop(config: &HuFullGameConfig) -> HuPreflopPlan {
    let mut builder = PreflopPlanner {
        config,
        nodes: 0,
        decisions: 0,
        fold_terminals: 0,
        all_in_terminals: 0,
        boundaries: Vec::new(),
        action_slots: 0,
    };
    builder.visit(PreflopState::root(config));
    builder.finish()
}

fn plan_representative_postflop(
    config: &HuFullGameConfig,
    boundary: &HuPostflopBoundaryGroup,
) -> Result<RepresentativePostflopPlan, String> {
    let sample_flop = Board::new(vec![
        "2c".parse().unwrap(),
        "2d".parse().unwrap(),
        "2h".parse().unwrap(),
    ])?;
    let spot = Spot {
        board: sample_flop,
        pot: boundary.pot,
        effective_stack: boundary.effective_stack,
        oop_range: config.bb_range.clone(),
        ip_range: config.sb_range.clone(),
        first_player: Player::Oop,
    };
    let tree = TreeBuilder::new(TreeTemplate {
        action_abstraction: config.postflop.clone(),
        chance_expansion: ChanceExpansion::Isomorphic,
    })
    .map_err(format_tree_build_error)?
    .build(spot)
    .map_err(format_tree_build_error)?;
    let stats = tree.stats();
    let action_slots_by_street =
        postflop_action_slots_by_street(&tree, &config.bb_range, &config.sb_range);
    let action_slots = action_slots_by_street.iter().sum();
    let terminal_work = postflop_terminal_work(&tree, &config.bb_range, &config.sb_range);
    Ok(RepresentativePostflopPlan {
        nodes: stats.nodes as u128,
        decisions: stats.decisions as u128,
        decisions_by_street: stats.decisions_by_street.map(|value| value as u128),
        chances: stats.chances as u128,
        terminals: stats.terminals as u128,
        action_slots,
        action_slots_by_street,
        storage_bytes: action_slots
            * (config.storage.regret_bytes + config.storage.strategy_sum_bytes),
        terminal_cfv_calls: terminal_work.terminal_cfv_calls,
        terminal_pair_upper_bound: terminal_work.private_pair_upper_bound,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct PostflopTerminalWork {
    terminal_cfv_calls: u128,
    private_pair_upper_bound: u128,
}

fn postflop_action_slots_by_street(
    tree: &crate::PublicTree,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> [u128; 3] {
    let mut slots = [0u128; 3];
    for node in &tree.nodes {
        if let PublicNodeKind::Decision { player, actions } = &node.kind {
            let combos = match player {
                Player::Oop => oop_range
                    .without_board_conflicts(&node.state.board)
                    .combos()
                    .len(),
                Player::Ip => ip_range
                    .without_board_conflicts(&node.state.board)
                    .combos()
                    .len(),
            };
            let street = match node.state.street {
                Street::Flop => 0,
                Street::Turn => 1,
                Street::River => 2,
            };
            slots[street] += combos as u128 * actions.len() as u128;
        }
    }
    slots
}

fn postflop_terminal_work(
    tree: &crate::PublicTree,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
) -> PostflopTerminalWork {
    let mut work = PostflopTerminalWork::default();
    for node in &tree.nodes {
        let PublicNodeKind::Terminal { reason } = node.kind else {
            continue;
        };
        let calls = match reason {
            TerminalReason::Fold => 0,
            TerminalReason::Showdown => 1,
            TerminalReason::AllIn => all_in_runout_upper_bound(node.state.street),
        };
        if calls == 0 {
            continue;
        }
        let oop = oop_range
            .without_board_conflicts(&node.state.board)
            .combos()
            .len() as u128;
        let ip = ip_range
            .without_board_conflicts(&node.state.board)
            .combos()
            .len() as u128;
        work.terminal_cfv_calls += calls;
        work.private_pair_upper_bound += calls * oop * ip;
    }
    work
}

fn all_in_runout_upper_bound(street: Street) -> u128 {
    match street {
        Street::Flop => 49 * 48,
        Street::Turn => 48,
        Street::River => 1,
    }
}

fn format_tree_build_error(error: TreeBuildError) -> String {
    format!("failed to build representative postflop tree: {error:?}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seat {
    Sb,
    Bb,
}

impl Seat {
    fn other(self) -> Self {
        match self {
            Self::Sb => Self::Bb,
            Self::Bb => Self::Sb,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreflopState {
    pot: u32,
    sb_stack: u32,
    bb_stack: u32,
    sb_commit: u32,
    bb_commit: u32,
    last_raise_size: u32,
    raises: u8,
    player: Seat,
    limped: bool,
}

impl PreflopState {
    fn root(config: &HuFullGameConfig) -> Self {
        let pot = config.small_blind + config.big_blind + config.ante * 2;
        Self {
            pot,
            sb_stack: config.sb_stack - config.small_blind,
            bb_stack: config.bb_stack - config.big_blind,
            sb_commit: config.small_blind,
            bb_commit: config.big_blind,
            last_raise_size: config.big_blind - config.small_blind,
            raises: 0,
            player: Seat::Sb,
            limped: false,
        }
    }

    fn stack(self, seat: Seat) -> u32 {
        match seat {
            Seat::Sb => self.sb_stack,
            Seat::Bb => self.bb_stack,
        }
    }

    fn commit(self, seat: Seat) -> u32 {
        match seat {
            Seat::Sb => self.sb_commit,
            Seat::Bb => self.bb_commit,
        }
    }

    fn to_call(self) -> u32 {
        self.commit(self.player.other())
            .saturating_sub(self.commit(self.player))
            .min(self.stack(self.player))
    }

    fn effective_stack_after_call(self) -> u32 {
        let to_call = self.to_call();
        match self.player {
            Seat::Sb => self.sb_stack.saturating_sub(to_call).min(self.bb_stack),
            Seat::Bb => self.bb_stack.saturating_sub(to_call).min(self.sb_stack),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreflopAction {
    Fold,
    Call,
    Check,
    Limp,
    RaiseTo(u32),
    AllIn,
}

struct PreflopPlanner<'a> {
    config: &'a HuFullGameConfig,
    nodes: usize,
    decisions: usize,
    fold_terminals: usize,
    all_in_terminals: usize,
    boundaries: Vec<HuPostflopBoundaryGroup>,
    action_slots: u128,
}

impl PreflopPlanner<'_> {
    fn finish(self) -> HuPreflopPlan {
        HuPreflopPlan {
            nodes: self.nodes,
            decisions: self.decisions,
            fold_terminals: self.fold_terminals,
            all_in_terminals: self.all_in_terminals,
            postflop_boundaries: merge_boundaries(self.boundaries),
            action_slots: self.action_slots,
            storage_bytes: self.action_slots
                * (self.config.storage.regret_bytes + self.config.storage.strategy_sum_bytes),
        }
    }

    fn visit(&mut self, state: PreflopState) {
        self.nodes += 1;
        let actions = self.legal_actions(state);
        if actions.is_empty() {
            self.push_boundary(state);
            return;
        }
        self.decisions += 1;
        self.action_slots += 1326u128 * actions.len() as u128;
        for action in actions {
            match self.apply(state, action) {
                PreflopTransition::State(next) => self.visit(next),
                PreflopTransition::Fold => {
                    self.nodes += 1;
                    self.fold_terminals += 1;
                }
                PreflopTransition::AllIn => {
                    self.nodes += 1;
                    self.all_in_terminals += 1;
                }
                PreflopTransition::Boundary(next) => {
                    self.nodes += 1;
                    self.push_boundary(next);
                }
            }
        }
    }

    fn legal_actions(&self, state: PreflopState) -> Vec<PreflopAction> {
        let to_call = state.to_call();
        let stack = state.stack(state.player);
        if stack == 0 {
            return Vec::new();
        }
        if state.player == Seat::Sb && state.raises == 0 && !state.limped {
            let mut actions = vec![PreflopAction::Fold, PreflopAction::Limp];
            for size_bb in self.config.preflop.open_raise_sizes_bb {
                if let Some(action) = self.raise_to_bb(state, *size_bb) {
                    push_unique_preflop_action(&mut actions, action);
                }
            }
            if self.all_in_allowed(state) {
                push_unique_preflop_action(&mut actions, PreflopAction::AllIn);
            }
            return actions;
        }
        if to_call > 0 {
            let mut actions = vec![PreflopAction::Fold, PreflopAction::Call];
            if state.raises < self.config.preflop.max_raises && stack > to_call {
                for action in self.raise_actions(state) {
                    push_unique_preflop_action(&mut actions, action);
                }
                if self.all_in_allowed(state) {
                    push_unique_preflop_action(&mut actions, PreflopAction::AllIn);
                }
            }
            return actions;
        }
        if state.limped && state.player == Seat::Bb {
            let mut actions = vec![PreflopAction::Check];
            for size_bb in self.config.preflop.limp_iso_sizes_bb {
                if let Some(action) = self.raise_to_bb(state, *size_bb) {
                    push_unique_preflop_action(&mut actions, action);
                }
            }
            if self.all_in_allowed(state) {
                push_unique_preflop_action(&mut actions, PreflopAction::AllIn);
            }
            return actions;
        }
        Vec::new()
    }

    fn raise_actions(&self, state: PreflopState) -> Vec<PreflopAction> {
        let mut actions = Vec::new();
        let opponent_commit = state.commit(state.player.other());
        for multiplier in self.config.preflop.facing_raise_size_multipliers {
            let to = ((opponent_commit as f32) * *multiplier).round() as u32;
            if let Some(action) = self.raise_to(state, to) {
                actions.push(action);
            }
        }
        actions
    }

    fn raise_to_bb(&self, state: PreflopState, size_bb: f32) -> Option<PreflopAction> {
        let to = (self.config.big_blind as f32 * size_bb).round() as u32;
        self.raise_to(state, to)
    }

    fn raise_to(&self, state: PreflopState, to: u32) -> Option<PreflopAction> {
        let actor_commit = state.commit(state.player);
        let opponent_commit = state.commit(state.player.other());
        let stack = state.stack(state.player);
        let max_to = actor_commit + stack;
        if max_to <= opponent_commit {
            return None;
        }
        let min_to = opponent_commit + state.last_raise_size.max(self.config.big_blind);
        let to = to.max(min_to);
        if to >= max_to {
            return Some(PreflopAction::AllIn);
        }
        let additional = to.saturating_sub(actor_commit);
        let remaining = stack.saturating_sub(additional);
        if remaining as f32
            <= max_to as f32 * self.config.preflop.force_all_in_remaining_stack_fraction
        {
            return Some(PreflopAction::AllIn);
        }
        Some(PreflopAction::RaiseTo(to))
    }

    fn all_in_allowed(&self, state: PreflopState) -> bool {
        let stack = state.stack(state.player);
        let spr = if state.pot == 0 {
            f32::INFINITY
        } else {
            stack as f32 / state.pot as f32
        };
        spr <= self.config.preflop.add_all_in_spr_threshold || state.raises > 0
    }

    fn apply(&self, state: PreflopState, action: PreflopAction) -> PreflopTransition {
        match action {
            PreflopAction::Fold => PreflopTransition::Fold,
            PreflopAction::Check => PreflopTransition::Boundary(state),
            PreflopAction::Call => {
                let to_call = state.to_call();
                let next = self.commit_additional(state, to_call);
                if next.sb_stack == 0 || next.bb_stack == 0 {
                    PreflopTransition::AllIn
                } else {
                    PreflopTransition::Boundary(next)
                }
            }
            PreflopAction::Limp => {
                let to_call = state.to_call();
                let mut next = self.commit_additional(state, to_call);
                next.player = state.player.other();
                next.limped = true;
                PreflopTransition::State(next)
            }
            PreflopAction::RaiseTo(to) => {
                let additional = to.saturating_sub(state.commit(state.player));
                let mut next = self.commit_additional(state, additional);
                next.player = state.player.other();
                next.last_raise_size = to.saturating_sub(state.commit(state.player.other()));
                next.raises = next.raises.saturating_add(1);
                next.limped = false;
                PreflopTransition::State(next)
            }
            PreflopAction::AllIn => {
                let additional = state.stack(state.player);
                let mut next = self.commit_additional(state, additional);
                next.player = state.player.other();
                next.last_raise_size = next
                    .commit(state.player)
                    .saturating_sub(state.commit(state.player.other()));
                next.raises = next.raises.saturating_add(1);
                PreflopTransition::State(next)
            }
        }
    }

    fn commit_additional(&self, mut state: PreflopState, amount: u32) -> PreflopState {
        let amount = amount.min(state.stack(state.player));
        state.pot = state.pot.saturating_add(amount);
        match state.player {
            Seat::Sb => {
                state.sb_stack = state.sb_stack.saturating_sub(amount);
                state.sb_commit = state.sb_commit.saturating_add(amount);
            }
            Seat::Bb => {
                state.bb_stack = state.bb_stack.saturating_sub(amount);
                state.bb_commit = state.bb_commit.saturating_add(amount);
            }
        }
        state
    }

    fn push_boundary(&mut self, state: PreflopState) {
        self.boundaries.push(HuPostflopBoundaryGroup {
            pot: state.pot,
            effective_stack: state.effective_stack_after_call(),
            sb_commit: state.sb_commit,
            bb_commit: state.bb_commit,
            count: 1,
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreflopTransition {
    State(PreflopState),
    Fold,
    AllIn,
    Boundary(PreflopState),
}

fn push_unique_preflop_action(actions: &mut Vec<PreflopAction>, action: PreflopAction) {
    if !actions.contains(&action) {
        actions.push(action);
    }
}

fn merge_boundaries(mut boundaries: Vec<HuPostflopBoundaryGroup>) -> Vec<HuPostflopBoundaryGroup> {
    boundaries.sort_by_key(|boundary| {
        (
            boundary.pot,
            boundary.effective_stack,
            boundary.sb_commit,
            boundary.bb_commit,
        )
    });
    let mut merged: Vec<HuPostflopBoundaryGroup> = Vec::new();
    for boundary in boundaries {
        if let Some(last) = merged.last_mut() {
            if last.pot == boundary.pot
                && last.effective_stack == boundary.effective_stack
                && last.sb_commit == boundary.sb_commit
                && last.bb_commit == boundary.bb_commit
            {
                last.count += boundary.count;
                continue;
            }
        }
        merged.push(boundary);
    }
    merged
}

fn bytes_to_gib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn bytes_to_mib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn ceil_div(value: u128, divisor: u128) -> u128 {
    if value == 0 {
        0
    } else {
        (value - 1) / divisor + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn default_preflop_template_has_postflop_boundaries() {
        let config = HuFullGameConfig::hu_spin_15bb(
            RangeSpec::from_str("AA").unwrap(),
            RangeSpec::from_str("KK").unwrap(),
            ActionAbstraction::postflop_solver_basic(),
        );
        let plan = plan_hu_full_game(&config).unwrap();
        assert!(plan.preflop.decisions > 0);
        assert!(plan.preflop.fold_terminals > 0);
        assert!(plan.preflop.all_in_terminals > 0);
        assert!(!plan.preflop.postflop_boundaries.is_empty());
        assert_eq!(plan.flop_classes, 1755);
        assert_eq!(
            plan.representative_subgames,
            plan.preflop.postflop_boundaries.len() as u128 * 1755
        );
        assert!(plan.streaming.postflop_chunks >= plan.representative_subgames);
        assert!(plan.streaming.max_resident_bytes < plan.total_storage_bytes);
        assert_eq!(
            plan.streaming.disk_state_bytes,
            plan.total_storage_bytes - plan.preflop.storage_bytes
        );
    }

    #[test]
    fn default_template_does_not_force_root_all_in() {
        let config = HuFullGameConfig::hu_spin_15bb(
            RangeSpec::from_str("AA").unwrap(),
            RangeSpec::from_str("KK").unwrap(),
            ActionAbstraction::postflop_solver_basic(),
        );
        let planner = PreflopPlanner {
            config: &config,
            nodes: 0,
            decisions: 0,
            fold_terminals: 0,
            all_in_terminals: 0,
            boundaries: Vec::new(),
            action_slots: 0,
        };
        let actions = planner.legal_actions(PreflopState::root(&config));
        assert!(actions.contains(&PreflopAction::Limp));
        assert!(actions.contains(&PreflopAction::RaiseTo(200)));
        assert!(actions.contains(&PreflopAction::RaiseTo(250)));
        assert!(!actions.contains(&PreflopAction::AllIn));
    }
}
