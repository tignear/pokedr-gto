use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use pokedr_core::cards::Card as PokedrCard;
use pokedr_core::hand_class::HandClass;
use pokedr_core::hand_class::all_hand_classes;
use pokedr_core::river::Combo;
use pokedr_core::subgame::{
    ActionAbstraction, ActionKind, BetSize, ChancePolicy, Player, PotState, RangeState,
    SubgameSolveRequest, SubgameSpec,
};
use rs_poker::{
    arena::{
        action::AgentAction,
        action::{Action, PlayedActionPayload},
        agent::Agent,
        game_state::{GameState, Round},
        historian::{Historian, HistorianError},
    },
    core::{Hand, Rank, Rankable},
    holdem::MonteCarloGame,
};

#[derive(Debug, Clone)]
pub struct PreflopRanges {
    open: HashSet<HandClass>,
    continue_vs_raise: HashSet<HandClass>,
    value_raise: HashSet<HandClass>,
}

impl PreflopRanges {
    pub fn open_class_count(&self) -> usize {
        self.open.len()
    }

    pub fn opens(&self, class: HandClass) -> bool {
        self.open.contains(&class)
    }

    pub fn continues_vs_raise(&self, class: HandClass) -> bool {
        self.continue_vs_raise.contains(&class)
    }

    pub fn value_raises(&self, class: HandClass) -> bool {
        self.value_raise.contains(&class)
    }
}

impl Default for PreflopRanges {
    fn default() -> Self {
        // Public 6-max references put BTN/SB opening ranges around the low-to-mid
        // 40% band. This is an implementable baseline, not a solved range.
        Self {
            open: hand_classes(&[
                "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
                "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "ATs", "AJs", "AQs", "AKs",
                "K2s", "K3s", "K4s", "K5s", "K6s", "K7s", "K8s", "K9s", "KTs", "KJs", "KQs", "Q5s",
                "Q6s", "Q7s", "Q8s", "Q9s", "QTs", "QJs", "J7s", "J8s", "J9s", "JTs", "T7s", "T8s",
                "T9s", "97s", "98s", "87s", "76s", "65s", "54s", "A2o", "A3o", "A4o", "A5o", "A6o",
                "A7o", "A8o", "A9o", "ATo", "AJo", "AQo", "AKo", "K8o", "K9o", "KTo", "KJo", "KQo",
                "Q9o", "QTo", "QJo", "J9o", "JTo", "T9o",
            ]),
            continue_vs_raise: hand_classes(&[
                "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA", "A8s", "A9s", "ATs",
                "AJs", "AQs", "AKs", "KTs", "KJs", "KQs", "QJs", "JTs", "ATo", "AJo", "AQo", "AKo",
                "KQo", "A5s", "A4s",
            ]),
            value_raise: hand_classes(&[
                "TT", "JJ", "QQ", "KK", "AA", "AQs", "AKs", "AKo", "A5s", "A4s",
            ]),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EquityPolicyAgent {
    preflop: PreflopRanges,
}

impl EquityPolicyAgent {
    pub fn new(preflop: PreflopRanges) -> Self {
        Self { preflop }
    }
}

#[derive(Debug)]
pub struct CfrPolicyAgent {
    fallback: EquityPolicyAgent,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
    belief: Arc<Mutex<PublicBelief>>,
}

impl Default for CfrPolicyAgent {
    fn default() -> Self {
        Self {
            fallback: EquityPolicyAgent::default(),
            iterations: 1_000,
            range_classes: 48,
            runouts: 8,
            belief: Arc::new(Mutex::new(PublicBelief::default())),
        }
    }
}

impl Clone for CfrPolicyAgent {
    fn clone(&self) -> Self {
        Self {
            fallback: self.fallback.clone(),
            iterations: self.iterations,
            range_classes: self.range_classes,
            runouts: self.runouts,
            belief: Arc::new(Mutex::new(PublicBelief::default())),
        }
    }
}

impl CfrPolicyAgent {
    pub fn new(iterations: usize, range_classes: usize, runouts: usize) -> Self {
        Self {
            fallback: EquityPolicyAgent::default(),
            iterations,
            range_classes,
            runouts,
            belief: Arc::new(Mutex::new(PublicBelief::default())),
        }
    }
}

impl Agent for CfrPolicyAgent {
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        if game_state.round == Round::Preflop || game_state.board.len() < 3 {
            return self.fallback.act(id, game_state);
        }

        let idx = game_state.round_data.to_act_idx;
        let initial_belief = self
            .belief
            .lock()
            .map(|belief| belief.snapshot())
            .unwrap_or_default();
        ensure_cfr_likelihoods_for_observed_actions(
            &self.belief,
            game_state,
            &initial_belief,
            self.iterations,
            self.range_classes,
            self.runouts,
        )
        .expect("observed postflop action must be resolved by CFR before acting");
        let belief = self
            .belief
            .lock()
            .map(|belief| belief.snapshot())
            .unwrap_or_default();
        cfr_postflop_action(
            game_state,
            idx,
            self.iterations,
            self.range_classes,
            self.runouts,
            &belief,
            &self.belief,
        )
        .expect("postflop CFR action failed; missing CFR-derived belief must be solved explicitly")
    }

    fn historian(&self) -> Option<Box<dyn Historian>> {
        Some(Box::new(PublicBeliefHistorian {
            belief: Arc::clone(&self.belief),
        }))
    }
}

#[derive(Debug, Clone, Default)]
struct PublicBelief {
    simulation_id: Option<u128>,
    actions: Vec<PublicAction>,
    cfr_likelihoods: Vec<CfrActionLikelihood>,
}

#[derive(Debug, Clone, Default)]
struct PublicBeliefSnapshot {
    actions: Vec<PublicAction>,
    cfr_likelihoods: Vec<CfrActionLikelihood>,
}

#[derive(Debug, Clone)]
struct PublicAction {
    player: usize,
    round: Round,
    sequence_index: usize,
    kind: PublicActionKind,
    size_ratio: f32,
    board: Vec<PokedrCard>,
    starting_pot: f32,
    starting_bet: f32,
    starting_min_raise: f32,
    starting_player_bet: f32,
    final_player_bet: f32,
    stacks_after: Vec<f32>,
    player_bet_after: Vec<f32>,
}

#[derive(Debug, Clone)]
struct CfrActionLikelihood {
    player: usize,
    round: Round,
    sequence_index: usize,
    history: Vec<PublicActionKind>,
    kind: PublicActionKind,
    combo_mask: u64,
    frequency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublicActionKind {
    Check,
    Call,
    BetOrRaise,
    AllIn,
    Fold,
}

#[derive(Clone)]
struct PublicBeliefHistorian {
    belief: Arc<Mutex<PublicBelief>>,
}

impl PublicBelief {
    fn snapshot(&self) -> PublicBeliefSnapshot {
        PublicBeliefSnapshot {
            actions: self.actions.clone(),
            cfr_likelihoods: self.cfr_likelihoods.clone(),
        }
    }

    fn reset(&mut self, simulation_id: u128) {
        self.simulation_id = Some(simulation_id);
        self.actions.clear();
        self.cfr_likelihoods.clear();
    }

    fn record_action(&mut self, simulation_id: u128, action: PublicAction) {
        if self.simulation_id != Some(simulation_id) {
            self.reset(simulation_id);
        }
        self.actions.push(action);
    }

    fn replace_cfr_likelihoods(
        &mut self,
        round: Round,
        from_sequence_index: usize,
        likelihoods: Vec<CfrActionLikelihood>,
    ) {
        self.cfr_likelihoods
            .retain(|entry| entry.round != round || entry.sequence_index < from_sequence_index);
        self.cfr_likelihoods.extend(likelihoods);
    }
}

impl Historian for PublicBeliefHistorian {
    fn record_action(
        &mut self,
        id: u128,
        _game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        let mut belief = self
            .belief
            .lock()
            .map_err(|_| HistorianError::UnableToRecordAction)?;
        match action {
            Action::GameStart(_) => belief.reset(id),
            Action::PlayedAction(payload) => {
                if let Some(mut public_action) = public_action_from_payload(_game_state, &payload) {
                    public_action.sequence_index = belief
                        .actions
                        .iter()
                        .filter(|action| action.round == public_action.round)
                        .count();
                    belief.record_action(id, public_action);
                }
            }
            Action::FailedAction(payload) => {
                if let Some(mut public_action) =
                    public_action_from_payload(_game_state, &payload.result)
                {
                    public_action.sequence_index = belief
                        .actions
                        .iter()
                        .filter(|action| action.round == public_action.round)
                        .count();
                    belief.record_action(id, public_action);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Agent for EquityPolicyAgent {
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        let idx = game_state.round_data.to_act_idx;
        let to_call = amount_to_call(game_state, idx);
        let stack = game_state.stacks[idx];
        let equity = estimate_equity(game_state, idx).unwrap_or(0.0);

        if game_state.round == Round::Preflop {
            return self.preflop_action(game_state, idx, to_call, stack, equity);
        }

        postflop_action(game_state, idx, to_call, stack, equity)
    }
}

impl EquityPolicyAgent {
    fn preflop_action(
        &self,
        game_state: &GameState,
        idx: usize,
        to_call: f32,
        stack: f32,
        equity: f32,
    ) -> AgentAction {
        let Some(class) = hand_class(game_state, idx) else {
            return AgentAction::Call;
        };

        if to_call <= 0.0 {
            if self.preflop.value_raise.contains(&class) {
                AgentAction::Bet((game_state.big_blind * 3.0).min(stack))
            } else if self.preflop.open.contains(&class) || equity >= 0.53 {
                AgentAction::Call
            } else {
                AgentAction::Fold
            }
        } else if self.preflop.value_raise.contains(&class) && to_call <= game_state.big_blind * 4.0
        {
            AgentAction::Bet((game_state.round_data.bet + game_state.big_blind * 3.0).min(stack))
        } else if self.preflop.continue_vs_raise.contains(&class)
            || equity * (game_state.total_pot + to_call) >= to_call
        {
            AgentAction::Call
        } else {
            AgentAction::Fold
        }
    }
}

fn postflop_action(
    game_state: &GameState,
    idx: usize,
    to_call: f32,
    stack: f32,
    equity: f32,
) -> AgentAction {
    let strength = made_hand_strength(game_state, idx);
    if to_call <= 0.0 {
        if strength >= 2 || equity >= 0.62 {
            let size = (game_state.total_pot * 0.75).max(game_state.big_blind * 2.0);
            AgentAction::Bet((game_state.round_data.bet + size).min(stack))
        } else {
            AgentAction::Call
        }
    } else if strength >= 2 || equity * (game_state.total_pot + to_call) >= to_call {
        AgentAction::Call
    } else {
        AgentAction::Fold
    }
}

fn cfr_postflop_action(
    game_state: &GameState,
    idx: usize,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
    belief: &PublicBeliefSnapshot,
    belief_store: &Arc<Mutex<PublicBelief>>,
) -> Option<AgentAction> {
    let hero_class = hand_class(game_state, idx)?;
    let board = pokedr_board(game_state)?;
    let to_call = amount_to_call(game_state, idx);
    let stack = game_state.stacks[idx];
    let (oop_idx, ip_idx) = heads_up_postflop_positions(game_state)?;
    let root_player = if idx == oop_idx {
        Player::Oop
    } else if idx == ip_idx {
        Player::Ip
    } else {
        return None;
    };
    let hero_mask = pokedr_hand_mask(game_state, idx)?;

    let oop_range = mc_range_prior(
        game_state,
        oop_idx,
        range_classes,
        (idx == oop_idx).then_some(hero_class),
        if idx == oop_idx { 0 } else { hero_mask },
        belief,
        None,
        board.as_slice(),
    )?;
    let ip_range = mc_range_prior(
        game_state,
        ip_idx,
        range_classes,
        (idx == ip_idx).then_some(hero_class),
        if idx == ip_idx { 0 } else { hero_mask },
        belief,
        None,
        board.as_slice(),
    )?;
    if oop_range.is_empty() || ip_range.is_empty() {
        return None;
    }

    let pot = PotState {
        pot: game_state.total_pot as f64,
        stacks: [
            game_state.stacks[oop_idx] as f64,
            game_state.stacks[ip_idx] as f64,
        ],
        committed: [
            game_state.round_data.player_bet[oop_idx] as f64,
            game_state.round_data.player_bet[ip_idx] as f64,
        ],
        current_bet: game_state.round_data.bet as f64,
        min_raise: game_state.round_data.min_raise as f64,
    };
    let spec = SubgameSpec::postflop(
        board.clone(),
        pot,
        RangeState::weighted_combos(oop_range, ip_range),
        cfr_action_abstraction(game_state),
        ChancePolicy::Sample(runouts.max(1)),
    )
    .ok()
    .map(|spec| spec.with_root_player(root_player))?;
    let result = spec
        .solve_cfr_with_request(SubgameSolveRequest {
            iterations,
            focused_oop_combo_mask: (root_player == Player::Oop).then_some(hero_mask),
            focused_ip_combo_mask: (root_player == Player::Ip).then_some(hero_mask),
            focused_sampling_rate: 0.5,
        })
        .ok()?;
    store_cfr_likelihoods(
        belief_store,
        game_state,
        belief,
        oop_idx,
        ip_idx,
        result.strategies.as_slice(),
    );
    let root = result.strategies.iter().find(|strategy| {
        strategy.node == result.root
            && strategy.player == root_player
            && strategy.combo.mask == hero_mask
    })?;
    let action = sample_strategy_action(game_state, idx, root.actions.as_slice())?;

    Some(match action {
        ActionKind::Fold => AgentAction::Fold,
        ActionKind::Check | ActionKind::Call => {
            if to_call <= 0.0 {
                AgentAction::Call
            } else {
                AgentAction::Bet(game_state.round_data.bet)
            }
        }
        ActionKind::Bet(amount) | ActionKind::Raise(amount) | ActionKind::AllIn(amount) => {
            let capped = amount
                .max(game_state.round_data.bet as f64)
                .min((game_state.round_data.player_bet[idx] + stack) as f64);
            AgentAction::Bet(capped as f32)
        }
    })
}

fn sample_strategy_action(
    game_state: &GameState,
    idx: usize,
    actions: &[pokedr_core::subgame::SubgameActionFrequency],
) -> Option<ActionKind> {
    let total: f64 = actions.iter().map(|action| action.frequency.max(0.0)).sum();
    if total <= 0.0 {
        return actions.first().map(|action| action.action);
    }

    let mut threshold = deterministic_action_roll(game_state, idx) * total;
    for action in actions {
        threshold -= action.frequency.max(0.0);
        if threshold <= 0.0 {
            return Some(action.action);
        }
    }
    actions.last().map(|action| action.action)
}

fn store_cfr_likelihoods(
    belief_store: &Arc<Mutex<PublicBelief>>,
    game_state: &GameState,
    belief: &PublicBeliefSnapshot,
    oop_idx: usize,
    ip_idx: usize,
    strategies: &[pokedr_core::subgame::SubgameComboStrategy],
) {
    let from_sequence_index = belief
        .actions
        .iter()
        .filter(|action| action.round == game_state.round)
        .count();
    let prior_history = public_history_before(belief, game_state.round, from_sequence_index);
    store_cfr_likelihoods_from_sequence(
        belief_store,
        game_state.round,
        from_sequence_index,
        prior_history.as_slice(),
        oop_idx,
        ip_idx,
        strategies,
    );
}

fn store_cfr_likelihoods_from_sequence(
    belief_store: &Arc<Mutex<PublicBelief>>,
    round: Round,
    from_sequence_index: usize,
    prior_history: &[PublicActionKind],
    oop_idx: usize,
    ip_idx: usize,
    strategies: &[pokedr_core::subgame::SubgameComboStrategy],
) {
    let mut likelihoods = Vec::new();
    for strategy in strategies {
        let player = match strategy.player {
            Player::Oop => oop_idx,
            Player::Ip => ip_idx,
        };
        let Some(local_history) = cfr_history_to_public(strategy.history.as_slice()) else {
            continue;
        };
        let sequence_index = from_sequence_index + local_history.len();
        let mut history = Vec::with_capacity(prior_history.len() + local_history.len());
        history.extend_from_slice(prior_history);
        history.extend(local_history);
        for action in &strategy.actions {
            let Some(kind) = public_kind_from_cfr_action(action.action) else {
                continue;
            };
            likelihoods.push(CfrActionLikelihood {
                player,
                round,
                sequence_index,
                history: history.clone(),
                kind,
                combo_mask: strategy.combo.mask,
                frequency: action.frequency,
            });
        }
    }

    if let Ok(mut belief) = belief_store.lock() {
        belief.replace_cfr_likelihoods(round, from_sequence_index, likelihoods);
    }
}

fn public_kind_from_cfr_action(action: ActionKind) -> Option<PublicActionKind> {
    match action {
        ActionKind::Check => Some(PublicActionKind::Check),
        ActionKind::Fold => Some(PublicActionKind::Fold),
        ActionKind::Call => Some(PublicActionKind::Call),
        ActionKind::Bet(_) | ActionKind::Raise(_) => Some(PublicActionKind::BetOrRaise),
        ActionKind::AllIn(_) => Some(PublicActionKind::AllIn),
    }
}

fn cfr_history_to_public(history: &[ActionKind]) -> Option<Vec<PublicActionKind>> {
    history
        .iter()
        .copied()
        .map(public_kind_from_cfr_action)
        .collect()
}

fn public_history_before(
    belief: &PublicBeliefSnapshot,
    round: Round,
    sequence_index: usize,
) -> Vec<PublicActionKind> {
    let mut actions: Vec<_> = belief
        .actions
        .iter()
        .filter(|action| action.round == round && action.sequence_index < sequence_index)
        .collect();
    actions.sort_by_key(|action| action.sequence_index);
    actions.into_iter().map(|action| action.kind).collect()
}

fn ensure_cfr_likelihoods_for_observed_actions(
    belief_store: &Arc<Mutex<PublicBelief>>,
    game_state: &GameState,
    belief: &PublicBeliefSnapshot,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
) -> Option<()> {
    let mut snapshot = belief.clone();
    for action in belief
        .actions
        .iter()
        .filter(|action| is_postflop_round(action.round))
    {
        if has_cfr_likelihood(&snapshot, action) {
            continue;
        }
        solve_observed_action_node(
            belief_store,
            game_state,
            &snapshot,
            action,
            iterations,
            range_classes,
            runouts,
        )?;
        snapshot = belief_store.lock().ok()?.snapshot();
    }
    Some(())
}

fn solve_observed_action_node(
    belief_store: &Arc<Mutex<PublicBelief>>,
    game_state: &GameState,
    belief: &PublicBeliefSnapshot,
    action: &PublicAction,
    iterations: usize,
    range_classes: usize,
    runouts: usize,
) -> Option<()> {
    let (oop_idx, ip_idx) = heads_up_postflop_positions(game_state)?;
    let root_player = if action.player == oop_idx {
        Player::Oop
    } else if action.player == ip_idx {
        Player::Ip
    } else {
        return None;
    };
    let action_limit = Some((action.round, action.sequence_index));
    let oop_range = mc_range_prior(
        game_state,
        oop_idx,
        range_classes,
        None,
        0,
        belief,
        action_limit,
        action.board.as_slice(),
    )?;
    let ip_range = mc_range_prior(
        game_state,
        ip_idx,
        range_classes,
        None,
        0,
        belief,
        action_limit,
        action.board.as_slice(),
    )?;
    let pot = PotState {
        pot: action.starting_pot as f64,
        stacks: [
            stack_before_action(action, oop_idx) as f64,
            stack_before_action(action, ip_idx) as f64,
        ],
        committed: [
            player_bet_before_action(action, oop_idx) as f64,
            player_bet_before_action(action, ip_idx) as f64,
        ],
        current_bet: action.starting_bet as f64,
        min_raise: action.starting_min_raise as f64,
    };
    let spec = SubgameSpec::postflop(
        action.board.clone(),
        pot,
        RangeState::weighted_combos(oop_range, ip_range),
        action_abstraction_for_pot(game_state.big_blind, action.starting_pot),
        ChancePolicy::Sample(runouts.max(1)),
    )
    .ok()?
    .with_root_player(root_player);
    let result = spec.solve_cfr(iterations).ok()?;
    let prior_history = public_history_before(belief, action.round, action.sequence_index);
    store_cfr_likelihoods_from_sequence(
        belief_store,
        action.round,
        action.sequence_index,
        prior_history.as_slice(),
        oop_idx,
        ip_idx,
        result.strategies.as_slice(),
    );
    Some(())
}

fn has_cfr_likelihood(belief: &PublicBeliefSnapshot, action: &PublicAction) -> bool {
    let history = public_history_before(belief, action.round, action.sequence_index);
    belief.cfr_likelihoods.iter().any(|entry| {
        entry.player == action.player
            && entry.round == action.round
            && entry.sequence_index == action.sequence_index
            && entry.history == history
            && entry.kind == action.kind
    })
}

fn stack_before_action(action: &PublicAction, player: usize) -> f32 {
    let stack_after = action.stacks_after.get(player).copied().unwrap_or(0.0);
    if player == action.player {
        stack_after + (action.final_player_bet - action.starting_player_bet).max(0.0)
    } else {
        stack_after
    }
}

fn player_bet_before_action(action: &PublicAction, player: usize) -> f32 {
    if player == action.player {
        action.starting_player_bet
    } else {
        action.player_bet_after.get(player).copied().unwrap_or(0.0)
    }
}

fn action_abstraction_for_pot(big_blind: f32, pot: f32) -> ActionAbstraction {
    let pot = pot.max(big_blind);
    ActionAbstraction {
        bet_sizes: vec![
            BetSize::Chips((pot * 0.5) as f64),
            BetSize::Chips((pot * 0.75) as f64),
        ],
        raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
        reraise_sizes: vec![BetSize::CurrentBetMultiple(2.2)],
        allow_all_in: true,
        max_raises: 2,
    }
}

fn is_postflop_round(round: Round) -> bool {
    matches!(round, Round::Flop | Round::Turn | Round::River)
}

fn deterministic_action_roll(game_state: &GameState, idx: usize) -> f64 {
    let mut value = idx as u64 + 0x9e37_79b9_7f4a_7c15;
    value ^= (game_state.total_pot.to_bits() as u64).rotate_left(7);
    value ^= (game_state.round_data.bet.to_bits() as u64).rotate_left(17);
    value ^= (game_state.round as u8 as u64).rotate_left(23);
    for card in &game_state.board {
        value ^= (u8::from(*card) as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = value.rotate_left(11);
    }
    for card in private_cards(game_state, idx) {
        value ^= (u8::from(card) as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        value = value.rotate_left(13);
    }
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let mixed = value ^ (value >> 31);
    (mixed as f64) / (u64::MAX as f64)
}

fn cfr_action_abstraction(game_state: &GameState) -> ActionAbstraction {
    let pot = game_state.total_pot.max(game_state.big_blind);
    ActionAbstraction {
        bet_sizes: vec![
            BetSize::Chips((pot * 0.5) as f64),
            BetSize::Chips((pot * 0.75) as f64),
        ],
        raise_sizes: vec![BetSize::CurrentBetMultiple(2.5)],
        reraise_sizes: vec![BetSize::CurrentBetMultiple(2.2)],
        allow_all_in: true,
        max_raises: 2,
    }
}

fn amount_to_call(game_state: &GameState, idx: usize) -> f32 {
    (game_state.round_data.bet - game_state.round_data.player_bet[idx]).max(0.0)
}

fn estimate_equity(game_state: &GameState, idx: usize) -> Option<f32> {
    let mut default_hand = Hand::new();
    default_hand.extend(game_state.board.iter().copied());
    let hands = game_state
        .hands
        .iter()
        .enumerate()
        .map(
            |(hand_idx, hand)| {
                if hand_idx == idx { *hand } else { default_hand }
            },
        )
        .collect();
    let mut monte = MonteCarloGame::new(hands).ok()?;
    monte.estimate_equity(400).get(idx).copied()
}

fn mc_range_prior(
    game_state: &GameState,
    idx: usize,
    max_classes: usize,
    required: Option<HandClass>,
    extra_dead_mask: u64,
    belief: &PublicBeliefSnapshot,
    action_limit: Option<(Round, usize)>,
    board: &[PokedrCard],
) -> Option<Vec<(Combo, f64)>> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let dead_mask = board_mask | extra_dead_mask;
    let opponent_idx = heads_up_opponent(game_state, idx).unwrap_or(idx);
    let scored: Vec<_> = all_hand_classes()
        .into_iter()
        .flat_map(|class| {
            class
                .combos()
                .into_iter()
                .filter(move |cards| cards.iter().all(|card| card.mask() & dead_mask == 0))
                .filter_map(move |cards| {
                    let combo = Combo::new(cards[0], cards[1])?;
                    let equity = estimate_class_equity_on_board(
                        game_state,
                        board,
                        idx,
                        opponent_idx,
                        cards,
                    )?;
                    Some((combo, equity))
                })
        })
        .collect();

    let mut weighted_scored: Vec<_> = scored
        .into_iter()
        .filter_map(|(combo, equity)| {
            public_belief_weight(belief, idx, combo, action_limit)
                .map(|public_weight| (combo, equity, public_weight))
        })
        .collect();
    weighted_scored.sort_by(|left, right| right.1.total_cmp(&left.1));
    let score_view: Vec<_> = weighted_scored
        .iter()
        .map(|(combo, equity, _)| (*combo, *equity))
        .collect();
    let combos = if has_postflop_belief_before(belief, idx, action_limit) {
        score_view.iter().map(|(combo, _)| *combo).collect()
    } else {
        stratified_combos(&score_view, max_classes.max(1))
    };
    let min_equity = weighted_scored
        .last()
        .map(|(_, equity, _)| *equity)
        .unwrap_or(0.0);
    let max_equity = weighted_scored
        .first()
        .map(|(_, equity, _)| *equity)
        .unwrap_or(1.0);
    let span = (max_equity - min_equity).max(0.001);
    let mut combos: Vec<_> = combos
        .into_iter()
        .map(|combo| {
            let (equity, public_weight) = weighted_scored
                .iter()
                .find(|(candidate, _, _)| *candidate == combo)
                .map(|(_, equity, public_weight)| (*equity, *public_weight))
                .unwrap_or((min_equity, 1.0));
            let percentile = ((equity - min_equity) / span).clamp(0.0, 1.0);
            (combo, (0.05 + f64::from(percentile) * 0.95) * public_weight)
        })
        .collect();
    if let Some(required) = required {
        for cards in required
            .combos()
            .into_iter()
            .filter(|cards| cards.iter().all(|card| card.mask() & dead_mask == 0))
        {
            let Some(combo) = Combo::new(cards[0], cards[1]) else {
                continue;
            };
            if !combos.iter().any(|(candidate, _)| *candidate == combo) {
                combos.push((combo, 1.0));
            }
        }
    }
    Some(combos)
}

fn public_action_from_payload(
    game_state: &GameState,
    payload: &PlayedActionPayload,
) -> Option<PublicAction> {
    let added = (payload.final_player_bet - payload.starting_player_bet).max(0.0);
    let size_ratio = added / payload.starting_pot.max(1.0);
    let kind = match payload.action {
        AgentAction::Fold => PublicActionKind::Fold,
        AgentAction::AllIn => PublicActionKind::AllIn,
        AgentAction::Call => {
            if payload.starting_bet <= payload.starting_player_bet {
                PublicActionKind::Check
            } else {
                PublicActionKind::Call
            }
        }
        AgentAction::Bet(_) => {
            if payload.final_bet > payload.starting_bet {
                PublicActionKind::BetOrRaise
            } else if payload.starting_bet > payload.starting_player_bet {
                PublicActionKind::Call
            } else {
                PublicActionKind::Check
            }
        }
    };

    Some(PublicAction {
        player: payload.idx,
        round: payload.round,
        sequence_index: 0,
        kind,
        size_ratio,
        board: pokedr_board(game_state).unwrap_or_default(),
        starting_pot: payload.starting_pot,
        starting_bet: payload.starting_bet,
        starting_min_raise: payload.starting_min_raise,
        starting_player_bet: payload.starting_player_bet,
        final_player_bet: payload.final_player_bet,
        stacks_after: game_state.stacks.clone(),
        player_bet_after: game_state.round_data.player_bet.clone(),
    })
}

fn public_belief_weight(
    belief: &PublicBeliefSnapshot,
    player: usize,
    combo: Combo,
    action_limit: Option<(Round, usize)>,
) -> Option<f64> {
    let mut weight = 1.0;
    let class = hand_class_from_combo(combo);
    for action in belief.actions.iter().filter(|action| {
        action.player == player
            && action_limit
                .map(|limit| action_precedes(*action, limit))
                .unwrap_or(true)
    }) {
        weight *= match action.round {
            Round::Preflop => preflop_action_likelihood(class, action),
            Round::Flop | Round::Turn | Round::River => {
                cfr_action_likelihood(belief, action, combo.mask())?
            }
            _ => 1.0,
        };
    }
    Some(weight.clamp(0.01, 100.0))
}

fn has_postflop_belief_before(
    belief: &PublicBeliefSnapshot,
    player: usize,
    action_limit: Option<(Round, usize)>,
) -> bool {
    belief.actions.iter().any(|action| {
        action.player == player
            && is_postflop_round(action.round)
            && action_limit
                .map(|limit| action_precedes(action, limit))
                .unwrap_or(true)
    })
}

fn cfr_action_likelihood(
    belief: &PublicBeliefSnapshot,
    action: &PublicAction,
    combo_mask: u64,
) -> Option<f64> {
    let history = public_history_before(belief, action.round, action.sequence_index);
    let matching: Vec<_> = belief
        .cfr_likelihoods
        .iter()
        .filter(|entry| {
            entry.player == action.player
                && entry.round == action.round
                && entry.sequence_index == action.sequence_index
                && entry.history == history
                && entry.kind == action.kind
        })
        .collect();
    if matching.is_empty() {
        return None;
    }
    let average = matching.iter().map(|entry| entry.frequency).sum::<f64>() / matching.len() as f64;
    let combo_matching: Vec<_> = matching
        .iter()
        .copied()
        .filter(|entry| entry.combo_mask == combo_mask)
        .collect();
    if combo_matching.is_empty() {
        return None;
    }
    let frequency = combo_matching
        .iter()
        .map(|entry| entry.frequency)
        .sum::<f64>()
        / combo_matching.len() as f64;
    Some(((frequency + 0.01) / (average + 0.01)).clamp(0.05, 8.0))
}

fn action_precedes(action: &PublicAction, limit: (Round, usize)) -> bool {
    let action_round = round_order(action.round);
    let limit_round = round_order(limit.0);
    action_round < limit_round || action_round == limit_round && action.sequence_index < limit.1
}

fn round_order(round: Round) -> u8 {
    match round {
        Round::Preflop => 0,
        Round::Flop => 1,
        Round::Turn => 2,
        Round::River => 3,
        _ => 4,
    }
}

fn preflop_action_likelihood(class: HandClass, action: &PublicAction) -> f64 {
    match action.kind {
        PublicActionKind::BetOrRaise => {
            let base = if is_premium_class(class) {
                3.2
            } else if is_strong_playable_class(class) {
                1.8
            } else if is_speculative_playable_class(class) {
                1.1
            } else {
                0.35
            };
            base * aggressive_size_factor(action.size_ratio)
        }
        PublicActionKind::AllIn => {
            if is_premium_class(class) {
                5.0
            } else if is_strong_playable_class(class) {
                1.7
            } else {
                0.18
            }
        }
        PublicActionKind::Call => {
            if is_premium_class(class) {
                0.75
            } else if is_pair(class) || is_speculative_playable_class(class) {
                1.8
            } else if is_strong_playable_class(class) {
                1.25
            } else {
                0.55
            }
        }
        PublicActionKind::Check => 1.0,
        PublicActionKind::Fold => 0.01,
    }
}

fn aggressive_size_factor(size_ratio: f32) -> f64 {
    if size_ratio >= 2.0 {
        1.55
    } else if size_ratio >= 0.75 {
        1.25
    } else {
        1.0
    }
}

fn is_pair(class: HandClass) -> bool {
    class.high == class.low
}

fn is_premium_class(class: HandClass) -> bool {
    is_pair(class) && class.high >= 10
        || class.high == 14 && class.low >= 12
        || class.high == 13 && class.low == 12 && class.suited
}

fn is_strong_playable_class(class: HandClass) -> bool {
    is_pair(class) && class.high >= 7
        || class.high == 14 && class.low >= 9
        || class.high >= 13 && class.low >= 10
        || class.suited && class.high >= 12 && class.low >= 9
}

fn is_speculative_playable_class(class: HandClass) -> bool {
    is_pair(class)
        || class.suited && class.high >= 10 && class.low >= 6
        || class.suited && connector_gap(class) <= 2 && class.low >= 4
        || !class.suited && connector_gap(class) <= 1 && class.high >= 10
}

fn connector_gap(class: HandClass) -> u8 {
    class.high.saturating_sub(class.low).saturating_sub(1)
}

fn stratified_combos(scored: &[(Combo, f32)], max_combos: usize) -> Vec<Combo> {
    if scored.is_empty() {
        return Vec::new();
    }
    if scored.len() <= max_combos {
        return scored.iter().map(|(combo, _)| *combo).collect();
    }

    let mut combos = Vec::with_capacity(max_combos);
    let mut seen = HashSet::new();
    for slot in 0..max_combos {
        let index = if max_combos == 1 {
            0
        } else {
            slot * (scored.len() - 1) / (max_combos - 1)
        };
        let combo = scored[index].0;
        if seen.insert(combo) {
            combos.push(combo);
        }
    }
    combos
}

fn estimate_class_equity_on_board(
    game_state: &GameState,
    board: &[PokedrCard],
    idx: usize,
    opponent_idx: usize,
    combo: [PokedrCard; 2],
) -> Option<f32> {
    let mut hero_hand = Hand::new();
    hero_hand.insert(rs_card(combo[0]));
    hero_hand.insert(rs_card(combo[1]));
    hero_hand.extend(board.iter().map(|card| rs_card(*card)));

    let mut opponent_default = Hand::new();
    opponent_default.extend(board.iter().map(|card| rs_card(*card)));

    let mut hands = game_state.hands.clone();
    hands[idx] = hero_hand;
    hands[opponent_idx] = opponent_default;
    let mut monte = MonteCarloGame::new(hands).ok()?;
    monte.estimate_equity(32).get(idx).copied()
}

fn made_hand_strength(game_state: &GameState, idx: usize) -> u8 {
    if game_state.board.len() + game_state.hands[idx].count() < 5 {
        return 0;
    }
    rank_category(combined_hand_rank(game_state, idx))
}

fn heads_up_opponent(game_state: &GameState, idx: usize) -> Option<usize> {
    (0..game_state.num_players).find(|&other| {
        other != idx && (game_state.player_active.get(other) || game_state.player_all_in.get(other))
    })
}

fn heads_up_postflop_positions(game_state: &GameState) -> Option<(usize, usize)> {
    if game_state.num_players != 2 {
        return None;
    }
    let ip_idx = game_state.dealer_idx;
    let oop_idx = (0..game_state.num_players).find(|&idx| idx != ip_idx)?;
    if game_state.player_active.get(ip_idx) || game_state.player_all_in.get(ip_idx) {
        Some((oop_idx, ip_idx))
    } else {
        None
    }
}

fn pokedr_hand_mask(game_state: &GameState, idx: usize) -> Option<u64> {
    let mut mask = 0_u64;
    for card in private_cards(game_state, idx) {
        mask |= pokedr_card(card)?.mask();
    }
    Some(mask)
}

fn pokedr_board(game_state: &GameState) -> Option<Vec<PokedrCard>> {
    game_state
        .board
        .iter()
        .map(|card| pokedr_card(*card))
        .collect()
}

fn pokedr_card(card: rs_poker::core::Card) -> Option<PokedrCard> {
    Some(PokedrCard::new(
        u8::from(card.value) + 2,
        match card.suit {
            rs_poker::core::Suit::Club => 0,
            rs_poker::core::Suit::Diamond => 1,
            rs_poker::core::Suit::Heart => 2,
            rs_poker::core::Suit::Spade => 3,
        },
    ))
}

fn rs_card(card: PokedrCard) -> rs_poker::core::Card {
    rs_poker::core::Card::new(
        rs_poker::core::Value::from(card.rank() - 2),
        match card.suit() {
            0 => rs_poker::core::Suit::Club,
            1 => rs_poker::core::Suit::Diamond,
            2 => rs_poker::core::Suit::Heart,
            _ => rs_poker::core::Suit::Spade,
        },
    )
}

fn combined_hand_rank(game_state: &GameState, idx: usize) -> Rank {
    let mut hand = Hand::new();
    hand.extend(game_state.board.iter().copied());
    hand.extend(game_state.hands[idx].iter());
    hand.rank()
}

fn rank_category(rank: Rank) -> u8 {
    match rank {
        Rank::HighCard(_) => 0,
        Rank::OnePair(_) => 1,
        Rank::TwoPair(_) => 2,
        Rank::ThreeOfAKind(_) => 3,
        Rank::Straight(_) => 4,
        Rank::Flush(_) => 5,
        Rank::FullHouse(_) => 6,
        Rank::FourOfAKind(_) => 7,
        Rank::StraightFlush(_) => 8,
    }
}

fn hand_class(game_state: &GameState, idx: usize) -> Option<HandClass> {
    let cards = private_cards(game_state, idx);
    if cards.len() != 2 {
        return None;
    }
    let mut cards = cards.into_iter();
    let first = cards.next()?;
    let second = cards.next()?;
    let first_rank = u8::from(first.value) + 2;
    let second_rank = u8::from(second.value) + 2;
    let high = first_rank.max(second_rank);
    let low = first_rank.min(second_rank);
    Some(HandClass::new(
        high,
        low,
        first.suit == second.suit && high != low,
    ))
}

fn hand_class_from_combo(combo: Combo) -> HandClass {
    HandClass::new(
        combo.first.rank().max(combo.second.rank()),
        combo.first.rank().min(combo.second.rank()),
        combo.first.suit() == combo.second.suit() && combo.first.rank() != combo.second.rank(),
    )
}

fn private_cards(game_state: &GameState, idx: usize) -> Vec<rs_poker::core::Card> {
    let board: HashSet<_> = game_state.board.iter().copied().collect();
    game_state.hands[idx]
        .iter()
        .filter(|card| !board.contains(card))
        .collect()
}

fn hand_classes(labels: &[&str]) -> HashSet<HandClass> {
    labels
        .iter()
        .map(|label| hand_class_from_label(label))
        .collect()
}

fn hand_class_from_label(label: &str) -> HandClass {
    let bytes = label.as_bytes();
    let high = rank_from_byte(bytes[0]);
    let low = rank_from_byte(bytes[1]);
    let suited = bytes.get(2).copied() == Some(b's');
    HandClass::new(high, low, suited && high != low)
}

fn rank_from_byte(rank: u8) -> u8 {
    match rank {
        b'A' => 14,
        b'K' => 13,
        b'Q' => 12,
        b'J' => 11,
        b'T' => 10,
        b'9' => 9,
        b'8' => 8,
        b'7' => 7,
        b'6' => 6,
        b'5' => 5,
        b'4' => 4,
        b'3' => 3,
        b'2' => 2,
        _ => panic!("invalid hand class rank"),
    }
}
