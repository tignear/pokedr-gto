pub const DEFAULT_DCFR_PLUS_ALPHA: f32 = 1.5;
pub const DEFAULT_DCFR_PLUS_GAMMA: f32 = 4.0;
pub const DEFAULT_DCFR_SCHEDULE_ALPHA_START: f32 = 1.5;
pub const DEFAULT_DCFR_SCHEDULE_ALPHA_END: f32 = 2.5;
pub const DEFAULT_DCFR_SCHEDULE_GAMMA_START: f32 = 4.0;
pub const DEFAULT_DCFR_SCHEDULE_GAMMA_END: f32 = 8.0;
pub const DEFAULT_DCFR_SCHEDULE_HORIZON: usize = 128;
pub const DEFAULT_PDCFR_PLUS_ALPHA: f32 = 2.5;
pub const DEFAULT_PDCFR_PLUS_GAMMA: f32 = 32.0;
pub const DEFAULT_PDCFR_PLUS_ETA_START: f32 = 1.0;
pub const DEFAULT_PDCFR_PLUS_ETA: f32 = 0.0;
pub const DEFAULT_PDCFR_PLUS_ETA_HORIZON: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CfrVariant {
    CfrPlus,
    Discounted,
    DcfrPlus {
        alpha: f32,
        gamma: f32,
    },
    DcfrSchedule {
        alpha_start: f32,
        alpha_end: f32,
        gamma_start: f32,
        gamma_end: f32,
        horizon: usize,
    },
    PdcfrPlus {
        alpha: f32,
        gamma: f32,
        eta_start: f32,
        eta_end: f32,
        eta_horizon: usize,
    },
}

impl CfrVariant {
    pub const fn dcfr_plus_default() -> Self {
        Self::DcfrPlus {
            alpha: DEFAULT_DCFR_PLUS_ALPHA,
            gamma: DEFAULT_DCFR_PLUS_GAMMA,
        }
    }

    pub const fn pdcfr_plus_default() -> Self {
        Self::PdcfrPlus {
            alpha: DEFAULT_PDCFR_PLUS_ALPHA,
            gamma: DEFAULT_PDCFR_PLUS_GAMMA,
            eta_start: DEFAULT_PDCFR_PLUS_ETA_START,
            eta_end: DEFAULT_PDCFR_PLUS_ETA,
            eta_horizon: DEFAULT_PDCFR_PLUS_ETA_HORIZON,
        }
    }

    pub const fn dcfr_schedule_default() -> Self {
        Self::DcfrSchedule {
            alpha_start: DEFAULT_DCFR_SCHEDULE_ALPHA_START,
            alpha_end: DEFAULT_DCFR_SCHEDULE_ALPHA_END,
            gamma_start: DEFAULT_DCFR_SCHEDULE_GAMMA_START,
            gamma_end: DEFAULT_DCFR_SCHEDULE_GAMMA_END,
            horizon: DEFAULT_DCFR_SCHEDULE_HORIZON,
        }
    }

    pub fn is_dcfr_plus(self) -> bool {
        matches!(
            self,
            Self::DcfrPlus { .. } | Self::DcfrSchedule { .. } | Self::PdcfrPlus { .. }
        )
    }

    pub fn uses_prediction(self) -> bool {
        matches!(self, Self::PdcfrPlus { .. })
    }
}

pub mod gpu;

#[derive(Debug, Clone)]
pub struct DenseCfrConfig {
    pub infosets: usize,
    pub actions: usize,
    pub variant: CfrVariant,
}

#[derive(Debug, Clone)]
pub struct DenseCfrState {
    infosets: usize,
    actions: usize,
    variant: CfrVariant,
    legal_actions: Vec<bool>,
    legal_action_counts: Vec<usize>,
    regrets: Vec<f32>,
    prediction: Vec<f32>,
    strategy_sum: Vec<f32>,
}

impl DenseCfrState {
    pub fn new(config: DenseCfrConfig) -> Self {
        assert!(config.infosets > 0, "infosets must be non-empty");
        assert!(config.actions > 0, "actions must be non-empty");
        let len = config.infosets * config.actions;
        Self {
            infosets: config.infosets,
            actions: config.actions,
            variant: config.variant,
            legal_actions: vec![true; len],
            legal_action_counts: vec![config.actions; config.infosets],
            regrets: vec![0.0; len],
            prediction: vec![0.0; len],
            strategy_sum: vec![0.0; len],
        }
    }

    pub fn new_with_legal_actions(config: DenseCfrConfig, legal_actions: Vec<bool>) -> Self {
        assert!(config.infosets > 0, "infosets must be non-empty");
        assert!(config.actions > 0, "actions must be non-empty");
        assert_eq!(legal_actions.len(), config.infosets * config.actions);
        let legal_action_counts =
            legal_action_counts(config.infosets, config.actions, &legal_actions);
        assert!(
            legal_action_counts.iter().all(|count| *count > 0),
            "each infoset must have at least one legal action"
        );
        let len = config.infosets * config.actions;
        Self {
            infosets: config.infosets,
            actions: config.actions,
            variant: config.variant,
            legal_actions,
            legal_action_counts,
            regrets: vec![0.0; len],
            prediction: vec![0.0; len],
            strategy_sum: vec![0.0; len],
        }
    }

    pub fn infosets(&self) -> usize {
        self.infosets
    }

    pub fn actions(&self) -> usize {
        self.actions
    }

    pub fn regrets(&self) -> &[f32] {
        &self.regrets
    }

    pub fn strategy_sum(&self) -> &[f32] {
        &self.strategy_sum
    }

    pub fn prediction(&self) -> &[f32] {
        &self.prediction
    }

    pub fn legal_actions(&self) -> &[bool] {
        &self.legal_actions
    }

    pub fn average_strategy_profile_state(&self) -> Self {
        let mut profile = Self::new_with_legal_actions(
            DenseCfrConfig {
                infosets: self.infosets,
                actions: self.actions,
                variant: CfrVariant::CfrPlus,
            },
            self.legal_actions.clone(),
        );
        let mut strategy = vec![0.0; self.actions];
        for infoset in 0..self.infosets {
            self.average_strategy_for(infoset, &mut strategy);
            let offset = infoset * self.actions;
            profile.regrets[offset..offset + self.actions].copy_from_slice(&strategy);
        }
        profile
    }

    pub fn strategy_for(&self, infoset: usize, out: &mut [f32]) {
        self.strategy_for_at(infoset, out, usize::MAX);
    }

    fn strategy_for_at(&self, infoset: usize, out: &mut [f32], iteration: usize) {
        assert!(infoset < self.infosets);
        assert!(out.len() >= self.actions);
        let offset = self.offset(infoset);
        let regrets = &self.regrets[offset..offset + self.actions];
        let prediction = &self.prediction[offset..offset + self.actions];
        let legal = &self.legal_actions[offset..offset + self.actions];
        let normalizer: f32 = regrets
            .iter()
            .zip(prediction)
            .zip(legal)
            .filter(|(_, is_legal)| **is_legal)
            .map(|((value, predicted), _)| {
                effective_regret_at(self.variant, *value, *predicted, iteration).max(0.0)
            })
            .sum();
        if normalizer > f32::EPSILON {
            for action in 0..self.actions {
                out[action] = if legal[action] {
                    effective_regret_at(
                        self.variant,
                        regrets[action],
                        prediction[action],
                        iteration,
                    )
                    .max(0.0)
                        / normalizer
                } else {
                    0.0
                };
            }
        } else {
            let uniform = 1.0 / self.legal_action_counts[infoset] as f32;
            for action in 0..self.actions {
                out[action] = if legal[action] { uniform } else { 0.0 };
            }
        }
    }

    pub fn average_strategy_for(&self, infoset: usize, out: &mut [f32]) {
        assert!(infoset < self.infosets);
        assert!(out.len() >= self.actions);
        let offset = self.offset(infoset);
        let sum = &self.strategy_sum[offset..offset + self.actions];
        let legal = &self.legal_actions[offset..offset + self.actions];
        let normalizer: f32 = sum
            .iter()
            .zip(legal)
            .filter(|(_, is_legal)| **is_legal)
            .map(|(value, _)| *value)
            .sum();
        if normalizer > f32::EPSILON {
            for action in 0..self.actions {
                out[action] = if legal[action] {
                    sum[action] / normalizer
                } else {
                    0.0
                };
            }
        } else {
            self.strategy_for(infoset, out);
        }
    }

    pub fn update_infoset(
        &mut self,
        infoset: usize,
        action_values: &[f32],
        reach_weight: f32,
        strategy_weight: f32,
        iteration: usize,
    ) {
        assert!(infoset < self.infosets);
        assert!(action_values.len() >= self.actions);
        assert!(reach_weight.is_finite() && reach_weight >= 0.0);
        assert!(strategy_weight.is_finite() && strategy_weight >= 0.0);

        let offset = self.offset(infoset);
        let mut strategy = vec![0.0; self.actions];
        self.strategy_for_at(infoset, &mut strategy, iteration);
        let node_value: f32 = strategy
            .iter()
            .zip(action_values.iter())
            .map(|(probability, value)| probability * value)
            .sum();

        let discount = regret_discount(self.variant, iteration);
        for action in 0..self.actions {
            if !self.legal_actions[offset + action] {
                self.regrets[offset + action] = 0.0;
                if self.variant.uses_prediction() {
                    self.prediction[offset + action] = 0.0;
                }
                self.strategy_sum[offset + action] = 0.0;
                continue;
            }
            let regret = (action_values[action] - node_value) * reach_weight;
            let slot = &mut self.regrets[offset + action];
            *slot *= discount;
            *slot += regret;
            if matches!(
                self.variant,
                CfrVariant::CfrPlus
                    | CfrVariant::DcfrPlus { .. }
                    | CfrVariant::DcfrSchedule { .. }
                    | CfrVariant::PdcfrPlus { .. }
            ) {
                *slot = slot.max(0.0);
            }
            if self.variant.uses_prediction() {
                self.prediction[offset + action] = regret;
            }
            self.strategy_sum[offset + action] *=
                average_strategy_discount(self.variant, iteration);
            self.strategy_sum[offset + action] += strategy_weight * strategy[action];
        }
    }

    pub fn update_all_infosets(
        &mut self,
        action_values: &[f32],
        reach_weights: &[f32],
        strategy_weights: &[f32],
        iteration: usize,
    ) {
        assert_eq!(action_values.len(), self.infosets * self.actions);
        assert_eq!(reach_weights.len(), self.infosets);
        assert_eq!(strategy_weights.len(), self.infosets);
        for infoset in 0..self.infosets {
            let offset = self.offset(infoset);
            self.update_infoset(
                infoset,
                &action_values[offset..offset + self.actions],
                reach_weights[infoset],
                strategy_weights[infoset],
                iteration,
            );
        }
    }

    fn offset(&self, infoset: usize) -> usize {
        infoset * self.actions
    }
}

#[derive(Debug, Clone)]
pub struct DenseCfrIteration {
    pub action_values: Vec<f32>,
    pub reach_weights: Vec<f32>,
    pub strategy_weights: Vec<f32>,
}

impl DenseCfrIteration {
    pub fn new(config: &DenseCfrConfig) -> Self {
        Self {
            action_values: vec![0.0; config.infosets * config.actions],
            reach_weights: vec![0.0; config.infosets],
            strategy_weights: vec![0.0; config.infosets],
        }
    }

    pub fn validate(&self, config: &DenseCfrConfig) {
        assert_eq!(self.action_values.len(), config.infosets * config.actions);
        assert_eq!(self.reach_weights.len(), config.infosets);
        assert_eq!(self.strategy_weights.len(), config.infosets);
        assert!(self.action_values.iter().all(|value| value.is_finite()));
        assert!(
            self.reach_weights
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(
            self.strategy_weights
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseCfrRunStats {
    pub iterations: usize,
}

pub struct DenseCfrSolver {
    config: DenseCfrConfig,
    state: DenseCfrState,
    iterations: usize,
}

impl DenseCfrSolver {
    pub fn new(config: DenseCfrConfig) -> Self {
        Self {
            state: DenseCfrState::new(config.clone()),
            config,
            iterations: 0,
        }
    }

    pub fn state(&self) -> &DenseCfrState {
        &self.state
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn run_iterations(
        &mut self,
        count: usize,
        mut fill_iteration: impl FnMut(usize, &DenseCfrState, &mut DenseCfrIteration),
    ) -> DenseCfrRunStats {
        let mut batch = DenseCfrIteration::new(&self.config);
        for _ in 0..count {
            let iteration = self.iterations + 1;
            fill_iteration(iteration, &self.state, &mut batch);
            batch.validate(&self.config);
            self.state.update_all_infosets(
                &batch.action_values,
                &batch.reach_weights,
                &batch.strategy_weights,
                iteration,
            );
            self.iterations = iteration;
        }
        DenseCfrRunStats { iterations: count }
    }

    pub fn into_state(self) -> DenseCfrState {
        self.state
    }
}

fn regret_discount(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::CfrPlus => 1.0,
        CfrVariant::Discounted => {
            let t = iteration.max(1) as f32;
            t / (t + 1.0)
        }
        CfrVariant::DcfrPlus { .. }
        | CfrVariant::DcfrSchedule { .. }
        | CfrVariant::PdcfrPlus { .. } => {
            if iteration <= 1 {
                0.0
            } else {
                let alpha = dcfr_alpha(variant, iteration);
                let weighted = ((iteration - 1) as f32).powf(alpha);
                weighted / (weighted + 1.5)
            }
        }
    }
}

fn average_strategy_discount(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::DcfrPlus { .. }
        | CfrVariant::DcfrSchedule { .. }
        | CfrVariant::PdcfrPlus { .. }
            if iteration > 1 =>
        {
            let gamma = dcfr_gamma(variant, iteration);
            (((iteration - 1) as f32) / iteration as f32).powf(gamma)
        }
        _ => 1.0,
    }
}

fn dcfr_alpha(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::DcfrPlus { alpha, .. } | CfrVariant::PdcfrPlus { alpha, .. } => alpha,
        CfrVariant::DcfrSchedule {
            alpha_start,
            alpha_end,
            horizon,
            ..
        } => scheduled_value(alpha_start, alpha_end, iteration, horizon),
        _ => DEFAULT_DCFR_PLUS_ALPHA,
    }
}

fn dcfr_gamma(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::DcfrPlus { gamma, .. } | CfrVariant::PdcfrPlus { gamma, .. } => gamma,
        CfrVariant::DcfrSchedule {
            gamma_start,
            gamma_end,
            horizon,
            ..
        } => scheduled_value(gamma_start, gamma_end, iteration, horizon),
        _ => DEFAULT_DCFR_PLUS_GAMMA,
    }
}

fn scheduled_value(start: f32, end: f32, iteration: usize, horizon: usize) -> f32 {
    let horizon = horizon.max(2);
    let progress = (iteration.saturating_sub(1) as f32 / (horizon - 1) as f32).clamp(0.0, 1.0);
    start + (end - start) * progress
}

fn pdcfr_eta(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::PdcfrPlus {
            eta_start,
            eta_end,
            eta_horizon,
            ..
        } => scheduled_value(eta_start, eta_end, iteration, eta_horizon),
        _ => 0.0,
    }
}

fn effective_regret_at(variant: CfrVariant, regret: f32, prediction: f32, iteration: usize) -> f32 {
    match variant {
        CfrVariant::PdcfrPlus { .. } => regret + pdcfr_eta(variant, iteration) * prediction,
        _ => regret,
    }
}

fn legal_action_counts(infosets: usize, actions: usize, legal_actions: &[bool]) -> Vec<usize> {
    (0..infosets)
        .map(|infoset| {
            let offset = infoset * actions;
            legal_actions[offset..offset + actions]
                .iter()
                .filter(|is_legal| **is_legal)
                .count()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_matching_starts_uniform() {
        let state = DenseCfrState::new(DenseCfrConfig {
            infosets: 2,
            actions: 3,
            variant: CfrVariant::CfrPlus,
        });
        let mut strategy = [0.0; 3];
        state.strategy_for(0, &mut strategy);
        assert_eq!(strategy, [1.0 / 3.0; 3]);
    }

    #[test]
    fn cfr_plus_updates_positive_regrets_and_average_strategy() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::CfrPlus,
        });
        state.update_infoset(0, &[1.0, -1.0], 1.0, 1.0, 1);

        assert!(state.regrets()[0] > 0.0);
        assert_eq!(state.regrets()[1], 0.0);

        let mut average = [0.0; 2];
        state.average_strategy_for(0, &mut average);
        assert_eq!(average, [0.5, 0.5]);
    }

    #[test]
    fn strategy_moves_toward_positive_regret_action() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::CfrPlus,
        });
        state.update_infoset(0, &[2.0, -1.0], 1.0, 1.0, 1);

        let mut strategy = [0.0; 2];
        state.strategy_for(0, &mut strategy);
        assert_eq!(strategy, [1.0, 0.0]);
    }

    #[test]
    fn dcfr_plus_clips_regrets_and_discounts_average_strategy() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::dcfr_plus_default(),
        });
        state.update_infoset(0, &[1.0, -1.0], 1.0, 1.0, 1);
        assert_eq!(state.regrets()[1], 0.0);
        assert_eq!(state.strategy_sum(), &[0.5, 0.5]);

        state.update_infoset(0, &[1.0, -1.0], 1.0, 1.0, 2);
        let discount = (1.0_f32 / 2.0).powf(4.0);
        assert!((state.strategy_sum()[0] - (0.5 * discount + 1.0)).abs() < 1e-6);
        assert!((state.strategy_sum()[1] - (0.5 * discount)).abs() < 1e-6);
        assert!(state.regrets().iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn dcfr_schedule_interpolates_discount_parameters() {
        let variant = CfrVariant::DcfrSchedule {
            alpha_start: 0.5,
            alpha_end: 2.5,
            gamma_start: 16.0,
            gamma_end: 8.0,
            horizon: 5,
        };

        assert!((dcfr_alpha(variant, 1) - 0.5).abs() < 1e-6);
        assert!((dcfr_alpha(variant, 3) - 1.5).abs() < 1e-6);
        assert!((dcfr_alpha(variant, 9) - 2.5).abs() < 1e-6);
        assert!((dcfr_gamma(variant, 1) - 16.0).abs() < 1e-6);
        assert!((dcfr_gamma(variant, 3) - 12.0).abs() < 1e-6);
        assert!((dcfr_gamma(variant, 9) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn pdcfr_plus_strategy_uses_previous_instant_regret() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::PdcfrPlus {
                alpha: 2.5,
                gamma: 8.0,
                eta_start: 1.0,
                eta_end: 1.0,
                eta_horizon: 2,
            },
        });
        state.regrets.copy_from_slice(&[0.0, 1.0]);
        state.prediction.copy_from_slice(&[1.0, 0.0]);

        let mut strategy = [0.0; 2];
        state.strategy_for(0, &mut strategy);

        assert_eq!(strategy, [0.5, 0.5]);
    }

    #[test]
    fn pdcfr_plus_eta_can_be_scheduled_by_iteration() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::PdcfrPlus {
                alpha: 2.5,
                gamma: 8.0,
                eta_start: 0.0,
                eta_end: 1.0,
                eta_horizon: 3,
            },
        });
        state.regrets.copy_from_slice(&[0.0, 1.0]);
        state.prediction.copy_from_slice(&[1.0, 0.0]);

        let mut strategy = [0.0; 2];
        state.strategy_for_at(0, &mut strategy, 1);
        assert_eq!(strategy, [0.0, 1.0]);
        state.strategy_for_at(0, &mut strategy, 2);
        assert!((strategy[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((strategy[1] - 2.0 / 3.0).abs() < 1e-6);
        state.strategy_for_at(0, &mut strategy, 3);
        assert_eq!(strategy, [0.5, 0.5]);
    }

    #[test]
    fn pdcfr_plus_records_current_instant_regret_as_prediction() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::PdcfrPlus {
                alpha: 2.5,
                gamma: 8.0,
                eta_start: 1.0,
                eta_end: 1.0,
                eta_horizon: 2,
            },
        });

        state.update_infoset(0, &[1.0, -1.0], 1.0, 1.0, 1);

        assert_eq!(state.prediction(), &[1.0, -1.0]);
        assert_eq!(state.regrets(), &[1.0, 0.0]);
    }

    #[test]
    fn legal_action_mask_excludes_padding_actions() {
        let mut state = DenseCfrState::new_with_legal_actions(
            DenseCfrConfig {
                infosets: 1,
                actions: 4,
                variant: CfrVariant::CfrPlus,
            },
            vec![true, false, true, false],
        );

        let mut strategy = [0.0; 4];
        state.strategy_for(0, &mut strategy);
        assert_eq!(strategy, [0.5, 0.0, 0.5, 0.0]);

        state.update_infoset(0, &[1.0, 100.0, -1.0, 100.0], 1.0, 1.0, 1);
        state.strategy_for(0, &mut strategy);
        assert_eq!(strategy[1], 0.0);
        assert_eq!(strategy[3], 0.0);
        assert_eq!(state.regrets()[1], 0.0);
        assert_eq!(state.regrets()[3], 0.0);
    }

    #[test]
    fn solver_reuses_iteration_batch_and_tracks_iterations() {
        let config = DenseCfrConfig {
            infosets: 3,
            actions: 2,
            variant: CfrVariant::Discounted,
        };
        let mut solver = DenseCfrSolver::new(config);
        let stats = solver.run_iterations(4, |iteration, _state, batch| {
            for (index, value) in batch.action_values.iter_mut().enumerate() {
                *value = ((index + iteration) as f32 * 0.5).sin();
            }
            batch.reach_weights.fill(1.0);
            batch.strategy_weights.fill(0.25);
        });

        assert_eq!(stats.iterations, 4);
        assert_eq!(solver.iterations(), 4);
        assert!(
            solver
                .state()
                .regrets()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            solver
                .state()
                .strategy_sum()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
