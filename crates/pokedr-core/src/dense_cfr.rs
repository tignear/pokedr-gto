#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfrVariant {
    CfrPlus,
    Discounted,
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
    regrets: Vec<f32>,
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
            regrets: vec![0.0; len],
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

    pub fn strategy_for(&self, infoset: usize, out: &mut [f32]) {
        assert!(infoset < self.infosets);
        assert!(out.len() >= self.actions);
        let offset = self.offset(infoset);
        let regrets = &self.regrets[offset..offset + self.actions];
        let normalizer: f32 = regrets.iter().map(|value| value.max(0.0)).sum();
        if normalizer > 0.0 {
            for action in 0..self.actions {
                out[action] = regrets[action].max(0.0) / normalizer;
            }
        } else {
            let uniform = 1.0 / self.actions as f32;
            for value in out.iter_mut().take(self.actions) {
                *value = uniform;
            }
        }
    }

    pub fn average_strategy_for(&self, infoset: usize, out: &mut [f32]) {
        assert!(infoset < self.infosets);
        assert!(out.len() >= self.actions);
        let offset = self.offset(infoset);
        let sum = &self.strategy_sum[offset..offset + self.actions];
        let normalizer: f32 = sum.iter().sum();
        if normalizer > 0.0 {
            for action in 0..self.actions {
                out[action] = sum[action] / normalizer;
            }
        } else {
            let uniform = 1.0 / self.actions as f32;
            for value in out.iter_mut().take(self.actions) {
                *value = uniform;
            }
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
        self.strategy_for(infoset, &mut strategy);
        let node_value: f32 = strategy
            .iter()
            .zip(action_values.iter())
            .map(|(probability, value)| probability * value)
            .sum();

        let discount = regret_discount(self.variant, iteration);
        for action in 0..self.actions {
            let regret = (action_values[action] - node_value) * reach_weight;
            let slot = &mut self.regrets[offset + action];
            *slot *= discount;
            *slot += regret;
            if matches!(self.variant, CfrVariant::CfrPlus) {
                *slot = slot.max(0.0);
            }
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

fn regret_discount(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::CfrPlus => 1.0,
        CfrVariant::Discounted => {
            let t = iteration.max(1) as f32;
            t / (t + 1.0)
        }
    }
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
}
