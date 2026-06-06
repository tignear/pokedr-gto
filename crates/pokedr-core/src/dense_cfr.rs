pub const DEFAULT_DCFR_PLUS_ALPHA: f32 = 1.5;
pub const DEFAULT_DCFR_PLUS_GAMMA: f32 = 4.0;
pub const DEFAULT_DCFR_ALPHA: f32 = 1.5;
pub const DEFAULT_DCFR_BETA: f32 = 0.5;
pub const DEFAULT_DCFR_GAMMA: f32 = 2.0;
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
const AVERAGE_STRATEGY_DELAY_ENV: &str = "POKEDR_AVG_DELAY";
const AVERAGE_STRATEGY_POWER_ENV: &str = "POKEDR_AVG_POWER";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CfrVariant {
    CfrPlus,
    Discounted,
    Dcfr {
        alpha: f32,
        beta: f32,
        gamma: f32,
    },
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

    pub const fn dcfr_default() -> Self {
        Self::Dcfr {
            alpha: DEFAULT_DCFR_ALPHA,
            beta: DEFAULT_DCFR_BETA,
            gamma: DEFAULT_DCFR_GAMMA,
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
            Self::Dcfr { .. }
                | Self::DcfrPlus { .. }
                | Self::DcfrSchedule { .. }
                | Self::PdcfrPlus { .. }
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

#[derive(Debug, Clone)]
pub struct CompactCfrConfig {
    pub action_offsets: Vec<usize>,
    pub variant: CfrVariant,
}

impl CompactCfrConfig {
    pub fn infosets(&self) -> usize {
        self.action_offsets.len().saturating_sub(1)
    }

    pub fn total_actions(&self) -> usize {
        self.action_offsets.last().copied().unwrap_or(0)
    }

    pub fn validate(&self) {
        assert!(
            self.action_offsets.len() >= 2,
            "action offsets must contain at least one infoset"
        );
        assert_eq!(self.action_offsets[0], 0);
        for window in self.action_offsets.windows(2) {
            assert!(
                window[1] > window[0],
                "each compact infoset must contain at least one action"
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactCfrState {
    action_offsets: Vec<usize>,
    variant: CfrVariant,
    regrets: Vec<f32>,
    prediction: Vec<f32>,
    strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CompactPrivateCfrConfig {
    pub public_action_offsets: Vec<usize>,
    pub combos: usize,
    pub variant: CfrVariant,
}

#[derive(Debug, Clone)]
pub struct BatchedPrivateCfrConfig {
    pub batches: usize,
    pub public_infosets: usize,
    pub combos: usize,
    pub actions: usize,
    pub variant: CfrVariant,
}

#[derive(Debug, Clone)]
pub struct BatchedPrivateCfrState {
    config: BatchedPrivateCfrConfig,
    legal_actions: Vec<bool>,
    regrets: Vec<f32>,
    prediction: Vec<f32>,
    strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactPrivateCfrChunk {
    pub public_start: usize,
    pub public_end: usize,
    pub public_action_start: usize,
    pub public_action_end: usize,
    pub action_slots: usize,
}

impl CompactPrivateCfrConfig {
    pub fn public_infosets(&self) -> usize {
        self.public_action_offsets.len().saturating_sub(1)
    }

    pub fn public_actions(&self) -> usize {
        self.public_action_offsets.last().copied().unwrap_or(0)
    }

    pub fn total_action_slots(&self) -> usize {
        self.public_actions() * self.combos
    }

    pub fn validate(&self) {
        assert!(self.combos > 0, "combo count must be non-empty");
        assert!(
            self.public_action_offsets.len() >= 2,
            "action offsets must contain at least one public infoset"
        );
        assert_eq!(self.public_action_offsets[0], 0);
        for window in self.public_action_offsets.windows(2) {
            assert!(
                window[1] > window[0],
                "each public infoset must contain at least one action"
            );
        }
    }

    pub fn chunk_by_action_bytes(&self, max_bytes: usize) -> Vec<CompactPrivateCfrChunk> {
        self.validate();
        let bytes_per_public_action = self.combos * std::mem::size_of::<f32>();
        assert!(
            max_bytes >= bytes_per_public_action,
            "max bytes must fit at least one public action"
        );
        let max_public_actions = (max_bytes / bytes_per_public_action).max(1);
        let mut chunks = Vec::new();
        let mut public_start = 0usize;
        while public_start < self.public_infosets() {
            let action_start = self.public_action_offsets[public_start];
            let mut public_end = public_start + 1;
            while public_end < self.public_infosets() {
                let action_len = self.public_action_offsets[public_end + 1] - action_start;
                if action_len > max_public_actions {
                    break;
                }
                public_end += 1;
            }
            let action_end = self.public_action_offsets[public_end];
            chunks.push(CompactPrivateCfrChunk {
                public_start,
                public_end,
                public_action_start: action_start,
                public_action_end: action_end,
                action_slots: (action_end - action_start) * self.combos,
            });
            public_start = public_end;
        }
        chunks
    }
}

#[derive(Debug, Clone)]
pub struct CompactPrivateCfrState {
    public_action_offsets: Vec<usize>,
    combos: usize,
    variant: CfrVariant,
    regrets: Vec<f32>,
    prediction: Vec<f32>,
    strategy_sum: Vec<f32>,
}

impl CompactPrivateCfrState {
    pub fn new(config: CompactPrivateCfrConfig) -> Self {
        config.validate();
        let len = config.total_action_slots();
        Self {
            public_action_offsets: config.public_action_offsets,
            combos: config.combos,
            variant: config.variant,
            regrets: vec![0.0; len],
            prediction: vec![0.0; len],
            strategy_sum: vec![0.0; len],
        }
    }

    pub fn from_dense_private_state(
        dense: &DenseCfrState,
        config: CompactPrivateCfrConfig,
    ) -> Self {
        config.validate();
        assert_eq!(
            dense.infosets(),
            config.public_infosets() * config.combos,
            "dense state must be indexed as public_infoset * combos + combo"
        );
        assert!(
            dense.actions()
                >= config
                    .public_action_offsets
                    .windows(2)
                    .map(|window| window[1] - window[0])
                    .max()
                    .unwrap_or(0),
            "dense max actions must cover compact public action counts"
        );
        let mut compact = Self::new(config);
        for public_infoset in 0..compact.public_infosets() {
            let action_count = compact.action_count(public_infoset);
            for combo in 0..compact.combos {
                let dense_infoset = public_infoset * compact.combos + combo;
                for action in 0..action_count {
                    let dense_slot = dense_infoset * dense.actions() + action;
                    let compact_slot = compact.slot(public_infoset, combo, action);
                    compact.regrets[compact_slot] = dense.regrets[dense_slot];
                    compact.prediction[compact_slot] = dense.prediction[dense_slot];
                    compact.strategy_sum[compact_slot] = dense.strategy_sum[dense_slot];
                }
            }
        }
        compact
    }

    pub fn public_infosets(&self) -> usize {
        self.public_action_offsets.len() - 1
    }

    pub fn combos(&self) -> usize {
        self.combos
    }

    pub fn public_actions(&self) -> usize {
        self.public_action_offsets.last().copied().unwrap_or(0)
    }

    pub fn total_action_slots(&self) -> usize {
        self.regrets.len()
    }

    pub fn action_count(&self, public_infoset: usize) -> usize {
        self.public_action_offsets[public_infoset + 1] - self.public_action_offsets[public_infoset]
    }

    pub fn slot(&self, public_infoset: usize, combo: usize, action: usize) -> usize {
        assert!(public_infoset < self.public_infosets());
        assert!(combo < self.combos);
        assert!(action < self.action_count(public_infoset));
        (self.public_action_offsets[public_infoset] + action) * self.combos + combo
    }

    pub fn regrets(&self) -> &[f32] {
        &self.regrets
    }

    pub fn prediction(&self) -> &[f32] {
        &self.prediction
    }

    pub fn strategy_sum(&self) -> &[f32] {
        &self.strategy_sum
    }

    pub fn strategy_for(&self, public_infoset: usize, combo: usize, out: &mut [f32]) {
        self.strategy_for_at(public_infoset, combo, out, usize::MAX);
    }

    fn strategy_for_at(
        &self,
        public_infoset: usize,
        combo: usize,
        out: &mut [f32],
        iteration: usize,
    ) {
        let action_count = self.action_count(public_infoset);
        assert!(combo < self.combos);
        assert!(out.len() >= action_count);
        let normalizer: f32 = (0..action_count)
            .map(|action| {
                let slot = self.slot(public_infoset, combo, action);
                effective_regret_at(
                    self.variant,
                    self.regrets[slot],
                    self.prediction[slot],
                    iteration,
                )
                .max(0.0)
            })
            .sum();
        if normalizer > f32::EPSILON {
            for (action, output) in out.iter_mut().take(action_count).enumerate() {
                let slot = self.slot(public_infoset, combo, action);
                *output = effective_regret_at(
                    self.variant,
                    self.regrets[slot],
                    self.prediction[slot],
                    iteration,
                )
                .max(0.0)
                    / normalizer;
            }
        } else {
            out[..action_count].fill(1.0 / action_count as f32);
        }
    }

    pub fn average_strategy_for(&self, public_infoset: usize, combo: usize, out: &mut [f32]) {
        let action_count = self.action_count(public_infoset);
        assert!(combo < self.combos);
        assert!(out.len() >= action_count);
        let normalizer: f32 = (0..action_count)
            .map(|action| self.strategy_sum[self.slot(public_infoset, combo, action)])
            .sum();
        if normalizer > f32::EPSILON {
            for (action, output) in out.iter_mut().take(action_count).enumerate() {
                *output = self.strategy_sum[self.slot(public_infoset, combo, action)] / normalizer;
            }
        } else {
            self.strategy_for(public_infoset, combo, out);
        }
    }

    pub fn update_infoset(
        &mut self,
        public_infoset: usize,
        combo: usize,
        action_values: &[f32],
        reach_weight: f32,
        strategy_weight: f32,
        iteration: usize,
    ) {
        let action_count = self.action_count(public_infoset);
        assert!(combo < self.combos);
        assert!(action_values.len() >= action_count);
        assert!(reach_weight.is_finite() && reach_weight >= 0.0);
        assert!(strategy_weight.is_finite() && strategy_weight >= 0.0);

        let mut strategy = vec![0.0; action_count];
        self.strategy_for_at(public_infoset, combo, &mut strategy, iteration);
        let node_value: f32 = strategy
            .iter()
            .zip(action_values.iter())
            .map(|(probability, value)| probability * value)
            .sum();

        for action in 0..action_count {
            let slot_index = self.slot(public_infoset, combo, action);
            let regret = (action_values[action] - node_value) * reach_weight;
            let slot = &mut self.regrets[slot_index];
            *slot *= regret_discount(self.variant, iteration, *slot);
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
                self.prediction[slot_index] = regret;
            }
            self.strategy_sum[slot_index] *= average_strategy_discount(self.variant, iteration);
            self.strategy_sum[slot_index] +=
                strategy_weight * average_strategy_weight_multiplier(iteration) * strategy[action];
        }
    }
}

impl CompactCfrState {
    pub fn new(config: CompactCfrConfig) -> Self {
        config.validate();
        let len = config.total_actions();
        Self {
            action_offsets: config.action_offsets,
            variant: config.variant,
            regrets: vec![0.0; len],
            prediction: vec![0.0; len],
            strategy_sum: vec![0.0; len],
        }
    }

    pub fn infosets(&self) -> usize {
        self.action_offsets.len() - 1
    }

    pub fn total_actions(&self) -> usize {
        self.regrets.len()
    }

    pub fn action_count(&self, infoset: usize) -> usize {
        self.range(infoset).len()
    }

    pub fn action_offsets(&self) -> &[usize] {
        &self.action_offsets
    }

    pub fn regrets(&self) -> &[f32] {
        &self.regrets
    }

    pub fn prediction(&self) -> &[f32] {
        &self.prediction
    }

    pub fn strategy_sum(&self) -> &[f32] {
        &self.strategy_sum
    }

    pub fn strategy_for(&self, infoset: usize, out: &mut [f32]) {
        self.strategy_for_at(infoset, out, usize::MAX);
    }

    fn strategy_for_at(&self, infoset: usize, out: &mut [f32], iteration: usize) {
        let range = self.range(infoset);
        assert!(out.len() >= range.len());
        let regrets = &self.regrets[range.clone()];
        let prediction = &self.prediction[range.clone()];
        let normalizer: f32 = regrets
            .iter()
            .zip(prediction)
            .map(|(value, predicted)| {
                effective_regret_at(self.variant, *value, *predicted, iteration).max(0.0)
            })
            .sum();
        if normalizer > f32::EPSILON {
            for action in 0..range.len() {
                out[action] = effective_regret_at(
                    self.variant,
                    regrets[action],
                    prediction[action],
                    iteration,
                )
                .max(0.0)
                    / normalizer;
            }
        } else {
            let uniform = 1.0 / range.len() as f32;
            out[..range.len()].fill(uniform);
        }
    }

    pub fn average_strategy_for(&self, infoset: usize, out: &mut [f32]) {
        let range = self.range(infoset);
        assert!(out.len() >= range.len());
        let sum = &self.strategy_sum[range.clone()];
        let normalizer: f32 = sum.iter().sum();
        if normalizer > f32::EPSILON {
            for action in 0..range.len() {
                out[action] = sum[action] / normalizer;
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
        let range = self.range(infoset);
        assert!(action_values.len() >= range.len());
        assert!(reach_weight.is_finite() && reach_weight >= 0.0);
        assert!(strategy_weight.is_finite() && strategy_weight >= 0.0);

        let mut strategy = vec![0.0; range.len()];
        self.strategy_for_at(infoset, &mut strategy, iteration);
        let node_value: f32 = strategy
            .iter()
            .zip(action_values.iter())
            .map(|(probability, value)| probability * value)
            .sum();

        for action in 0..range.len() {
            let slot_index = range.start + action;
            let regret = (action_values[action] - node_value) * reach_weight;
            let slot = &mut self.regrets[slot_index];
            *slot *= regret_discount(self.variant, iteration, *slot);
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
                self.prediction[slot_index] = regret;
            }
            self.strategy_sum[slot_index] *= average_strategy_discount(self.variant, iteration);
            self.strategy_sum[slot_index] +=
                strategy_weight * average_strategy_weight_multiplier(iteration) * strategy[action];
        }
    }

    fn range(&self, infoset: usize) -> std::ops::Range<usize> {
        assert!(infoset < self.infosets());
        self.action_offsets[infoset]..self.action_offsets[infoset + 1]
    }
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
            *slot *= regret_discount(self.variant, iteration, *slot);
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
            self.strategy_sum[offset + action] +=
                strategy_weight * average_strategy_weight_multiplier(iteration) * strategy[action];
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

impl BatchedPrivateCfrConfig {
    pub fn validate(&self) {
        assert!(self.batches > 0, "batch count must be non-empty");
        assert!(
            self.public_infosets > 0,
            "public infosets must be non-empty"
        );
        assert!(self.combos > 0, "combo count must be non-empty");
        assert!(self.actions > 0, "actions must be non-empty");
    }

    pub fn private_infosets_per_batch(&self) -> usize {
        self.public_infosets * self.combos
    }

    pub fn private_infosets(&self) -> usize {
        self.batches * self.private_infosets_per_batch()
    }

    pub fn action_slots_per_batch(&self) -> usize {
        self.private_infosets_per_batch() * self.actions
    }

    pub fn action_slots(&self) -> usize {
        self.batches * self.action_slots_per_batch()
    }

    pub fn offset(
        &self,
        batch: usize,
        public_infoset: usize,
        combo: usize,
        action: usize,
    ) -> usize {
        self.validate_indices(batch, public_infoset, combo, action);
        (((batch * self.public_infosets + public_infoset) * self.combos + combo) * self.actions)
            + action
    }

    fn validate_indices(&self, batch: usize, public_infoset: usize, combo: usize, action: usize) {
        assert!(batch < self.batches, "batch index out of range");
        assert!(
            public_infoset < self.public_infosets,
            "public infoset index out of range"
        );
        assert!(combo < self.combos, "combo index out of range");
        assert!(action < self.actions, "action index out of range");
    }
}

impl BatchedPrivateCfrState {
    pub fn new(config: BatchedPrivateCfrConfig, legal_actions_per_public: &[bool]) -> Self {
        config.validate();
        assert_eq!(
            legal_actions_per_public.len(),
            config.public_infosets * config.actions
        );
        let mut legal_actions = vec![false; config.action_slots()];
        for batch in 0..config.batches {
            for public_infoset in 0..config.public_infosets {
                let public_offset = public_infoset * config.actions;
                for combo in 0..config.combos {
                    for action in 0..config.actions {
                        let target = config.offset(batch, public_infoset, combo, action);
                        legal_actions[target] = legal_actions_per_public[public_offset + action];
                    }
                }
            }
        }
        Self {
            regrets: vec![0.0; config.action_slots()],
            prediction: vec![0.0; config.action_slots()],
            strategy_sum: vec![0.0; config.action_slots()],
            legal_actions,
            config,
        }
    }

    pub fn config(&self) -> &BatchedPrivateCfrConfig {
        &self.config
    }

    pub fn legal_actions(&self) -> &[bool] {
        &self.legal_actions
    }

    pub fn regrets(&self) -> &[f32] {
        &self.regrets
    }

    pub fn prediction(&self) -> &[f32] {
        &self.prediction
    }

    pub fn strategy_sum(&self) -> &[f32] {
        &self.strategy_sum
    }

    pub fn dense_state_for_batch(&self, batch: usize) -> DenseCfrState {
        assert!(batch < self.config.batches, "batch index out of range");
        let dense_config = DenseCfrConfig {
            infosets: self.config.private_infosets_per_batch(),
            actions: self.config.actions,
            variant: self.config.variant,
        };
        let batch_start = batch * self.config.action_slots_per_batch();
        let batch_end = batch_start + self.config.action_slots_per_batch();
        let legal_actions = self.legal_actions[batch_start..batch_end].to_vec();
        let legal_action_counts =
            legal_action_counts(dense_config.infosets, dense_config.actions, &legal_actions);
        DenseCfrState {
            infosets: dense_config.infosets,
            actions: dense_config.actions,
            variant: dense_config.variant,
            legal_actions,
            legal_action_counts,
            regrets: self.regrets[batch_start..batch_end].to_vec(),
            prediction: self.prediction[batch_start..batch_end].to_vec(),
            strategy_sum: self.strategy_sum[batch_start..batch_end].to_vec(),
        }
    }

    pub fn average_strategy_profile_state(&self) -> Self {
        let mut profile_config = self.config.clone();
        profile_config.variant = CfrVariant::CfrPlus;
        let mut profile = Self::new(profile_config, &self.legal_actions_per_public());
        for batch in 0..self.config.batches {
            let dense_profile = self
                .dense_state_for_batch(batch)
                .average_strategy_profile_state();
            profile.overwrite_batch_from_dense(batch, &dense_profile);
        }
        profile
    }

    fn legal_actions_per_public(&self) -> Vec<bool> {
        let mut legal = vec![false; self.config.public_infosets * self.config.actions];
        for public_infoset in 0..self.config.public_infosets {
            let source = self.config.offset(0, public_infoset, 0, 0);
            let target = public_infoset * self.config.actions;
            legal[target..target + self.config.actions]
                .copy_from_slice(&self.legal_actions[source..source + self.config.actions]);
        }
        legal
    }

    pub fn overwrite_batch_from_dense(&mut self, batch: usize, state: &DenseCfrState) {
        assert!(batch < self.config.batches, "batch index out of range");
        assert_eq!(state.infosets, self.config.private_infosets_per_batch());
        assert_eq!(state.actions, self.config.actions);
        assert_eq!(state.variant, self.config.variant);
        let batch_start = batch * self.config.action_slots_per_batch();
        let batch_end = batch_start + self.config.action_slots_per_batch();
        self.legal_actions[batch_start..batch_end].copy_from_slice(&state.legal_actions);
        self.regrets[batch_start..batch_end].copy_from_slice(&state.regrets);
        self.prediction[batch_start..batch_end].copy_from_slice(&state.prediction);
        self.strategy_sum[batch_start..batch_end].copy_from_slice(&state.strategy_sum);
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

fn regret_discount(variant: CfrVariant, iteration: usize, regret: f32) -> f32 {
    match variant {
        CfrVariant::CfrPlus => 1.0,
        CfrVariant::Discounted => {
            let t = iteration.max(1) as f32;
            t / (t + 1.0)
        }
        CfrVariant::Dcfr { alpha, beta, .. } => {
            if iteration <= 1 {
                0.0
            } else {
                let exponent = if regret > 0.0 { alpha } else { beta };
                let weighted = ((iteration - 1) as f32).powf(exponent);
                weighted / (weighted + 1.0)
            }
        }
        CfrVariant::DcfrPlus { .. }
        | CfrVariant::DcfrSchedule { .. }
        | CfrVariant::PdcfrPlus { .. } => {
            if iteration <= 1 {
                0.0
            } else {
                let alpha = dcfr_alpha(variant, iteration);
                let weighted = ((iteration - 1) as f32).powf(alpha);
                weighted / (weighted + 1.0)
            }
        }
    }
}

fn average_strategy_discount(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::Dcfr { gamma, .. } if iteration > 1 => {
            (((iteration - 1) as f32) / iteration as f32).powf(gamma)
        }
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

pub(super) fn average_strategy_delay() -> usize {
    std::env::var(AVERAGE_STRATEGY_DELAY_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

pub(super) fn average_strategy_power() -> f32 {
    std::env::var(AVERAGE_STRATEGY_POWER_ENV)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

fn average_strategy_weight_multiplier(iteration: usize) -> f32 {
    let delay = average_strategy_delay();
    if delay == 0 && average_strategy_power() == 0.0 {
        return 1.0;
    }
    if iteration <= delay {
        return 0.0;
    }
    ((iteration - delay) as f32).powf(average_strategy_power())
}

fn dcfr_alpha(variant: CfrVariant, iteration: usize) -> f32 {
    match variant {
        CfrVariant::Dcfr { alpha, .. } => alpha,
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
        CfrVariant::Dcfr { gamma, .. } => gamma,
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
    fn dcfr_plus_regret_discount_matches_weighted_cfr_plus_formula() {
        let variant = CfrVariant::DcfrPlus {
            alpha: 2.0,
            gamma: 4.0,
        };

        assert_eq!(regret_discount(variant, 1, 1.0), 0.0);
        assert!((regret_discount(variant, 2, 1.0) - 0.5).abs() < 1e-6);
        assert!((regret_discount(variant, 3, 1.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn dcfr_discounts_negative_regrets_with_beta_without_clipping() {
        let mut state = DenseCfrState::new(DenseCfrConfig {
            infosets: 1,
            actions: 2,
            variant: CfrVariant::Dcfr {
                alpha: 1.5,
                beta: 0.5,
                gamma: 2.0,
            },
        });
        state.regrets.copy_from_slice(&[-4.0, 2.0]);
        state.update_infoset(0, &[0.0, 0.0], 0.0, 0.0, 2);

        assert!(state.regrets()[0] < 0.0);
        assert!(state.regrets()[1] > 0.0);
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
    fn compact_state_matches_dense_state_without_padding_slots() {
        let dense_config = DenseCfrConfig {
            infosets: 3,
            actions: 4,
            variant: CfrVariant::PdcfrPlus {
                alpha: 2.5,
                gamma: 8.0,
                eta_start: 1.0,
                eta_end: 0.25,
                eta_horizon: 8,
            },
        };
        let legal = vec![
            true, true, false, false, true, true, true, false, true, false, true, true,
        ];
        let mut dense = DenseCfrState::new_with_legal_actions(dense_config, legal);
        let mut compact = CompactCfrState::new(CompactCfrConfig {
            action_offsets: vec![0, 2, 5, 8],
            variant: dense.variant,
        });

        let updates = [
            (0, [3.0, -1.0, 0.0, 0.0], [3.0, -1.0, 0.0]),
            (1, [0.25, 1.0, -2.0, 0.0], [0.25, 1.0, -2.0]),
            (2, [-1.0, 0.0, 0.5, 2.0], [-1.0, 0.5, 2.0]),
        ];

        for iteration in 1..=4 {
            for (infoset, dense_values, compact_values) in updates {
                let reach_weight = 0.25 + infoset as f32 * 0.5;
                let strategy_weight = 1.0 + iteration as f32 * 0.25;
                dense.update_infoset(
                    infoset,
                    &dense_values,
                    reach_weight,
                    strategy_weight,
                    iteration,
                );
                compact.update_infoset(
                    infoset,
                    &compact_values,
                    reach_weight,
                    strategy_weight,
                    iteration,
                );
            }
        }

        for infoset in 0..3 {
            let mut dense_strategy = [0.0; 4];
            let mut compact_strategy = [0.0; 3];
            dense.strategy_for(infoset, &mut dense_strategy);
            compact.strategy_for(infoset, &mut compact_strategy);
            let mut dense_average = [0.0; 4];
            let mut compact_average = [0.0; 3];
            dense.average_strategy_for(infoset, &mut dense_average);
            compact.average_strategy_for(infoset, &mut compact_average);

            let compact_range =
                compact.action_offsets()[infoset]..compact.action_offsets()[infoset + 1];
            let mut compact_action = 0;
            for action in 0..4 {
                if dense.legal_actions[dense.offset(infoset) + action] {
                    assert!(
                        (dense_strategy[action] - compact_strategy[compact_action]).abs() < 1e-6
                    );
                    assert!((dense_average[action] - compact_average[compact_action]).abs() < 1e-6);
                    assert!(
                        (dense.regrets()[dense.offset(infoset) + action]
                            - compact.regrets()[compact_range.start + compact_action])
                            .abs()
                            < 1e-6
                    );
                    assert!(
                        (dense.prediction()[dense.offset(infoset) + action]
                            - compact.prediction()[compact_range.start + compact_action])
                            .abs()
                            < 1e-6
                    );
                    assert!(
                        (dense.strategy_sum()[dense.offset(infoset) + action]
                            - compact.strategy_sum()[compact_range.start + compact_action])
                            .abs()
                            < 1e-6
                    );
                    compact_action += 1;
                }
            }
            assert_eq!(compact_action, compact.action_count(infoset));
        }
    }

    #[test]
    fn compact_private_state_matches_dense_private_infosets_without_padding_slots() {
        let public_infosets = 2;
        let combos = 3;
        let max_actions = 3;
        let variant = CfrVariant::dcfr_plus_default();
        let public_action_counts = [2, 3];
        let mut legal = vec![false; public_infosets * combos * max_actions];
        for public_infoset in 0..public_infosets {
            for combo in 0..combos {
                let private_infoset = public_infoset * combos + combo;
                for action in 0..public_action_counts[public_infoset] {
                    legal[private_infoset * max_actions + action] = true;
                }
            }
        }
        let mut dense = DenseCfrState::new_with_legal_actions(
            DenseCfrConfig {
                infosets: public_infosets * combos,
                actions: max_actions,
                variant,
            },
            legal,
        );
        let mut compact = CompactPrivateCfrState::new(CompactPrivateCfrConfig {
            public_action_offsets: vec![0, 2, 5],
            combos,
            variant,
        });

        for iteration in 1..=5 {
            for public_infoset in 0..public_infosets {
                for combo in 0..combos {
                    let private_infoset = public_infoset * combos + combo;
                    let dense_values = [
                        1.0 + public_infoset as f32,
                        -0.5 + combo as f32 * 0.25,
                        0.75,
                    ];
                    let compact_values = &dense_values[..public_action_counts[public_infoset]];
                    let reach_weight = 0.5 + combo as f32 * 0.25;
                    let strategy_weight = 0.25 + iteration as f32 * 0.5;
                    dense.update_infoset(
                        private_infoset,
                        &dense_values,
                        reach_weight,
                        strategy_weight,
                        iteration,
                    );
                    compact.update_infoset(
                        public_infoset,
                        combo,
                        compact_values,
                        reach_weight,
                        strategy_weight,
                        iteration,
                    );
                }
            }
        }

        for public_infoset in 0..public_infosets {
            for combo in 0..combos {
                let private_infoset = public_infoset * combos + combo;
                let mut dense_strategy = [0.0; 3];
                let mut compact_strategy = [0.0; 3];
                dense.strategy_for(private_infoset, &mut dense_strategy);
                compact.strategy_for(public_infoset, combo, &mut compact_strategy);
                let mut dense_average = [0.0; 3];
                let mut compact_average = [0.0; 3];
                dense.average_strategy_for(private_infoset, &mut dense_average);
                compact.average_strategy_for(public_infoset, combo, &mut compact_average);

                for action in 0..public_action_counts[public_infoset] {
                    let dense_slot = private_infoset * max_actions + action;
                    let compact_slot = compact.slot(public_infoset, combo, action);
                    assert!((dense_strategy[action] - compact_strategy[action]).abs() < 1e-6);
                    assert!((dense_average[action] - compact_average[action]).abs() < 1e-6);
                    assert!(
                        (dense.regrets()[dense_slot] - compact.regrets()[compact_slot]).abs()
                            < 1e-6
                    );
                    assert!(
                        (dense.strategy_sum()[dense_slot] - compact.strategy_sum()[compact_slot])
                            .abs()
                            < 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn compact_private_state_can_be_built_from_dense_private_state() {
        let public_infosets = 2;
        let combos = 3;
        let max_actions = 3;
        let variant = CfrVariant::CfrPlus;
        let mut legal = vec![false; public_infosets * combos * max_actions];
        for public_infoset in 0..public_infosets {
            let action_count = if public_infoset == 0 { 2 } else { 3 };
            for combo in 0..combos {
                let private_infoset = public_infoset * combos + combo;
                for action in 0..action_count {
                    legal[private_infoset * max_actions + action] = true;
                }
            }
        }
        let mut dense = DenseCfrState::new_with_legal_actions(
            DenseCfrConfig {
                infosets: public_infosets * combos,
                actions: max_actions,
                variant,
            },
            legal,
        );
        for iteration in 1..=3 {
            for public_infoset in 0..public_infosets {
                for combo in 0..combos {
                    let private_infoset = public_infoset * combos + combo;
                    dense.update_infoset(
                        private_infoset,
                        &[combo as f32 + 1.0, public_infoset as f32 - 0.5, 0.25],
                        1.0,
                        1.0,
                        iteration,
                    );
                }
            }
        }

        let compact = CompactPrivateCfrState::from_dense_private_state(
            &dense,
            CompactPrivateCfrConfig {
                public_action_offsets: vec![0, 2, 5],
                combos,
                variant,
            },
        );

        for public_infoset in 0..public_infosets {
            let action_count = compact.action_count(public_infoset);
            for combo in 0..combos {
                let private_infoset = public_infoset * combos + combo;
                let mut dense_strategy = [0.0; 3];
                let mut compact_strategy = [0.0; 3];
                dense.average_strategy_for(private_infoset, &mut dense_strategy);
                compact.average_strategy_for(public_infoset, combo, &mut compact_strategy);
                for action in 0..action_count {
                    let dense_slot = private_infoset * max_actions + action;
                    let compact_slot = compact.slot(public_infoset, combo, action);
                    assert_eq!(dense.regrets()[dense_slot], compact.regrets()[compact_slot]);
                    assert_eq!(
                        dense.strategy_sum()[dense_slot],
                        compact.strategy_sum()[compact_slot]
                    );
                    assert!((dense_strategy[action] - compact_strategy[action]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn compact_private_config_chunks_by_public_action_bytes() {
        let config = CompactPrivateCfrConfig {
            public_action_offsets: vec![0, 2, 5, 7, 11],
            combos: 10,
            variant: CfrVariant::CfrPlus,
        };
        let chunks = config.chunk_by_action_bytes(5 * 10 * std::mem::size_of::<f32>());

        assert_eq!(
            chunks,
            vec![
                CompactPrivateCfrChunk {
                    public_start: 0,
                    public_end: 2,
                    public_action_start: 0,
                    public_action_end: 5,
                    action_slots: 50,
                },
                CompactPrivateCfrChunk {
                    public_start: 2,
                    public_end: 3,
                    public_action_start: 5,
                    public_action_end: 7,
                    action_slots: 20,
                },
                CompactPrivateCfrChunk {
                    public_start: 3,
                    public_end: 4,
                    public_action_start: 7,
                    public_action_end: 11,
                    action_slots: 40,
                },
            ]
        );
    }

    #[test]
    fn batched_private_state_uses_batch_major_offsets() {
        let config = BatchedPrivateCfrConfig {
            batches: 3,
            public_infosets: 2,
            combos: 5,
            actions: 4,
            variant: CfrVariant::CfrPlus,
        };
        let legal = vec![true, true, false, false, true, false, true, false];
        let state = BatchedPrivateCfrState::new(config.clone(), &legal);

        assert_eq!(config.private_infosets_per_batch(), 10);
        assert_eq!(config.private_infosets(), 30);
        assert_eq!(config.action_slots_per_batch(), 40);
        assert_eq!(config.action_slots(), 120);
        assert_eq!(config.offset(2, 1, 4, 3), 119);
        assert_eq!(state.regrets().len(), 120);
        assert_eq!(state.prediction().len(), 120);
        assert_eq!(state.strategy_sum().len(), 120);
        assert!(state.legal_actions()[config.offset(1, 0, 3, 1)]);
        assert!(!state.legal_actions()[config.offset(1, 0, 3, 2)]);
        assert!(state.legal_actions()[config.offset(2, 1, 4, 2)]);
        assert!(!state.legal_actions()[config.offset(2, 1, 4, 3)]);
    }

    #[test]
    fn batched_private_state_extracts_and_overwrites_dense_batch() {
        let config = BatchedPrivateCfrConfig {
            batches: 2,
            public_infosets: 3,
            combos: 4,
            actions: 3,
            variant: CfrVariant::dcfr_plus_default(),
        };
        let legal = vec![true, false, true, true, true, false, false, true, true];
        let mut state = BatchedPrivateCfrState::new(config.clone(), &legal);
        let mut replacement = DenseCfrState::new_with_legal_actions(
            DenseCfrConfig {
                infosets: config.private_infosets_per_batch(),
                actions: config.actions,
                variant: config.variant,
            },
            state.dense_state_for_batch(1).legal_actions().to_vec(),
        );
        for (index, value) in replacement.regrets.iter_mut().enumerate() {
            *value = index as f32 * 0.25;
        }
        for (index, value) in replacement.prediction.iter_mut().enumerate() {
            *value = -(index as f32) * 0.125;
        }
        for (index, value) in replacement.strategy_sum.iter_mut().enumerate() {
            *value = 1.0 + index as f32 * 0.5;
        }

        state.overwrite_batch_from_dense(1, &replacement);
        let batch0 = state.dense_state_for_batch(0);
        let batch1 = state.dense_state_for_batch(1);

        assert!(batch0.regrets().iter().all(|value| *value == 0.0));
        assert_eq!(batch1.regrets(), replacement.regrets());
        assert_eq!(batch1.prediction(), replacement.prediction());
        assert_eq!(batch1.strategy_sum(), replacement.strategy_sum());
        assert_eq!(batch1.legal_actions(), replacement.legal_actions());
    }

    #[test]
    fn batched_private_average_profile_preserves_batches() {
        let config = BatchedPrivateCfrConfig {
            batches: 2,
            public_infosets: 1,
            combos: 2,
            actions: 3,
            variant: CfrVariant::CfrPlus,
        };
        let legal = vec![true, true, false];
        let mut state = BatchedPrivateCfrState::new(config.clone(), &legal);
        state.strategy_sum[config.offset(0, 0, 0, 0)] = 1.0;
        state.strategy_sum[config.offset(0, 0, 0, 1)] = 3.0;
        state.strategy_sum[config.offset(1, 0, 1, 0)] = 2.0;
        state.strategy_sum[config.offset(1, 0, 1, 1)] = 2.0;

        let profile = state.average_strategy_profile_state();
        let batch0 = profile.dense_state_for_batch(0);
        let batch1 = profile.dense_state_for_batch(1);

        assert_eq!(batch0.regrets()[0], 0.25);
        assert_eq!(batch0.regrets()[1], 0.75);
        assert_eq!(batch0.regrets()[2], 0.0);
        assert_eq!(batch1.regrets()[3], 0.5);
        assert_eq!(batch1.regrets()[4], 0.5);
        assert_eq!(batch1.regrets()[5], 0.0);
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
