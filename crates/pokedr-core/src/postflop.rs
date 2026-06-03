use std::cmp::Ordering;

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

        let all_in = request.stack;
        insert_or_replace_near(
            &mut sizings,
            AggressiveSizing {
                amount: all_in,
                source: SizingSource::Forced,
            },
            self.config.merge_log_ratio,
        );

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

    fn builder() -> ActionSetBuilder {
        ActionSetBuilder::new(ActionSetConfig::default())
    }

    #[test]
    fn check_spot_includes_standard_bets_and_all_in() {
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
            actions
                .iter()
                .any(|candidate| candidate.action == PlayerAction::AllIn { amount: 500 })
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
    fn aggressive_action_cap_preserves_observed_and_all_in() {
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
            aggressive
                .iter()
                .any(|candidate| candidate.action == PlayerAction::AllIn { amount: 1000 })
        );
    }
}
