use crate::plan::{CfrStorageConfig, live_private_combos, public_board_multiplicity};
use crate::tree::{PublicNodeKind, PublicTree, Street};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfrVariant {
    CfrPlus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionSlotLayout {
    pub records: Vec<ActionSlotRecord>,
    pub total_action_slots: u128,
    pub regret_bytes: u128,
    pub strategy_sum_bytes: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionSlotRecord {
    pub node_id: usize,
    pub street: Street,
    pub actions: usize,
    pub public_boards: u128,
    pub private_combos: u128,
    pub start: u128,
    pub len: u128,
}

#[derive(Debug)]
pub struct CfrPlusState {
    pub layout: ActionSlotLayout,
    pub regret: Vec<f32>,
    pub strategy_sum: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfrPrefixUpdateSummary {
    pub requested_slots: usize,
    pub updated_slots: usize,
    pub strategy_sum_delta: f64,
    pub regret_checksum: f64,
    pub strategy_sum_checksum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotChunk {
    pub index: u128,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfrIterationDryRun {
    pub iteration: u32,
    pub records_visited: usize,
    pub action_slots_visited: u128,
    pub infosets_visited: u128,
    pub strategy_sum_writes: u128,
    pub strategy_sum_delta: f64,
    pub regret_reads: u128,
    pub regret_writes: u128,
    pub checksum: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfrStateAllocError {
    TooManySlots { slots: u128 },
}

impl ActionSlotLayout {
    pub fn storage_bytes(&self) -> u128 {
        self.total_action_slots * (self.regret_bytes + self.strategy_sum_bytes)
    }

    pub fn storage_gib(&self) -> f64 {
        self.storage_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn flop_slots(&self) -> u128 {
        self.street_slots(Street::Flop)
    }

    pub fn turn_slots(&self) -> u128 {
        self.street_slots(Street::Turn)
    }

    pub fn river_slots(&self) -> u128 {
        self.street_slots(Street::River)
    }

    fn street_slots(&self, street: Street) -> u128 {
        self.records
            .iter()
            .filter(|record| record.street == street)
            .map(|record| record.len)
            .sum()
    }

    pub fn slot_chunk(&self, index: u128, target_bytes: u128) -> Option<SlotChunk> {
        let slot_bytes = self.regret_bytes + self.strategy_sum_bytes;
        let slots_per_chunk = (target_bytes / slot_bytes).max(1);
        let start = index.checked_mul(slots_per_chunk)?;
        if start >= self.total_action_slots {
            return None;
        }
        let end = (start + slots_per_chunk).min(self.total_action_slots);
        Some(SlotChunk {
            index,
            start: usize::try_from(start).ok()?,
            end: usize::try_from(end).ok()?,
        })
    }
}

impl CfrPlusState {
    pub fn allocate(layout: ActionSlotLayout) -> Result<Self, CfrStateAllocError> {
        let slots = usize::try_from(layout.total_action_slots).map_err(|_| {
            CfrStateAllocError::TooManySlots {
                slots: layout.total_action_slots,
            }
        })?;
        Ok(Self {
            layout,
            regret: vec![0.0; slots],
            strategy_sum: vec![0.0; slots],
        })
    }

    pub fn storage_gib(&self) -> f64 {
        (self.regret.len() + self.strategy_sum.len()) as f64 * std::mem::size_of::<f32>() as f64
            / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn update_prefix_slots(&mut self, requested_slots: usize) -> CfrPrefixUpdateSummary {
        self.update_slot_range(0, requested_slots.min(self.regret.len()), requested_slots)
    }

    pub fn update_slot_chunk(
        &mut self,
        chunk: SlotChunk,
        requested_slots: usize,
    ) -> CfrPrefixUpdateSummary {
        self.update_slot_range(chunk.start, chunk.end, requested_slots)
    }

    fn update_slot_range(
        &mut self,
        range_start: usize,
        range_end: usize,
        requested_slots: usize,
    ) -> CfrPrefixUpdateSummary {
        let mut updated_slots = 0usize;
        let mut strategy_sum_delta = 0.0f64;
        let mut regret_checksum = 0.0f64;
        let mut strategy_sum_checksum = 0.0f64;

        for record in &self.layout.records {
            let record_start = record.start as usize;
            let record_end = (record.start + record.len) as usize;
            if record_end <= range_start {
                continue;
            }
            if record_start >= range_end {
                break;
            }
            let start = record_start.max(range_start);
            let end = record_end.min(range_end);
            let actions = record.actions;
            let aligned_start =
                record_start + ceil_div_usize(start - record_start, actions) * actions;
            let full_infosets = end.saturating_sub(aligned_start) / actions;
            for infoset in 0..full_infosets {
                let slot = aligned_start + infoset * actions;
                let regrets = &mut self.regret[slot..slot + actions];
                let strategy_sum = &mut self.strategy_sum[slot..slot + actions];
                let mut positive_sum = 0.0f32;
                for regret in regrets.iter_mut() {
                    if *regret < 0.0 {
                        *regret = 0.0;
                    }
                    positive_sum += *regret;
                }
                if positive_sum > 0.0 {
                    for (regret, sum) in regrets.iter().zip(strategy_sum.iter_mut()) {
                        let strategy = *regret / positive_sum;
                        *sum += strategy;
                        strategy_sum_delta += strategy as f64;
                    }
                } else {
                    let strategy = 1.0f32 / actions as f32;
                    for sum in strategy_sum.iter_mut() {
                        *sum += strategy;
                        strategy_sum_delta += strategy as f64;
                    }
                }
                updated_slots += actions;
            }
        }

        let checksum_end = range_start + updated_slots;
        for index in range_start..checksum_end {
            regret_checksum += self.regret[index] as f64 * (index as f64 + 1.0);
            strategy_sum_checksum += self.strategy_sum[index] as f64 * (index as f64 + 1.0);
        }

        CfrPrefixUpdateSummary {
            requested_slots,
            updated_slots,
            strategy_sum_delta,
            regret_checksum,
            strategy_sum_checksum,
        }
    }
}

fn ceil_div_usize(value: usize, divisor: usize) -> usize {
    if value == 0 {
        0
    } else {
        (value - 1) / divisor + 1
    }
}

pub fn dry_run_cfr_plus_iteration(layout: &ActionSlotLayout, iteration: u32) -> CfrIterationDryRun {
    let mut infosets_visited = 0u128;
    let mut action_slots_visited = 0u128;
    let mut checksum = 0.0f64;
    for record in &layout.records {
        let infosets = record.public_boards * record.private_combos;
        let slots = infosets * record.actions as u128;
        debug_assert_eq!(slots, record.len);
        infosets_visited += infosets;
        action_slots_visited += slots;
        checksum += record.start as f64 * 0.000_000_001;
        checksum += slots as f64 / record.actions as f64;
    }
    CfrIterationDryRun {
        iteration,
        records_visited: layout.records.len(),
        action_slots_visited,
        infosets_visited,
        strategy_sum_writes: action_slots_visited,
        strategy_sum_delta: infosets_visited as f64,
        regret_reads: action_slots_visited,
        regret_writes: action_slots_visited,
        checksum,
    }
}

pub fn build_action_slot_layout(tree: &PublicTree, config: CfrStorageConfig) -> ActionSlotLayout {
    let mut records = Vec::new();
    let mut cursor = 0u128;
    for node in &tree.nodes {
        let PublicNodeKind::Decision { actions, .. } = &node.kind else {
            continue;
        };
        let public_boards = public_board_multiplicity(node.state.street);
        let private_combos = live_private_combos(node.state.street);
        let len = public_boards * private_combos * actions.len() as u128;
        records.push(ActionSlotRecord {
            node_id: node.id,
            street: node.state.street,
            actions: actions.len(),
            public_boards,
            private_combos,
            start: cursor,
            len,
        });
        cursor += len;
    }
    ActionSlotLayout {
        records,
        total_action_slots: cursor,
        regret_bytes: config.regret_bytes,
        strategy_sum_bytes: config.strategy_sum_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, Player, RangeSpec, Spot, TreeBuilder, TreeTemplate};
    use std::str::FromStr;

    #[test]
    fn default_layout_matches_planner_slots() {
        let tree = TreeBuilder::new(TreeTemplate::conservative_default())
            .unwrap()
            .build(Spot {
                board: Board::from_str("As7h2c").unwrap(),
                pot: 650,
                effective_stack: 9700,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let config = CfrStorageConfig::default();
        let layout = build_action_slot_layout(&tree, config);
        assert_eq!(layout.records.len(), tree.stats().decisions);
        assert_eq!(layout.flop_slots(), 39_984);
        assert_eq!(layout.turn_slots(), 14_481_264);
        assert_eq!(layout.river_slots(), 3_218_820_192);
        assert_eq!(layout.total_action_slots, 3_233_341_440);
        assert_eq!(layout.records.first().unwrap().start, 0);
        let last = layout.records.last().unwrap();
        assert_eq!(last.start + last.len, layout.total_action_slots);
    }

    #[test]
    fn dry_run_iteration_visits_every_action_slot_once() {
        let tree = TreeBuilder::new(TreeTemplate::conservative_default())
            .unwrap()
            .build(Spot {
                board: Board::from_str("As7h2c").unwrap(),
                pot: 650,
                effective_stack: 9700,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let layout = build_action_slot_layout(&tree, CfrStorageConfig::default());
        let dry_run = dry_run_cfr_plus_iteration(&layout, 1);
        assert_eq!(dry_run.records_visited, layout.records.len());
        assert_eq!(dry_run.action_slots_visited, layout.total_action_slots);
        assert_eq!(dry_run.strategy_sum_writes, layout.total_action_slots);
        assert_eq!(dry_run.regret_reads, layout.total_action_slots);
        assert_eq!(dry_run.regret_writes, layout.total_action_slots);
        assert!(dry_run.infosets_visited < dry_run.action_slots_visited);
    }

    #[test]
    fn cfr_plus_prefix_update_writes_uniform_initial_strategy() {
        let tree = TreeBuilder::new(TreeTemplate::conservative_default())
            .unwrap()
            .build(Spot {
                board: Board::from_str("As7h2c").unwrap(),
                pot: 650,
                effective_stack: 9700,
                oop_range: RangeSpec::full_deck_uniform(),
                ip_range: RangeSpec::full_deck_uniform(),
                first_player: Player::Oop,
            })
            .unwrap();
        let layout = build_action_slot_layout(&tree, CfrStorageConfig::default());
        let first_actions = layout.records[0].actions;
        let mut state = CfrPlusState::allocate(ActionSlotLayout {
            records: layout.records[..1].to_vec(),
            total_action_slots: layout.records[0].len,
            regret_bytes: layout.regret_bytes,
            strategy_sum_bytes: layout.strategy_sum_bytes,
        })
        .unwrap();
        let summary = state.update_prefix_slots(first_actions * 4);
        assert_eq!(summary.updated_slots, first_actions * 4);
        assert!((summary.strategy_sum_delta - 4.0).abs() < 1e-6);
        for chunk in state.strategy_sum[..summary.updated_slots].chunks(first_actions) {
            let sum = chunk.iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
