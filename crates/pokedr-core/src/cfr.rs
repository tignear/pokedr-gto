use crate::plan::{CfrStorageConfig, live_private_combos, public_board_multiplicity};
use crate::tree::{ActionKind, Player, PublicNodeKind, PublicState, PublicTree, Street};
use rayon::prelude::*;
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfrStateIterationSummary {
    pub chunks: u128,
    pub updated_slots: usize,
    pub strategy_sum_delta: f64,
    pub regret_checksum: f64,
    pub strategy_sum_checksum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordChunk {
    record_start: usize,
    record_end: usize,
    slot_start: usize,
    slot_end: usize,
}

struct CfrUpdateJob<'a> {
    records: Vec<ActionSlotRecord>,
    slot_start: usize,
    regret: &'a mut [f32],
    strategy_sum: &'a mut [f32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicStateDuplicateReport {
    pub decision_nodes: usize,
    pub exact_unique: usize,
    pub exact_duplicates: usize,
    pub boardless_unique: usize,
    pub boardless_duplicates: usize,
    pub action_compatible_unique: usize,
    pub action_compatible_duplicates: usize,
    pub history_exact_unique: usize,
    pub history_exact_duplicates: usize,
    pub history_boardless_unique: usize,
    pub history_boardless_duplicates: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CfrStorageScenarioReport {
    pub total_slots: u128,
    pub regret_f32_strategy_f32_gib: f64,
    pub regret_f32_strategy_u16_gib: f64,
    pub regret_f32_only_gib: f64,
    pub river_slots: u128,
    pub river_ordered_board_slots: u128,
    pub river_unordered_board_slots: u128,
    pub river_unordered_regret_f32_strategy_f32_gib: f64,
    pub river_unordered_regret_f32_strategy_u16_gib: f64,
    pub river_unordered_regret_f32_only_gib: f64,
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

    pub fn apply_regret_matching_iteration(
        &mut self,
        target_chunk_bytes: u128,
    ) -> CfrStateIterationSummary {
        self.apply_regret_matching_iteration_parallel(target_chunk_bytes, 1)
    }

    pub fn apply_regret_matching_iteration_parallel(
        &mut self,
        target_chunk_bytes: u128,
        requested_threads: usize,
    ) -> CfrStateIterationSummary {
        let record_chunks = self.record_chunks(target_chunk_bytes);
        let threads = if requested_threads == 0 {
            rayon::current_num_threads()
        } else {
            requested_threads
        }
        .max(1)
        .min(record_chunks.len().max(1));

        if threads == 1 {
            let mut summary = CfrStateIterationSummary {
                chunks: record_chunks.len() as u128,
                ..CfrStateIterationSummary::default()
            };
            for record in self.layout.records.clone() {
                summary += self.update_whole_record(record);
            }
            return summary;
        }

        let work = merge_record_chunks_for_threads(&record_chunks, threads);
        let records = work
            .iter()
            .map(|chunk| self.layout.records[chunk.record_start..chunk.record_end].to_vec())
            .collect::<Vec<_>>();

        let mut regret_tail = self.regret.as_mut_slice();
        let mut strategy_tail = self.strategy_sum.as_mut_slice();
        let mut previous_slot_end = 0usize;
        let mut jobs = Vec::with_capacity(work.len());
        for (chunk, records) in work.iter().copied().zip(records) {
            let gap = chunk.slot_start - previous_slot_end;
            let (_, rest) = regret_tail.split_at_mut(gap);
            regret_tail = rest;
            let (_, rest) = strategy_tail.split_at_mut(gap);
            strategy_tail = rest;

            let len = chunk.slot_end - chunk.slot_start;
            let (regret_chunk, rest) = regret_tail.split_at_mut(len);
            regret_tail = rest;
            let (strategy_chunk, rest) = strategy_tail.split_at_mut(len);
            strategy_tail = rest;
            previous_slot_end = chunk.slot_end;
            jobs.push(CfrUpdateJob {
                records,
                slot_start: chunk.slot_start,
                regret: regret_chunk,
                strategy_sum: strategy_chunk,
            });
        }
        let partials = update_cfr_jobs(jobs);

        let mut summary = CfrStateIterationSummary {
            chunks: record_chunks.len() as u128,
            ..CfrStateIterationSummary::default()
        };
        for partial in partials {
            summary += partial;
        }
        summary
    }

    fn record_chunks(&self, target_chunk_bytes: u128) -> Vec<RecordChunk> {
        let slot_bytes = self.layout.regret_bytes + self.layout.strategy_sum_bytes;
        let target_chunk_slots = (target_chunk_bytes / slot_bytes).max(1);
        let mut chunks = Vec::new();
        let mut record_start = 0usize;
        let mut slot_start = 0usize;
        let mut current_chunk_slots = 0u128;

        for (record_index, record) in self.layout.records.iter().enumerate() {
            if current_chunk_slots > 0 && current_chunk_slots + record.len > target_chunk_slots {
                let previous = self.layout.records[record_index - 1];
                chunks.push(RecordChunk {
                    record_start,
                    record_end: record_index,
                    slot_start,
                    slot_end: (previous.start + previous.len) as usize,
                });
                record_start = record_index;
                slot_start = record.start as usize;
                current_chunk_slots = 0;
            }
            current_chunk_slots += record.len;
        }
        if record_start < self.layout.records.len() {
            let last = *self
                .layout
                .records
                .last()
                .expect("records must not be empty");
            chunks.push(RecordChunk {
                record_start,
                record_end: self.layout.records.len(),
                slot_start,
                slot_end: (last.start + last.len) as usize,
            });
        }
        chunks
    }

    fn update_whole_record(&mut self, record: ActionSlotRecord) -> CfrPrefixUpdateSummary {
        self.update_slot_range(
            record.start as usize,
            (record.start + record.len) as usize,
            record.len as usize,
        )
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

impl Default for CfrStateIterationSummary {
    fn default() -> Self {
        Self {
            chunks: 0,
            updated_slots: 0,
            strategy_sum_delta: 0.0,
            regret_checksum: 0.0,
            strategy_sum_checksum: 0.0,
        }
    }
}

impl std::ops::AddAssign<CfrPrefixUpdateSummary> for CfrStateIterationSummary {
    fn add_assign(&mut self, rhs: CfrPrefixUpdateSummary) {
        self.updated_slots += rhs.updated_slots;
        self.strategy_sum_delta += rhs.strategy_sum_delta;
        self.regret_checksum += rhs.regret_checksum;
        self.strategy_sum_checksum += rhs.strategy_sum_checksum;
    }
}

impl std::ops::AddAssign<CfrStateIterationSummary> for CfrStateIterationSummary {
    fn add_assign(&mut self, rhs: CfrStateIterationSummary) {
        self.chunks += rhs.chunks;
        self.updated_slots += rhs.updated_slots;
        self.strategy_sum_delta += rhs.strategy_sum_delta;
        self.regret_checksum += rhs.regret_checksum;
        self.strategy_sum_checksum += rhs.strategy_sum_checksum;
    }
}

fn merge_record_chunks_for_threads(chunks: &[RecordChunk], threads: usize) -> Vec<RecordChunk> {
    let total_slots = chunks
        .iter()
        .map(|chunk| chunk.slot_end - chunk.slot_start)
        .sum::<usize>();
    let target_slots = total_slots.div_ceil(threads);
    let mut merged = Vec::new();
    let mut start = 0usize;
    while start < chunks.len() {
        let mut end = start + 1;
        let slot_start = chunks[start].slot_start;
        let mut slot_end = chunks[start].slot_end;
        while end < chunks.len()
            && slot_end - slot_start < target_slots
            && merged.len() + 1 < threads
        {
            slot_end = chunks[end].slot_end;
            end += 1;
        }
        merged.push(RecordChunk {
            record_start: chunks[start].record_start,
            record_end: chunks[end - 1].record_end,
            slot_start,
            slot_end,
        });
        start = end;
    }
    merged
}

fn update_cfr_jobs(jobs: Vec<CfrUpdateJob<'_>>) -> Vec<CfrStateIterationSummary> {
    jobs.into_par_iter()
        .map(|job| {
            update_records_in_slices(&job.records, job.slot_start, job.regret, job.strategy_sum)
        })
        .collect()
}

fn update_records_in_slices(
    records: &[ActionSlotRecord],
    base_slot: usize,
    regret: &mut [f32],
    strategy_sum: &mut [f32],
) -> CfrStateIterationSummary {
    let mut summary = CfrStateIterationSummary::default();
    for record in records {
        let local_start = record.start as usize - base_slot;
        let local_end = local_start + record.len as usize;
        let record_regret = &mut regret[local_start..local_end];
        let record_strategy_sum = &mut strategy_sum[local_start..local_end];
        let actions = record.actions;
        for (infoset, (regrets, sums)) in record_regret
            .chunks_exact_mut(actions)
            .zip(record_strategy_sum.chunks_exact_mut(actions))
            .enumerate()
        {
            let mut positive_sum = 0.0f32;
            for regret in regrets.iter_mut() {
                if *regret < 0.0 {
                    *regret = 0.0;
                }
                positive_sum += *regret;
            }
            if positive_sum > 0.0 {
                for (regret, sum) in regrets.iter().zip(sums.iter_mut()) {
                    let strategy = *regret / positive_sum;
                    *sum += strategy;
                    summary.strategy_sum_delta += strategy as f64;
                }
            } else {
                let strategy = 1.0f32 / actions as f32;
                for sum in sums.iter_mut() {
                    *sum += strategy;
                    summary.strategy_sum_delta += strategy as f64;
                }
            }

            let global_slot = record.start as usize + infoset * actions;
            for action in 0..actions {
                let index = global_slot + action;
                summary.regret_checksum += regrets[action] as f64 * (index as f64 + 1.0);
                summary.strategy_sum_checksum += sums[action] as f64 * (index as f64 + 1.0);
            }
            summary.updated_slots += actions;
        }
    }
    summary
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

pub fn analyze_cfr_storage_scenarios(layout: &ActionSlotLayout) -> CfrStorageScenarioReport {
    let river_slots = layout.river_slots();
    let river_unordered_board_slots = layout
        .records
        .iter()
        .filter(|record| record.street == Street::River)
        .map(|record| {
            let ordered_boards = record.public_boards.max(1);
            let unordered_boards = ordered_boards / 2;
            record.len / ordered_boards * unordered_boards
        })
        .sum::<u128>();
    let non_river_slots = layout.total_action_slots - river_slots;
    let unordered_total_slots = non_river_slots + river_unordered_board_slots;
    CfrStorageScenarioReport {
        total_slots: layout.total_action_slots,
        regret_f32_strategy_f32_gib: slots_to_gib(layout.total_action_slots, 4, 4),
        regret_f32_strategy_u16_gib: slots_to_gib(layout.total_action_slots, 4, 2),
        regret_f32_only_gib: slots_to_gib(layout.total_action_slots, 4, 0),
        river_slots,
        river_ordered_board_slots: river_slots,
        river_unordered_board_slots,
        river_unordered_regret_f32_strategy_f32_gib: slots_to_gib(unordered_total_slots, 4, 4),
        river_unordered_regret_f32_strategy_u16_gib: slots_to_gib(unordered_total_slots, 4, 2),
        river_unordered_regret_f32_only_gib: slots_to_gib(unordered_total_slots, 4, 0),
    }
}

fn slots_to_gib(slots: u128, regret_bytes: u128, strategy_sum_bytes: u128) -> f64 {
    slots as f64 * (regret_bytes + strategy_sum_bytes) as f64 / (1024.0 * 1024.0 * 1024.0)
}

pub fn analyze_public_state_duplicates(tree: &PublicTree) -> PublicStateDuplicateReport {
    let mut exact = BTreeMap::<StateKey, usize>::new();
    let mut boardless = BTreeMap::<StateKey, usize>::new();
    let mut action_compatible = BTreeMap::<StateActionKey, usize>::new();
    let mut history_exact = BTreeMap::<HistoryStateActionKey, usize>::new();
    let mut history_boardless = BTreeMap::<HistoryStateActionKey, usize>::new();
    let mut decision_nodes = 0usize;

    for node in &tree.nodes {
        let PublicNodeKind::Decision { actions, .. } = &node.kind else {
            continue;
        };
        decision_nodes += 1;
        *exact
            .entry(StateKey::from_state(&node.state, true))
            .or_default() += 1;
        *boardless
            .entry(StateKey::from_state(&node.state, false))
            .or_default() += 1;
        *action_compatible
            .entry(StateActionKey {
                state: StateKey::from_state(&node.state, false),
                actions: actions.clone(),
            })
            .or_default() += 1;
    }
    if !tree.nodes.is_empty() {
        visit_history_keys(
            tree,
            0,
            &mut Vec::new(),
            &mut history_exact,
            &mut history_boardless,
        );
    }

    PublicStateDuplicateReport {
        decision_nodes,
        exact_unique: exact.len(),
        exact_duplicates: duplicate_count(&exact),
        boardless_unique: boardless.len(),
        boardless_duplicates: duplicate_count(&boardless),
        action_compatible_unique: action_compatible.len(),
        action_compatible_duplicates: duplicate_count(&action_compatible),
        history_exact_unique: history_exact.len(),
        history_exact_duplicates: duplicate_count(&history_exact),
        history_boardless_unique: history_boardless.len(),
        history_boardless_duplicates: duplicate_count(&history_boardless),
    }
}

fn visit_history_keys(
    tree: &PublicTree,
    node_id: usize,
    history: &mut Vec<HistoryItem>,
    exact: &mut BTreeMap<HistoryStateActionKey, usize>,
    boardless: &mut BTreeMap<HistoryStateActionKey, usize>,
) {
    let node = &tree.nodes[node_id];
    match &node.kind {
        PublicNodeKind::Decision { actions, .. } => {
            *exact
                .entry(HistoryStateActionKey {
                    state: StateKey::from_state(&node.state, true),
                    history: history.clone(),
                    actions: actions.clone(),
                })
                .or_default() += 1;
            *boardless
                .entry(HistoryStateActionKey {
                    state: StateKey::from_state(&node.state, false),
                    history: history.clone(),
                    actions: actions.clone(),
                })
                .or_default() += 1;

            for (action, child) in actions.iter().zip(&node.children) {
                history.push(HistoryItem::Action(*action));
                visit_history_keys(tree, *child, history, exact, boardless);
                history.pop();
            }
        }
        PublicNodeKind::Chance(chance) => {
            history.push(HistoryItem::Chance(chance.next_street));
            for child in &node.children {
                visit_history_keys(tree, *child, history, exact, boardless);
            }
            history.pop();
        }
        PublicNodeKind::Terminal { .. } => {}
    }
}

fn duplicate_count<K: Ord>(counts: &BTreeMap<K, usize>) -> usize {
    counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StateKey {
    street: Street,
    board: Option<Vec<u8>>,
    pot: u32,
    oop_stack: u32,
    ip_stack: u32,
    oop_street_commit: u32,
    ip_street_commit: u32,
    last_raise_size: u32,
    raises_this_street: u8,
    checks_this_street: u8,
    player: Player,
}

impl StateKey {
    fn from_state(state: &PublicState, include_board: bool) -> Self {
        Self {
            street: state.street,
            board: include_board.then(|| {
                state
                    .board
                    .cards()
                    .iter()
                    .map(|card| card.index() as u8)
                    .collect()
            }),
            pot: state.pot,
            oop_stack: state.oop_stack,
            ip_stack: state.ip_stack,
            oop_street_commit: state.oop_street_commit,
            ip_street_commit: state.ip_street_commit,
            last_raise_size: state.last_raise_size,
            raises_this_street: state.raises_this_street,
            checks_this_street: state.checks_this_street,
            player: state.player,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct StateActionKey {
    state: StateKey,
    actions: Vec<ActionKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum HistoryItem {
    Action(ActionKind),
    Chance(Street),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HistoryStateActionKey {
    state: StateKey,
    history: Vec<HistoryItem>,
    actions: Vec<ActionKind>,
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

    #[test]
    fn full_state_iteration_updates_every_slot_once() {
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
        let records = layout.records[..3].to_vec();
        let total_action_slots = records.iter().map(|record| record.len).sum();
        let total_infosets = records
            .iter()
            .map(|record| record.len / record.actions as u128)
            .sum::<u128>();
        let mut state = CfrPlusState::allocate(ActionSlotLayout {
            records,
            total_action_slots,
            regret_bytes: layout.regret_bytes,
            strategy_sum_bytes: layout.strategy_sum_bytes,
        })
        .unwrap();
        let summary = state.apply_regret_matching_iteration(1024);
        assert_eq!(summary.updated_slots as u128, total_action_slots);
        assert!(
            (summary.strategy_sum_delta - total_infosets as f64).abs() < 1e-2,
            "strategy_sum_delta={} total_infosets={}",
            summary.strategy_sum_delta,
            total_infosets,
        );
    }

    #[test]
    fn parallel_state_iteration_matches_sequential() {
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
        let records = layout.records[..8].to_vec();
        let total_action_slots = records.iter().map(|record| record.len).sum();
        let small_layout = ActionSlotLayout {
            records,
            total_action_slots,
            regret_bytes: layout.regret_bytes,
            strategy_sum_bytes: layout.strategy_sum_bytes,
        };
        let mut sequential = CfrPlusState::allocate(small_layout.clone()).unwrap();
        let mut parallel = CfrPlusState::allocate(small_layout).unwrap();
        let seq = sequential.apply_regret_matching_iteration_parallel(4096, 1);
        let par = parallel.apply_regret_matching_iteration_parallel(4096, 4);
        assert_eq!(seq.updated_slots, par.updated_slots);
        assert!((seq.strategy_sum_delta - par.strategy_sum_delta).abs() < 1e-6);
        assert!((seq.regret_checksum - par.regret_checksum).abs() < 1e-6);
        let checksum_delta = (seq.strategy_sum_checksum - par.strategy_sum_checksum).abs();
        let checksum_scale = seq.strategy_sum_checksum.abs().max(1.0);
        assert!(
            checksum_delta / checksum_scale < 1e-12,
            "seq={} par={} delta={}",
            seq.strategy_sum_checksum,
            par.strategy_sum_checksum,
            checksum_delta,
        );
        assert_eq!(sequential.regret, parallel.regret);
        assert_eq!(sequential.strategy_sum, parallel.strategy_sum);
    }
}
