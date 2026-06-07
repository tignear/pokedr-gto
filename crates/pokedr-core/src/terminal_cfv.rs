use crate::cards::{Board, Card};
use crate::range::RangeSpec;
use crate::tree::{PublicNodeKind, PublicTree, TerminalReason};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivateCombo {
    pub first: Card,
    pub second: Card,
}

#[derive(Debug, Clone)]
pub struct TerminalCfvInput {
    pub board: Board,
    pub hero_reach: Vec<f32>,
    pub villain_reach: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TerminalCfvOutput {
    pub hero_values: Vec<f32>,
    pub villain_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PreparedTerminalBoard {
    combos: Vec<PrivateCombo>,
    combo_index_by_key: BTreeMap<u64, usize>,
    strengths: Vec<u64>,
    order: Vec<usize>,
    group_bounds: Vec<(usize, usize)>,
    weaker_blocker_ranges: Vec<(usize, usize)>,
    weaker_blockers: Vec<u16>,
    stronger_blocker_ranges: Vec<(usize, usize)>,
    stronger_blockers: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct TerminalCfvScratch {
    prefix: Vec<f32>,
    hero_values: Vec<f32>,
    villain_values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalCfvParallelSmoke {
    pub board_count: usize,
    pub calls: usize,
    pub threads: usize,
    pub prepare_elapsed_ms: f64,
    pub eval_elapsed_ms: f64,
    pub calls_per_second: f64,
    pub checksum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalCfvTreePass {
    pub terminals: usize,
    pub board_evals: usize,
    pub threads: usize,
    pub prepare_elapsed_ms: f64,
    pub eval_elapsed_ms: f64,
    pub checksum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalCfvBatchSmoke {
    pub columns: usize,
    pub batch_width: usize,
    pub threads: usize,
    pub baseline_elapsed_ms: f64,
    pub batch_elapsed_ms: f64,
    pub speedup: f64,
    pub max_delta: f32,
    pub baseline_checksum: f64,
    pub batch_checksum: f64,
}

#[derive(Debug, Clone)]
pub struct PreparedTerminalCfvSmoke {
    prepared: Vec<PreparedTerminalBoard>,
}

#[derive(Debug, Clone)]
struct PreparedTerminalCfvItem {
    prepared: PreparedTerminalBoard,
    oop_reach: Vec<f32>,
    ip_reach: Vec<f32>,
}

impl PreparedTerminalBoard {
    pub fn new(board: &Board) -> Result<Self, String> {
        let combos = live_combos(board)?;
        let combo_index_by_key = combos
            .iter()
            .enumerate()
            .map(|(index, combo)| (combo.key(), index))
            .collect::<BTreeMap<_, _>>();
        let strengths = combo_strengths(board, &combos);
        let mut order = (0..combos.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|index| strengths[*index]);
        let sorted_strengths = order
            .iter()
            .map(|combo_index| strengths[*combo_index])
            .collect::<Vec<_>>();
        let mut group_bounds = vec![(0usize, 0usize); combos.len()];
        let mut lower = 0usize;
        while lower < order.len() {
            let strength = sorted_strengths[lower];
            let mut upper = lower + 1;
            while upper < order.len() && sorted_strengths[upper] == strength {
                upper += 1;
            }
            for sorted_index in lower..upper {
                group_bounds[order[sorted_index]] = (lower, upper);
            }
            lower = upper;
        }
        let split_blockers = split_blocker_tables(&combos, &strengths);
        Ok(Self {
            combos,
            combo_index_by_key,
            strengths,
            order,
            group_bounds,
            weaker_blocker_ranges: split_blockers.weaker_ranges,
            weaker_blockers: split_blockers.weaker,
            stronger_blocker_ranges: split_blockers.stronger_ranges,
            stronger_blockers: split_blockers.stronger,
        })
    }

    pub fn combos(&self) -> &[PrivateCombo] {
        &self.combos
    }

    pub fn combo_index(&self, first: Card, second: Card) -> Option<usize> {
        self.combo_index_by_key
            .get(&private_combo_key(first, second))
            .copied()
    }

    pub fn reach_from_range(&self, range: &RangeSpec) -> Vec<f32> {
        let mut reach = vec![0.0f32; self.combos.len()];
        for combo in range.combos() {
            if let Some(index) = self
                .combo_index_by_key
                .get(&private_combo_key(combo.first, combo.second))
            {
                reach[*index] += combo.weight;
            }
        }
        reach
    }
}

impl PreparedTerminalCfvSmoke {
    pub fn new(flop: &Board) -> Result<Self, String> {
        if flop.cards().len() != 3 {
            return Err("terminal CFV smoke requires a three-card flop".to_string());
        }
        let boards = river_boards_from_flop(flop)?;
        let prepared = boards
            .iter()
            .map(PreparedTerminalBoard::new)
            .collect::<Result<Vec<_>, _>>()?;
        if prepared.is_empty() {
            return Err("no terminal boards generated".to_string());
        }
        Ok(Self { prepared })
    }

    pub fn board_count(&self) -> usize {
        self.prepared.len()
    }

    pub fn run(
        &self,
        calls: usize,
        requested_threads: usize,
    ) -> Result<TerminalCfvParallelSmoke, String> {
        let eval =
            run_terminal_cfv_parallel_smoke_prepared(&self.prepared, calls, requested_threads)?;
        Ok(TerminalCfvParallelSmoke {
            prepare_elapsed_ms: 0.0,
            ..eval
        })
    }
}

pub fn terminal_cfv_parallel_smoke(
    flop: &Board,
    calls: usize,
    requested_threads: usize,
) -> Result<TerminalCfvParallelSmoke, String> {
    if flop.cards().len() != 3 {
        return Err("terminal CFV smoke requires a three-card flop".to_string());
    }
    let started_prepare = Instant::now();
    let boards = river_boards_from_flop(flop)?;
    let prepared = boards
        .iter()
        .map(PreparedTerminalBoard::new)
        .collect::<Result<Vec<_>, _>>()?;
    let prepare_elapsed_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;
    if prepared.is_empty() {
        return Err("no terminal boards generated".to_string());
    }
    let mut eval = run_terminal_cfv_parallel_smoke_prepared(&prepared, calls, requested_threads)?;
    eval.prepare_elapsed_ms = prepare_elapsed_ms;
    Ok(eval)
}

pub fn terminal_cfv_tree_pass(
    tree: &PublicTree,
    oop_range: &RangeSpec,
    ip_range: &RangeSpec,
    requested_threads: usize,
) -> Result<TerminalCfvTreePass, String> {
    let started_prepare = Instant::now();
    let river_boards = river_boards_from_flop(&tree.spot.board)?;
    let mut board_index_by_key = BTreeMap::new();
    let mut prepared = Vec::with_capacity(river_boards.len());
    for (index, board) in river_boards.iter().enumerate() {
        let prepared_board = PreparedTerminalBoard::new(board)?;
        let oop_reach = prepared_board.reach_from_range(oop_range);
        let ip_reach = prepared_board.reach_from_range(ip_range);
        board_index_by_key.insert(board_key(board), index);
        prepared.push(PreparedTerminalCfvItem {
            prepared: prepared_board,
            oop_reach,
            ip_reach,
        });
    }
    let mut board_indices = Vec::new();
    for node in &tree.nodes {
        let PublicNodeKind::Terminal { reason } = node.kind else {
            continue;
        };
        if !matches!(reason, TerminalReason::Showdown | TerminalReason::AllIn) {
            continue;
        }
        for board in terminal_boards(&node.state.board)? {
            let key = board_key(&board);
            let Some(index) = board_index_by_key.get(&key) else {
                return Err("terminal board is outside the flop river board cache".to_string());
            };
            board_indices.push(*index);
        }
    }
    let prepare_elapsed_ms = started_prepare.elapsed().as_secs_f64() * 1000.0;
    let board_evals = board_indices.len();
    if board_indices.is_empty() {
        return Ok(TerminalCfvTreePass {
            terminals: 0,
            board_evals: 0,
            threads: 0,
            prepare_elapsed_ms,
            eval_elapsed_ms: 0.0,
            checksum: 0.0,
        });
    }

    let available_threads = rayon::current_num_threads();
    let threads = if requested_threads == 0 {
        available_threads
    } else {
        requested_threads.min(available_threads)
    }
    .max(1)
    .min(board_indices.len());

    let started_eval = Instant::now();
    let checksum = (0..threads)
        .into_par_iter()
        .map(|thread_index| -> Result<f64, String> {
            let mut checksum = 0.0f64;
            let mut scratch = TerminalCfvScratch::new(&prepared[0].prepared);
            let mut task_index = thread_index;
            while task_index < board_indices.len() {
                let item = &prepared[board_indices[task_index]];
                terminal_cfv_prefix_blocker_into(
                    &item.prepared,
                    &item.oop_reach,
                    &item.ip_reach,
                    &mut scratch,
                )?;
                checksum += terminal_cfv_output_checksum(&scratch);
                task_index += threads;
            }
            Ok(checksum)
        })
        .try_reduce(|| 0.0f64, |left, right| Ok(left + right))?;
    let eval_elapsed_ms = started_eval.elapsed().as_secs_f64() * 1000.0;
    let terminals = tree
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                PublicNodeKind::Terminal {
                    reason: TerminalReason::Showdown | TerminalReason::AllIn
                }
            )
        })
        .count();

    Ok(TerminalCfvTreePass {
        terminals,
        board_evals,
        threads,
        prepare_elapsed_ms,
        eval_elapsed_ms,
        checksum,
    })
}

pub fn terminal_cfv_batch_smoke(
    board: &Board,
    columns: usize,
    batch_width: usize,
    requested_threads: usize,
) -> Result<TerminalCfvBatchSmoke, String> {
    let terminal_board = first_terminal_board(board)?;
    let prepared = PreparedTerminalBoard::new(&terminal_board)?;
    let combos = prepared.combos().len();
    let columns = columns.max(1);
    let batch_width = batch_width.max(1).min(columns);
    let available_threads = rayon::current_num_threads();
    let threads = if requested_threads == 0 {
        available_threads
    } else {
        requested_threads.min(available_threads)
    }
    .max(1)
    .min(columns);

    let mut hero_reaches = vec![0.0f32; columns * combos];
    let mut villain_reaches = vec![0.0f32; columns * combos];
    for column in 0..columns {
        let hero = deterministic_reach(
            combos,
            column * 11,
            17 + column % 7,
            0.25,
            0.0175 + (column % 5) as f32 * 0.001,
        );
        let villain = deterministic_reach(
            combos,
            7 + column * 13,
            23 + column % 11,
            0.50,
            0.01125 + (column % 3) as f32 * 0.001,
        );
        hero_reaches[column * combos..(column + 1) * combos].copy_from_slice(&hero);
        villain_reaches[column * combos..(column + 1) * combos].copy_from_slice(&villain);
    }

    let baseline_started = Instant::now();
    let baseline_parts = (0..threads)
        .into_par_iter()
        .map(
            |thread_index| -> Result<Vec<(usize, Vec<f32>, Vec<f32>)>, String> {
                let mut scratch = TerminalCfvScratch::new(&prepared);
                let mut outputs = Vec::new();
                let mut column = thread_index;
                while column < columns {
                    terminal_cfv_prefix_blocker_into(
                        &prepared,
                        &hero_reaches[column * combos..(column + 1) * combos],
                        &villain_reaches[column * combos..(column + 1) * combos],
                        &mut scratch,
                    )?;
                    outputs.push((
                        column,
                        scratch.hero_values.clone(),
                        scratch.villain_values.clone(),
                    ));
                    column += threads;
                }
                Ok(outputs)
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let baseline_elapsed_ms = baseline_started.elapsed().as_secs_f64() * 1000.0;
    let mut baseline_hero = vec![0.0f32; columns * combos];
    let mut baseline_villain = vec![0.0f32; columns * combos];
    for part in baseline_parts {
        for (column, hero, villain) in part {
            baseline_hero[column * combos..(column + 1) * combos].copy_from_slice(&hero);
            baseline_villain[column * combos..(column + 1) * combos].copy_from_slice(&villain);
        }
    }

    let batch_started = Instant::now();
    let batch_ranges = (0..columns)
        .step_by(batch_width)
        .map(|start| (start, (start + batch_width).min(columns)))
        .collect::<Vec<_>>();
    let batch_parts = batch_ranges
        .into_par_iter()
        .map(|(start, end)| {
            let width = end - start;
            let mut hero_values = vec![0.0f32; width * combos];
            let mut villain_values = vec![0.0f32; width * combos];
            terminal_cfv_prefix_blocker_columns_into(
                &prepared,
                &hero_reaches[start * combos..end * combos],
                &villain_reaches[start * combos..end * combos],
                width,
                &mut hero_values,
                &mut villain_values,
            );
            (start, hero_values, villain_values)
        })
        .collect::<Vec<_>>();
    let batch_elapsed_ms = batch_started.elapsed().as_secs_f64() * 1000.0;
    let mut batch_hero = vec![0.0f32; columns * combos];
    let mut batch_villain = vec![0.0f32; columns * combos];
    for (start, hero, villain) in batch_parts {
        let width = hero.len() / combos;
        batch_hero[start * combos..(start + width) * combos].copy_from_slice(&hero);
        batch_villain[start * combos..(start + width) * combos].copy_from_slice(&villain);
    }

    let max_delta = baseline_hero
        .iter()
        .chain(&baseline_villain)
        .zip(batch_hero.iter().chain(&batch_villain))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let baseline_checksum =
        terminal_cfv_columns_checksum(&baseline_hero, &baseline_villain, combos);
    let batch_checksum = terminal_cfv_columns_checksum(&batch_hero, &batch_villain, combos);
    let speedup = if batch_elapsed_ms > 0.0 {
        baseline_elapsed_ms / batch_elapsed_ms
    } else {
        0.0
    };

    Ok(TerminalCfvBatchSmoke {
        columns,
        batch_width,
        threads,
        baseline_elapsed_ms,
        batch_elapsed_ms,
        speedup,
        max_delta,
        baseline_checksum,
        batch_checksum,
    })
}

fn run_terminal_cfv_parallel_smoke_prepared(
    prepared: &[PreparedTerminalBoard],
    calls: usize,
    requested_threads: usize,
) -> Result<TerminalCfvParallelSmoke, String> {
    let available_threads = rayon::current_num_threads();
    let threads = if requested_threads == 0 {
        available_threads
    } else {
        requested_threads.min(available_threads)
    }
    .max(1)
    .min(calls.max(1));

    let started_eval = Instant::now();
    let checksum = (0..threads)
        .into_par_iter()
        .map(|thread_index| -> Result<f64, String> {
            let combos = prepared[0].combos().len();
            let hero_reach = deterministic_reach(combos, 0, 17, 0.25, 0.03125);
            let villain_reach = deterministic_reach(combos, 7, 23, 0.50, 0.02125);
            let mut scratch = TerminalCfvScratch::new(&prepared[0]);
            let mut checksum = 0.0f64;
            let mut task = thread_index;
            while task < calls {
                checksum += run_terminal_cfv_smoke_call(
                    prepared,
                    &hero_reach,
                    &villain_reach,
                    &mut scratch,
                    task,
                    combos,
                    task % prepared.len(),
                )?;
                task += threads;
            }
            Ok(checksum)
        })
        .try_reduce(|| 0.0f64, |left, right| Ok(left + right))?;
    let eval_elapsed_ms = started_eval.elapsed().as_secs_f64() * 1000.0;
    let calls_per_second = if eval_elapsed_ms > 0.0 {
        calls as f64 / (eval_elapsed_ms / 1000.0)
    } else {
        0.0
    };

    Ok(TerminalCfvParallelSmoke {
        board_count: prepared.len(),
        calls,
        threads,
        prepare_elapsed_ms: 0.0,
        eval_elapsed_ms,
        calls_per_second,
        checksum,
    })
}

fn first_terminal_board(board: &Board) -> Result<Board, String> {
    if board.cards().len() == 5 {
        return Ok(board.clone());
    }
    terminal_boards(board)?
        .into_iter()
        .next()
        .ok_or_else(|| "no terminal boards generated".to_string())
}

fn run_terminal_cfv_smoke_call(
    prepared: &[PreparedTerminalBoard],
    hero_reach: &[f32],
    villain_reach: &[f32],
    scratch: &mut TerminalCfvScratch,
    task: usize,
    combos: usize,
    board_index: usize,
) -> Result<f64, String> {
    let board = &prepared[board_index];
    terminal_cfv_prefix_blocker_into(board, hero_reach, villain_reach, scratch)?;
    let sample = task % combos;
    Ok(scratch.hero_values[sample] as f64 * 0.5
        + scratch.villain_values[(sample * 37) % combos] as f64 * 0.25)
}

fn terminal_cfv_prefix_blocker_columns_into(
    prepared: &PreparedTerminalBoard,
    hero_reaches: &[f32],
    villain_reaches: &[f32],
    columns: usize,
    hero_values: &mut [f32],
    villain_values: &mut [f32],
) {
    side_values_prefix_blocker_columns_into(prepared, villain_reaches, columns, hero_values);
    side_values_prefix_blocker_columns_into(prepared, hero_reaches, columns, villain_values);
}

fn side_values_prefix_blocker_columns_into(
    prepared: &PreparedTerminalBoard,
    opponent_reaches: &[f32],
    columns: usize,
    values: &mut [f32],
) {
    let combos = prepared.combos.len();
    let mut prefix = vec![0.0f32; (combos + 1) * columns];
    for (sorted_index, combo_index) in prepared.order.iter().enumerate() {
        let previous = sorted_index * columns;
        let next = previous + columns;
        for column in 0..columns {
            prefix[next + column] =
                prefix[previous + column] + opponent_reaches[column * combos + *combo_index];
        }
    }
    let total_start = combos * columns;

    for hero in 0..combos {
        let (lower, upper) = prepared.group_bounds[hero];
        let lower_start = lower * columns;
        let upper_start = upper * columns;
        let (weak_start, weak_end) = prepared.weaker_blocker_ranges[hero];
        let (strong_start, strong_end) = prepared.stronger_blocker_ranges[hero];
        for column in 0..columns {
            let mut value = prefix[lower_start + column]
                - (prefix[total_start + column] - prefix[upper_start + column]);
            for blocker in &prepared.weaker_blockers[weak_start..weak_end] {
                value -= opponent_reaches[column * combos + *blocker as usize];
            }
            for blocker in &prepared.stronger_blockers[strong_start..strong_end] {
                value += opponent_reaches[column * combos + *blocker as usize];
            }
            values[column * combos + hero] = value;
        }
    }
}

impl TerminalCfvScratch {
    pub fn new(prepared: &PreparedTerminalBoard) -> Self {
        let combos = prepared.combos.len();
        Self {
            prefix: vec![0.0; combos + 1],
            hero_values: vec![0.0; combos],
            villain_values: vec![0.0; combos],
        }
    }

    pub fn hero_values(&self) -> &[f32] {
        &self.hero_values
    }

    pub fn villain_values(&self) -> &[f32] {
        &self.villain_values
    }
}

pub fn live_combos(board: &Board) -> Result<Vec<PrivateCombo>, String> {
    if board.cards().len() != 5 {
        return Err("terminal CFV requires a five-card board".to_string());
    }
    let deck = board.remaining_deck();
    let mut combos = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
    for i in 0..deck.len() {
        for j in i + 1..deck.len() {
            combos.push(PrivateCombo {
                first: deck[i],
                second: deck[j],
            });
        }
    }
    Ok(combos)
}

pub fn terminal_cfv_bruteforce(input: &TerminalCfvInput) -> Result<TerminalCfvOutput, String> {
    let prepared = PreparedTerminalBoard::new(&input.board)?;
    terminal_cfv_bruteforce_prepared(&prepared, input)
}

pub fn terminal_cfv_bruteforce_prepared(
    prepared: &PreparedTerminalBoard,
    input: &TerminalCfvInput,
) -> Result<TerminalCfvOutput, String> {
    let combos = &prepared.combos;
    let strengths = &prepared.strengths;
    validate_reach(input, combos.len())?;
    let mut hero_values = vec![0.0; combos.len()];
    let mut villain_values = vec![0.0; combos.len()];
    for hero in 0..combos.len() {
        for villain in 0..combos.len() {
            if combos[hero].collides(combos[villain]) {
                continue;
            }
            let payoff = compare_strength(strengths[hero], strengths[villain]) as f32;
            hero_values[hero] += input.villain_reach[villain] * payoff;
            villain_values[villain] -= input.hero_reach[hero] * payoff;
        }
    }
    Ok(TerminalCfvOutput {
        hero_values,
        villain_values,
    })
}

pub fn terminal_cfv_prefix_blocker(input: &TerminalCfvInput) -> Result<TerminalCfvOutput, String> {
    let prepared = PreparedTerminalBoard::new(&input.board)?;
    terminal_cfv_prefix_blocker_prepared(&prepared, input)
}

pub fn terminal_cfv_prefix_blocker_prepared(
    prepared: &PreparedTerminalBoard,
    input: &TerminalCfvInput,
) -> Result<TerminalCfvOutput, String> {
    let combos = &prepared.combos;
    validate_reach(input, combos.len())?;
    let mut scratch = TerminalCfvScratch::new(prepared);
    terminal_cfv_prefix_blocker_into(
        prepared,
        &input.hero_reach,
        &input.villain_reach,
        &mut scratch,
    )?;
    Ok(TerminalCfvOutput {
        hero_values: scratch.hero_values,
        villain_values: scratch.villain_values,
    })
}

pub fn terminal_cfv_prefix_blocker_into(
    prepared: &PreparedTerminalBoard,
    hero_reach: &[f32],
    villain_reach: &[f32],
    scratch: &mut TerminalCfvScratch,
) -> Result<(), String> {
    let combos = &prepared.combos;
    if hero_reach.len() != combos.len() || villain_reach.len() != combos.len() {
        return Err(format!("reach vectors must have {} entries", combos.len()));
    }
    side_values_prefix_blocker_into(
        prepared,
        villain_reach,
        scratch.prefix.as_mut_slice(),
        &mut scratch.hero_values,
    );
    side_values_prefix_blocker_into(
        prepared,
        hero_reach,
        scratch.prefix.as_mut_slice(),
        &mut scratch.villain_values,
    );
    Ok(())
}

pub fn terminal_cfv_prefix_blocker_targets_into(
    prepared: &PreparedTerminalBoard,
    hero_reach: &[f32],
    villain_reach: &[f32],
    hero_targets: &[Option<usize>],
    villain_targets: &[Option<usize>],
    scratch: &mut TerminalCfvScratch,
) -> Result<(), String> {
    let combos = &prepared.combos;
    if hero_reach.len() != combos.len() || villain_reach.len() != combos.len() {
        return Err(format!("reach vectors must have {} entries", combos.len()));
    }
    side_values_prefix_blocker_targets_into(
        prepared,
        villain_reach,
        hero_targets,
        scratch.prefix.as_mut_slice(),
        &mut scratch.hero_values,
    );
    side_values_prefix_blocker_targets_into(
        prepared,
        hero_reach,
        villain_targets,
        scratch.prefix.as_mut_slice(),
        &mut scratch.villain_values,
    );
    Ok(())
}

pub fn terminal_cfv_prefix_blocker_board_targets_into(
    prepared: &PreparedTerminalBoard,
    hero_reach: &[f32],
    villain_reach: &[f32],
    hero_targets: &[u16],
    villain_targets: &[u16],
    scratch: &mut TerminalCfvScratch,
) -> Result<(), String> {
    let combos = &prepared.combos;
    if hero_reach.len() != combos.len() || villain_reach.len() != combos.len() {
        return Err(format!("reach vectors must have {} entries", combos.len()));
    }
    side_values_prefix_blocker_board_targets_into(
        prepared,
        villain_reach,
        hero_targets,
        scratch.prefix.as_mut_slice(),
        &mut scratch.hero_values,
    );
    side_values_prefix_blocker_board_targets_into(
        prepared,
        hero_reach,
        villain_targets,
        scratch.prefix.as_mut_slice(),
        &mut scratch.villain_values,
    );
    Ok(())
}

pub fn terminal_cfv_sparse_targets_into(
    prepared: &PreparedTerminalBoard,
    hero_reach: &[f32],
    villain_reach: &[f32],
    hero_nonzero: &[u16],
    villain_nonzero: &[u16],
    hero_targets: &[Option<usize>],
    villain_targets: &[Option<usize>],
    scratch: &mut TerminalCfvScratch,
) -> Result<(), String> {
    let combos = &prepared.combos;
    if hero_reach.len() != combos.len() || villain_reach.len() != combos.len() {
        return Err(format!("reach vectors must have {} entries", combos.len()));
    }
    side_values_sparse_targets_into(
        prepared,
        villain_reach,
        villain_nonzero,
        hero_targets,
        &mut scratch.hero_values,
    );
    side_values_sparse_targets_into(
        prepared,
        hero_reach,
        hero_nonzero,
        villain_targets,
        &mut scratch.villain_values,
    );
    Ok(())
}

pub fn terminal_cfv_sparse_board_targets_into(
    prepared: &PreparedTerminalBoard,
    hero_reach: &[f32],
    villain_reach: &[f32],
    hero_nonzero: &[u16],
    villain_nonzero: &[u16],
    hero_targets: &[u16],
    villain_targets: &[u16],
    scratch: &mut TerminalCfvScratch,
) -> Result<(), String> {
    let combos = &prepared.combos;
    if hero_reach.len() != combos.len() || villain_reach.len() != combos.len() {
        return Err(format!("reach vectors must have {} entries", combos.len()));
    }
    side_values_sparse_board_targets_into(
        prepared,
        villain_reach,
        villain_nonzero,
        hero_targets,
        &mut scratch.hero_values,
    );
    side_values_sparse_board_targets_into(
        prepared,
        hero_reach,
        hero_nonzero,
        villain_targets,
        &mut scratch.villain_values,
    );
    Ok(())
}

fn side_values_prefix_blocker_into(
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    prefix: &mut [f32],
    values: &mut [f32],
) {
    let combos = &prepared.combos;
    prefix[0] = 0.0f32;
    for (sorted_index, combo_index) in prepared.order.iter().enumerate() {
        prefix[sorted_index + 1] = prefix[sorted_index] + opponent_reach[*combo_index];
    }
    let total = prefix[combos.len()];

    for hero in 0..combos.len() {
        let (lower, upper) = prepared.group_bounds[hero];
        let weaker = prefix[lower];
        let stronger = total - prefix[upper];
        let mut value = weaker - stronger;

        let (weak_start, weak_end) = prepared.weaker_blocker_ranges[hero];
        for blocker in &prepared.weaker_blockers[weak_start..weak_end] {
            value -= opponent_reach[*blocker as usize];
        }
        let (strong_start, strong_end) = prepared.stronger_blocker_ranges[hero];
        for blocker in &prepared.stronger_blockers[strong_start..strong_end] {
            value += opponent_reach[*blocker as usize];
        }
        values[hero] = value;
    }
}

fn side_values_sparse_targets_into(
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    opponent_nonzero: &[u16],
    targets: &[Option<usize>],
    values: &mut [f32],
) {
    for target in targets.iter().flatten() {
        let hero = *target;
        let hero_combo = prepared.combos[hero];
        let hero_strength = prepared.strengths[hero];
        let mut value = 0.0f32;
        for opponent in opponent_nonzero {
            let villain = *opponent as usize;
            if hero_combo.collides(prepared.combos[villain]) {
                continue;
            }
            value += opponent_reach[villain]
                * compare_strength(hero_strength, prepared.strengths[villain]) as f32;
        }
        values[hero] = value;
    }
}

fn side_values_sparse_board_targets_into(
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    opponent_nonzero: &[u16],
    targets: &[u16],
    values: &mut [f32],
) {
    for target in targets {
        let hero = *target as usize;
        let hero_combo = prepared.combos[hero];
        let hero_strength = prepared.strengths[hero];
        let mut value = 0.0f32;
        for opponent in opponent_nonzero {
            let villain = *opponent as usize;
            if hero_combo.collides(prepared.combos[villain]) {
                continue;
            }
            value += opponent_reach[villain]
                * compare_strength(hero_strength, prepared.strengths[villain]) as f32;
        }
        values[hero] = value;
    }
}

fn side_values_prefix_blocker_targets_into(
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    targets: &[Option<usize>],
    prefix: &mut [f32],
    values: &mut [f32],
) {
    let combos = &prepared.combos;
    prefix[0] = 0.0f32;
    for (sorted_index, combo_index) in prepared.order.iter().enumerate() {
        prefix[sorted_index + 1] = prefix[sorted_index] + opponent_reach[*combo_index];
    }
    let total = prefix[combos.len()];

    for target in targets.iter().flatten() {
        let hero = *target;
        let (lower, upper) = prepared.group_bounds[hero];
        let weaker = prefix[lower];
        let stronger = total - prefix[upper];
        let mut value = weaker - stronger;

        let (weak_start, weak_end) = prepared.weaker_blocker_ranges[hero];
        for blocker in &prepared.weaker_blockers[weak_start..weak_end] {
            value -= opponent_reach[*blocker as usize];
        }
        let (strong_start, strong_end) = prepared.stronger_blocker_ranges[hero];
        for blocker in &prepared.stronger_blockers[strong_start..strong_end] {
            value += opponent_reach[*blocker as usize];
        }
        values[hero] = value;
    }
}

fn side_values_prefix_blocker_board_targets_into(
    prepared: &PreparedTerminalBoard,
    opponent_reach: &[f32],
    targets: &[u16],
    prefix: &mut [f32],
    values: &mut [f32],
) {
    let combos = &prepared.combos;
    prefix[0] = 0.0f32;
    for (sorted_index, combo_index) in prepared.order.iter().enumerate() {
        prefix[sorted_index + 1] = prefix[sorted_index] + opponent_reach[*combo_index];
    }
    let total = prefix[combos.len()];

    for target in targets {
        let hero = *target as usize;
        let (lower, upper) = prepared.group_bounds[hero];
        let weaker = prefix[lower];
        let stronger = total - prefix[upper];
        let mut value = weaker - stronger;

        let (weak_start, weak_end) = prepared.weaker_blocker_ranges[hero];
        for blocker in &prepared.weaker_blockers[weak_start..weak_end] {
            value -= opponent_reach[*blocker as usize];
        }
        let (strong_start, strong_end) = prepared.stronger_blocker_ranges[hero];
        for blocker in &prepared.stronger_blockers[strong_start..strong_end] {
            value += opponent_reach[*blocker as usize];
        }
        values[hero] = value;
    }
}

fn card_combo_table(combos: &[PrivateCombo]) -> Vec<Vec<u16>> {
    let mut card_lists = vec![Vec::new(); 52];
    for (index, combo) in combos.iter().enumerate() {
        card_lists[combo.first.index()].push(index as u16);
        card_lists[combo.second.index()].push(index as u16);
    }
    card_lists
}

struct SplitBlockerTables {
    weaker_ranges: Vec<(usize, usize)>,
    weaker: Vec<u16>,
    stronger_ranges: Vec<(usize, usize)>,
    stronger: Vec<u16>,
}

fn split_blocker_tables(combos: &[PrivateCombo], strengths: &[u64]) -> SplitBlockerTables {
    let card_lists = card_combo_table(combos);
    let mut weaker_ranges = Vec::with_capacity(combos.len());
    let mut weaker = Vec::with_capacity(combos.len() * 46);
    let mut stronger_ranges = Vec::with_capacity(combos.len());
    let mut stronger = Vec::with_capacity(combos.len() * 46);

    for (hero, combo) in combos.iter().enumerate() {
        let hero_strength = strengths[hero];
        let weaker_start = weaker.len();
        let stronger_start = stronger.len();

        for card in [combo.first, combo.second] {
            for blocker in &card_lists[card.index()] {
                if *blocker == hero as u16 {
                    continue;
                }
                let villain = *blocker as usize;
                match strengths[villain].cmp(&hero_strength) {
                    Ordering::Less => weaker.push(*blocker),
                    Ordering::Greater => stronger.push(*blocker),
                    Ordering::Equal => {}
                }
            }
        }
        weaker_ranges.push((weaker_start, weaker.len()));
        stronger_ranges.push((stronger_start, stronger.len()));
    }

    SplitBlockerTables {
        weaker_ranges,
        weaker,
        stronger_ranges,
        stronger,
    }
}

fn validate_reach(input: &TerminalCfvInput, combos: usize) -> Result<(), String> {
    if input.hero_reach.len() != combos || input.villain_reach.len() != combos {
        return Err(format!(
            "reach vectors must have {combos} entries for this board"
        ));
    }
    Ok(())
}

fn river_boards_from_flop(flop: &Board) -> Result<Vec<Board>, String> {
    let deck = flop.remaining_deck();
    let mut boards = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
    for turn_index in 0..deck.len() {
        for river_index in turn_index + 1..deck.len() {
            let board = flop.push(deck[turn_index])?.push(deck[river_index])?;
            boards.push(board);
        }
    }
    Ok(boards)
}

fn terminal_boards(board: &Board) -> Result<Vec<Board>, String> {
    match board.cards().len() {
        5 => Ok(vec![board.clone()]),
        4 => {
            let deck = board.remaining_deck();
            let mut boards = Vec::with_capacity(deck.len());
            for river in deck {
                boards.push(board.push(river)?);
            }
            Ok(boards)
        }
        3 => river_boards_from_flop(board),
        other => Err(format!(
            "terminal CFV requires a flop, turn, or river board, got {other} cards"
        )),
    }
}

fn board_key(board: &Board) -> u64 {
    let mut key = 0u64;
    for card in board.cards() {
        key |= 1u64 << card.index();
    }
    key
}

fn terminal_cfv_output_checksum(scratch: &TerminalCfvScratch) -> f64 {
    scratch
        .hero_values
        .iter()
        .chain(scratch.villain_values.iter())
        .enumerate()
        .map(|(index, value)| *value as f64 * (index as f64 + 1.0))
        .sum()
}

fn terminal_cfv_columns_checksum(
    hero_values: &[f32],
    villain_values: &[f32],
    combos: usize,
) -> f64 {
    hero_values
        .chunks(combos)
        .chain(villain_values.chunks(combos))
        .enumerate()
        .map(|(column, values)| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| *value as f64 * (column as f64 + 1.0) * (index as f64 + 1.0))
                .sum::<f64>()
        })
        .sum()
}

fn deterministic_reach(
    combos: usize,
    offset: usize,
    period: usize,
    base: f32,
    step: f32,
) -> Vec<f32> {
    (0..combos)
        .map(|index| base + ((index + offset) % period) as f32 * step)
        .collect()
}

fn combo_strengths(board: &Board, combos: &[PrivateCombo]) -> Vec<u64> {
    let mut board_acc = SevenCardAccum::new();
    for card in board.cards() {
        board_acc.add(*card);
    }
    combos
        .iter()
        .map(|combo| {
            let mut acc = board_acc;
            acc.add(combo.first);
            acc.add(combo.second);
            acc.rank()
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct SevenCardAccum {
    rank_counts: [u8; 13],
    count_to_rank_mask: [u16; 5],
    suit_rank_masks: [u16; 4],
    rank_mask: u16,
}

impl SevenCardAccum {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, card: Card) {
        let rank = card.rank.index();
        let bit = 1u16 << rank;
        self.rank_mask |= bit;
        self.suit_rank_masks[card.suit.index()] |= bit;

        let previous = self.rank_counts[rank] as usize;
        self.rank_counts[rank] += 1;
        if previous >= 2 {
            self.count_to_rank_mask[previous] &= !bit;
        }
        let current = previous + 1;
        if current >= 2 {
            self.count_to_rank_mask[current] |= bit;
        }
    }

    fn rank(self) -> u64 {
        let flush_mask = self.flush_mask();
        if let Some(flush_mask) = flush_mask {
            if let Some(high) = straight_high_from_mask(flush_mask) {
                return pack(8, &[high]);
            }
        }

        let quads = self.count_to_rank_mask[4];
        if quads != 0 {
            let quad = highest_rank(quads);
            let kicker = highest_rank(self.rank_mask & !quads);
            return pack(7, &[quad, kicker]);
        }

        let trips = self.count_to_rank_mask[3];
        let pairs = self.count_to_rank_mask[2];
        if trips.count_ones() >= 2 {
            let set = highest_rank(trips);
            let pair = highest_rank(trips & !(1u16 << rank_index(set)));
            return pack(6, &[set, pair]);
        }
        if trips != 0 && pairs != 0 {
            return pack(6, &[highest_rank(trips), highest_rank(pairs)]);
        }

        if let Some(flush_mask) = flush_mask {
            return pack_mask(5, flush_mask, 5);
        }

        if let Some(high) = straight_high_from_mask(self.rank_mask) {
            return pack(4, &[high]);
        }

        if trips != 0 {
            return pack_masks(3, trips, self.rank_mask & !trips, 2);
        }

        if pairs.count_ones() >= 2 {
            let top_pairs = keep_top_n(pairs, 2);
            return pack_masks(2, top_pairs, self.rank_mask & !top_pairs, 1);
        }

        if pairs != 0 {
            return pack_masks(1, pairs, self.rank_mask & !pairs, 3);
        }

        pack_mask(0, self.rank_mask, 5)
    }

    fn flush_mask(self) -> Option<u16> {
        self.suit_rank_masks
            .into_iter()
            .find(|mask| mask.count_ones() >= 5)
    }
}

fn straight_high_from_mask(mask: u16) -> Option<u8> {
    const WHEEL: u16 = (1 << 12) | 0b1111;
    for high_index in (4usize..=12).rev() {
        let straight = 0b1_1111u16 << (high_index - 4);
        if mask & straight == straight {
            return Some(rank_value(high_index));
        }
    }
    if mask & WHEEL == WHEEL {
        return Some(5);
    }
    None
}

fn pack(category: u8, ranks: &[u8]) -> u64 {
    let mut value = category as u64;
    for rank in ranks {
        value = (value << 4) | *rank as u64;
    }
    value << (4 * (5 - ranks.len()))
}

fn pack_mask(category: u8, mask: u16, count: usize) -> u64 {
    let mut ranks = [0u8; 5];
    let mut written = 0usize;
    for rank in ranks_desc(mask) {
        ranks[written] = rank;
        written += 1;
        if written == count {
            break;
        }
    }
    pack(category, &ranks[..written])
}

fn pack_masks(category: u8, primary_mask: u16, kicker_mask: u16, kicker_count: usize) -> u64 {
    let mut ranks = [0u8; 5];
    let mut written = 0usize;
    for rank in ranks_desc(primary_mask) {
        ranks[written] = rank;
        written += 1;
    }
    for rank in ranks_desc(kicker_mask) {
        ranks[written] = rank;
        written += 1;
        if written == primary_mask.count_ones() as usize + kicker_count {
            break;
        }
    }
    pack(category, &ranks[..written])
}

fn keep_top_n(mask: u16, count: usize) -> u16 {
    let mut kept = 0u16;
    let mut written = 0usize;
    for index in (0usize..13).rev() {
        let bit = 1u16 << index;
        if mask & bit == 0 {
            continue;
        }
        kept |= bit;
        written += 1;
        if written == count {
            break;
        }
    }
    kept
}

fn highest_rank(mask: u16) -> u8 {
    rank_value(15 - mask.leading_zeros() as usize)
}

fn rank_value(index: usize) -> u8 {
    index as u8 + 2
}

fn rank_index(value: u8) -> usize {
    (value - 2) as usize
}

fn ranks_desc(mask: u16) -> impl Iterator<Item = u8> {
    (0usize..13)
        .rev()
        .filter(move |index| mask & (1u16 << index) != 0)
        .map(rank_value)
}

fn compare_strength(hero: u64, villain: u64) -> i8 {
    match hero.cmp(&villain) {
        Ordering::Greater => 1,
        Ordering::Equal => 0,
        Ordering::Less => -1,
    }
}

impl PrivateCombo {
    fn collides(self, other: Self) -> bool {
        self.first == other.first
            || self.first == other.second
            || self.second == other.first
            || self.second == other.second
    }

    fn key(self) -> u64 {
        private_combo_key(self.first, self.second)
    }
}

fn private_combo_key(first: Card, second: Card) -> u64 {
    (1u64 << first.index()) | (1u64 << second.index())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn accumulator_strength_matches_slow_best_of_seven_for_terminal_board() {
        let board = Board::from_str("As7h2c2d2h").unwrap();
        let combos = live_combos(&board).unwrap();
        let fast = combo_strengths(&board, &combos);
        for (index, combo) in combos.iter().enumerate() {
            let cards = [
                combo.first,
                combo.second,
                board.cards()[0],
                board.cards()[1],
                board.cards()[2],
                board.cards()[3],
                board.cards()[4],
            ];
            assert_eq!(
                fast[index],
                slow_best_7_card_strength(cards),
                "combo={combo:?}"
            );
        }
    }

    #[test]
    fn prefix_blocker_matches_bruteforce() {
        let board = Board::from_str("As7h2c2d2h").unwrap();
        let combos = live_combos(&board).unwrap();
        let hero_reach = (0..combos.len())
            .map(|index| 0.25 + (index % 17) as f32 * 0.03125)
            .collect();
        let villain_reach = (0..combos.len())
            .map(|index| 0.5 + (index % 23) as f32 * 0.02125)
            .collect();
        let input = TerminalCfvInput {
            board,
            hero_reach,
            villain_reach,
        };
        let brute = terminal_cfv_bruteforce(&input).unwrap();
        let fast = terminal_cfv_prefix_blocker(&input).unwrap();
        let max_delta = brute
            .hero_values
            .iter()
            .chain(&brute.villain_values)
            .zip(fast.hero_values.iter().chain(&fast.villain_values))
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta < 2e-3, "max_delta={max_delta}");
    }

    #[test]
    fn prefix_blocker_into_reuses_scratch_for_repeated_calls() {
        let board = Board::from_str("As7h2c2d2h").unwrap();
        let prepared = PreparedTerminalBoard::new(&board).unwrap();
        let combos = prepared.combos().len();
        let hero_reach = (0..combos)
            .map(|index| 0.25 + (index % 17) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let villain_reach = (0..combos)
            .map(|index| 0.5 + (index % 23) as f32 * 0.02125)
            .collect::<Vec<_>>();
        let mut scratch = TerminalCfvScratch::new(&prepared);

        terminal_cfv_prefix_blocker_into(&prepared, &hero_reach, &villain_reach, &mut scratch)
            .unwrap();
        let expected_hero = scratch.hero_values.clone();
        let expected_villain = scratch.villain_values.clone();
        let prefix_capacity = scratch.prefix.capacity();
        let hero_capacity = scratch.hero_values.capacity();
        let villain_capacity = scratch.villain_values.capacity();

        for _ in 0..16 {
            terminal_cfv_prefix_blocker_into(&prepared, &hero_reach, &villain_reach, &mut scratch)
                .unwrap();
            assert_eq!(scratch.hero_values, expected_hero);
            assert_eq!(scratch.villain_values, expected_villain);
            assert_eq!(scratch.prefix.capacity(), prefix_capacity);
            assert_eq!(scratch.hero_values.capacity(), hero_capacity);
            assert_eq!(scratch.villain_values.capacity(), villain_capacity);
        }
    }

    #[test]
    fn prefix_blocker_targets_match_full_values_on_requested_combos() {
        let board = Board::from_str("As7h2c2d2h").unwrap();
        let prepared = PreparedTerminalBoard::new(&board).unwrap();
        let combos = prepared.combos().len();
        let hero_reach = (0..combos)
            .map(|index| 0.25 + (index % 17) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let villain_reach = (0..combos)
            .map(|index| 0.5 + (index % 23) as f32 * 0.02125)
            .collect::<Vec<_>>();
        let hero_targets = (0..32)
            .map(|index| Some((index * 19) % combos))
            .collect::<Vec<_>>();
        let villain_targets = (0..32)
            .map(|index| Some((index * 31 + 7) % combos))
            .collect::<Vec<_>>();

        let mut full = TerminalCfvScratch::new(&prepared);
        terminal_cfv_prefix_blocker_into(&prepared, &hero_reach, &villain_reach, &mut full)
            .unwrap();
        let mut subset = TerminalCfvScratch::new(&prepared);
        terminal_cfv_prefix_blocker_targets_into(
            &prepared,
            &hero_reach,
            &villain_reach,
            &hero_targets,
            &villain_targets,
            &mut subset,
        )
        .unwrap();
        let mut board_subset = TerminalCfvScratch::new(&prepared);
        terminal_cfv_prefix_blocker_board_targets_into(
            &prepared,
            &hero_reach,
            &villain_reach,
            &flatten_targets(&hero_targets),
            &flatten_targets(&villain_targets),
            &mut board_subset,
        )
        .unwrap();

        for target in hero_targets.iter().flatten() {
            assert_eq!(subset.hero_values[*target], full.hero_values[*target]);
            assert_eq!(board_subset.hero_values[*target], full.hero_values[*target]);
        }
        for target in villain_targets.iter().flatten() {
            assert_eq!(subset.villain_values[*target], full.villain_values[*target]);
            assert_eq!(
                board_subset.villain_values[*target],
                full.villain_values[*target]
            );
        }
    }

    #[test]
    fn sparse_targets_match_full_values_on_requested_combos() {
        let board = Board::from_str("As7h2c2d2h").unwrap();
        let prepared = PreparedTerminalBoard::new(&board).unwrap();
        let combos = prepared.combos().len();
        let mut hero_reach = vec![0.0f32; combos];
        let mut villain_reach = vec![0.0f32; combos];
        let hero_nonzero = [3u16, 97, 251, 509];
        let villain_nonzero = [11u16, 173, 431];
        for index in hero_nonzero {
            hero_reach[index as usize] = 0.25 + index as f32 * 0.001;
        }
        for index in villain_nonzero {
            villain_reach[index as usize] = 0.5 + index as f32 * 0.001;
        }
        let hero_targets = (0..32)
            .map(|index| Some((index * 19) % combos))
            .collect::<Vec<_>>();
        let villain_targets = (0..32)
            .map(|index| Some((index * 31 + 7) % combos))
            .collect::<Vec<_>>();

        let mut full = TerminalCfvScratch::new(&prepared);
        terminal_cfv_prefix_blocker_into(&prepared, &hero_reach, &villain_reach, &mut full)
            .unwrap();
        let mut sparse = TerminalCfvScratch::new(&prepared);
        terminal_cfv_sparse_targets_into(
            &prepared,
            &hero_reach,
            &villain_reach,
            &hero_nonzero,
            &villain_nonzero,
            &hero_targets,
            &villain_targets,
            &mut sparse,
        )
        .unwrap();
        let mut board_sparse = TerminalCfvScratch::new(&prepared);
        terminal_cfv_sparse_board_targets_into(
            &prepared,
            &hero_reach,
            &villain_reach,
            &hero_nonzero,
            &villain_nonzero,
            &flatten_targets(&hero_targets),
            &flatten_targets(&villain_targets),
            &mut board_sparse,
        )
        .unwrap();

        for target in hero_targets.iter().flatten() {
            assert!(
                (sparse.hero_values[*target] - full.hero_values[*target]).abs() < 1e-5,
                "hero target={target} sparse={} full={}",
                sparse.hero_values[*target],
                full.hero_values[*target],
            );
            assert!(
                (board_sparse.hero_values[*target] - full.hero_values[*target]).abs() < 1e-5,
                "hero target={target} board_sparse={} full={}",
                board_sparse.hero_values[*target],
                full.hero_values[*target],
            );
        }
        for target in villain_targets.iter().flatten() {
            assert!(
                (sparse.villain_values[*target] - full.villain_values[*target]).abs() < 1e-5,
                "villain target={target} sparse={} full={}",
                sparse.villain_values[*target],
                full.villain_values[*target],
            );
            assert!(
                (board_sparse.villain_values[*target] - full.villain_values[*target]).abs() < 1e-5,
                "villain target={target} board_sparse={} full={}",
                board_sparse.villain_values[*target],
                full.villain_values[*target],
            );
        }
    }

    #[test]
    fn parallel_smoke_generates_river_boards_and_calls_cfv() {
        let flop = Board::from_str("As7h2c").unwrap();
        let smoke = terminal_cfv_parallel_smoke(&flop, 64, 2).unwrap();
        assert_eq!(smoke.board_count, 1176);
        assert_eq!(smoke.calls, 64);
        assert!(smoke.threads >= 1);
        assert!(smoke.calls_per_second > 0.0);
        assert!(smoke.checksum.is_finite());
    }

    #[test]
    fn batch_smoke_matches_scalar_prefix_for_flop_input() {
        let flop = Board::from_str("As7h2c").unwrap();
        let smoke = terminal_cfv_batch_smoke(&flop, 8, 4, 2).unwrap();
        assert_eq!(smoke.columns, 8);
        assert_eq!(smoke.batch_width, 4);
        assert!(smoke.baseline_elapsed_ms >= 0.0);
        assert!(smoke.batch_elapsed_ms >= 0.0);
        assert!(smoke.max_delta < 1e-4, "max_delta={}", smoke.max_delta);
        assert!(smoke.baseline_checksum.is_finite());
        assert!(smoke.batch_checksum.is_finite());
    }

    fn flatten_targets(targets: &[Option<usize>]) -> Vec<u16> {
        targets
            .iter()
            .flatten()
            .map(|target| *target as u16)
            .collect()
    }

    fn slow_best_7_card_strength(cards: [Card; 7]) -> u64 {
        let mut best = 0;
        for a in 0..3 {
            for b in a + 1..4 {
                for c in b + 1..5 {
                    for d in c + 1..6 {
                        for e in d + 1..7 {
                            best = best.max(slow_rank_5([
                                cards[a], cards[b], cards[c], cards[d], cards[e],
                            ]));
                        }
                    }
                }
            }
        }
        best
    }

    fn slow_rank_5(cards: [Card; 5]) -> u64 {
        let mut counts = [0u8; 13];
        let mut suit_counts = [0u8; 4];
        let mut rank_mask = 0u16;
        let mut suit_masks = [0u16; 4];
        for card in cards {
            let rank = card.rank.index();
            let suit = card.suit.index();
            counts[rank] += 1;
            suit_counts[suit] += 1;
            rank_mask |= 1u16 << rank;
            suit_masks[suit] |= 1u16 << rank;
        }

        let flush_mask = suit_counts
            .iter()
            .position(|count| *count == 5)
            .map(|suit| suit_masks[suit]);
        if let Some(flush_mask) = flush_mask {
            if let Some(high) = straight_high_from_mask(flush_mask) {
                return pack(8, &[high]);
            }
        }

        let mut quads = 0u16;
        let mut trips = 0u16;
        let mut pairs = 0u16;
        for (rank, count) in counts.iter().enumerate() {
            let bit = 1u16 << rank;
            match count {
                4 => quads |= bit,
                3 => trips |= bit,
                2 => pairs |= bit,
                _ => {}
            }
        }

        if quads != 0 {
            return pack(7, &[highest_rank(quads), highest_rank(rank_mask & !quads)]);
        }
        if trips != 0 && pairs != 0 {
            return pack(6, &[highest_rank(trips), highest_rank(pairs)]);
        }
        if let Some(flush_mask) = flush_mask {
            return pack_mask(5, flush_mask, 5);
        }
        if let Some(high) = straight_high_from_mask(rank_mask) {
            return pack(4, &[high]);
        }
        if trips != 0 {
            return pack_masks(3, trips, rank_mask & !trips, 2);
        }
        if pairs.count_ones() == 2 {
            return pack_masks(2, pairs, rank_mask & !pairs, 1);
        }
        if pairs != 0 {
            return pack_masks(1, pairs, rank_mask & !pairs, 3);
        }
        pack_mask(0, rank_mask, 5)
    }
}
