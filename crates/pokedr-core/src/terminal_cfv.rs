use crate::cards::{Board, Card};
use std::cmp::Ordering;
use std::thread;
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

impl PreparedTerminalBoard {
    pub fn new(board: &Board) -> Result<Self, String> {
        let combos = live_combos(board)?;
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
    let available_threads = thread::available_parallelism().map_or(1, usize::from);
    let threads = if requested_threads == 0 {
        available_threads
    } else {
        requested_threads.min(available_threads)
    }
    .max(1)
    .min(calls.max(1));

    let started_eval = Instant::now();
    let checksum = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for thread_index in 0..threads {
            let prepared = &prepared;
            handles.push(scope.spawn(move || -> Result<f64, String> {
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
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "terminal CFV worker panicked".to_string())?
            })
            .try_fold(0.0f64, |total, checksum| {
                checksum.map(|value| total + value)
            })
    })?;
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
        prepare_elapsed_ms,
        eval_elapsed_ms,
        calls_per_second,
        checksum,
    })
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

impl TerminalCfvScratch {
    pub fn new(prepared: &PreparedTerminalBoard) -> Self {
        let combos = prepared.combos.len();
        Self {
            prefix: vec![0.0; combos + 1],
            hero_values: vec![0.0; combos],
            villain_values: vec![0.0; combos],
        }
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
    fn parallel_smoke_generates_river_boards_and_calls_cfv() {
        let flop = Board::from_str("As7h2c").unwrap();
        let smoke = terminal_cfv_parallel_smoke(&flop, 64, 2).unwrap();
        assert_eq!(smoke.board_count, 1176);
        assert_eq!(smoke.calls, 64);
        assert!(smoke.threads >= 1);
        assert!(smoke.calls_per_second > 0.0);
        assert!(smoke.checksum.is_finite());
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
