use crate::blinds::{BlindClock, BlindLevel};
use crate::death_race::{DeathRaceState, death_race_value};
use crate::equity::{
    EquityCache, EquityPot, heads_up_equity_vs_range_cached, multi_way_range_showdown_payouts,
    multi_way_showdown_payouts, three_way_equity_vs_ranges_cached,
};
use crate::hand_class::{HandClass, all_hand_classes};
use crate::structure::orbit_cost;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct ShortStackConfig {
    pub level: BlindLevel,
    pub alive_players: u8,
    pub stack: u32,
    pub stacks: Vec<u32>,
    pub players_behind: u8,
    pub elapsed_in_level_seconds: u32,
    pub hand_duration_seconds: u32,
    pub max_boards_per_combo: usize,
    pub range_sample_limit: usize,
    pub iterations: usize,
    pub spot_iterations: usize,
    pub include_overcall: bool,
}

#[derive(Debug, Clone)]
pub struct ShortStackReport {
    pub dead_pot: u32,
    pub stack_in_big_blinds: f64,
    pub orbit_cost: u32,
    pub converged: bool,
    pub max_iterations: usize,
    pub max_spot_iterations: usize,
    pub overcall_analyzed: bool,
    pub single_call_required_equity: f64,
    pub overcall_required_equity: f64,
    pub seats: Vec<SeatRanges>,
}

#[derive(Debug, Clone)]
pub struct SeatRanges {
    pub seat_index: u8,
    pub players_behind: u8,
    pub posted_amount: u32,
    pub iterations_run: usize,
    pub converged: bool,
    pub call_required_equity: f64,
    pub overcall_required_equity: f64,
    pub shove_range: Vec<HandResult>,
    pub call_range: Vec<HandResult>,
    pub call_spots: Vec<CallSpot>,
    pub overcall_range: Vec<HandResult>,
}

#[derive(Debug, Clone)]
pub struct CallSpot {
    pub opener_seat: u8,
    pub effective_all_in_cost: u32,
    pub iterations_run: usize,
    pub converged: bool,
    pub required_equity: f64,
    pub range: Vec<HandResult>,
    pub patterns: Vec<CallPattern>,
    pub next_response: Option<ResponseNode>,
}

#[derive(Debug, Clone)]
pub struct CallPattern {
    pub callers: Vec<u8>,
    pub probability: f64,
    pub range: Vec<HandResult>,
}

#[derive(Debug, Clone)]
pub struct ResponseNode {
    pub actor_seat: u8,
    pub prior_callers: Vec<u8>,
    pub range: Vec<HandResult>,
}

#[derive(Debug, Clone)]
pub struct HandResult {
    pub hand: HandClass,
    pub equity: f64,
    pub ev: f64,
}

pub fn analyze_short_stack(config: &ShortStackConfig) -> ShortStackReport {
    let stacks = normalized_stacks(config);
    let report_stack = stacks.first().copied().unwrap_or(config.stack);
    let dead_pot = config.level.small_blind
        + config.level.big_blind
        + config
            .level
            .per_player_ante
            .saturating_mul(config.alive_players as u32);
    let single_call_required_equity =
        report_stack as f64 / (dead_pot + report_stack.saturating_mul(2)) as f64;
    let overcall_required_equity =
        report_stack as f64 / (dead_pot + report_stack.saturating_mul(3)) as f64;

    let classes = all_hand_classes();
    let mut cache = EquityCache::new();
    let seats: Vec<_> = (0..config.alive_players)
        .map(|seat_index| {
            let players_behind = config
                .alive_players
                .saturating_sub(seat_index)
                .saturating_sub(1);
            analyze_seat(
                config,
                &stacks,
                &classes,
                dead_pot,
                seat_index,
                players_behind,
                &mut cache,
            )
        })
        .collect();
    let converged = seats
        .iter()
        .all(|seat| seat.converged && seat.call_spots.iter().all(|spot| spot.converged));

    ShortStackReport {
        dead_pot,
        stack_in_big_blinds: report_stack as f64 / config.level.big_blind as f64,
        orbit_cost: orbit_cost(config.level, config.alive_players),
        converged,
        max_iterations: config.iterations,
        max_spot_iterations: config.spot_iterations,
        overcall_analyzed: config.include_overcall,
        single_call_required_equity,
        overcall_required_equity,
        seats,
    }
}

fn analyze_seat(
    config: &ShortStackConfig,
    stacks: &[u32],
    classes: &[HandClass],
    dead_pot: u32,
    seat_index: u8,
    players_behind: u8,
    cache: &mut EquityCache,
) -> SeatRanges {
    let seat_config = ShortStackConfig {
        players_behind,
        ..config.clone()
    };
    let posted_amount = posted_amount(config.level, config.alive_players, seat_index);
    let seat_stack = stacks[seat_index as usize];
    let all_in_cost = seat_stack.saturating_sub(posted_amount);
    let posted_stacks = posted_stacks(config.level, stacks);
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };
    let hand_duration_seconds = config.hand_duration_seconds;
    let fold_value = state_value(
        clock,
        hand_duration_seconds,
        &posted_stacks,
        seat_index as usize,
    );
    let call_win_value = state_value(
        clock,
        hand_duration_seconds,
        &call_win_stacks(
            &posted_stacks,
            seat_index as usize,
            opener_seat(seat_index),
            dead_pot,
            all_in_cost,
        ),
        seat_index as usize,
    );
    let call_lose_value = state_value(
        clock,
        hand_duration_seconds,
        &call_lose_stacks(
            &posted_stacks,
            seat_index as usize,
            opener_seat(seat_index),
            dead_pot,
            all_in_cost,
        ),
        seat_index as usize,
    );
    let overcall_win_value = state_value(
        clock,
        hand_duration_seconds,
        &call_win_stacks(
            &posted_stacks,
            seat_index as usize,
            opener_seat(seat_index),
            dead_pot,
            all_in_cost,
        ),
        seat_index as usize,
    );
    let overcall_lose_value = call_lose_value;
    let call_required_equity = state_required_equity(fold_value, call_win_value, call_lose_value);
    let overcall_required_equity =
        state_required_equity(fold_value, overcall_win_value, overcall_lose_value);
    let response_required_equity =
        first_response_required_equity(config, stacks, dead_pot, seat_index as usize, all_in_cost);
    let solution = solve_shove_response(
        classes,
        response_required_equity,
        &seat_config,
        stacks,
        dead_pot,
        all_in_cost,
        seat_index as usize,
        config.iterations,
        cache,
    );
    let modeled_shove_range = solution.shove_range;
    let displayed_shove_range = if players_behind == 0 {
        Vec::new()
    } else {
        modeled_shove_range.clone()
    };
    let call_range_results = profitable_call_range(
        classes,
        &modeled_shove_range
            .iter()
            .map(|result| result.hand)
            .collect::<Vec<_>>(),
        call_required_equity,
        config.max_boards_per_combo,
        config.range_sample_limit,
        cache,
    );
    let call_spots = analyze_call_spots(
        config,
        stacks,
        classes,
        dead_pot,
        seat_index as usize,
        cache,
    );
    let overcall_range = if config.include_overcall {
        profitable_overcall_range(
            classes,
            &modeled_shove_range
                .iter()
                .map(|result| result.hand)
                .collect::<Vec<_>>(),
            &solution.call_range,
            overcall_required_equity,
            config.max_boards_per_combo,
            config.range_sample_limit,
            cache,
        )
    } else {
        Vec::new()
    };

    SeatRanges {
        seat_index,
        players_behind,
        posted_amount,
        iterations_run: solution.iterations_run,
        converged: solution.converged,
        call_required_equity,
        overcall_required_equity,
        shove_range: displayed_shove_range,
        call_range: call_range_results,
        call_spots,
        overcall_range,
    }
}

fn analyze_call_spots(
    config: &ShortStackConfig,
    stacks: &[u32],
    classes: &[HandClass],
    dead_pot: u32,
    caller_seat: usize,
    cache: &mut EquityCache,
) -> Vec<CallSpot> {
    let mut spots = Vec::new();
    let posted = posted_stacks(config.level, stacks);
    let caller_cost = stacks[caller_seat].saturating_sub(posted_amount(
        config.level,
        stacks.len() as u8,
        caller_seat as u8,
    ));
    for opener_seat in 0..caller_seat {
        let opener_cost = stacks[opener_seat].saturating_sub(posted_amount(
            config.level,
            stacks.len() as u8,
            opener_seat as u8,
        ));
        let effective_cost = caller_cost.min(opener_cost);
        if effective_cost == 0 {
            continue;
        }

        let opener_config = ShortStackConfig {
            players_behind: stacks.len().saturating_sub(opener_seat).saturating_sub(1) as u8,
            ..config.clone()
        };
        let required_equity = call_required_equity_for(
            config,
            &posted,
            dead_pot,
            caller_seat,
            opener_seat,
            effective_cost,
        );
        let solution = solve_shove_response(
            classes,
            required_equity,
            &opener_config,
            stacks,
            dead_pot,
            opener_cost,
            opener_seat,
            config.spot_iterations,
            cache,
        );
        let range = profitable_call_range(
            classes,
            &solution
                .shove_range
                .iter()
                .map(|result| result.hand)
                .collect::<Vec<_>>(),
            required_equity,
            config.max_boards_per_combo,
            config.range_sample_limit,
            cache,
        );
        let tree_result = profitable_tree_call_range(
            classes,
            &solution
                .shove_range
                .iter()
                .map(|result| result.hand)
                .collect::<Vec<_>>(),
            config,
            stacks,
            dead_pot,
            caller_seat,
            opener_seat,
            range,
            cache,
        );
        let range = tree_result.range;
        let next_response = next_response_node(
            classes,
            &solution
                .shove_range
                .iter()
                .map(|result| result.hand)
                .collect::<Vec<_>>(),
            &range.iter().map(|result| result.hand).collect::<Vec<_>>(),
            config,
            stacks,
            dead_pot,
            opener_seat,
            caller_seat,
        );

        spots.push(CallSpot {
            opener_seat: opener_seat as u8,
            effective_all_in_cost: effective_cost,
            iterations_run: solution.iterations_run,
            converged: solution.converged,
            required_equity,
            range,
            patterns: tree_result.patterns,
            next_response,
        });
    }

    spots
}

struct TreeCallResult {
    range: Vec<HandResult>,
    patterns: Vec<CallPattern>,
}

fn profitable_tree_call_range(
    classes: &[HandClass],
    opener_range: &[HandClass],
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    caller_seat: usize,
    opener_seat: usize,
    heads_up_range: Vec<HandResult>,
    _cache: &mut EquityCache,
) -> TreeCallResult {
    let downstream_patterns = downstream_call_patterns(classes, stacks, caller_seat);

    let posted = posted_stacks(config.level, stacks);
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };
    let hand_duration_seconds = config.hand_duration_seconds;
    let fold_value = state_value(
        clock,
        hand_duration_seconds,
        &steal_stacks(&posted, opener_seat, dead_pot),
        caller_seat,
    );
    let sampled_opener_range = sample_range(opener_range, config.range_sample_limit);
    let tree_sample_limit = config.range_sample_limit.min(4);
    let sampled_opener_range = sample_range(&sampled_opener_range, tree_sample_limit);
    let candidate_hands: Vec<_> = heads_up_range
        .into_iter()
        .map(|result| result.hand)
        .collect();
    let mut pattern_outputs = Vec::new();

    for pattern in &downstream_patterns {
        let range = pattern_range(
            &candidate_hands,
            &sampled_opener_range,
            config,
            &posted,
            dead_pot,
            caller_seat,
            opener_seat,
            stacks,
            pattern,
        );
        pattern_outputs.push(CallPattern {
            callers: pattern.callers.iter().map(|&seat| seat as u8).collect(),
            probability: pattern.probability,
            range,
        });
    }

    let mut results: Vec<_> = candidate_hands
        .into_par_iter()
        .filter_map(|hand| {
            let mut equity_acc = 0.0;
            let mut call_value = 0.0;

            for pattern in &downstream_patterns {
                let outcome = pattern_outcome(
                    hand,
                    &sampled_opener_range,
                    config,
                    clock,
                    hand_duration_seconds,
                    &posted,
                    dead_pot,
                    caller_seat,
                    opener_seat,
                    stacks,
                    pattern,
                );
                equity_acc += pattern.probability * outcome.hero_share;
                call_value += pattern.probability * outcome.value;
            }

            let ev = call_value - fold_value;

            if ev >= 0.0 {
                Some(HandResult {
                    hand,
                    equity: equity_acc,
                    ev,
                })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|left, right| {
        right
            .ev
            .total_cmp(&left.ev)
            .then_with(|| right.equity.total_cmp(&left.equity))
    });
    TreeCallResult {
        range: results,
        patterns: pattern_outputs,
    }
}

fn next_response_node(
    classes: &[HandClass],
    opener_range: &[HandClass],
    caller_range: &[HandClass],
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    opener_seat: usize,
    caller_seat: usize,
) -> Option<ResponseNode> {
    let actor_seat = caller_seat + 1;
    if actor_seat >= stacks.len() || caller_range.is_empty() || opener_range.is_empty() {
        return None;
    }

    let prior_callers = vec![caller_seat];
    let prior_ranges = vec![sample_range(caller_range, config.range_sample_limit.min(4))];
    let sampled_opener_range = sample_range(
        &sample_range(opener_range, config.range_sample_limit),
        config.range_sample_limit.min(4),
    );
    let range = response_node_range(
        classes,
        &sampled_opener_range,
        &prior_ranges,
        config,
        stacks,
        dead_pot,
        actor_seat,
        opener_seat,
        &prior_callers,
    );

    Some(ResponseNode {
        actor_seat: actor_seat as u8,
        prior_callers: prior_callers.into_iter().map(|seat| seat as u8).collect(),
        range,
    })
}

fn response_node_range(
    classes: &[HandClass],
    sampled_opener_range: &[HandClass],
    prior_ranges: &[Vec<HandClass>],
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    actor_seat: usize,
    opener_seat: usize,
    prior_callers: &[usize],
) -> Vec<HandResult> {
    let posted = posted_stacks(config.level, stacks);
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };
    let future_patterns = downstream_call_patterns(classes, stacks, actor_seat);
    let fold_value = future_patterns
        .iter()
        .map(|pattern| {
            pattern.probability
                * folded_response_value(
                    config,
                    clock,
                    &posted,
                    dead_pot,
                    actor_seat,
                    opener_seat,
                    prior_callers,
                    stacks,
                    sampled_opener_range,
                    prior_ranges,
                    pattern,
                )
        })
        .sum::<f64>();

    let mut range: Vec<_> = classes
        .par_iter()
        .copied()
        .filter_map(|hand| {
            let mut hero_share = 0.0;
            let mut call_value = 0.0;

            for pattern in &future_patterns {
                let outcome = response_pattern_outcome(
                    hand,
                    sampled_opener_range,
                    prior_ranges,
                    config,
                    clock,
                    &posted,
                    dead_pot,
                    actor_seat,
                    opener_seat,
                    prior_callers,
                    stacks,
                    pattern,
                );
                hero_share += pattern.probability * outcome.hero_share;
                call_value += pattern.probability * outcome.value;
            }

            let ev = call_value - fold_value;
            (ev >= 0.0).then_some(HandResult {
                hand,
                equity: hero_share,
                ev,
            })
        })
        .collect();

    range.sort_by(|left, right| {
        right
            .ev
            .total_cmp(&left.ev)
            .then_with(|| right.equity.total_cmp(&left.equity))
    });
    range
}

struct DownstreamPattern {
    probability: f64,
    callers: Vec<usize>,
    ranges: Vec<Vec<HandClass>>,
}

fn downstream_call_patterns(
    classes: &[HandClass],
    stacks: &[u32],
    caller_seat: usize,
) -> Vec<DownstreamPattern> {
    let seats: Vec<_> = ((caller_seat + 1)..stacks.len()).collect();
    let range = top_fraction_by_heuristic(classes, 0.2);
    let call_probability = combo_fraction(&range);
    let pattern_count = 1_usize << seats.len();

    (0..pattern_count)
        .map(|mask| {
            let mut probability = 1.0;
            let mut callers = Vec::new();
            let mut ranges = Vec::new();

            for (index, &seat) in seats.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    probability *= 1.0 - call_probability;
                } else {
                    probability *= call_probability;
                    callers.push(seat);
                    ranges.push(range.clone());
                }
            }

            DownstreamPattern {
                probability,
                callers,
                ranges,
            }
        })
        .collect()
}

fn pattern_range(
    candidate_hands: &[HandClass],
    sampled_opener_range: &[HandClass],
    config: &ShortStackConfig,
    posted: &[u32],
    dead_pot: u32,
    caller_seat: usize,
    opener_seat: usize,
    stacks: &[u32],
    pattern: &DownstreamPattern,
) -> Vec<HandResult> {
    let mut range: Vec<_> = candidate_hands
        .par_iter()
        .copied()
        .filter_map(|hand| {
            let clock = BlindClock {
                current_level: config.level.level,
                elapsed_in_level_seconds: config.elapsed_in_level_seconds,
            };
            let outcome = pattern_outcome(
                hand,
                sampled_opener_range,
                config,
                clock,
                config.hand_duration_seconds,
                posted,
                dead_pot,
                caller_seat,
                opener_seat,
                stacks,
                pattern,
            );
            let fold_value = state_value(
                clock,
                config.hand_duration_seconds,
                &steal_stacks(posted, opener_seat, dead_pot),
                caller_seat,
            );
            let ev = outcome.value - fold_value;

            (ev >= 0.0).then_some(HandResult {
                hand,
                equity: outcome.hero_share,
                ev,
            })
        })
        .collect();
    range.sort_by(|left, right| {
        right
            .ev
            .total_cmp(&left.ev)
            .then_with(|| right.equity.total_cmp(&left.equity))
    });
    range
}

struct PatternOutcome {
    hero_share: f64,
    value: f64,
}

fn pattern_outcome(
    hand: HandClass,
    sampled_opener_range: &[HandClass],
    config: &ShortStackConfig,
    clock: BlindClock,
    hand_duration_seconds: u32,
    posted: &[u32],
    dead_pot: u32,
    caller_seat: usize,
    opener_seat: usize,
    stacks: &[u32],
    pattern: &DownstreamPattern,
) -> PatternOutcome {
    let tree_sample_limit = config.range_sample_limit.min(4);
    let mut opponent_ranges = Vec::with_capacity(pattern.ranges.len() + 1);
    opponent_ranges.push(sampled_opener_range.to_vec());
    opponent_ranges.extend(
        pattern
            .ranges
            .iter()
            .map(|range| sample_range(range, tree_sample_limit)),
    );

    let participants = pattern_participants(caller_seat, opener_seat, pattern);
    let commitments = all_in_commitments(config.level, stacks, &participants);
    let pots = side_pots(dead_pot, &commitments);
    let total_pot = pots.iter().map(|pot| pot.amount).sum::<f64>();
    let payouts =
        multi_way_showdown_payouts(hand, &opponent_ranges, &pots, config.max_boards_per_combo);
    let mut hero_payout = 0.0;
    let mut value = 0.0;

    for payout in &payouts {
        hero_payout += payout.first().copied().unwrap_or(0.0);
        value += state_value(
            clock,
            hand_duration_seconds,
            &showdown_stacks_from_payouts(posted, &participants, &commitments, payout),
            caller_seat,
        );
    }

    if !payouts.is_empty() {
        hero_payout /= payouts.len() as f64;
        value /= payouts.len() as f64;
    }

    PatternOutcome {
        hero_share: if total_pot > 0.0 {
            hero_payout / total_pot
        } else {
            0.0
        },
        value,
    }
}

fn response_pattern_outcome(
    hand: HandClass,
    sampled_opener_range: &[HandClass],
    prior_ranges: &[Vec<HandClass>],
    config: &ShortStackConfig,
    clock: BlindClock,
    posted: &[u32],
    dead_pot: u32,
    actor_seat: usize,
    opener_seat: usize,
    prior_callers: &[usize],
    stacks: &[u32],
    pattern: &DownstreamPattern,
) -> PatternOutcome {
    let tree_sample_limit = config.range_sample_limit.min(4);
    let mut opponent_ranges = Vec::with_capacity(prior_ranges.len() + pattern.ranges.len() + 1);
    opponent_ranges.push(sampled_opener_range.to_vec());
    opponent_ranges.extend(prior_ranges.iter().cloned());
    opponent_ranges.extend(
        pattern
            .ranges
            .iter()
            .map(|range| sample_range(range, tree_sample_limit)),
    );

    let mut participants = vec![actor_seat, opener_seat];
    participants.extend(prior_callers.iter().copied());
    participants.extend(pattern.callers.iter().copied());
    let commitments = all_in_commitments(config.level, stacks, &participants);
    let pots = side_pots(dead_pot, &commitments);
    let total_pot = pots.iter().map(|pot| pot.amount).sum::<f64>();
    let payouts =
        multi_way_showdown_payouts(hand, &opponent_ranges, &pots, config.max_boards_per_combo);
    let mut hero_payout = 0.0;
    let mut value = 0.0;

    for payout in &payouts {
        hero_payout += payout.first().copied().unwrap_or(0.0);
        value += state_value(
            clock,
            config.hand_duration_seconds,
            &showdown_stacks_from_payouts(posted, &participants, &commitments, payout),
            actor_seat,
        );
    }

    if !payouts.is_empty() {
        hero_payout /= payouts.len() as f64;
        value /= payouts.len() as f64;
    }

    PatternOutcome {
        hero_share: if total_pot > 0.0 {
            hero_payout / total_pot
        } else {
            0.0
        },
        value,
    }
}

fn folded_response_value(
    config: &ShortStackConfig,
    clock: BlindClock,
    posted: &[u32],
    dead_pot: u32,
    actor_seat: usize,
    opener_seat: usize,
    prior_callers: &[usize],
    stacks: &[u32],
    sampled_opener_range: &[HandClass],
    prior_ranges: &[Vec<HandClass>],
    pattern: &DownstreamPattern,
) -> f64 {
    let tree_sample_limit = config.range_sample_limit.min(4);
    let mut participants = vec![opener_seat];
    participants.extend(prior_callers.iter().copied());
    participants.extend(pattern.callers.iter().copied());
    let mut player_ranges = Vec::with_capacity(participants.len());
    player_ranges.push(sampled_opener_range.to_vec());
    player_ranges.extend(prior_ranges.iter().cloned());
    player_ranges.extend(
        pattern
            .ranges
            .iter()
            .map(|range| sample_range(range, tree_sample_limit)),
    );
    let commitments = all_in_commitments(config.level, stacks, &participants);
    let pots = side_pots(dead_pot, &commitments);
    let payouts =
        multi_way_range_showdown_payouts(&player_ranges, &pots, config.max_boards_per_combo);

    if payouts.is_empty() {
        return state_value(clock, config.hand_duration_seconds, posted, actor_seat);
    }

    payouts
        .iter()
        .map(|payout| {
            state_value(
                clock,
                config.hand_duration_seconds,
                &showdown_stacks_from_payouts(posted, &participants, &commitments, payout),
                actor_seat,
            )
        })
        .sum::<f64>()
        / payouts.len() as f64
}

fn pattern_participants(
    caller_seat: usize,
    opener_seat: usize,
    pattern: &DownstreamPattern,
) -> Vec<usize> {
    let mut participants = vec![caller_seat, opener_seat];
    participants.extend(pattern.callers.iter().copied());
    participants
}

fn all_in_commitments(
    level: BlindLevel,
    stacks: &[u32],
    participants: &[usize],
) -> Vec<(usize, u32)> {
    let full_commitments: Vec<_> = participants
        .iter()
        .map(|&seat| {
            (
                seat,
                stacks[seat].saturating_sub(posted_amount(level, stacks.len() as u8, seat as u8)),
            )
        })
        .collect();

    full_commitments
        .iter()
        .enumerate()
        .map(|(index, &(seat, amount))| {
            let max_called = full_commitments
                .iter()
                .enumerate()
                .filter(|(other_index, _)| *other_index != index)
                .map(|(_, &(_, other_amount))| other_amount)
                .max()
                .unwrap_or(0);
            (seat, amount.min(max_called))
        })
        .collect()
}

fn side_pots(dead_pot: u32, commitments: &[(usize, u32)]) -> Vec<EquityPot> {
    let mut levels: Vec<_> = commitments
        .iter()
        .map(|&(_, amount)| amount)
        .filter(|&amount| amount > 0)
        .collect();
    levels.sort_unstable();
    levels.dedup();

    let mut previous = 0;
    let mut dead_pot_left = dead_pot as f64;
    let mut pots = Vec::new();

    for level in levels {
        let eligible: Vec<_> = commitments
            .iter()
            .enumerate()
            .filter_map(|(index, &(_, amount))| (amount >= level).then_some(index))
            .collect();
        let layer = level.saturating_sub(previous);
        previous = level;

        if eligible.len() < 2 || layer == 0 {
            continue;
        }

        let mut amount = layer as f64 * eligible.len() as f64;
        if dead_pot_left > 0.0 {
            amount += dead_pot_left;
            dead_pot_left = 0.0;
        }
        pots.push(EquityPot { amount, eligible });
    }

    if dead_pot_left > 0.0 {
        pots.push(EquityPot {
            amount: dead_pot_left,
            eligible: (0..commitments.len()).collect(),
        });
    }

    pots
}

fn showdown_stacks_from_payouts(
    stacks: &[u32],
    participants: &[usize],
    commitments: &[(usize, u32)],
    payouts: &[f64],
) -> Vec<u32> {
    let mut next = stacks.to_vec();

    for &(seat, amount) in commitments {
        next[seat] = next[seat].saturating_sub(amount);
    }

    for (&seat, payout) in participants.iter().zip(payouts) {
        next[seat] = next[seat].saturating_add(payout.round() as u32);
    }

    next
}

fn first_response_required_equity(
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    opener_seat: usize,
    opener_cost: u32,
) -> f64 {
    if opener_seat + 1 >= stacks.len() {
        return 1.0;
    }

    let posted = posted_stacks(config.level, stacks);
    let caller_seat = caller_seat(opener_seat, stacks.len());
    let caller_cost = stacks[caller_seat].saturating_sub(posted_amount(
        config.level,
        stacks.len() as u8,
        caller_seat as u8,
    ));
    let effective_cost = opener_cost.min(caller_cost);

    call_required_equity_for(
        config,
        &posted,
        dead_pot,
        caller_seat,
        opener_seat,
        effective_cost,
    )
}

fn call_required_equity_for(
    config: &ShortStackConfig,
    posted_stacks: &[u32],
    dead_pot: u32,
    caller_seat: usize,
    opener_seat: usize,
    effective_cost: u32,
) -> f64 {
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };
    let fold_value = state_value(
        clock,
        config.hand_duration_seconds,
        &steal_stacks(posted_stacks, opener_seat, dead_pot),
        caller_seat,
    );
    let win_value = state_value(
        clock,
        config.hand_duration_seconds,
        &call_win_stacks(
            posted_stacks,
            caller_seat,
            opener_seat,
            dead_pot,
            effective_cost,
        ),
        caller_seat,
    );
    let lose_value = state_value(
        clock,
        config.hand_duration_seconds,
        &call_lose_stacks(
            posted_stacks,
            caller_seat,
            opener_seat,
            dead_pot,
            effective_cost,
        ),
        caller_seat,
    );

    state_required_equity(fold_value, win_value, lose_value)
}

struct ShoveResponseSolution {
    shove_range: Vec<HandResult>,
    call_range: Vec<HandClass>,
    iterations_run: usize,
    converged: bool,
}

fn solve_shove_response(
    classes: &[HandClass],
    response_required_equity: f64,
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    all_in_cost: u32,
    hero_seat: usize,
    max_iterations: usize,
    cache: &mut EquityCache,
) -> ShoveResponseSolution {
    let mut call_range = top_fraction_by_heuristic(classes, 0.25);
    let mut iterations_run = 0;
    let mut converged = false;

    for _ in 0..max_iterations {
        let shove_range = profitable_shove_range(
            classes,
            &call_range,
            config,
            stacks,
            dead_pot,
            all_in_cost,
            hero_seat,
            cache,
        );
        let next_call_range = profitable_call_range(
            classes,
            &shove_range,
            response_required_equity,
            config.max_boards_per_combo,
            config.range_sample_limit,
            cache,
        )
        .into_iter()
        .map(|result| result.hand)
        .collect();

        iterations_run += 1;
        if next_call_range == call_range {
            call_range = next_call_range;
            converged = true;
            break;
        }
        call_range = next_call_range;
    }

    let shove_range = profitable_shove_results(
        classes,
        &call_range,
        config,
        stacks,
        dead_pot,
        all_in_cost,
        hero_seat,
        cache,
    );

    ShoveResponseSolution {
        shove_range,
        call_range,
        iterations_run,
        converged,
    }
}

fn profitable_shove_range(
    classes: &[HandClass],
    call_range: &[HandClass],
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    all_in_cost: u32,
    hero_seat: usize,
    cache: &mut EquityCache,
) -> Vec<HandClass> {
    profitable_shove_results(
        classes,
        call_range,
        config,
        stacks,
        dead_pot,
        all_in_cost,
        hero_seat,
        cache,
    )
    .into_iter()
    .map(|result| result.hand)
    .collect()
}

fn profitable_shove_results(
    classes: &[HandClass],
    call_range: &[HandClass],
    config: &ShortStackConfig,
    stacks: &[u32],
    dead_pot: u32,
    all_in_cost: u32,
    hero_seat: usize,
    _cache: &mut EquityCache,
) -> Vec<HandResult> {
    let call_probability = combo_fraction(call_range);
    let sampled_call_range = sample_range(call_range, config.range_sample_limit);
    let fold_probability = (1.0 - call_probability).powi(config.players_behind as i32);
    let called_probability = 1.0 - fold_probability;
    let posted_stacks = posted_stacks(config.level, stacks);
    let caller_seat = caller_seat(hero_seat, stacks.len());
    let caller_posted = posted_amount(config.level, stacks.len() as u8, caller_seat as u8);
    let caller_cost = stacks[caller_seat].saturating_sub(caller_posted);
    let effective_cost = all_in_cost.min(caller_cost);
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };
    let hand_duration_seconds = config.hand_duration_seconds;
    let fold_value = state_value(clock, hand_duration_seconds, &posted_stacks, hero_seat);
    let steal_value = state_value(
        clock,
        hand_duration_seconds,
        &steal_stacks(&posted_stacks, hero_seat, dead_pot),
        hero_seat,
    );
    let win_value = state_value(
        clock,
        hand_duration_seconds,
        &call_win_stacks(
            &posted_stacks,
            hero_seat,
            caller_seat,
            dead_pot,
            effective_cost,
        ),
        hero_seat,
    );
    let lose_value = state_value(
        clock,
        hand_duration_seconds,
        &call_lose_stacks(
            &posted_stacks,
            hero_seat,
            caller_seat,
            dead_pot,
            effective_cost,
        ),
        hero_seat,
    );

    if sampled_call_range.is_empty() || called_probability == 0.0 {
        let ev = steal_value - fold_value;
        if ev >= 0.0 {
            return classes
                .iter()
                .copied()
                .map(|hand| HandResult {
                    hand,
                    equity: 0.0,
                    ev,
                })
                .collect();
        }
        return Vec::new();
    }

    let mut results: Vec<_> = classes
        .par_iter()
        .copied()
        .filter_map(|hand| {
            let mut cache = EquityCache::new();
            let equity = heads_up_equity_vs_range_cached(
                hand,
                &sampled_call_range,
                config.max_boards_per_combo,
                &mut cache,
            )
            .share();
            let shove_value = fold_probability * steal_value
                + called_probability * (equity * win_value + (1.0 - equity) * lose_value);
            let ev = shove_value - fold_value;

            if ev >= 0.0 {
                Some(HandResult { hand, equity, ev })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|left, right| {
        right
            .ev
            .total_cmp(&left.ev)
            .then_with(|| right.equity.total_cmp(&left.equity))
    });
    results
}

fn profitable_call_range(
    classes: &[HandClass],
    shove_range: &[HandClass],
    required_equity: f64,
    max_boards_per_combo: usize,
    range_sample_limit: usize,
    _cache: &mut EquityCache,
) -> Vec<HandResult> {
    let sampled_shove_range = sample_range(shove_range, range_sample_limit);

    if sampled_shove_range.is_empty() {
        return Vec::new();
    }

    if required_equity <= 0.0 {
        return classes
            .iter()
            .copied()
            .map(|hand| HandResult {
                hand,
                equity: 0.0,
                ev: 0.0,
            })
            .collect();
    }

    if required_equity >= 1.0 {
        return Vec::new();
    }

    let mut results: Vec<_> = classes
        .par_iter()
        .copied()
        .filter_map(|hand| {
            let mut cache = EquityCache::new();
            let equity = heads_up_equity_vs_range_cached(
                hand,
                &sampled_shove_range,
                max_boards_per_combo,
                &mut cache,
            )
            .share();
            let ev = equity - required_equity;

            if ev >= 0.0 {
                Some(HandResult { hand, equity, ev })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|left, right| right.equity.total_cmp(&left.equity));
    results
}

fn profitable_overcall_range(
    classes: &[HandClass],
    shove_range: &[HandClass],
    call_range: &[HandClass],
    required_equity: f64,
    max_boards_per_combo: usize,
    range_sample_limit: usize,
    _cache: &mut EquityCache,
) -> Vec<HandResult> {
    let candidate_classes = top_fraction_by_heuristic(classes, 0.2);
    let sampled_shove_range = sample_range(shove_range, range_sample_limit);
    let sampled_call_range = sample_range(call_range, range_sample_limit);

    let mut results: Vec<_> = candidate_classes
        .into_par_iter()
        .filter_map(|hand| {
            let mut cache = EquityCache::new();
            let equity = three_way_equity_vs_ranges_cached(
                hand,
                &sampled_shove_range,
                &sampled_call_range,
                max_boards_per_combo,
                &mut cache,
            )
            .share();
            let ev = equity - required_equity;

            if ev >= 0.0 {
                Some(HandResult { hand, equity, ev })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|left, right| right.equity.total_cmp(&left.equity));
    results
}

fn top_fraction_by_heuristic(classes: &[HandClass], fraction: f64) -> Vec<HandClass> {
    let mut scored: Vec<_> = classes
        .iter()
        .map(|&hand| (hand, heuristic_strength(hand)))
        .collect();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let target_combos = (1326.0 * fraction).round() as usize;
    let mut selected = Vec::new();
    let mut combos = 0;

    for (hand, _) in scored {
        if combos >= target_combos {
            break;
        }
        combos += hand.combos().len();
        selected.push(hand);
    }

    selected
}

fn sample_range(range: &[HandClass], limit: usize) -> Vec<HandClass> {
    if limit == 0 || range.len() <= limit {
        return range.to_vec();
    }

    (0..limit)
        .map(|index| {
            let sampled_index = (index * range.len() + range.len() / (limit * 2)) / limit;
            range[sampled_index.min(range.len() - 1)]
        })
        .collect()
}

fn heuristic_strength(hand: HandClass) -> f64 {
    if hand.high == hand.low {
        return 100.0 + hand.high as f64 * 4.0;
    }

    let high = hand.high as f64;
    let low = hand.low as f64;
    let gap = hand.high - hand.low - 1;

    high * 4.0 + low * 1.5 + if hand.suited { 3.0 } else { 0.0 } - gap as f64 * 2.0
}

fn combo_fraction(range: &[HandClass]) -> f64 {
    range.iter().map(|hand| hand.combos().len()).sum::<usize>() as f64 / 1326.0
}

fn normalized_stacks(config: &ShortStackConfig) -> Vec<u32> {
    if config.stacks.is_empty() {
        vec![config.stack; config.alive_players as usize]
    } else {
        config.stacks.clone()
    }
}

fn posted_amount(level: BlindLevel, alive_players: u8, seat_index: u8) -> u32 {
    let ante = level.per_player_ante;
    let small_blind_seat = alive_players.saturating_sub(2);
    let big_blind_seat = alive_players.saturating_sub(1);

    if seat_index == big_blind_seat {
        ante.saturating_add(level.big_blind)
    } else if seat_index == small_blind_seat {
        ante.saturating_add(level.small_blind)
    } else {
        ante
    }
}

fn posted_stacks(level: BlindLevel, stacks: &[u32]) -> Vec<u32> {
    stacks
        .iter()
        .enumerate()
        .map(|(seat, &stack)| {
            stack.saturating_sub(posted_amount(level, stacks.len() as u8, seat as u8))
        })
        .collect()
}

fn state_required_equity(fold_value: f64, win_value: f64, lose_value: f64) -> f64 {
    if win_value <= lose_value {
        1.0
    } else {
        ((fold_value - lose_value) / (win_value - lose_value)).clamp(0.0, 1.0)
    }
}

fn state_value(
    clock: BlindClock,
    hand_duration_seconds: u32,
    stacks: &[u32],
    hero_seat: usize,
) -> f64 {
    death_race_value(
        &DeathRaceState {
            clock,
            stacks: stacks.to_vec(),
            next_small_blind_seat: stacks.len().saturating_sub(2),
            hand_duration_seconds,
        },
        hero_seat,
    )
}

fn steal_stacks(stacks: &[u32], hero_seat: usize, dead_pot: u32) -> Vec<u32> {
    let mut next = stacks.to_vec();
    next[hero_seat] = next[hero_seat].saturating_add(dead_pot);
    next
}

fn call_win_stacks(
    stacks: &[u32],
    hero_seat: usize,
    opponent_seat: usize,
    dead_pot: u32,
    all_in_cost: u32,
) -> Vec<u32> {
    let mut next = stacks.to_vec();
    next[hero_seat] = next[hero_seat]
        .saturating_add(dead_pot)
        .saturating_add(all_in_cost);
    next[opponent_seat] = next[opponent_seat].saturating_sub(all_in_cost);
    next
}

fn call_lose_stacks(
    stacks: &[u32],
    hero_seat: usize,
    opponent_seat: usize,
    dead_pot: u32,
    all_in_cost: u32,
) -> Vec<u32> {
    let mut next = stacks.to_vec();
    next[hero_seat] = next[hero_seat].saturating_sub(all_in_cost);
    next[opponent_seat] = next[opponent_seat]
        .saturating_add(dead_pot)
        .saturating_add(all_in_cost);
    next
}

fn caller_seat(hero_seat: usize, player_count: usize) -> usize {
    (hero_seat + 1).min(player_count.saturating_sub(1))
}

fn opener_seat(seat_index: u8) -> usize {
    seat_index.saturating_sub(1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_initial_range_prefers_premium_hands() {
        let classes = all_hand_classes();
        let range = top_fraction_by_heuristic(&classes, 0.01);

        assert!(range.iter().any(|hand| hand.label() == "AA"));
        assert!(range.iter().any(|hand| hand.label() == "KK"));
    }

    #[test]
    fn report_contains_ranges_for_each_alive_player() {
        let report = analyze_short_stack(&ShortStackConfig {
            level: crate::blinds::blind_level(11).expect("level 11 exists"),
            alive_players: 3,
            stack: 40_000,
            stacks: vec![40_000; 3],
            players_behind: 2,
            elapsed_in_level_seconds: 0,
            hand_duration_seconds: 20,
            max_boards_per_combo: 1,
            range_sample_limit: 1,
            iterations: 0,
            spot_iterations: 0,
            include_overcall: false,
        });

        assert_eq!(report.seats.len(), 3);
        assert_eq!(report.seats[0].players_behind, 2);
        assert_eq!(report.seats[1].players_behind, 1);
        assert_eq!(report.seats[2].players_behind, 0);
    }

    #[test]
    fn posted_amount_accounts_for_blinds_and_antes() {
        let level = crate::blinds::blind_level(11).expect("level 11 exists");

        assert_eq!(posted_amount(level, 6, 0), 2_200);
        assert_eq!(posted_amount(level, 6, 4), 6_500);
        assert_eq!(posted_amount(level, 6, 5), 10_800);
    }

    #[test]
    fn side_pots_exclude_uncalled_all_in_excess() {
        let commitments = vec![(0, 50), (1, 50), (2, 20)];
        let pots = side_pots(10, &commitments);

        assert_eq!(pots.len(), 2);
        assert_eq!(pots[0].amount, 70.0);
        assert_eq!(pots[0].eligible, vec![0, 1, 2]);
        assert_eq!(pots[1].amount, 60.0);
        assert_eq!(pots[1].eligible, vec![0, 1]);
    }

    #[test]
    fn all_in_commitments_cap_the_covering_stack_at_the_largest_called_amount() {
        let level = BlindLevel {
            level: 1,
            big_blind: 0,
            small_blind: 0,
            per_player_ante: 0,
        };
        let commitments = all_in_commitments(level, &[100, 50, 20], &[0, 1, 2]);

        assert_eq!(commitments, vec![(0, 50), (1, 50), (2, 20)]);
    }

    #[test]
    fn state_required_equity_charges_fold_value() {
        let level = crate::blinds::blind_level(11).expect("level 11 exists");
        let posted = posted_amount(level, 6, 5);
        let stacks = vec![40_000; 6];
        let posted_stacks = posted_stacks(level, &stacks);
        let clock = BlindClock {
            current_level: 11,
            elapsed_in_level_seconds: 0,
        };
        let fold_value = state_value(clock, 20, &posted_stacks, 5);
        let all_in_cost = 40_000 - posted;
        let dead_pot = level.small_blind + level.big_blind + level.per_player_ante * 6;
        let win_value = state_value(
            clock,
            20,
            &call_win_stacks(&posted_stacks, 5, 4, dead_pot, all_in_cost),
            5,
        );
        let lose_value = state_value(
            clock,
            20,
            &call_lose_stacks(&posted_stacks, 5, 4, dead_pot, all_in_cost),
            5,
        );

        assert!((0.0..=1.0).contains(&state_required_equity(fold_value, win_value, lose_value)));
    }

    #[test]
    fn state_value_rewards_outlasting_an_opponent() {
        let level = crate::blinds::blind_level(11).expect("level 11 exists");
        let clock = BlindClock {
            current_level: level.level,
            elapsed_in_level_seconds: 0,
        };
        let hero_safe = state_value(clock, 20, &[1, 40_000], 1);
        let hero_dead = state_value(clock, 20, &[40_000, 1], 1);

        assert!(hero_safe > hero_dead);
    }

    #[test]
    fn covered_all_in_loss_does_not_eliminate_covering_player() {
        let stacks = vec![80_000, 20_000];
        let next = call_lose_stacks(&stacks, 0, 1, 10_000, 20_000);

        assert_eq!(next[0], 60_000);
        assert_eq!(next[1], 50_000);
    }
}
