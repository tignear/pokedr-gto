use crate::blinds::{BlindClock, BlindLevel};
use crate::death_race::{DeathRaceState, death_race_value};
use crate::equity::{
    EquityCache, heads_up_equity_vs_range_cached, three_way_equity_vs_ranges_cached,
};
use crate::hand_class::{HandClass, all_hand_classes};
use crate::structure::orbit_cost;

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
}

#[derive(Debug, Clone)]
pub struct ShortStackReport {
    pub dead_pot: u32,
    pub stack_in_big_blinds: f64,
    pub orbit_cost: u32,
    pub single_call_required_equity: f64,
    pub overcall_required_equity: f64,
    pub seats: Vec<SeatRanges>,
}

#[derive(Debug, Clone)]
pub struct SeatRanges {
    pub seat_index: u8,
    pub players_behind: u8,
    pub posted_amount: u32,
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
    pub required_equity: f64,
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
    let seats = (0..config.alive_players)
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

    ShortStackReport {
        dead_pot,
        stack_in_big_blinds: report_stack as f64 / config.level.big_blind as f64,
        orbit_cost: orbit_cost(config.level, config.alive_players),
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
    let mut call_range = top_fraction_by_heuristic(classes, 0.25);

    for _ in 0..config.iterations {
        let shove_range = profitable_shove_range(
            classes,
            &call_range,
            &seat_config,
            stacks,
            dead_pot,
            all_in_cost,
            seat_index as usize,
            cache,
        );
        call_range = profitable_call_range(
            classes,
            &shove_range,
            call_required_equity,
            config.max_boards_per_combo,
            cache,
        )
        .into_iter()
        .map(|result| result.hand)
        .collect();
    }

    let modeled_shove_range = profitable_shove_results(
        classes,
        &call_range,
        &seat_config,
        stacks,
        dead_pot,
        all_in_cost,
        seat_index as usize,
        cache,
    );
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
    let overcall_range = profitable_overcall_range(
        classes,
        &modeled_shove_range
            .iter()
            .map(|result| result.hand)
            .collect::<Vec<_>>(),
        &call_range,
        overcall_required_equity,
        config.max_boards_per_combo,
        config.range_sample_limit,
        cache,
    );

    SeatRanges {
        seat_index,
        players_behind,
        posted_amount,
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
    let clock = BlindClock {
        current_level: config.level.level,
        elapsed_in_level_seconds: config.elapsed_in_level_seconds,
    };

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
        let baseline_call_range = top_fraction_by_heuristic(classes, 0.25);
        let opener_range = profitable_shove_range(
            classes,
            &baseline_call_range,
            &opener_config,
            stacks,
            dead_pot,
            opener_cost,
            opener_seat,
            cache,
        );

        let fold_value = state_value(clock, config.hand_duration_seconds, &posted, caller_seat);
        let win_value = state_value(
            clock,
            config.hand_duration_seconds,
            &call_win_stacks(&posted, caller_seat, opener_seat, dead_pot, effective_cost),
            caller_seat,
        );
        let lose_value = state_value(
            clock,
            config.hand_duration_seconds,
            &call_lose_stacks(&posted, caller_seat, opener_seat, dead_pot, effective_cost),
            caller_seat,
        );
        let required_equity = state_required_equity(fold_value, win_value, lose_value);
        let range = profitable_call_range(
            classes,
            &opener_range,
            required_equity,
            config.max_boards_per_combo,
            cache,
        );

        spots.push(CallSpot {
            opener_seat: opener_seat as u8,
            effective_all_in_cost: effective_cost,
            required_equity,
            range,
        });
    }

    spots
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
    cache: &mut EquityCache,
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

    let mut results = Vec::new();

    for &hand in classes {
        let equity = heads_up_equity_vs_range_cached(
            hand,
            &sampled_call_range,
            config.max_boards_per_combo,
            cache,
        )
        .share();
        let shove_value = fold_probability * steal_value
            + called_probability * (equity * win_value + (1.0 - equity) * lose_value);
        let ev = shove_value - fold_value;

        if ev >= 0.0 {
            results.push(HandResult { hand, equity, ev });
        }
    }

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
    cache: &mut EquityCache,
) -> Vec<HandResult> {
    let mut results = Vec::new();
    let sampled_shove_range = sample_range(shove_range, 32);

    for &hand in classes {
        let equity = heads_up_equity_vs_range_cached(
            hand,
            &sampled_shove_range,
            max_boards_per_combo,
            cache,
        )
        .share();
        let ev = equity - required_equity;

        if ev >= 0.0 {
            results.push(HandResult { hand, equity, ev });
        }
    }

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
    cache: &mut EquityCache,
) -> Vec<HandResult> {
    let mut results = Vec::new();
    let candidate_classes = top_fraction_by_heuristic(classes, 0.2);
    let sampled_shove_range = sample_range(shove_range, range_sample_limit);
    let sampled_call_range = sample_range(call_range, range_sample_limit);

    for hand in candidate_classes {
        let equity = three_way_equity_vs_ranges_cached(
            hand,
            &sampled_shove_range,
            &sampled_call_range,
            max_boards_per_combo,
            cache,
        )
        .share();
        let ev = equity - required_equity;

        if ev >= 0.0 {
            results.push(HandResult { hand, equity, ev });
        }
    }

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
