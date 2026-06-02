use crate::blinds::BlindLevel;
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
    pub players_behind: u8,
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
    pub shove_range: Vec<HandResult>,
    pub call_range: Vec<HandResult>,
    pub overcall_range: Vec<HandResult>,
}

#[derive(Debug, Clone)]
pub struct HandResult {
    pub hand: HandClass,
    pub equity: f64,
    pub ev: f64,
}

pub fn analyze_short_stack(config: &ShortStackConfig) -> ShortStackReport {
    let dead_pot = config.level.small_blind
        + config.level.big_blind
        + config
            .level
            .per_player_ante
            .saturating_mul(config.alive_players as u32);
    let single_call_required_equity =
        config.stack as f64 / (dead_pot + config.stack.saturating_mul(2)) as f64;
    let overcall_required_equity =
        config.stack as f64 / (dead_pot + config.stack.saturating_mul(3)) as f64;

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
                &classes,
                dead_pot,
                single_call_required_equity,
                overcall_required_equity,
                seat_index,
                players_behind,
                &mut cache,
            )
        })
        .collect();

    ShortStackReport {
        dead_pot,
        stack_in_big_blinds: config.stack as f64 / config.level.big_blind as f64,
        orbit_cost: orbit_cost(config.level, config.alive_players),
        single_call_required_equity,
        overcall_required_equity,
        seats,
    }
}

fn analyze_seat(
    config: &ShortStackConfig,
    classes: &[HandClass],
    dead_pot: u32,
    single_call_required_equity: f64,
    overcall_required_equity: f64,
    seat_index: u8,
    players_behind: u8,
    cache: &mut EquityCache,
) -> SeatRanges {
    let seat_config = ShortStackConfig {
        players_behind,
        ..config.clone()
    };
    let mut call_range = top_fraction_by_heuristic(classes, 0.25);

    for _ in 0..config.iterations {
        let shove_range =
            profitable_shove_range(classes, &call_range, &seat_config, dead_pot, cache);
        call_range = profitable_call_range(
            classes,
            &shove_range,
            single_call_required_equity,
            config.max_boards_per_combo,
            cache,
        )
        .into_iter()
        .map(|result| result.hand)
        .collect();
    }

    let modeled_shove_range =
        profitable_shove_results(classes, &call_range, &seat_config, dead_pot, cache);
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
        single_call_required_equity,
        config.max_boards_per_combo,
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
        shove_range: displayed_shove_range,
        call_range: call_range_results,
        overcall_range,
    }
}

fn profitable_shove_range(
    classes: &[HandClass],
    call_range: &[HandClass],
    config: &ShortStackConfig,
    dead_pot: u32,
    cache: &mut EquityCache,
) -> Vec<HandClass> {
    profitable_shove_results(classes, call_range, config, dead_pot, cache)
        .into_iter()
        .map(|result| result.hand)
        .collect()
}

fn profitable_shove_results(
    classes: &[HandClass],
    call_range: &[HandClass],
    config: &ShortStackConfig,
    dead_pot: u32,
    cache: &mut EquityCache,
) -> Vec<HandResult> {
    let call_probability = combo_fraction(call_range);
    let sampled_call_range = sample_range(call_range, config.range_sample_limit);
    let fold_probability = (1.0 - call_probability).powi(config.players_behind as i32);
    let called_probability = 1.0 - fold_probability;
    let called_pot = (dead_pot + config.stack.saturating_mul(2)) as f64;
    let risk = config.stack as f64;

    let mut results = Vec::new();

    for &hand in classes {
        let equity = heads_up_equity_vs_range_cached(
            hand,
            &sampled_call_range,
            config.max_boards_per_combo,
            cache,
        )
        .share();
        let ev =
            fold_probability * dead_pot as f64 + called_probability * (equity * called_pot - risk);

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

    range.iter().take(limit).copied().collect()
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
            players_behind: 2,
            max_boards_per_combo: 1,
            range_sample_limit: 1,
            iterations: 0,
        });

        assert_eq!(report.seats.len(), 3);
        assert_eq!(report.seats[0].players_behind, 2);
        assert_eq!(report.seats[1].players_behind, 1);
        assert_eq!(report.seats[2].players_behind, 0);
    }
}
