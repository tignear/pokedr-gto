use crate::blinds::BlindClock;
use crate::scoring::rank_points;

#[derive(Debug, Clone)]
pub struct DeathRaceState {
    pub clock: BlindClock,
    pub stacks: Vec<u32>,
    pub next_small_blind_seat: usize,
    pub hand_duration_seconds: u32,
}

pub fn death_race_value(state: &DeathRaceState, hero_seat: usize) -> f64 {
    if hero_seat >= state.stacks.len() {
        return 0.0;
    }

    let mut stacks = state.stacks.clone();
    let mut next_small_blind_seat = state.next_small_blind_seat;
    let mut clock = state.clock;
    let mut alive = alive_count(&stacks);

    if stacks[hero_seat] == 0 {
        return rank_points(alive.saturating_add(1).max(1) as u8).unwrap_or(0) as f64;
    }

    while alive > 1 {
        let active = active_seats(&stacks);
        if active.len() <= 1 {
            break;
        }

        let level = clock.level();
        let small_blind_seat = next_active_seat(&active, next_small_blind_seat);
        let big_blind_seat = next_active_seat(&active, small_blind_seat + 1);
        let mut deaths = Vec::new();

        for &seat in &active {
            let mut cost = level.per_player_ante;
            if seat == small_blind_seat {
                cost = cost.saturating_add(level.small_blind);
            }
            if seat == big_blind_seat {
                cost = cost.saturating_add(level.big_blind);
            }

            stacks[seat] = stacks[seat].saturating_sub(cost);
            if stacks[seat] == 0 {
                deaths.push(seat);
            }
        }

        deaths.sort_unstable();
        for seat in deaths {
            let rank = alive as u8;
            if seat == hero_seat {
                return rank_points(rank).unwrap_or(0) as f64;
            }
            alive -= 1;
        }

        next_small_blind_seat = big_blind_seat + 1;
        clock = clock.next_hand_after(state.hand_duration_seconds);
    }

    rank_points(1).unwrap_or(0) as f64
}

fn active_seats(stacks: &[u32]) -> Vec<usize> {
    stacks
        .iter()
        .enumerate()
        .filter_map(|(seat, &stack)| (stack > 0).then_some(seat))
        .collect()
}

fn alive_count(stacks: &[u32]) -> usize {
    stacks.iter().filter(|&&stack| stack > 0).count()
}

fn next_active_seat(active: &[usize], start: usize) -> usize {
    active
        .iter()
        .copied()
        .filter(|&seat| seat >= start)
        .min()
        .unwrap_or(active[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_stack_that_dies_first_gets_last_place_points() {
        let state = DeathRaceState {
            clock: BlindClock {
                current_level: 11,
                elapsed_in_level_seconds: 0,
            },
            stacks: vec![1, 40_000, 40_000, 40_000, 40_000, 40_000],
            next_small_blind_seat: 4,
            hand_duration_seconds: 20,
        };

        assert_eq!(death_race_value(&state, 0), -40.0);
    }

    #[test]
    fn already_dead_hero_gets_rank_after_living_players() {
        let state = DeathRaceState {
            clock: BlindClock {
                current_level: 11,
                elapsed_in_level_seconds: 0,
            },
            stacks: vec![0, 40_000, 40_000, 40_000, 40_000, 40_000],
            next_small_blind_seat: 4,
            hand_duration_seconds: 20,
        };

        assert_eq!(death_race_value(&state, 0), -40.0);
    }

    #[test]
    fn last_survivor_gets_first_place_points() {
        let state = DeathRaceState {
            clock: BlindClock {
                current_level: 11,
                elapsed_in_level_seconds: 0,
            },
            stacks: vec![1, 1, 1, 1, 1, 40_000],
            next_small_blind_seat: 4,
            hand_duration_seconds: 20,
        };

        assert_eq!(death_race_value(&state, 5), 40.0);
    }

    #[test]
    fn death_race_value_stays_in_rank_point_range() {
        let state = DeathRaceState {
            clock: BlindClock {
                current_level: 10,
                elapsed_in_level_seconds: 239,
            },
            stacks: vec![100_000, 100_000],
            next_small_blind_seat: 0,
            hand_duration_seconds: 1,
        };
        let value = death_race_value(&state, 0);

        assert!((-40.0..=40.0).contains(&value));
    }
}
