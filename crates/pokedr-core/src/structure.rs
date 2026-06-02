use crate::blinds::BlindLevel;

pub const STARTING_STACK: u32 = 20_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlindPressure {
    pub level: u8,
    pub stack_in_big_blinds: f64,
    pub orbit_cost: u32,
    pub orbit_cost_as_stack_fraction: f64,
    pub stack_after_one_folded_orbit: u32,
    pub stack_after_one_folded_orbit_in_big_blinds: f64,
}

pub fn blind_pressure(level: BlindLevel, alive_players: u8, stack: u32) -> BlindPressure {
    let orbit_cost = orbit_cost(level, alive_players);
    let stack_after_one_folded_orbit = stack.saturating_sub(orbit_cost);

    BlindPressure {
        level: level.level,
        stack_in_big_blinds: stack as f64 / level.big_blind as f64,
        orbit_cost,
        orbit_cost_as_stack_fraction: orbit_cost as f64 / stack as f64,
        stack_after_one_folded_orbit,
        stack_after_one_folded_orbit_in_big_blinds: stack_after_one_folded_orbit as f64
            / level.big_blind as f64,
    }
}

pub fn orbit_cost(level: BlindLevel, alive_players: u8) -> u32 {
    level
        .per_player_ante
        .saturating_mul(alive_players as u32)
        .saturating_add(level.small_blind)
        .saturating_add(level.big_blind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blinds::blind_level;

    #[test]
    fn orbit_cost_uses_per_player_ante() {
        let level = blind_level(1).expect("level 1 exists");

        assert_eq!(orbit_cost(level, 6), 600);
    }

    #[test]
    fn level_eight_pressure_is_the_new_transition_point() {
        let pressure = blind_pressure(blind_level(8).expect("level 8 exists"), 6, STARTING_STACK);

        assert!((pressure.stack_in_big_blinds - 8.0).abs() < 0.001);
        assert_eq!(pressure.orbit_cost, 7_530);
        assert!((pressure.orbit_cost_as_stack_fraction - 0.3765).abs() < 0.001);
        assert_eq!(pressure.stack_after_one_folded_orbit, 12_470);
        assert!((pressure.stack_after_one_folded_orbit_in_big_blinds - 4.988).abs() < 0.001);
    }

    #[test]
    fn level_nine_pressure_is_already_short_from_starting_stack() {
        let pressure = blind_pressure(blind_level(9).expect("level 9 exists"), 6, STARTING_STACK);

        assert!((pressure.stack_in_big_blinds - 5.263).abs() < 0.001);
        assert_eq!(pressure.orbit_cost, 11_400);
        assert!((pressure.orbit_cost_as_stack_fraction - 0.57).abs() < 0.001);
        assert_eq!(pressure.stack_after_one_folded_orbit, 8_600);
        assert!((pressure.stack_after_one_folded_orbit_in_big_blinds - 2.263).abs() < 0.001);
    }

    #[test]
    fn level_ten_pressure_is_push_fold_like_from_starting_stack() {
        let pressure = blind_pressure(blind_level(10).expect("level 10 exists"), 6, STARTING_STACK);

        assert!((pressure.stack_in_big_blinds - 3.509).abs() < 0.001);
        assert_eq!(pressure.orbit_cost, 16_950);
        assert!((pressure.orbit_cost_as_stack_fraction - 0.8475).abs() < 0.001);
        assert_eq!(pressure.stack_after_one_folded_orbit, 3_050);
        assert!((pressure.stack_after_one_folded_orbit_in_big_blinds - 0.535).abs() < 0.001);
    }
}
