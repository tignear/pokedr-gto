#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindLevel {
    pub level: u8,
    pub big_blind: u32,
    pub small_blind: u32,
    pub per_player_ante: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindClock {
    pub current_level: u8,
    pub elapsed_in_level_seconds: u32,
}

pub const LEVEL_DURATION_SECONDS: u32 = 180;

pub const BLIND_LEVELS: [BlindLevel; 16] = [
    BlindLevel {
        level: 1,
        big_blind: 200,
        small_blind: 100,
        per_player_ante: 50,
    },
    BlindLevel {
        level: 2,
        big_blind: 280,
        small_blind: 140,
        per_player_ante: 70,
    },
    BlindLevel {
        level: 3,
        big_blind: 400,
        small_blind: 200,
        per_player_ante: 100,
    },
    BlindLevel {
        level: 4,
        big_blind: 560,
        small_blind: 280,
        per_player_ante: 140,
    },
    BlindLevel {
        level: 5,
        big_blind: 780,
        small_blind: 390,
        per_player_ante: 200,
    },
    BlindLevel {
        level: 6,
        big_blind: 1_100,
        small_blind: 550,
        per_player_ante: 280,
    },
    BlindLevel {
        level: 7,
        big_blind: 1_640,
        small_blind: 820,
        per_player_ante: 410,
    },
    BlindLevel {
        level: 8,
        big_blind: 2_500,
        small_blind: 1_250,
        per_player_ante: 630,
    },
    BlindLevel {
        level: 9,
        big_blind: 3_800,
        small_blind: 1_900,
        per_player_ante: 950,
    },
    BlindLevel {
        level: 10,
        big_blind: 5_700,
        small_blind: 2_850,
        per_player_ante: 1_400,
    },
    BlindLevel {
        level: 11,
        big_blind: 8_600,
        small_blind: 4_300,
        per_player_ante: 2_200,
    },
    BlindLevel {
        level: 12,
        big_blind: 13_000,
        small_blind: 6_500,
        per_player_ante: 3_200,
    },
    BlindLevel {
        level: 13,
        big_blind: 19_600,
        small_blind: 9_800,
        per_player_ante: 4_900,
    },
    BlindLevel {
        level: 14,
        big_blind: 29_500,
        small_blind: 14_750,
        per_player_ante: 7_400,
    },
    BlindLevel {
        level: 15,
        big_blind: 44_300,
        small_blind: 22_150,
        per_player_ante: 11_000,
    },
    BlindLevel {
        level: 16,
        big_blind: 60_000,
        small_blind: 30_000,
        per_player_ante: 15_000,
    },
];

pub fn blind_level(level: u8) -> Option<BlindLevel> {
    let index = level.checked_sub(1)? as usize;
    BLIND_LEVELS.get(index).copied()
}

impl BlindClock {
    pub fn new() -> Self {
        Self {
            current_level: 1,
            elapsed_in_level_seconds: 0,
        }
    }

    pub fn level(self) -> BlindLevel {
        blind_level(self.current_level)
            .unwrap_or(*BLIND_LEVELS.last().expect("blind levels must not be empty"))
    }

    pub fn seconds_until_level_up(self) -> u32 {
        LEVEL_DURATION_SECONDS.saturating_sub(self.elapsed_in_level_seconds)
    }

    pub fn next_hand_after(self, hand_duration_seconds: u32) -> Self {
        let elapsed_in_level_seconds = self
            .elapsed_in_level_seconds
            .saturating_add(hand_duration_seconds);

        if elapsed_in_level_seconds >= LEVEL_DURATION_SECONDS {
            Self {
                current_level: self.next_level(),
                elapsed_in_level_seconds: 0,
            }
        } else {
            Self {
                current_level: self.current_level,
                elapsed_in_level_seconds,
            }
        }
    }

    fn next_level(self) -> u8 {
        let last_level = BLIND_LEVELS
            .last()
            .expect("blind levels must not be empty")
            .level;

        self.current_level.saturating_add(1).min(last_level)
    }
}

impl Default for BlindClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blind_schedule_matches_table() {
        assert_eq!(
            blind_level(1),
            Some(BlindLevel {
                level: 1,
                big_blind: 200,
                small_blind: 100,
                per_player_ante: 50,
            })
        );
        assert_eq!(
            blind_level(10),
            Some(BlindLevel {
                level: 10,
                big_blind: 5_700,
                small_blind: 2_850,
                per_player_ante: 1_400,
            })
        );
        assert_eq!(
            blind_level(16),
            Some(BlindLevel {
                level: 16,
                big_blind: 60_000,
                small_blind: 30_000,
                per_player_ante: 15_000,
            })
        );
    }

    #[test]
    fn invalid_blind_level_is_none() {
        assert_eq!(blind_level(0), None);
        assert_eq!(blind_level(17), None);
    }

    #[test]
    fn blind_clock_advances_only_between_hands() {
        let clock = BlindClock::new();
        assert_eq!(clock.level().level, 1);

        let clock = clock.next_hand_after(179);
        assert_eq!(clock.level().level, 1);
        assert_eq!(clock.elapsed_in_level_seconds, 179);

        let clock = clock.next_hand_after(1);
        assert_eq!(clock.level().level, 2);
        assert_eq!(clock.elapsed_in_level_seconds, 0);

        let clock = clock.next_hand_after(179);
        assert_eq!(clock.level().level, 2);
        assert_eq!(clock.elapsed_in_level_seconds, 179);
    }

    #[test]
    fn new_level_gets_a_fresh_three_minute_clock() {
        let clock = BlindClock::new().next_hand_after(180);
        assert_eq!(clock.level().level, 2);
        assert_eq!(clock.seconds_until_level_up(), LEVEL_DURATION_SECONDS);

        let clock = clock.next_hand_after(120);
        assert_eq!(clock.level().level, 2);
        assert_eq!(clock.seconds_until_level_up(), 60);
    }

    #[test]
    fn long_hand_advances_only_one_level_and_resets_clock() {
        let clock = BlindClock::new().next_hand_after(600);

        assert_eq!(clock.level().level, 2);
        assert_eq!(clock.elapsed_in_level_seconds, 0);
    }

    #[test]
    fn blind_clock_stays_at_last_level() {
        let mut clock = BlindClock {
            current_level: 16,
            elapsed_in_level_seconds: 179,
        };

        clock = clock.next_hand_after(1);

        assert_eq!(clock.level().level, 16);
        assert_eq!(clock.elapsed_in_level_seconds, 0);
    }
}
