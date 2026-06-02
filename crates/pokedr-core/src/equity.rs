use crate::cards::{Card, deck};
use crate::hand_class::HandClass;
use crate::hand_eval::evaluate_seven;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equity {
    pub win: f64,
    pub tie: f64,
    pub total: f64,
}

impl Equity {
    pub fn share(self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            (self.win + self.tie) / self.total
        }
    }
}

pub fn heads_up_equity_vs_range(
    hero: HandClass,
    villain_range: &[HandClass],
    max_boards_per_combo: usize,
) -> Equity {
    let mut equity = Equity {
        win: 0.0,
        tie: 0.0,
        total: 0.0,
    };

    for hero_combo in hero.combos() {
        let hero_mask = hero_combo[0].mask() | hero_combo[1].mask();

        for villain in villain_range {
            for villain_combo in villain.combos() {
                let villain_mask = villain_combo[0].mask() | villain_combo[1].mask();
                if hero_mask & villain_mask != 0 {
                    continue;
                }

                let combo_equity =
                    heads_up_combo_equity(hero_combo, villain_combo, max_boards_per_combo);
                equity.win += combo_equity.win;
                equity.tie += combo_equity.tie;
                equity.total += combo_equity.total;
            }
        }
    }

    equity
}

pub fn three_way_equity_vs_ranges(
    hero: HandClass,
    first_range: &[HandClass],
    second_range: &[HandClass],
    max_boards_per_combo: usize,
) -> Equity {
    let mut equity = Equity {
        win: 0.0,
        tie: 0.0,
        total: 0.0,
    };

    for hero_combo in hero.combos() {
        let hero_mask = hero_combo[0].mask() | hero_combo[1].mask();

        for first in first_range {
            for first_combo in first.combos() {
                let first_mask = first_combo[0].mask() | first_combo[1].mask();
                if hero_mask & first_mask != 0 {
                    continue;
                }

                for second in second_range {
                    for second_combo in second.combos() {
                        let second_mask = second_combo[0].mask() | second_combo[1].mask();
                        if (hero_mask | first_mask) & second_mask != 0 {
                            continue;
                        }

                        let combo_equity = three_way_combo_equity(
                            hero_combo,
                            first_combo,
                            second_combo,
                            max_boards_per_combo,
                        );
                        equity.win += combo_equity.win;
                        equity.tie += combo_equity.tie;
                        equity.total += combo_equity.total;
                    }
                }
            }
        }
    }

    equity
}

fn heads_up_combo_equity(hero: [Card; 2], villain: [Card; 2], max_boards: usize) -> Equity {
    let dead_mask = hero[0].mask() | hero[1].mask() | villain[0].mask() | villain[1].mask();
    let available = available_cards(dead_mask);
    let mut equity = Equity {
        win: 0.0,
        tie: 0.0,
        total: 0.0,
    };

    if max_boards > 0 {
        let mut rng = SplitMix64::new(dead_mask ^ 0x9e37_79b9_7f4a_7c15);
        for _ in 0..max_boards {
            let board = sample_board(&available, &mut rng);
            let hero_value = evaluate_seven([
                hero[0], hero[1], board[0], board[1], board[2], board[3], board[4],
            ]);
            let villain_value = evaluate_seven([
                villain[0], villain[1], board[0], board[1], board[2], board[3], board[4],
            ]);

            if hero_value > villain_value {
                equity.win += 1.0;
            } else if hero_value == villain_value {
                equity.tie += 0.5;
            }
            equity.total += 1.0;
        }

        return equity;
    }

    for a in 0..(available.len() - 4) {
        for b in (a + 1)..(available.len() - 3) {
            for c in (b + 1)..(available.len() - 2) {
                for d in (c + 1)..(available.len() - 1) {
                    for e in (d + 1)..available.len() {
                        let board = [
                            available[a],
                            available[b],
                            available[c],
                            available[d],
                            available[e],
                        ];
                        let hero_value = evaluate_seven([
                            hero[0], hero[1], board[0], board[1], board[2], board[3], board[4],
                        ]);
                        let villain_value = evaluate_seven([
                            villain[0], villain[1], board[0], board[1], board[2], board[3],
                            board[4],
                        ]);

                        if hero_value > villain_value {
                            equity.win += 1.0;
                        } else if hero_value == villain_value {
                            equity.tie += 0.5;
                        }
                        equity.total += 1.0;
                    }
                }
            }
        }
    }

    equity
}

fn three_way_combo_equity(
    hero: [Card; 2],
    first: [Card; 2],
    second: [Card; 2],
    max_boards: usize,
) -> Equity {
    let dead_mask = hero[0].mask()
        | hero[1].mask()
        | first[0].mask()
        | first[1].mask()
        | second[0].mask()
        | second[1].mask();
    let available = available_cards(dead_mask);
    let mut equity = Equity {
        win: 0.0,
        tie: 0.0,
        total: 0.0,
    };

    if max_boards > 0 {
        let mut rng = SplitMix64::new(dead_mask ^ 0xbf58_476d_1ce4_e5b9);
        for _ in 0..max_boards {
            let board = sample_board(&available, &mut rng);
            let hero_value = evaluate_seven([
                hero[0], hero[1], board[0], board[1], board[2], board[3], board[4],
            ]);
            let first_value = evaluate_seven([
                first[0], first[1], board[0], board[1], board[2], board[3], board[4],
            ]);
            let second_value = evaluate_seven([
                second[0], second[1], board[0], board[1], board[2], board[3], board[4],
            ]);
            let best = hero_value.max(first_value).max(second_value);
            let winners = [hero_value, first_value, second_value]
                .iter()
                .filter(|&&value| value == best)
                .count();

            if hero_value == best {
                if winners == 1 {
                    equity.win += 1.0;
                } else {
                    equity.tie += 1.0 / winners as f64;
                }
            }
            equity.total += 1.0;
        }

        return equity;
    }

    for a in 0..(available.len() - 4) {
        for b in (a + 1)..(available.len() - 3) {
            for c in (b + 1)..(available.len() - 2) {
                for d in (c + 1)..(available.len() - 1) {
                    for e in (d + 1)..available.len() {
                        let board = [
                            available[a],
                            available[b],
                            available[c],
                            available[d],
                            available[e],
                        ];
                        let hero_value = evaluate_seven([
                            hero[0], hero[1], board[0], board[1], board[2], board[3], board[4],
                        ]);
                        let first_value = evaluate_seven([
                            first[0], first[1], board[0], board[1], board[2], board[3], board[4],
                        ]);
                        let second_value = evaluate_seven([
                            second[0], second[1], board[0], board[1], board[2], board[3], board[4],
                        ]);
                        let best = hero_value.max(first_value).max(second_value);
                        let winners = [hero_value, first_value, second_value]
                            .iter()
                            .filter(|&&value| value == best)
                            .count();

                        if hero_value == best {
                            if winners == 1 {
                                equity.win += 1.0;
                            } else {
                                equity.tie += 1.0 / winners as f64;
                            }
                        }
                        equity.total += 1.0;
                    }
                }
            }
        }
    }

    equity
}

fn available_cards(dead_mask: u64) -> Vec<Card> {
    deck()
        .into_iter()
        .filter(|card| dead_mask & card.mask() == 0)
        .collect()
}

fn sample_board(available: &[Card], rng: &mut SplitMix64) -> [Card; 5] {
    let mut indexes = [usize::MAX; 5];
    let mut cards = [available[0]; 5];

    for index in 0..5 {
        loop {
            let candidate = rng.next_usize(available.len());
            if !indexes[..index].contains(&candidate) {
                indexes[index] = candidate;
                cards[index] = available[candidate];
                break;
            }
        }
    }

    cards
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_usize(&mut self, upper_bound: usize) -> usize {
        (self.next_u64() as usize) % upper_bound
    }
}
