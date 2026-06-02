use std::collections::HashSet;

use crate::cards::{Card, deck};
use crate::hand_class::{HandClass, all_hand_classes};
use crate::hand_eval::evaluate_seven;
use crate::river::Combo;

#[derive(Debug, Clone)]
pub struct PostflopCombo {
    pub combo: Combo,
    pub class: HandClass,
    pub mask: u64,
}

#[derive(Debug, Clone)]
pub struct PostflopEquityReport {
    pub combo: PostflopCombo,
    pub equity: f64,
    pub win: f64,
    pub tie: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeParseError {
    InvalidToken,
}

impl PostflopCombo {
    pub fn label(&self) -> String {
        self.combo.label()
    }
}

pub fn parse_range(spec: &str) -> Result<Vec<HandClass>, RangeParseError> {
    let trimmed = spec.trim();
    if trimmed == "*" || trimmed.eq_ignore_ascii_case("all") {
        return Ok(all_hand_classes());
    }

    let mut classes = Vec::new();
    let mut seen = HashSet::new();
    for token in trimmed
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        for class in parse_range_token(token)? {
            if seen.insert(class) {
                classes.push(class);
            }
        }
    }

    Ok(classes)
}

pub fn postflop_combos(classes: &[HandClass], board: &[Card]) -> Vec<PostflopCombo> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let mut combos = Vec::new();
    let mut seen = HashSet::new();

    for &class in classes {
        for [first, second] in class.combos() {
            let Some(combo) = Combo::new(first, second) else {
                continue;
            };
            let mask = combo.mask();
            if mask & board_mask != 0 || !seen.insert(mask) {
                continue;
            }
            combos.push(PostflopCombo { combo, class, mask });
        }
    }

    combos
}

pub fn postflop_equity_reports(
    board: &[Card],
    hero_range: &[PostflopCombo],
    villain_range: &[PostflopCombo],
    max_runouts: usize,
) -> Vec<PostflopEquityReport> {
    let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());
    let mut reports: Vec<_> = hero_range
        .iter()
        .map(|hero| {
            let mut win = 0.0;
            let mut tie = 0.0;
            let mut total = 0.0;

            for villain in villain_range {
                if hero.mask & villain.mask != 0 {
                    continue;
                }

                let used_mask = board_mask | hero.mask | villain.mask;
                for completed_board in sampled_runouts(board, used_mask, max_runouts) {
                    let hero_value = evaluate_seven([
                        hero.combo.first,
                        hero.combo.second,
                        completed_board[0],
                        completed_board[1],
                        completed_board[2],
                        completed_board[3],
                        completed_board[4],
                    ]);
                    let villain_value = evaluate_seven([
                        villain.combo.first,
                        villain.combo.second,
                        completed_board[0],
                        completed_board[1],
                        completed_board[2],
                        completed_board[3],
                        completed_board[4],
                    ]);
                    match hero_value.cmp(&villain_value) {
                        std::cmp::Ordering::Greater => win += 1.0,
                        std::cmp::Ordering::Equal => tie += 1.0,
                        std::cmp::Ordering::Less => {}
                    }
                    total += 1.0;
                }
            }

            PostflopEquityReport {
                combo: hero.clone(),
                equity: if total == 0.0 {
                    0.0
                } else {
                    (win + tie * 0.5) / total
                },
                win,
                tie,
                total,
            }
        })
        .collect();

    reports.sort_by(|left, right| {
        right
            .equity
            .total_cmp(&left.equity)
            .then_with(|| left.combo.label().cmp(&right.combo.label()))
    });
    reports
}

fn parse_range_token(token: &str) -> Result<Vec<HandClass>, RangeParseError> {
    let plus = token.ends_with('+');
    let token = token.trim_end_matches('+');
    let chars: Vec<_> = token.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return Err(RangeParseError::InvalidToken);
    }

    let high = parse_rank(chars[0]).ok_or(RangeParseError::InvalidToken)?;
    let low = parse_rank(chars[1]).ok_or(RangeParseError::InvalidToken)?;
    let suitedness = chars.get(2).copied();
    if high < low {
        return Err(RangeParseError::InvalidToken);
    }

    if high == low {
        if suitedness.is_some() {
            return Err(RangeParseError::InvalidToken);
        }
        if plus {
            return Ok((low..=14)
                .rev()
                .map(|rank| HandClass::new(rank, rank, false))
                .collect());
        }
        return Ok(vec![HandClass::new(high, low, false)]);
    }

    let suited = match suitedness {
        Some('s') | Some('S') => Some(true),
        Some('o') | Some('O') => Some(false),
        None => None,
        _ => return Err(RangeParseError::InvalidToken),
    };
    let lows: Vec<_> = if plus {
        ((low + 1)..high).chain(std::iter::once(low)).collect()
    } else {
        vec![low]
    };

    let mut classes = Vec::new();
    for low in lows {
        match suited {
            Some(value) => classes.push(HandClass::new(high, low, value)),
            None => {
                classes.push(HandClass::new(high, low, true));
                classes.push(HandClass::new(high, low, false));
            }
        }
    }
    Ok(classes)
}

fn parse_rank(rank: char) -> Option<u8> {
    match rank.to_ascii_uppercase() {
        'A' => Some(14),
        'K' => Some(13),
        'Q' => Some(12),
        'J' => Some(11),
        'T' => Some(10),
        '9' => Some(9),
        '8' => Some(8),
        '7' => Some(7),
        '6' => Some(6),
        '5' => Some(5),
        '4' => Some(4),
        '3' => Some(3),
        '2' => Some(2),
        _ => None,
    }
}

fn sampled_runouts(board: &[Card], used_mask: u64, max_runouts: usize) -> Vec<[Card; 5]> {
    debug_assert!((3..=5).contains(&board.len()));
    if board.len() == 5 {
        return vec![[board[0], board[1], board[2], board[3], board[4]]];
    }

    let available: Vec<_> = deck()
        .into_iter()
        .filter(|card| used_mask & card.mask() == 0)
        .collect();
    let needed = 5 - board.len();
    let mut runouts = Vec::new();

    match needed {
        1 => {
            for &turn_or_river in &available {
                let mut completed = [Card(0); 5];
                completed[..board.len()].copy_from_slice(board);
                completed[board.len()] = turn_or_river;
                runouts.push(completed);
            }
        }
        2 => {
            for first_index in 0..available.len() {
                for second_index in (first_index + 1)..available.len() {
                    let mut completed = [Card(0); 5];
                    completed[..board.len()].copy_from_slice(board);
                    completed[board.len()] = available[first_index];
                    completed[board.len() + 1] = available[second_index];
                    runouts.push(completed);
                }
            }
        }
        _ => unreachable!("postflop board should have 3 to 5 cards"),
    }

    if max_runouts == 0 || runouts.len() <= max_runouts {
        return runouts;
    }

    let mut sampled = Vec::with_capacity(max_runouts);
    for iteration in 0..max_runouts {
        sampled.push(runouts[sampled_index(iteration, runouts.len())]);
    }
    sampled
}

fn sampled_index(iteration: usize, len: usize) -> usize {
    let mut value = iteration as u64 + 0x517c_c1b7_2722_0a95;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) as usize) % len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn parses_exact_and_plus_ranges() {
        assert_eq!(parse_range("AA,AKs,AKo").unwrap().len(), 3);
        assert_eq!(parse_range("TT+").unwrap().len(), 5);
        assert_eq!(parse_range("AJs+").unwrap().len(), 3);
    }

    #[test]
    fn postflop_combos_remove_board_cards() {
        let classes = parse_range("AA,AKs,AKo").unwrap();
        let board = [c(14, 0), c(13, 1), c(2, 2)];
        let combos = postflop_combos(&classes, &board);
        let board_mask = board.iter().fold(0_u64, |mask, card| mask | card.mask());

        assert!(combos.iter().all(|combo| combo.mask & board_mask == 0));
    }

    #[test]
    fn flop_equity_reports_are_sorted() {
        let board = [c(14, 0), c(13, 1), c(2, 2)];
        let oop = postflop_combos(&parse_range("AA,AKs,AKo").unwrap(), &board);
        let ip = postflop_combos(&parse_range("QQ,JJ,AQs").unwrap(), &board);
        let reports = postflop_equity_reports(&board, &oop, &ip, 16);

        assert!(!reports.is_empty());
        assert!(reports[0].equity >= reports[reports.len() - 1].equity);
    }
}
